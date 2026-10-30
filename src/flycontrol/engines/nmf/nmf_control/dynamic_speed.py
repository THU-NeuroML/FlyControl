"""Within-episode velocity tracking; isolated EXP04 dynamic-speed protocol v1.

The action at decision k sees c[k]. Its interval reward uses c[k]. The returned
observation exposes c[k+1]. Commands are independent of action/physics outcomes.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np

from .env import TerrainLocomotionEnv
from .tasks import _wrap_pi, speed_control_reward, speed_target_observation


def command_schedule(rng, duration=12.0, dt=.01, kind=None, spec=None):
    """Return a zero-order-held decision command and an auditable full schedule."""
    n = int(round(duration / dt))
    t = np.arange(n + 1) * dt
    spec = dict(spec or {})
    kind = kind or str(rng.choice(['constant', 'steps', 'ramps'], p=[.2, .4, .4]))
    knots = []
    if kind == 'fixed_steps':
        times = np.asarray(spec['times'], float)
        values = np.asarray(spec['values'], float)
        if len(times) != len(values) or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError('fixed_steps requires increasing times starting at zero')
        c = values[np.clip(np.searchsorted(times, t + 1e-9, side='right') - 1, 0, len(values)-1)]
        knots = list(zip(times.tolist(), values.tolist()))
    elif kind == 'fixed_ramps':
        times = np.asarray(spec['times'], float)
        values = np.asarray(spec['values'], float)
        if len(times) != len(values) or times[0] != 0 or np.any(np.diff(times) <= 0):
            raise ValueError('fixed_ramps requires increasing times starting at zero')
        c = np.interp(t, times, values)
        knots = list(zip(times.tolist(), values.tolist()))
    elif kind == 'sine':
        c = 12 + 4 * np.sin(2 * np.pi * float(spec.get('frequency_hz', .25)) * t)
    elif kind == 'constant':
        v = float(spec.get('value', rng.uniform(8, 16)))
        c = np.full(n + 1, v)
        knots = [(0., v)]
    elif kind in ('steps', 'ramps'):
        # Alternating independently sampled low/high targets guarantee exposure
        # to both signs; they do not expose future commands to the policy.
        high = bool(rng.integers(2))
        v = float(rng.uniform(13, 16) if high else rng.uniform(8, 11))
        c = np.full(n + 1, v)
        k = int(round(1.0 / dt))
        knots = [(0., v), (k * dt, v)]
        while k < n:
            high = not high
            nxt = float(rng.uniform(13, 16) if high else rng.uniform(8, 11))
            if kind == 'steps':
                c[k:] = nxt
                knots.append((k * dt, nxt))
                k += int(round(rng.uniform(1., 2.) / dt))
            else:
                rate = float(rng.uniform(2., 8.))
                count = max(1, int(np.ceil(abs(nxt-v) / rate / dt)))
                end = min(n, k + count)
                c[k:end+1] = v + (nxt-v) * np.arange(end-k+1) / count
                knots.extend([(k * dt, v), (end * dt, float(c[end]))])
                c[end+1:] = nxt
                k = end + int(round(rng.uniform(.6, 1.) / dt))
            v = nxt
        if not (np.any(np.diff(c) > 1e-8) and np.any(np.diff(c) < -1e-8)):
            raise ValueError('dynamic episode too short for both directions')
    else:
        raise ValueError(f'unknown command kind: {kind}')
    if not np.all(np.isfinite(c)) or c.min() < 8-1e-8 or c.max() > 16+1e-8:
        raise ValueError('speed commands must stay in [8,16] mm/s')
    meta = dict(kind=kind, duration_s=duration, decision_dt_s=dt, knots=knots,
                command_sha256=hashlib.sha256(c.astype('<f8').tobytes()).hexdigest(),
                increases=int(np.sum(np.diff(c) > 1e-8)),
                decreases=int(np.sum(np.diff(c) < -1e-8)))
    return c, meta


class DynamicSpeedEnv(TerrainLocomotionEnv):
    """Same body/CPG, 51D observation and 2D action as the matched Adult task.

    Speed is interval displacement projected onto the midpoint body heading.
    Initial-heading speed remains a separate diagnostic. The formerly unused
    fifth task channel now reports signed heading error / pi for all models.
    """
    def __init__(self, *, dynamic_command_seed=40000, dynamic_heading_weight=1.,
                 dynamic_audit_dir=None, **kwargs):
        if kwargs.get('task') != 'speed_control' or kwargs.get('observation_mode') != 'proprio_task':
            raise ValueError('requires speed_control / proprio_task')
        if kwargs.get('observation_delay', 0) or kwargs.get('action_delay', 0):
            raise ValueError('v1 does not support delay')
        if np.any(np.asarray(kwargs.get('observation_noise_std', 0)) != 0):
            raise ValueError('v1 does not support observation noise')
        self.dynamic_command_seed = int(dynamic_command_seed)
        self.dynamic_heading_weight = float(dynamic_heading_weight)
        if self.dynamic_heading_weight < 0:
            raise ValueError('heading weight must be nonnegative')
        self._command_rng = None
        self._dynamic_filtered_v = 0.
        self._audit_dir = None if dynamic_audit_dir is None else Path(dynamic_audit_dir)
        self._episode_serial = 0
        super().__init__(**kwargs)

    def _audit(self, payload):
        if self._audit_dir is not None:
            self._audit_dir.mkdir(parents=True, exist_ok=True)
            with (self._audit_dir/f'worker_{os.getpid()}.jsonl').open('a') as f:
                f.write(json.dumps(dict(episode=self._episode_serial, **payload))+'\n')

    def _audit_outcome(self, info, terminated, truncated):
        self._audit(dict(event='end',elapsed_s=info.get('elapsed_s'),
            terminated=bool(terminated),truncated=bool(truncated),
            flip=bool(info.get('flip')),physics_error=bool(info.get('physics_error')),
            nan_detected=bool(info.get('nan_detected')),
            command_kind=self._command_metadata['kind'],decisions=self._decision_index,
            increases_executed=int(np.sum(np.diff(self._commands[:self._decision_index])>1e-8)),
            decreases_executed=int(np.sum(np.diff(self._commands[:self._decision_index])<-1e-8))))

    def _task_obs(self, raw_obs):
        task, info = speed_target_observation(
            self._target_speed_mm_s, self.speed_target_scale_mm_s,
            self._dynamic_filtered_v, self._last_heading_rate,
            self.speed_heading_rate_scale_rad_s)
        err = _wrap_pi(self._initial_heading_angle - self._raw_heading_angle(raw_obs))
        task[4] = err / np.pi
        info.update(heading_error_rad=err, heading_error_abs_rad=abs(err))
        return task, info

    def reset(self, *, seed=None, options=None):
        if seed is not None or self._command_rng is None:
            self._command_rng = np.random.default_rng(np.random.SeedSequence(
                [self.dynamic_command_seed, 0 if seed is None else int(seed)]))
        options = dict(options or {})
        override_seed = options.pop('command_seed', None)
        rng = self._command_rng if override_seed is None else np.random.default_rng(int(override_seed))
        kind = options.pop('command_kind', None)
        spec = options.pop('command_spec', None)
        self._commands, self._command_metadata = command_schedule(
            rng, self.max_time, self.timestep * self.decision_interval, kind, spec)
        self._decision_index = 0
        self._episode_serial += 1
        self._dynamic_filtered_v = 0.
        options['target_speed_mm_s'] = float(self._commands[0])
        obs, info = super().reset(seed=seed, options=options)
        info['dynamic_command'] = self._command_metadata
        self._audit(dict(event='start',reset_seed=seed,physics_seed=options.get('physics_seed'),
            command=self._command_metadata,commands_mm_s=self._commands.tolist()))
        return obs, info

    def step(self, action):
        applied = float(self._target_speed_mm_s)
        xy_before = np.asarray(self._last_raw_obs['fly'][0, :2], float).copy()
        heading_before = self._raw_heading_angle(self._last_raw_obs)
        t_before = self._physics_steps * self.timestep
        obs, _, terminated, truncated, info = super().step(action)
        if info.get('physics_error') or info.get('nan_detected'):
            info.update(target_applied_mm_s=applied, dynamic_task_failure=True)
            self._audit_outcome(info,terminated,truncated)
            return obs, -self.flip_penalty, terminated, truncated, info
        dt = self._physics_steps * self.timestep - t_before
        if dt <= 0:
            raise RuntimeError('no elapsed physics time')
        xy = np.asarray(self._last_raw_obs['fly'][0, :2], float)
        heading = self._raw_heading_angle(self._last_raw_obs)
        middle = heading_before + .5 * _wrap_pi(heading-heading_before)
        delta = xy - xy_before
        body_v = float(delta @ np.array([np.cos(middle), np.sin(middle)]) / dt)
        lateral_v = float(delta @ np.array([-np.sin(middle), np.cos(middle)]) / dt)
        self._dynamic_filtered_v = ((1-self.speed_filter_alpha) * self._dynamic_filtered_v
                                    + self.speed_filter_alpha * body_v)
        h_error = _wrap_pi(self._initial_heading_angle - heading)
        speed_reward = speed_control_reward(
            self._dynamic_filtered_v, 0., applied,
            speed_error_weight=self.speed_error_weight, heading_rate_weight=0.,
            huber_delta_mm_s=self.speed_huber_delta_mm_s)
        yaw_reward = -self.speed_heading_rate_weight * abs(self._last_heading_rate)
        heading_reward = -self.dynamic_heading_weight * abs(h_error)
        failure_reward = -self.flip_penalty if info.get('flip') else 0.
        # The 12 s task must not make early death cheaper than accumulated
        # negative tracking costs. A bounded nonnegative per-step score removes
        # that shortcut while retaining the old Huber cost in its linear range.
        tracking_reward = float(np.clip(1. + (speed_reward+yaw_reward+heading_reward)/20., 0., 1.))
        reward = tracking_reward + failure_reward
        # Diagnostics refer to the interval just executed, never the next target.
        info.update(target_speed_mm_s=applied, target_applied_mm_s=applied,
                    current_speed_mm_s=self._dynamic_filtered_v,
                    body_forward_velocity_mm_s=body_v,
                    filtered_body_forward_velocity_mm_s=self._dynamic_filtered_v,
                    body_lateral_velocity_mm_s=lateral_v,
                    speed_error_mm_s=applied-self._dynamic_filtered_v,
                    speed_error_abs_mm_s=abs(applied-self._dynamic_filtered_v),
                    heading_error_rad=h_error, heading_error_abs_rad=abs(h_error),
                    reward_speed=speed_reward, reward_yaw=yaw_reward,
                    reward_heading=heading_reward, reward_failure=failure_reward,
                    reward_tracking=tracking_reward,
                    interval_start_s=t_before, interval_duration_s=dt,
                    command_kind=self._command_metadata['kind'],
                    dynamic_task_failure=bool(info.get('flip')))
        self._decision_index += 1
        self._target_speed_mm_s = float(self._commands[min(self._decision_index, len(self._commands)-1)])
        obs = self._noisy_obs(self._last_raw_obs)
        info['target_next_mm_s'] = self._target_speed_mm_s
        if terminated or truncated:
            self._audit_outcome(info,terminated,truncated)
        return obs, float(reward), terminated, truncated, info
