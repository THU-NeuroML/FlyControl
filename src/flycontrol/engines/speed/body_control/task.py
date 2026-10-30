"""Continuous commands and matched physical time; frozen speed body and CPG."""

from .context import ROOT, CONFIG
import copy, hashlib, json
from collections import deque
from nmf_control.tasks import speed_control_reward, speed_target_observation, _wrap_pi
import numpy as np
import gymnasium as gym
from nmf_control.dynamic_speed import DynamicSpeedEnv
from .commands import schedule, PLAN

OBS_DIM, ACTION_DIM = 51, 2


def array_hash(*xs):
    h = hashlib.sha256()
    for x in xs:
        a = np.ascontiguousarray(x)
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class ContinuousSpeedTask(DynamicSpeedEnv):
    def __init__(
        self, clock_ms=40, rank=0, log_path=None, fixed_case=None, fast_observation=True
    ):
        self.cycle_window = deque(maxlen=10)
        self.cycle_speed = 0.0
        self.cycle_yaw = 0.0
        self.previous_potential = 0.0
        self.repeat = int(clock_ms) // 10
        self.fast_observation = bool(fast_observation)
        assert self.repeat in (1, 4)
        cfg = json.loads((CONFIG / "environment.json").read_text())
        cfg.pop("dynamic_audit_dir", None)
        cfg.update(max_time=PLAN["horizon_s"])
        super().__init__(**cfg)
        self.rank = rank
        self.log_path = log_path
        self.fixed_case = fixed_case
        self.command_stream = None
        self.phase_stream = None
        self.episode_index = 0
        self.restored = False
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
        self.last_micro_infos = []

    def _public_obs(self, obs):
        obs = obs.copy()
        obs[:18] /= 2.0
        obs[18:24] /= 10.0
        assert obs.shape == (51,) and np.isfinite(obs).all()
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _log(self, row):
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")

    def reset(self, *, seed=None, options=None):
        self.cycle_window.clear()
        self.cycle_speed = 0.0
        self.cycle_yaw = 0.0
        if self.fixed_case is None:
            if (seed is not None and not self.restored) or self.command_stream is None:
                if seed is None:
                    raise ValueError("Explicit initial seed required")
                self.command_stream = np.random.default_rng(
                    np.random.SeedSequence([seed, self.rank, 400000])
                )
                self.phase_stream = np.random.default_rng(
                    np.random.SeedSequence([seed, self.rank, 500000])
                )
                self.episode_index = 0
            self.restored = False
            case = dict(
                command_seed=int(self.command_stream.integers(1, 2**31 - 1)),
                physics_seed=int(self.phase_stream.integers(1, 2**31 - 1)),
            )
        else:
            case = dict(self.fixed_case)
        c, meta = schedule(case["command_seed"], case.get("kind"))
        # Parent reset constructs the exact archived body; replace only the
        # exogenous schedule after its initial constant target has been set.
        obs, info = super().reset(
            seed=case["physics_seed"],
            options=dict(
                physics_seed=case["physics_seed"],
                command_kind="constant",
                command_spec=dict(value=float(c[0])),
            ),
        )
        self._commands = c
        self._command_metadata = meta
        self._target_speed_mm_s = float(c[0])
        self._decision_index = 0
        self.case = case
        self.command_meta = meta
        self.episode_index += 1
        self.episode_return = 0.0
        obs = self._public_obs(self._noisy_obs(self._last_raw_obs))
        self.initial_hash = array_hash(
            self._sim.physics.data.qpos,
            self._sim.physics.data.qvel,
            self._controller.cpg.curr_phases,
            self._controller.cpg.curr_magnitudes,
        )
        info.update(
            case=case,
            command=meta,
            initial_state_sha256=self.initial_hash,
            initial_observation_sha256=array_hash(obs),
        )
        self._log(
            dict(
                event="start",
                episode=self.episode_index,
                case=case,
                command=meta,
                initial_state_sha256=self.initial_hash,
            )
        )
        if self.fast_observation:
            from .fast_observation import install

            install(self)
        self.previous_potential = self._potential(
            self._target_speed_mm_s, self.cycle_speed
        )
        return obs, info

    def _task_obs(self, raw_obs):
        task, info = speed_target_observation(
            self._target_speed_mm_s,
            self.speed_target_scale_mm_s,
            self.cycle_speed,
            self.cycle_yaw,
            self.speed_heading_rate_scale_rad_s,
        )
        error = _wrap_pi(self._initial_heading_angle - self._raw_heading_angle(raw_obs))
        task[4] = error / np.pi
        info.update(heading_error_rad=error, heading_error_abs_rad=abs(error))
        return task, info

    def _potential(self, target, speed):
        cfg = PLAN["potential"]
        return -cfg["coefficient"] * abs(target - speed) / cfg["speed_scale_mm_s"]

    def _aligned_reward(self, info, failed, terminal):
        old_tracking = float(info.get("reward_tracking", 0.0))
        old_speed = float(info.get("filtered_body_forward_velocity_mm_s", 0.0))
        velocity = float(info.get("body_forward_velocity_mm_s", 0.0))
        yaw = float(info.get("heading_rate_rad_s", 0.0))
        dt = float(info.get("interval_duration_s", PLAN["micro_dt_s"]))
        if not np.isfinite([velocity, yaw, dt]).all():
            assert failed
            velocity = 0.0
            yaw = 0.0
            dt = PLAN["micro_dt_s"]
        self.cycle_window.append((velocity, yaw, dt))
        x = np.asarray(self.cycle_window)
        self.cycle_speed = float(np.average(x[:, 0], weights=x[:, 2]))
        self.cycle_yaw = float(np.average(x[:, 1], weights=x[:, 2]))
        target = float(info.get("target_applied_mm_s", self._target_speed_mm_s))
        speed_reward = speed_control_reward(
            self.cycle_speed,
            0.0,
            target,
            speed_error_weight=self.speed_error_weight,
            heading_rate_weight=0.0,
            huber_delta_mm_s=self.speed_huber_delta_mm_s,
        )
        yaw_reward = -self.speed_heading_rate_weight * abs(self.cycle_yaw)
        heading = float(info.get("heading_error_rad", 0.0))
        if not np.isfinite(heading):
            assert failed
            heading = 0.0
        heading_reward = -self.dynamic_heading_weight * abs(heading)
        base = float(
            np.clip(1 + (speed_reward + yaw_reward + heading_reward) / 20, 0, 1)
        )
        failure_reward = -self.flip_penalty if failed else 0.0
        potential = (
            0.0
            if terminal
            else self._potential(self._target_speed_mm_s, self.cycle_speed)
        )
        shaping = PLAN["micro_gamma"] * potential - self.previous_potential
        self.previous_potential = potential
        info.update(
            reward_legacy_tracking=old_tracking,
            legacy_filtered_speed_mm_s=old_speed,
            reward_tracking=base,
            reward_speed=speed_reward,
            reward_yaw=yaw_reward,
            reward_heading=heading_reward,
            reward_failure=failure_reward,
            reward_shaping=shaping,
            current_speed_mm_s=self.cycle_speed,
            filtered_body_forward_velocity_mm_s=self.cycle_speed,
            gait_averaged_yaw_rad_s=self.cycle_yaw,
            speed_error_mm_s=target - self.cycle_speed,
            speed_error_abs_mm_s=abs(target - self.cycle_speed),
        )
        return base + failure_reward + shaping

    def step(self, raw_action):
        raw_action = np.asarray(raw_action, dtype=np.float32)
        assert raw_action.shape == (2,) and np.isfinite(raw_action).all()
        physical = np.tanh(raw_action)
        reward = 0.0
        self.last_micro_infos = []
        for k in range(self.repeat):
            seen = float(self._target_speed_mm_s)
            obs, r, term, trunc, info = super().step(physical)
            assert abs(info["target_applied_mm_s"] - seen) < 1e-9
            failed = bool(
                info.get("flip")
                or info.get("physics_error")
                or info.get("nan_detected")
            )
            finite = all(np.isfinite(v).all() for v in self._last_raw_obs.values())
            failed = failed or not finite
            # This study defines a finite 12 s task. Its deadline is terminal,
            # so PPO must not bootstrap beyond the zero-terminal potential.
            deadline = bool(trunc)
            term = bool(term or failed or deadline)
            trunc = False
            info.update(finite_horizon_terminal=True, deadline_reached=deadline)
            r = self._aligned_reward(info, failed, term or trunc)
            obs = self._noisy_obs(self._last_raw_obs)
            row = dict(
                info,
                micro_reward=float(r),
                physical_failure=failed,
                physical_action=physical.tolist(),
                gaussian_action=raw_action.tolist(),
            )
            self.last_micro_infos.append(row)
            reward += PLAN["micro_gamma"] ** k * float(r)
            self.episode_return += float(r)
            if term or trunc:
                break
        info = dict(
            info,
            physical_failure=failed,
            action_hold_micro_steps=k + 1,
            macro_reward=reward,
            episode_return=self.episode_return,
        )
        if term or trunc:
            executed = self._commands[: self._decision_index]
            self._log(
                dict(
                    event="end",
                    episode=self.episode_index,
                    case=self.case,
                    kind=self.command_meta["kind"],
                    elapsed_s=info["elapsed_s"],
                    failure=failed,
                    episode_return=self.episode_return,
                    increases_executed=int(np.sum(np.diff(executed) > 1e-8)),
                    decreases_executed=int(np.sum(np.diff(executed) < -1e-8)),
                )
            )
        safe_obs = (
            np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0) if failed else obs
        )
        return self._public_obs(safe_obs), float(reward), term, trunc, info

    def get_state(self):
        return dict(
            command_stream=(
                copy.deepcopy(self.command_stream.bit_generator.state)
                if self.command_stream
                else None
            ),
            phase_stream=(
                copy.deepcopy(self.phase_stream.bit_generator.state)
                if self.phase_stream
                else None
            ),
            episode_index=self.episode_index,
        )

    def set_state(self, state):
        self.command_stream = np.random.default_rng()
        self.phase_stream = np.random.default_rng()
        self.command_stream.bit_generator.state = state["command_stream"]
        self.phase_stream.bit_generator.state = state["phase_stream"]
        self.episode_index = state["episode_index"]
        self.restored = True


def make_env(rank, n_envs, clock_ms, run=None):
    return ContinuousSpeedTask(
        clock_ms=clock_ms,
        rank=rank,
        log_path=ROOT / "runs" / run / f"env_{rank:02d}.jsonl" if run else None,
    )
