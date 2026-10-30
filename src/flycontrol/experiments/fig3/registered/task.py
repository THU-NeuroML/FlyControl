"""Fresh, paired CPG command tasks in the archived NeuromechFly body.

The environment alone squashes raw Gaussian actions. No teacher or trained
policy supplies the CPG. Force values use the original FlyGym model units.
"""

from .common import DATA, PROTOCOL, REPO, ROOT
import copy, hashlib, json
from collections import deque
import numpy as np
import gymnasium as gym
from nmf_control.env import TerrainLocomotionEnv

OBS_DIM, ACTION_DIM = 64, 2
ANGLES = (-90.0, -60.0, -30.0, 30.0, 60.0, 90.0)


def wrap(x):
    return float((x + np.pi) % (2 * np.pi) - np.pi)


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        x = np.ascontiguousarray(a)
        h.update(str(x.shape).encode())
        h.update(str(x.dtype).encode())
        h.update(x.tobytes())
    return h.hexdigest()


class PushTurnTask(TerrainLocomotionEnv):
    def __init__(
        self,
        family="push",
        objective="aligned",
        rank=0,
        n_envs=1,
        fixed_case=None,
        log_path=None,
        horizon=2.0,
    ):
        super().__init__(
            terrains=["flat"],
            terrain_seeds=[0],
            timestep=1e-4,
            decision_interval=100,
            max_time=horizon,
            action_mode="command2",
            task="heading_hold" if family == "push" else "reorientation",
            observation_mode="proprio_task",
            reorientation_targets_rad=[np.pi / 6.0],
            flip_penalty=1000.0,
        )
        assert family in ("push", "angle") and objective in ("legacy", "aligned")
        self.family, self.objective = family, objective
        self.rank, self.n_envs = rank, n_envs
        self.fixed_case, self.log_path = fixed_case, log_path
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        # Inherited reset/step observations are discarded; public observations
        # use the new fixed-scale encoding below. Sensor noise is disabled.
        self._default_observation_noise_std = np.zeros(OBS_DIM, dtype=np.float32)
        self.observation_noise_std = self._default_observation_noise_std.copy()
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (ACTION_DIM,), np.float32)
        self.stream = None
        self.base_seed = None
        self.episode_index = 0
        self.restored = False

    def reset(self, *, seed=None, options=None):
        if self.fixed_case is not None:
            if seed is None:
                raise ValueError("Evaluation requires explicit phase seed")
            phase_seed = int(seed)
            case = dict(self.fixed_case)
        else:
            if seed is not None and not self.restored:
                self.base_seed = int(seed)
                self.stream = np.random.default_rng(
                    np.random.SeedSequence([seed, self.rank, 20261002])
                )
                self.episode_index = 0
            self.restored = False
            if self.stream is None:
                raise ValueError("Missing training seed")
            phase_seed = (
                int(
                    self.stream.integers(
                        10000000 // self.n_envs, (2**31 - 1) // self.n_envs
                    )
                )
                * self.n_envs
                + self.rank
            )
            case = (
                dict(
                    force=float(self.stream.choice([0.0, 10.0, 20.0])),
                    side=float(self.stream.choice([-1.0, 1.0])),
                    onset=float(self.stream.uniform(0.5, 0.9)),
                )
                if self.family == "push"
                else dict(angle_deg=float(self.stream.choice(ANGLES)), onset=0.3)
            )
        self.case = case
        self.phase_seed = phase_seed
        self.elapsed = 0.0
        self.onset = float(case.get("onset", 0.75 if self.family == "push" else 0.3))
        self.goal_delta = np.deg2rad(float(case.get("angle_deg", 0.0)))
        self.previous_action = np.zeros(2)
        self.window = deque(maxlen=20)
        perturbation = (
            dict(
                type="side_push",
                force=case["force"],
                start_time_s=self.onset,
                duration_s=0.04,
                direction="lateral",
            )
            if self.family == "push"
            else dict(type="none")
        )
        _, info = super().reset(
            seed=phase_seed,
            options=dict(physics_seed=phase_seed, perturbation=perturbation),
        )
        self._perturbation_side = float(case.get("side", 1.0))
        self.start = self._last_raw_obs["fly"][0].copy()
        self.initial_heading = self._initial_heading_angle
        self.heading_integral = 0.0
        self.postpush_auc = 0.0
        self.max_lateral = 0.0
        self.total_rotation = 0.0
        self.previous_heading = self.initial_heading
        self.episode_return = 0.0
        self.first_stable_s = None
        self._update_target()
        self.initial_hash = array_hash(
            self._sim.physics.data.qpos,
            self._sim.physics.data.qvel,
            self._controller.cpg.curr_phases,
            self._controller.cpg.curr_magnitudes,
        )
        obs = self.observation()
        info.update(
            phase_seed=phase_seed,
            case=case,
            initial_state_sha256=self.initial_hash,
            initial_observation_sha256=array_hash(obs),
        )
        self.episode_index += 1
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(
                    json.dumps(
                        dict(episode_index=self.episode_index, **info), default=str
                    )
                    + "\n"
                )
        return obs, info

    def _update_target(self):
        active = self.family == "push" or self.elapsed >= self.onset - 1e-9
        self._target_angle_rad = self.goal_delta if active else 0.0
        self._target_heading_rad = wrap(self.initial_heading + self._target_angle_rad)
        self.command_active = active

    def observation(self):
        raw = self._last_raw_obs
        if not all(np.isfinite(v).all() for v in raw.values()):
            return np.zeros(OBS_DIM, dtype=np.float32)
        base = self._controller.proprio_obs(raw).copy()
        base[:18] /= 2.0
        base[18:24] /= 10.0
        e = wrap(self._target_heading_rad - self._raw_heading_angle(raw))
        obs = np.r_[
            base,
            raw["fly"][1] / 20.0,
            raw["fly"][3] / 50.0,
            raw["cardinal_vectors"][2],
            np.sin(e),
            np.cos(e),
            self._target_angle_rad / np.pi,
            self.elapsed / self.max_time,
            float(self.command_active),
            self.previous_action,
            (
                float(np.mean([x[3] for x in list(self.window)[-10:]]))
                if self.window
                else 0.0
            )
            / 5.0,
            (raw["fly"][0, 1] - self.start[1]) / 2.0,
        ].astype(np.float32)
        assert obs.shape == (OBS_DIM,) and np.isfinite(obs).all()
        return np.clip(obs, -10.0, 10.0)

    def step(self, gaussian_action):
        x = np.asarray(gaussian_action, dtype=np.float32)
        if x.shape != (2,) or not np.isfinite(x).all():
            raise ValueError("Invalid action")
        action = np.tanh(x)
        before = self._physics_steps
        _, old_reward, physical_term, physical_timeout, info = super().step(action)
        dt = (self._physics_steps - before) * self.timestep
        self.elapsed = self._physics_steps * self.timestep
        self._update_target()
        raw = self._last_raw_obs
        finite = all(np.isfinite(v).all() for v in raw.values())
        numerical = bool(
            info.get("physics_error") or info.get("nan_detected") or not finite
        )
        failure = bool(physical_term or numerical)
        heading = self._raw_heading_angle(raw) if finite else self.previous_heading
        self.total_rotation += wrap(heading - self.previous_heading)
        self.previous_heading = heading
        e = wrap(self._target_heading_rad - heading)
        up = float(raw["cardinal_vectors"][2, 2]) if finite else -1.0
        speed = float(raw["fly"][1] @ raw["fly_orientation"]) if finite else 0.0
        lateral = (
            float(raw["fly"][0, 1] - self.start[1]) if finite else self.max_lateral
        )
        self.max_lateral = max(self.max_lateral, abs(lateral))
        rate = float(self._last_heading_rate) if finite else 0.0
        self.window.append((abs(e), up, speed, rate))
        w = np.asarray(self.window)
        stable = bool(
            len(w) == 20
            and np.all(w[:, 0] <= np.deg2rad(10.0))
            and np.all(w[:, 1] >= 0.8)
            and w[:, 2].mean() >= 3.0
        )
        after_event = (
            self.elapsed
            >= self.onset + (0.04 if self.family == "push" else 0.0) + 0.2 - 1e-8
        )
        stable = bool(stable and after_event and not failure)
        if stable and self.first_stable_s is None:
            self.first_stable_s = self.elapsed
        self.heading_integral += abs(e) * dt
        if self.family == "push":
            left, right = self.onset + 0.04, self.onset + 0.54
            overlap = max(0.0, min(self.elapsed, right) - max(self.elapsed - dt, left))
            self.postpush_auc += abs(e) * overlap
            if failure:
                self.postpush_auc += np.pi * max(0.0, right - max(self.elapsed, left))
        deadline = bool(physical_timeout or self.elapsed >= self.max_time - 1e-8)
        terminated = bool(failure or deadline)
        success = bool(deadline and stable and not failure)
        terms = dict(
            heading=2.0 * np.cos(e) * dt,
            speed=0.5 * float(np.clip(w[-10:, 2].mean() / 10.0, -1.0, 1.0)) * dt,
            settling=-0.25
            * min((w[-10:, 3].mean() / 5.0) ** 2, 4.0)
            * np.exp(-((e / 0.35) ** 2))
            * dt,
            posture=-0.5 * float(np.clip(1.0 - up, 0, 2)) * dt,
            action=-0.02 * float(np.mean(action**2)) * dt,
            path=-0.25 * min((lateral / 1.5) ** 2, 4.0) * dt
            if self.family == "push"
            else 0.0,
            physical_failure=-5.0 * failure,
            terminal_goal=(5.0 if success else -2.0)
            if deadline and not failure
            else 0.0,
        )
        reward = (
            float(old_reward)
            if self.objective == "legacy"
            else float(sum(terms.values()))
        )
        self.previous_action = action.copy()
        self.episode_return += reward
        info.update(
            phase_seed=self.phase_seed,
            case=self.case,
            initial_state_sha256=self.initial_hash,
            elapsed_s=self.elapsed,
            physical_failure=failure,
            numerical_error=numerical,
            task_deadline_reached=deadline,
            success=success,
            stable_now=stable,
            final_heading_error_deg=float(np.rad2deg(abs(e))),
            heading_error_rad=e,
            heading_auc_rad_s=self.heading_integral,
            postpush_auc_rad_s=self.postpush_auc,
            total_rotation_deg=float(np.rad2deg(self.total_rotation)),
            body_forward_speed_mm_s=speed,
            up_z=up,
            yaw_rate_rad_s=rate,
            max_lateral_mm=self.max_lateral,
            gait_averaged_yaw_rad_s=float(w[-10:, 3].mean()),
            gait_averaged_speed_mm_s=float(w[-10:, 2].mean()),
            first_stable_s=self.first_stable_s,
            reward_terms=terms,
            episode_return=self.episode_return,
            action_rms=float(np.sqrt(np.mean(action**2))),
            finite_horizon_terminal=True,
        )
        return self.observation(), reward, terminated, False, info

    def get_state(self):
        return dict(
            base_seed=self.base_seed,
            stream=copy.deepcopy(self.stream.bit_generator.state)
            if self.stream
            else None,
            episode_index=self.episode_index,
        )

    def set_state(self, state):
        self.base_seed = state["base_seed"]
        self.stream = np.random.default_rng()
        self.stream.bit_generator.state = state["stream"]
        self.episode_index = state["episode_index"]
        self.restored = True


def make_env(rank, n_envs, family, objective, run=None):
    folder = ROOT / "runs" / run if run else None
    return PushTurnTask(
        family=family,
        objective=objective,
        rank=rank,
        n_envs=n_envs,
        log_path=folder / f"env_{rank:02d}_episodes.jsonl" if folder else None,
    )


class PotentialPushTurnTask(PushTurnTask):
    """Potential-based shaping with the same finite-horizon task objective."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conf = json.loads((PROTOCOL).read_text())
        self.shaping_gamma = float(conf["action_hold"]["micro_gamma"])
        self.potential_coefficient = float(conf["potential_shaping"]["coefficient"])

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        error = wrap(
            self._target_heading_rad - self._raw_heading_angle(self._last_raw_obs)
        )
        self.previous_potential = -self.potential_coefficient * abs(error)
        self.shaped_episode_return = 0.0
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        potential = (
            0.0
            if term or trunc
            else -self.potential_coefficient * abs(info["heading_error_rad"])
        )
        shaping = self.shaping_gamma * potential - self.previous_potential
        self.previous_potential = potential
        shaped_reward = float(reward + shaping)
        self.shaped_episode_return += shaped_reward
        info.update(
            unshaped_reward=reward,
            potential_shaping=shaping,
            potential=potential,
            unshaped_episode_return=info["episode_return"],
            episode_return=self.shaped_episode_return,
        )
        info["reward_terms"]["potential_shaping"] = shaping
        return obs, shaped_reward, term, trunc, info


# Both the training factory and evaluator resolve this common task class.
PushTurnTask = PotentialPushTurnTask


class HeldActionTask(PushTurnTask):
    """Four audited10ms steps per policy decision; event timing stays10ms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.micro_gamma = float(
            json.loads((PROTOCOL).read_text())["action_hold"]["micro_gamma"]
        )

    def step(self, action):
        reward = 0.0
        self.last_micro_infos = []
        for index in range(4):
            obs, micro_reward, term, trunc, info = super().step(action)
            self.last_micro_infos.append(dict(info, micro_reward=micro_reward))
            reward += self.micro_gamma**index * micro_reward
            if term or trunc:
                break
        info = dict(
            info,
            action_hold_micro_steps=index + 1,
            policy_control_dt_s=0.04,
            macro_reward=reward,
            unweighted_micro_episode_return=info["episode_return"],
        )
        return obs, float(reward), term, trunc, info


PushTurnTask = HeldActionTask
