"""Registered checkpoint evaluation for this body."""

import json
import os
import numpy as np
import torch
import hashlib
from fly_control.io import write_json, CheckpointYield
from fly_control.training import TrainingPPO
from .env import make_env, BODY_CONFIG


@torch.no_grad()
def evaluate(policy, body, device, seeds, command, interface="native"):
    from flycontrol.paths import run_root, read_json
    smoke = read_json(run_root() / "run.json")["smoke"]
    horizon = 0.2 if smoke else 20.0
    if smoke:
        seeds = list(seeds)[:1]
    envs = [make_env(body, eval_speed=command, interface=interface) for _ in seeds]
    try:
        obs = np.stack([e.reset(seed=s)[0] for e, s in zip(envs, seeds)])
        state = policy.init_state(len(envs), device)
        done = np.zeros(len(envs), bool)
        failed = done.copy()
        length = np.zeros(len(envs), int)
        rewards = np.zeros(len(envs))
        discounted = rewards.copy()
        lat = rewards.copy()
        tilt = rewards.copy()
        lateral_squared_time = rewards.copy()
        late_x = np.zeros(len(envs))
        late_time = np.zeros(len(envs))
        elapsed = np.zeros(len(envs))
        action_hash = [hashlib.sha256() for _ in seeds]
        for step in range(round(horizon / envs[0].dt)):
            action, _, _, state = policy.act(
                torch.as_tensor(obs, device=device), state, deterministic=True
            )
            actions = action.cpu().numpy()
            for i, e in enumerate(envs):
                if done[i]:
                    continue
                old_x = float(e.data.qpos[0])
                obs[i], reward, term, trunc, info = e.step(actions[i])
                action_hash[i].update(actions[i].tobytes())
                rewards[i] += reward
                discounted[i] += np.exp(-elapsed[i] / 5) * reward
                elapsed[i] += info["control_duration_s"]
                length[i] += 1
                if elapsed[i] > 10 + 1e-8:
                    late_x[i] += e.data.qpos[0] - old_x
                    late_time[i] += info["control_duration_s"]
                lat[i] = max(lat[i], abs(e.data.qpos[1] - e.physics.origin[1]))
                lateral_squared_time[i] += (
                    e.data.qpos[1] - e.physics.origin[1]
                ) ** 2 * info["control_duration_s"]
                tilt[i] = max(tilt[i], info["torso_tilt_degrees"])
                failed[i] = term
                done[i] = term or trunc
            if done.all():
                break
        result = []
        for i, (e, seed) in enumerate(zip(envs, seeds)):
            speed = float((e.data.qpos[0] - e.physics.origin[0]) / horizon)
            survived = bool(not failed[i] and elapsed[i] >= horizon - 1e-6)
            result.append(
                dict(
                    body=body,
                    seed=seed,
                    command=command,
                    duration_s=float(elapsed[i]),
                    survived=survived,
                    velocity_mps=speed,
                    late_velocity_mps=(
                        float(late_x[i] / late_time[i]) if late_time[i] > 0 else None
                    ),
                    lateral_max_m=float(lat[i]),
                    max_tilt_deg=float(tilt[i]),
                    return_raw=float(rewards[i]),
                    lateral_rms_m=float(
                        np.sqrt(lateral_squared_time[i] / max(elapsed[i], 1e-9))
                    ),
                    discounted_return=float(discounted[i]),
                    success=bool(
                        survived
                        and (
                            abs(speed) <= 0.05
                            if command < 0.02
                            else speed >= 0.7 * command
                        )
                        and lat[i] <= 0.75
                    ),
                    action_sha256=action_hash[i].hexdigest(),
                    teacher_used=False,
                    body_interface=interface,
                    cpg_used=interface in {"hexapod_cpg_residual", "hexapod_cpg_path"},
                    prescribed_joint_gait=interface
                    in {"hexapod_cpg_residual", "hexapod_cpg_path"},
                )
            )
        return result
    finally:
        for env in envs:
            env.close()


class CorrectPPO(TrainingPPO):
    def _save_checkpoint_atomic(self, ckpt, path):
        super()._save_checkpoint_atomic(ckpt, path)
        if path.name not in ["checkpoint.pt", "initial_checkpoint.pt"]:
            return
        step = int(ckpt["timesteps"])
        write_json(
            self.folder / "status.json",
            dict(
                state="training",
                timesteps=step,
                target=self.cfg.total_timesteps,
                elapsed_s=ckpt["elapsed_wall"],
                body=self.body,
                pid=os.getpid(),
                environment_curriculum_state=ckpt.get("env_states"),
                last=ckpt["history"][-1] if ckpt["history"] else None,
            ),
        )
        budgets = json.loads((self.root / "protocol.json").read_text())[
            "evaluation_budgets"
        ]
        if (
            step in set(budgets + [self.cfg.total_timesteps])
            and not (self.folder / f"evaluation_{step:07d}.json").exists()
        ):
            values = []
            commands = [
                BODY_CONFIG[self.body]["speed"],
                max(0.1, BODY_CONFIG[self.body]["initial_speed"]),
            ]
            if self.body == "cassie":
                commands.append(0.0)
            for command in sorted(set(commands)):
                values.extend(
                    evaluate(
                        self.policy,
                        self.body,
                        str(self.device),
                        (
                            list(range(899100, 899120))
                            if step == self.cfg.total_timesteps
                            else list(range(891000, 891004))
                        ),
                        command,
                        self.interface,
                    )
                )
            write_json(self.folder / f"evaluation_{step:07d}.json", values)
            slim = dict(
                policy_state=ckpt["policy_state"],
                timesteps=step,
                protocol_hash=ckpt["protocol_hash"],
                teacher_used=False,
            )
            super()._save_checkpoint_atomic(slim, self.folder / f"policy_{step:07d}.pt")
        control = json.loads((self.root / "control.json").read_text())
        if control.get("stop_all") or self.folder.name in control.get("yield_runs", []):
            raise CheckpointYield()
