"""Registered checkpoint evaluation for this body."""

import json
import os
import numpy as np
import torch
import hashlib
from fly_control.io import write_json, CheckpointYield
from fly_control.training import TrainingPPO
from .env import make_env, BODY_CONFIG
from fly_control.training import evaluate


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
