"""Exact two-stage PPO recipe, with per-model historical optimizer settings."""

from functools import partial
import json
import time
from .context import REPO, CONFIG, configure_process, write_json, sha
from .task import make_env
from .policy import build_policy
from .audited_ppo import AuditedPPO, tensor_hash
from fly_control.ppo import PPOConfig
from fly_control.vector_env import SubprocVecEnv


class ContinuationPPO(AuditedPPO):
    def load_checkpoint(self, path):
        result = super().load_checkpoint(path)
        spec = json.loads((CONFIG / "experiments" / f"{self.method}_seed{self.training_seed}.json").read_text())
        base = spec["base_ppo"]
        base_updates = base["total_timesteps"] // (base["n_envs"] * base["n_steps"])
        if (result["timesteps"], result["update"]) != (base["total_timesteps"], base_updates):
            raise ValueError(
                "Continuation requires the original 12,800-decision checkpoint"
            )
        self.protocol_hash = self.continuation_hash
        write_json(
            self.folder / "RESUME_AUDIT.json",
            dict(
                source=str(path),
                source_sha256=sha(path),
                original_protocol_hash=result["protocol_hash"],
                original_update=base_updates,
                original_decisions=base["total_timesteps"],
                optimizer_restored=True,
                rng_streams_restored=True,
                physical_simulation="Reset once using restored environment RNG streams",
                adaptive_policy_kl=self.rule,
                protocol_hash=self.protocol_hash,
            ),
        )
        return result


def train(method, seed, output, device="cuda:0", stage="continue", source=None):
    configure_process()
    config_path = CONFIG / "experiments" / f"{method}_seed{seed}.json"
    spec = json.loads(config_path.read_text())
    prefix = "base" if stage == "base" else "continuation"
    cfg = PPOConfig(**spec[prefix + "_ppo"])
    conf = json.loads((CONFIG / "protocol.json").read_text())
    conf["adaptive_policy_kl"] = spec[prefix + "_adaptive_policy_kl"]
    conf["evaluation_budgets"] = [cfg.total_timesteps]
    output.mkdir(parents=True, exist_ok=False)
    policy = build_policy(method, seed, device, 40)
    graph = policy.actor.core if method == "connectome" else None
    before = (
        tensor_hash([graph.edge_pre, graph.edge_post, graph.edge_weight])
        if graph
        else None
    )
    env = SubprocVecEnv(
        [partial(make_env, i, cfg.n_envs, 40, str(output)) for i in range(cfg.n_envs)]
    )
    cls = AuditedPPO if stage == "base" else ContinuationPPO
    trainer = cls(
        env,
        policy,
        cfg,
        device,
        folder=output,
        conf=conf,
        protocol_hash=spec["base_protocol_hash"],
    )
    trainer.continuation_hash = sha(config_path)
    trainer.method, trainer.training_seed = method, seed
    if stage != "base":
        if source is None:
            raise ValueError("Continuation requires the checkpoint from this run's base stage")
    elif source is not None:
        raise ValueError("Base stage starts from initialization; omit --source")
    write_json(
        output / "initialization.json",
        dict(
            method=method,
            seed=seed,
            stage=stage,
            config=vars(cfg),
            adaptive_policy_kl=conf["adaptive_policy_kl"],
            config_sha256=sha(config_path),
        ),
    )
    start = time.time()
    try:
        trainer.learn(output, resume_from=source)
        if graph:
            assert before == tensor_hash(
                [graph.edge_pre, graph.edge_post, graph.edge_weight]
            )
        write_json(
            output / "COMPLETE.json",
            dict(
                total_decisions=cfg.total_timesteps,
                graph_frozen=True,
                seconds=time.time() - start,
                policy_sha256=sha(output / f"policy_{cfg.total_timesteps:08d}.pt"),
            ),
        )
    finally:
        env.close()
