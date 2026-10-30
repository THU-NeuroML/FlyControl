"""Locomotion training and evaluation pipeline.

Public objects are loaded lazily. Importing a lightweight utility such as
``larva.locomotion.models`` must not initialize FlyGym/dm_control/MuJoCo;
that eager side effect unnecessarily coupled every policy user to
NeuroMechFly and made independent Gymnasium tasks fragile on headless hosts.
"""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "TerrainLocomotionEnv": (".env", "TerrainLocomotionEnv"),
    "build_policy": (".models", "build_policy"),
    "Policy": (".models", "Policy"),
    "RecurrentPPO": (".ppo", "RecurrentPPO"),
    "PPOConfig": (".ppo", "PPOConfig"),
    "RunningMeanStd": (".ppo", "RunningMeanStd"),
    "SyncVecEnv": (".vec_env", "SyncVecEnv"),
    "SubprocVecEnv": (".vec_env", "SubprocVecEnv"),
    "make_vec_env": (".vec_env", "make_vec_env"),
    "make_env_from_spec": (".vec_env", "make_env_from_spec"),
    "load_larva": (".connectome", "load_larva"),
    "load_adult": (".connectome", "load_adult"),
    "randomize": (".connectome", "randomize"),
    "ConnectomeData": (".connectome", "ConnectomeData"),
    "ConnectomeCfC": (".cfc", "ConnectomeCfC"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, object_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), object_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
