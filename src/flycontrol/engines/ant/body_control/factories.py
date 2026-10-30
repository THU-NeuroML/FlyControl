"""Picklable Ant environment factories using the shared vector workers."""

from typing import Sequence, Callable
from fly_control.vector_env import SyncVecEnv, SubprocVecEnv


def make_vec_env(
    env_fns: Sequence[Callable], kind: str = "sync", start_method: str = "spawn"
):
    """Construct a vec env of the given kind.

    kind:
      'sync'    -> SyncVecEnv (single process, for-loop)
      'subproc' -> SubprocVecEnv (one process per env, real parallel)
    """
    kind = kind.lower()
    if kind == "sync":
        return SyncVecEnv(env_fns)
    if kind in ("subproc", "subprocess", "subprocvec"):
        return SubprocVecEnv(env_fns, start_method=start_method)
    raise ValueError(
        f"Unknown vec_env kind: {kind!r} " f"(expected 'sync' or 'subproc')"
    )


def make_env_from_spec(class_path: str, init_kwargs: dict, reset_seed: int):
    import importlib

    (module_path, class_name) = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    env = cls(**init_kwargs)
    env.reset(seed=reset_seed)
    return env
