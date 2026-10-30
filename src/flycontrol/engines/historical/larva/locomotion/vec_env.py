"""Vectorized env wrappers for PPO rollout collection.

Two implementations:

* ``SyncVecEnv``      — single-process, sequential for-loop. Cheap to
                        spin up, debuggable. Throughput bound by one
                        Python interpreter / one MuJoCo core.
* ``SubprocVecEnv``   — one OS process per env. Real parallelism (no GIL
                        contention, no MuJoCo state sharing). On 8 cores
                        gives 6–8x throughput in our setting.

Why this file exists (and is not folded into ppo.py):

  SubprocVecEnv uses ``multiprocessing.get_context("spawn")``. Spawn
  re-imports the entry module from scratch in each child, which means
  the parent's ``if __name__ == '__main__'`` block runs only in the
  parent. Worker entry points therefore *must* be importable
  module-level functions (no closures, no lambdas). Keeping them in a
  dedicated module makes the import contract obvious.

macOS notes:

  fork() after MuJoCo init corrupts internal state (GL context, OpenMP
  pools, etc.). We force ``spawn`` and never let the parent build a
  MuJoCo model before forking workers. Each env is built inside its
  child process via the env_fn callable that was pickled across.

Picklability contract:

  env_fns passed to SubprocVecEnv must be picklable. Use
  ``functools.partial(top_level_fn, *picklable_args)`` rather than
  closures. ``train._build_env`` is the canonical factory and is safe.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import sys
import time
import traceback
from multiprocessing.connection import Connection
from typing import Callable, Optional, Sequence

import numpy as np


# ----------------------------------------------------------------------
# Synchronous (single-process) vec env
# ----------------------------------------------------------------------


class SyncVecEnv:
    """For-loop over envs in this process. Useful for debugging and as a
    fallback when subprocess parallelism is unavailable."""

    def __init__(self, env_fns: Sequence[Callable]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        first = self.envs[0]
        self.observation_space = first.observation_space
        self.action_space = first.action_space

    def reset(self, seeds: Optional[Sequence[int]] = None):
        obs_list = []
        for i, env in enumerate(self.envs):
            seed = seeds[i] if seeds is not None else None
            o, _ = env.reset(seed=seed)
            obs_list.append(o)
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        obs, rewards, dones, infos = [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, term, trunc, info = env.step(actions[i])
            done = bool(term or trunc)
            info = dict(info) if info else {}
            info["terminated"] = bool(term)
            info["truncated"] = bool(trunc)
            if done:
                info["terminal_observation"] = np.asarray(o, dtype=np.float32)
                o, _ = env.reset()
            obs.append(o)
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return (np.stack(obs, 0),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_),
                infos)

    def get_states(self) -> list[dict | None]:
        """Return optional per-environment protocol state."""
        return [
            env.get_state() if hasattr(env, "get_state") else None
            for env in self.envs
        ]

    def set_states(self, states: Sequence[dict | None]) -> None:
        if len(states) != self.num_envs:
            raise ValueError("State count does not match vector environment")
        for env, state in zip(self.envs, states):
            if state is None:
                continue
            set_state = getattr(env, "set_state", None)
            if set_state is None:
                raise AttributeError(
                    f"{type(env).__name__} does not support set_state")
            set_state(state)

    def close(self) -> None:
        for env in self.envs:
            close = getattr(env, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:
                    pass


# ----------------------------------------------------------------------
# Subprocess (real-parallel) vec env
# ----------------------------------------------------------------------


# Protocol over the pipe is a 2-tuple (tag, payload):
#   parent -> child: ("step", action_np) | ("reset", seed_or_None)
#                  | ("close", None)
#   child  -> parent: ("ok",    payload) | ("error", (msg, traceback))
# Initial handshake sent unprompted by child after env construction:
#                  : ("ready", (observation_space, action_space))
#                  : ("error", (msg, traceback)) on build failure.


def _safe_send(remote: Connection, msg) -> bool:
    """Send a message, swallowing pipe-closed errors.

    When the parent dies (SIGKILL / SIGTERM / crash) the worker's next
    send() raises BrokenPipeError or OSError. That's normal shutdown,
    not a bug — there's no one left to read. Swallow it so the worker
    exits quietly.
    """
    try:
        remote.send(msg)
        return True
    except (BrokenPipeError, EOFError, OSError):
        return False


def _worker_loop(remote: Connection, env_fn_pickle: bytes,
                 worker_idx: int) -> None:
    """Child process entry point.

    Builds the env, sends a handshake with its spaces, then services a
    request/response loop until ``close`` arrives or the pipe breaks.
    All exceptions are caught and reported back as ('error', ...) so the
    parent can fail fast instead of hanging on a dead pipe.

    Signal handling: the child resets SIGINT to default (so Ctrl-C in a
    spawned subprocess kills it directly instead of being inherited
    from a debugger / IPython context). Parent signal propagation is
    independent of this.
    """
    # Default SIGINT in child: terminate. Without this, Ctrl-C
    # delivered to the whole process group may produce noisy
    # KeyboardInterrupts inside C-level recv() in workers.
    import signal as _signal
    try:
        _signal.signal(_signal.SIGINT, _signal.SIG_DFL)
    except Exception:
        pass

    # Best-effort tag for diagnostic prints; safe to ignore failures.
    try:
        import setproctitle  # type: ignore
        setproctitle.setproctitle(f"flygym-worker-{worker_idx}")
    except Exception:
        pass

    try:
        env_fn = pickle.loads(env_fn_pickle)
        env = env_fn()
        _safe_send(remote,
                   ("ready", (env.observation_space, env.action_space)))
    except Exception as e:
        _safe_send(remote, ("error", (repr(e), traceback.format_exc())))
        try:
            remote.close()
        except Exception:
            pass
        return

    try:
        while True:
            try:
                cmd, data = remote.recv()
            except (EOFError, ConnectionResetError, OSError):
                break

            if cmd == "close":
                break

            try:
                if cmd == "step":
                    obs, reward, term, trunc, info = env.step(data)
                    done = bool(term or trunc)
                    info = dict(info) if info else {}
                    info["terminated"] = bool(term)
                    info["truncated"] = bool(trunc)
                    if done:
                        info["terminal_observation"] = np.asarray(obs, dtype=np.float32)
                        obs, _ = env.reset()
                    if not _safe_send(remote,
                                      ("ok", (obs, float(reward), done, info))):
                        break
                elif cmd == "reset":
                    obs, info = env.reset(seed=data)
                    if not _safe_send(remote, ("ok", (obs, info))):
                        break
                elif cmd == "get_state":
                    get_state = getattr(env, "get_state", None)
                    state = get_state() if get_state is not None else None
                    if not _safe_send(remote, ("ok", state)):
                        break
                elif cmd == "set_state":
                    if data is None:
                        if not _safe_send(remote, ("ok", None)):
                            break
                        continue
                    set_state = getattr(env, "set_state", None)
                    if set_state is None:
                        raise AttributeError(
                            f"{type(env).__name__} does not support set_state")
                    set_state(data)
                    if not _safe_send(remote, ("ok", None)):
                        break
                else:
                    raise ValueError(f"Unknown cmd: {cmd!r}")
            except KeyboardInterrupt:
                # Treat ^C as a normal close. Parent will reap us.
                break
            except Exception as e:
                _safe_send(remote,
                           ("error", (repr(e), traceback.format_exc())))
                break  # don't try to recover, parent will tear us down
    finally:
        try:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        except Exception:
            pass
        try:
            remote.close()
        except Exception:
            pass


class SubprocWorkerError(RuntimeError):
    """Raised in the parent when any worker reports an exception."""


class SubprocVecEnv:
    """Real multi-process vec env, spawn-based, MuJoCo-safe on macOS.

    Each env runs in its own process. ``step()`` fans out N action
    messages, then collects N responses — the wall-time per step is
    roughly max(per-worker step time), not sum.

    Args:
      env_fns: picklable zero-arg callables. Each is sent (pickled) to a
        worker which calls it to construct the env *inside* that
        worker. Closures over local state will fail under spawn — use
        ``functools.partial(top_level_fn, *args)``.
      start_method: 'spawn' (default, required on macOS with MuJoCo) or
        'forkserver'. Do not use 'fork'.

    Lifecycle:
      __init__       -> spawns N workers; blocks until each sends 'ready'.
      reset(seeds)   -> sends 'reset' to all; collects all obs.
      step(actions)  -> sends 'step' to all; collects all results.
      close()        -> sends 'close' to all; joins; force-kills laggards.
    """

    def __init__(self, env_fns: Sequence[Callable],
                 start_method: str = "spawn") -> None:
        if start_method == "fork":
            raise ValueError(
                "fork() is unsafe with MuJoCo: forking after MuJoCo "
                "initialization corrupts GL/OMP state. Use 'spawn'.")

        ctx = mp.get_context(start_method)
        self.parent_pipes: list[Connection] = []
        self.processes: list[mp.process.BaseProcess] = []
        self.closed: bool = False
        self.num_envs: int = len(env_fns)

        for idx, fn in enumerate(env_fns):
            try:
                fn_pickle = pickle.dumps(fn)
            except Exception as e:
                self._cleanup_partial()
                raise TypeError(
                    f"env_fns[{idx}] is not picklable for {start_method}: "
                    f"{e!r}. Use functools.partial over a top-level "
                    f"function, not a closure or lambda.") from e

            parent_conn, child_conn = ctx.Pipe(duplex=True)
            proc = ctx.Process(
                target=_worker_loop,
                args=(child_conn, fn_pickle, idx),
                name=f"flygym-worker-{idx}",
                daemon=True,
            )
            proc.start()
            # Parent never writes to the child end of the pipe; closing
            # it here prevents the child's pipe.recv() from blocking
            # forever on a zombie file descriptor in the parent.
            child_conn.close()
            self.parent_pipes.append(parent_conn)
            self.processes.append(proc)

        # Handshake: every worker sends 'ready' (or 'error') once env is
        # built. Using all workers gives a parallel warm-up: the
        # parent's wall-time blocks on max(build_times), not sum.
        spaces = None
        for idx, pipe in enumerate(self.parent_pipes):
            tag, payload = pipe.recv()
            if tag != "ready":
                self.close()
                msg, tb = payload
                raise SubprocWorkerError(
                    f"worker {idx} failed during env construction: "
                    f"{msg}\n{tb}")
            obs_space, act_space = payload
            if spaces is None:
                spaces = (obs_space, act_space)

        assert spaces is not None
        self.observation_space, self.action_space = spaces

        print(f"[vec_env] SubprocVecEnv ready: {self.num_envs} workers "
              f"(start_method={start_method})",
              file=sys.stderr, flush=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seeds: Optional[Sequence[int]] = None):
        self._check_open()
        for i, pipe in enumerate(self.parent_pipes):
            seed = seeds[i] if seeds is not None else None
            pipe.send(("reset", seed))
        obs_list = []
        for idx, pipe in enumerate(self.parent_pipes):
            tag, payload = pipe.recv()
            self._check_response(idx, tag, payload)
            obs, _info = payload
            obs_list.append(obs)
        return np.stack(obs_list, axis=0)

    def step(self, actions: np.ndarray):
        self._check_open()
        if len(actions) != self.num_envs:
            raise ValueError(
                f"actions has length {len(actions)} but vec_env has "
                f"{self.num_envs} workers")

        # Fan-out: send all step commands. This is the parallelism
        # gate — if we recv between sends, the workers serialize.
        for i, pipe in enumerate(self.parent_pipes):
            pipe.send(("step", np.asarray(actions[i])))

        obs_list, rewards, dones, infos = [], [], [], []
        for idx, pipe in enumerate(self.parent_pipes):
            tag, payload = pipe.recv()
            self._check_response(idx, tag, payload)
            obs, r, done, info = payload
            obs_list.append(obs)
            rewards.append(r)
            dones.append(done)
            infos.append(info)
        return (np.stack(obs_list, 0),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_),
                infos)

    def get_states(self) -> list[dict | None]:
        self._check_open()
        for pipe in self.parent_pipes:
            pipe.send(("get_state", None))
        states: list[dict | None] = []
        for idx, pipe in enumerate(self.parent_pipes):
            tag, payload = pipe.recv()
            self._check_response(idx, tag, payload)
            states.append(payload)
        return states

    def set_states(self, states: Sequence[dict | None]) -> None:
        self._check_open()
        if len(states) != self.num_envs:
            raise ValueError("State count does not match vector environment")
        for pipe, state in zip(self.parent_pipes, states):
            pipe.send(("set_state", state))
        for idx, pipe in enumerate(self.parent_pipes):
            tag, payload = pipe.recv()
            self._check_response(idx, tag, payload)

    def close(self, graceful_timeout: float = 3.0,
              terminate_timeout: float = 1.0) -> None:
        """Tear down all workers.

        Two-phase: (1) send 'close' command to every worker in parallel
        and wait up to ``graceful_timeout`` total for them to exit, then
        (2) SIGTERM + SIGKILL the laggards. Done in *parallel* across
        workers so total close time is O(timeout), not O(N*timeout).
        """
        if self.closed:
            return
        self.closed = True

        # Phase 1: broadcast 'close' to all workers (best effort).
        for pipe in self.parent_pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass

        # Phase 1b: wait up to graceful_timeout for *all* workers, in
        # parallel. We poll instead of join(timeout) per process so a
        # slow worker doesn't extend the budget for the others.
        deadline = time.monotonic() + graceful_timeout
        while time.monotonic() < deadline:
            if not any(p.is_alive() for p in self.processes):
                break
            time.sleep(0.05)

        # Phase 2: SIGTERM any laggards (in parallel: signal first, then
        # wait once).
        for p in self.processes:
            if p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        deadline = time.monotonic() + terminate_timeout
        while time.monotonic() < deadline:
            if not any(p.is_alive() for p in self.processes):
                break
            time.sleep(0.05)

        # Phase 3: SIGKILL stubborn workers. Should never happen with a
        # well-behaved worker loop, but we must not leak zombies.
        for p in self.processes:
            if p.is_alive():
                try:
                    p.kill()
                except Exception:
                    pass
        for p in self.processes:
            p.join(timeout=0.5)

        for pipe in self.parent_pipes:
            try:
                pipe.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_open(self) -> None:
        if self.closed:
            raise RuntimeError("SubprocVecEnv is closed")

    def _check_response(self, idx: int, tag: str, payload) -> None:
        if tag == "ok":
            return
        if tag == "error":
            msg, tb = payload
            self.close()
            raise SubprocWorkerError(
                f"worker {idx} raised: {msg}\n{tb}")
        self.close()
        raise SubprocWorkerError(
            f"worker {idx} sent unknown tag {tag!r}")

    def _cleanup_partial(self) -> None:
        """Tear down whatever has been spawned so far. Used when
        __init__ fails partway through (e.g. an env_fn doesn't pickle)."""
        for pipe in getattr(self, "parent_pipes", []):
            try:
                pipe.close()
            except Exception:
                pass
        for p in getattr(self, "processes", []):
            try:
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1)
            except Exception:
                pass


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def make_vec_env(env_fns: Sequence[Callable], kind: str = "sync",
                 start_method: str = "spawn"):
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
    raise ValueError(f"Unknown vec_env kind: {kind!r} "
                     f"(expected 'sync' or 'subproc')")


# ----------------------------------------------------------------------
# Picklable env factory (top-level for spawn)
# ----------------------------------------------------------------------


def make_env_from_spec(class_path: str, init_kwargs: dict,
                       reset_seed: int):
    """Build an env by import path + kwargs.

    This factory is *module-level on purpose*: ``SubprocVecEnv`` uses
    ``spawn``, which means the worker has to unpickle the factory from a
    proper module (not from ``__main__``). Defining the factory here
    sidesteps the well-known "can't pickle <function thunk>" trap that
    closure-based env factories hit under spawn.

    Args:
      class_path: dotted path like "larva.locomotion.env.TerrainLocomotionEnv".
      init_kwargs: passed straight to ``cls(**init_kwargs)``.
      reset_seed: seed passed to env.reset() immediately after construction.

    Used as::

      from functools import partial
      env_fn = partial(make_env_from_spec, class_path, env_cfg, seed)
    """
    import importlib
    module_path, class_name = class_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    env = cls(**init_kwargs)
    env.reset(seed=reset_seed)
    return env
