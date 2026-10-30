"""Full paired steady-start evaluation; exact same-model MjData snapshots.

Snapshots preserve all native solver/contact state with MuJoCo's mj_copyData.
The in-place restore leaves dm_control bindings valid. No body translation,
resampling, checkpoint update, or phase exclusion is allowed.
"""

from .steady_state import *
import argparse, copy, ctypes, time, traceback, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import OrderedDict
import mujoco, torch
from .steady_task import SteadyObstacleTask
from terrain_control.policy import build_policy
from fly_control.ppo import ObservationRunningMeanStd

from fig6.paths import ROOT as REPO

BASE = REPO / "data/figure2"
HEIGHTS = [0.0, 0.2, 0.3, 0.45, 0.525]
SEEDS = list(range(2011, 2031))
STUDY = json.loads((REPO / "configs/figure2/study.json").read_text())
PHASES = [int(x) for b in STUDY["phase_bins"] for x in b]
assert len(PHASES) == 64 and len(set(PHASES)) == 64
assert set(PHASES) == set(STUDY["phase_panels"]["confirmation"])
POLICIES = OrderedDict()
LIB = ctypes.CDLL(str(next(Path(mujoco.__file__).parent.glob("libmujoco.so*"))))
COPY_DATA = LIB.mj_copyData
COPY_DATA.argtypes = [ctypes.c_void_p] * 3
COPY_DATA.restype = ctypes.c_void_p


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def checkpoint_path(method, seed):
    from flycontrol.pipeline import checkpoint
    from flycontrol.paths import read_json
    return checkpoint(REPO, read_json(REPO / "run.json"), method, seed)


def policy_for(method, seed):
    key = (method, seed)
    if key not in POLICIES:
        checkpoint = checkpoint_path(method, seed)
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        init = json.loads(
            (checkpoint.parent / "initialization.json").read_text()
        )
        policy = build_policy(
            BASE / "graph.npz",
            688,
            6,
            method,
            seed,
            "cpu",
            "normalized",
            init.get("initial_log_std", -1.0),
        )
        policy.load_state_dict(ck["policy_state"])
        policy.eval()
        normalizer = None
        if ck.get("obs_rms_state"):
            normalizer = ObservationRunningMeanStd((688,))
            normalizer.load_state_dict(ck["obs_rms_state"])
        POLICIES[key] = (policy, normalizer, sha(checkpoint))
        while len(POLICIES) > 4:
            POLICIES.popitem(last=False)
    POLICIES.move_to_end(key)
    return POLICIES[key]


class Snapshot:
    def __init__(self, env):
        self.native = copy.copy(env._sim.physics.data.ptr)
        self.task = {
            k: copy.deepcopy(v)
            for k, v in vars(env).items()
            if isinstance(
                v,
                (np.ndarray, int, float, str, bool, list, dict, tuple, set, type(None)),
            )
            and k != "forward_cache_stats"
        }
        self.cpg = copy.deepcopy(vars(env._cpg))
        self.fly = {
            k: copy.deepcopy(getattr(env._sim.fly, k))
            for k in [
                "_last_adhesion",
                "_active_adhesion",
                "_last_observation",
                "_last_neck_actuation",
                "_curr_visual_input",
                "_last_vision_update_time",
            ]
            if hasattr(env._sim.fly, k)
        }
        self.sim_time = env._sim.curr_time
        self.cache = copy.deepcopy(env.forward_cache_stats)
        self.dirty = env._sim.physics._dirty

    def restore(self, env):
        p = env._sim.physics
        COPY_DATA(p.data.ptr._address, p.model.ptr._address, self.native._address)
        for k, v in self.task.items():
            setattr(env, k, copy.deepcopy(v))
        vars(env._cpg).clear()
        vars(env._cpg).update(copy.deepcopy(self.cpg))
        for k, v in self.fly.items():
            setattr(env._sim.fly, k, copy.deepcopy(v))
        env._sim.curr_time = self.sim_time
        env.forward_cache_stats.clear()
        env.forward_cache_stats.update(self.cache)
        p._dirty = self.dirty
        return env.observation()


def rollout(env, method, seed):
    policy, normalizer, checkpoint_hash = policy_for(method, seed)
    obs = env.observation()
    state = policy.init_state(1, "cpu")
    r = raw(env)
    initial = dict(
        qpos_sha256=hashlib.sha256(env._sim.physics.data.qpos.tobytes()).hexdigest(),
        qvel_sha256=hashlib.sha256(env._sim.physics.data.qvel.tobytes()).hexdigest(),
        observation_sha256=hashlib.sha256(obs.tobytes()).hexdigest(),
        body_position=r["fly"][0].tolist(),
        body_up=float(r["cardinal_vectors"][2, 2]),
        forward_speed=float(r["fly"][1, 0]),
        heading_rad=float(np.arctan2(r["fly_orientation"][1], r["fly_orientation"][0])),
        cpg_phase_rad=float(env._cpg.curr_phases[0] % (2 * np.pi)),
        cpg_min_magnitude=float(env._cpg.curr_magnitudes.min()),
    )
    trace = {
        k: []
        for k in [
            "time_s",
            "qpos",
            "qvel",
            "position",
            "feet",
            "up",
            "action",
            "observation",
        ]
    }
    fields = dict(
        time_s=0.0,
        qpos=env._sim.physics.data.qpos,
        qvel=env._sim.physics.data.qvel,
        position=r["fly"][0],
        feet=r["end_effectors"],
        up=r["cardinal_vectors"][2],
        action=np.zeros(6),
        observation=obs,
    )
    for k, v in fields.items():
        trace[k].append(np.asarray(v).copy())
    with torch.inference_mode():
        while True:
            x = normalizer.normalize(obs[None], clip=10.0) if normalizer else obs[None]
            action, _, _, state = policy.act(
                torch.as_tensor(x), state, deterministic=True
            )
            obs, _, term, trunc, info = env.step(action[0].numpy())
            for frame in env.last_micro_trace:
                for k, v in frame.items():
                    trace[k].append(
                        np.asarray(float(v) - env.warmup_sim_time)
                        if k == "time_s"
                        else v.copy()
                    )
            if term or trunc:
                break
    z = {k: np.asarray(v) for k, v in trace.items()}
    duration = env.elapsed
    changes = np.abs(np.diff(z["action"], axis=0)).mean(axis=1)
    offset = env.coordinate_offset_xy.tolist()
    result = dict(
        method=method,
        model="Adult" if method == "connectome" else "MLP",
        training_seed=seed,
        height=env.value,
        phase_seed=env.episode_seed,
        mode="steady_start_full",
        success=bool(info["success"]),
        crossed_within_corridor=bool(info["success"] and env.max_lateral <= 1.5),
        duration_s=duration,
        completion_time_with_failure_horizon_s=duration if info["success"] else 1.2,
        max_lateral_mm=float(env.max_lateral),
        physical_failure=bool(info["physical_termination"]),
        numerical_failure=bool(info["numerical_error"]),
        final_last_foot_x_mm=float(z["feet"][-1, :, 0].min()) - offset[0],
        final_body_x_mm=float(z["position"][-1, 0]) - offset[0],
        final_up_z=float(z["up"][-1, 2]),
        minimum_up_z=float(z["up"][:, 2].min()),
        final_stable_dwell_s=float(env.clear_duration),
        residual_variation_per_observed_second=float(changes.sum() / duration),
        residual_max_step_change_mean6_l1=float(changes.max()),
        action_rms=float(np.sqrt(np.mean(z["action"][1:] ** 2))),
        initial=initial,
        warmup_metadata=env.runin_diagnostics,
        checkpoint_sha256=checkpoint_hash,
        coordinate_offset_xy=offset,
        actual_obstacle_near_x_mm=env.near,
        actual_obstacle_far_x_mm=env.far,
        minimum_ground_edge_margin_mm=float(
            min(
                (80 - z["feet"][:, :, 0]).min(),
                (z["feet"][:, :, 0] + 10).min(),
                (40 - np.abs(z["feet"][:, :, 1])).min(),
            )
        ),
        body_state_not_translated=True,
        geometry_audit=env.geometry_audit,
        obstacle_contact_macro_samples=env.obstacle_contact_macro_samples,
    )
    return result, z
