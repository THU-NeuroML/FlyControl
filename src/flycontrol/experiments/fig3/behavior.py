"""Original episode metrics and complete registered Holm families, with relocated inputs."""

import json, hashlib, itertools
import numpy as np
import pandas as pd
from scipy.stats import t as student_t

METHODS = ["connectome", "mlp", "er"]
SECONDARY = {
    "angle": [
        "command_variation",
        "success",
        "failure",
        "speed",
        "relative_command_roughness",
        "late_heading_upright_fraction",
        "last_window_error",
        "worst_direction_error",
    ],
    "push": [
        "command_variation",
        "terminal_error",
        "failure",
        "speed",
        "relative_command_roughness",
        "late_heading_upright_fraction",
        "last_window_error",
        "recovery_auc",
    ],
}


def exact_p(diff):
    diff = np.asarray(diff, dtype=float)
    observed = abs(diff.sum())
    tol = 1e-12 * max(1.0, abs(diff).sum())
    if observed <= tol:
        return 1.0

    def sums(d):
        a = np.array([0.0])
        for v in d:
            a = np.r_[a + v, a - v]
        return a

    split = len(diff) // 2
    left = sums(diff[:split])
    right = np.sort(sums(diff[split:]))
    low = np.searchsorted(right, -observed + tol - left, side="right")
    high = len(right) - np.searchsorted(right, observed - tol - left, side="left")
    return float(np.sum(low + high, dtype=np.int64) / (len(left) * len(right)))


def self_check():
    for n in [3, 6, 12]:
        d = np.random.default_rng(771 + n).normal(size=n)
        sums = np.array(list(itertools.product([-1.0, 1.0], repeat=n))) @ d
        expected = np.mean(abs(sums) >= abs(d.sum()) - 1e-10)
        assert exact_p(d) == expected, (n, exact_p(d), expected)
    assert exact_p(np.zeros(24)) == 1
    assert exact_p(np.ones(24)) == 2 ** (-23)


def metrics(path):
    payload = path.read_bytes()
    r = json.loads(payload)
    tr = r["micro_trajectory"]
    times = np.array([x["elapsed_s"] for x in tr])
    grid = np.arange(1, 201) * 0.01
    valid = len(tr)
    if not np.isclose(times[-1], grid[valid - 1], atol=1e-9):
        valid -= 1
    error = np.full(200, 180.0)
    up = np.zeros(200)
    speed = np.zeros(200)
    assert np.allclose(times[:valid], grid[:valid], atol=1e-9)
    error[:valid] = np.rad2deg(np.abs([x["heading_error_rad"] for x in tr[:valid]]))
    up[:valid] = [x["up_z"] for x in tr[:valid]]
    speed[:valid] = [x["body_forward_speed_mm_s"] for x in tr[:valid]]
    macro = [x for x in r["trajectory"] if 0.4 + 1e-9 < x["elapsed_s"] <= 2 + 1e-9]
    rough = np.nan
    if len(macro) == 40 and not r["final"]["physical_failure"]:
        actions = np.array([x["action"] for x in macro])
        centered = actions - actions.mean(axis=0)
        den = np.sqrt(np.mean(np.sum(centered**2, axis=1)))
        rough = (
            float(np.sqrt(np.mean(np.sum(np.diff(actions, axis=0) ** 2, axis=1))) / den)
            if den > 1e-12
            else 0.0
        )
    return dict(
        family=r["family"],
        seed=r["training_seed"] or -1,
        method=r["method"],
        phase=r["phase_seed"],
        angle=r["case"].get("angle_deg", 0.0),
        force=r["case"].get("force", 0.0),
        side=r["case"].get("side", 0.0),
        terminal_error=r["primary_angle_error_deg"],
        success=float(r["final"]["success"]),
        failure=float(r["final"]["physical_failure"]),
        command_variation=r["action_total_variation"],
        speed=float(speed.mean()),
        relative_command_roughness=rough,
        late_heading_upright_fraction=float(
            ((error[-50:] <= 10) & (up[-50:] >= 0.8)).mean()
        ),
        last_window_error=float(error[-50:].mean()),
        recovery_auc=r["primary_push_auc_rad_s"],
        source=str(path),
        source_sha256=hashlib.sha256(payload).hexdigest(),
        initial_state=r["initial"]["initial_state_sha256"],
        initial_obs=r["initial"]["initial_observation_sha256"],
    )
