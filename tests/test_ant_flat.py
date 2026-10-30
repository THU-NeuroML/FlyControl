"""Regression checks against the published capacity-control observations."""
import numpy as np
import pandas as pd
import pytest
import yaml

from flycontrol.analysis import signflip, holm
from flycontrol.paths import CONFIGS, read_json
from flycontrol.pipeline import initialize, checkpoint
from flycontrol.experiments.fig5.flat_analysis import REFERENCE, ENDPOINTS, audit, compute
from flycontrol.experiments.fig5.workflow import configuration


@pytest.fixture(scope="module")
def archived():
    data = read_json(REFERENCE)
    frames = {k: pd.DataFrame(v["data"], columns=v["columns"]) for k, v in data["tables"].items()}
    return data, frames, dict(smoke=False, methods=["connectome", "mlp"], seeds=list(range(20)))


def test_flat_capacity_and_maze_configs_are_separate(tmp_path):
    spec = initialize(tmp_path, "ant-flat", smoke=True)
    for method in ("adult", "mlp"):
        for stage in ("base", "flat"):
            name = f"{method}_{stage}"
            flat = yaml.safe_load((CONFIGS / f"internal/ant_flat/{name}.yaml").read_text())
            maze = yaml.safe_load((CONFIGS / f"internal/maze/{name}.yaml").read_text())
            assert flat["ppo"]["n_minibatches"] == 4
            if method == "mlp":
                assert flat["policy"]["actor_kwargs"]["hidden"] == [1378, 1378]
                assert maze["policy"]["actor_kwargs"]["hidden"] == [256, 256]
                assert maze["ppo"]["n_minibatches"] == 32
            path, short = configuration(tmp_path, spec, name, "ant_flat")
            assert path.is_file() and short["ppo"]["total_timesteps"] == 128
    assert checkpoint(tmp_path, spec, "mlp", 0).parts[-2:] == ("flat", "best_checkpoint.pt")


def test_archived_complete_grid_and_failure_definition(archived):
    data, frames, spec = archived
    for role, frame in frames.items():
        audit(frame, role, spec, data["protocol"])
        assert len(frame) == 4000
        with pytest.raises(ValueError, match="Incomplete/duplicate"):
            audit(frame.iloc[:-1], role, spec, data["protocol"])
    broken = frames["friction"].copy()
    broken.loc[0, "functional_failure"] = not broken.loc[0, "functional_failure"]
    with pytest.raises(ValueError, match="Functional failure"):
        audit(broken, "friction", spec, data["protocol"])


def test_published_p_values_and_means_are_reproduced(archived):
    data, frames, spec = archived
    pvalues = []
    for endpoint, (role, metric) in ENDPOINTS.items():
        wide = frames[role].groupby(["training_seed", "method"])[metric].mean().unstack()
        pvalues.append(signflip(wide.connectome-wide.mlp))
        expected = {"velocity": (1.9363904282638256, 1.2357185753815991),
                    "efficiency": (.8845663553073095, .7932812787417302),
                    "failure": (.386, .2985)}[endpoint]
        assert np.allclose([wide.connectome.mean(), wide.mlp.mean()], expected, atol=1e-12)
    corrected = holm([r["p_exact_two_sided"] for r in data["original_26"]] + pvalues)
    assert len(corrected) == 29
    assert np.allclose(corrected[-3:], [.059108734130859, .0446338653564447, .2183303833007812], atol=1e-12)


def test_full_flat_analysis_rejects_incomplete_training_cohort(tmp_path, archived):
    from flycontrol.paths import write_json
    data, frames, spec = archived
    write_json(tmp_path / "protocol.json", data["protocol"])
    with pytest.raises(ValueError, match="all 20"):
        compute(tmp_path, dict(spec, seeds=list(range(19))), frames, pd.DataFrame(), {}, "test")


def test_archived_seed_curves_have_all_seeds_and_native_horizon(archived):
    data, _, _ = archived
    assert np.allclose(data["time_seconds"], np.arange(1001)*.05)
    for metrics in data["seed_timecourses"].values():
        for values in metrics.values():
            a = np.asarray(values)
            assert a.shape == (20, 1001) and np.isfinite(a).all()
            assert np.all(a[:, 0] == 0)
