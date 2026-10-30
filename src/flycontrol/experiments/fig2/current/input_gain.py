"""Measure every point in the final 21-by-21 input-gain plane from native replay."""

import argparse, time, json
import numpy as np
import pandas as pd
from flycontrol.experiments.fig2.current.paths import OUTPUT
from flycontrol.experiments.fig2.current.probe import Probe, BLOCKS


def run(method, seed):
    out = OUTPUT / "input_gain" / f"{method}_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    p = Probe(method, seed)
    screen=[]
    for axis in BLOCKS:
        for dose in (-.05,.05):
            screen.extend(dict(method=method,training_seed=seed,axis=axis,dose=dose,**v) for v in p.evaluate({axis:dose}))
    pd.DataFrame(screen).to_csv(out/'axis_screen.csv',index=False)
    (out/'AXIS_SELECTION.json').write_text(json.dumps(dict(axes=['Velocity','Joint velocity'],exploratory=True,selection='Manuscript axes held fixed; screening is regenerated and not used to reselect axes.')))
    rows = []
    for y in np.linspace(-0.1, 0.1, 21):
        for x in np.linspace(-0.1, 0.1, 21):
            rows.extend(
                dict(method=method, training_seed=seed, x=x, y=y, **v)
                for v in p.evaluate({"Velocity": x, "Joint velocity": y})
            )
        pd.DataFrame(rows).to_csv(out / "surface.csv", index=False)
    assert len(rows) == 441 * len(np.unique(p.heights))
    (out / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                method=method,
                seed=seed,
                measured_points=441,
                checkpoint=p.checkpoint,
                native_error=p.native_error,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("method", choices=["mlp", "connectome"])
    p.add_argument("seed", type=int)
    a = p.parse_args()
    run(a.method, a.seed)
