"""Common-history local input responses and actor-parameter sensitivity."""

from flycontrol.experiments.fig4.mechanisms.study import *
import numpy as np, json, time
import argparse, time, pandas as pd
from flycontrol.experiments.shared.perturbation import actor_filter_relative_direction, multiply_perturbed_actor, policy_mean_on_windows

STARTS = [25, 75, 125, 175, 225, 275]
BLOCKS = dict(
    feet=(0, 18),
    contact=(18, 24),
    posture=(24, 28),
    cpg=(28, 46),
    speed=(46, 49),
    heading=(49, 51),
)


@torch.inference_mode()
def run(method, seed, device):
    start = time.time()
    folder = O / "probes" / f"{method}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "COMPLETE.json").exists():
        return
    cp = checkpoint(method, seed)
    cph = sha(cp)
    p = load_policy(method, seed, device)
    names = [baseline_path(m, seed, c) for m in METHODS for c in CASES]
    histories = [np.load(n) for n in names]
    obs = np.stack([z["observations"] for z in histories])
    speeds = np.stack([z["raw_measured_speed"] for z in histories])
    assert obs.shape == (4, 300, 51)
    x = torch.as_tensor(obs, device=device)
    raw_speed = torch.as_tensor(speeds, device=device, dtype=torch.float32)
    h = p.init_state(4, device)
    states = []
    windows = []
    raw = []
    means = []
    for i in range(300):
        if i in STARTS:
            states.append(h.clone())
            windows.append(x[:, i : i + 16])
            raw.append(raw_speed[:, i : i + 16])
        mu, h, _ = p(x[:, i], h)
        means.append(mu.cpu().numpy())
    replay = np.stack(means, axis=1)
    own_offset = 0 if method == "connectome" else 2
    replay_error = max(
        float(
            np.max(
                np.abs(
                    replay[own_offset + i]
                    - histories[own_offset + i]["gaussian_actions"]
                )
            )
        )
        for i in range(2)
    )
    assert replay_error < 3e-5, replay_error
    windows = torch.cat(windows)
    h = torch.cat(states)
    raw = torch.cat(raw)
    ref = policy_mean_on_windows(p, windows, h)
    variance = p.log_std.exp().square()[None, None, :]
    np.savez_compressed(
        folder / "windows.npz",
        observations=windows.cpu().numpy(),
        initial_hidden=h.cpu().numpy(),
        raw_measured_speed=raw.cpu().numpy(),
        reference_gaussian_means=ref.cpu().numpy(),
    )

    def metrics(mu, r):
        sq = (mu - r).square()
        return [
            float((0.5 * sq / variance).sum(-1).mean()),
            float(0.5 * sq.sum(-1).mean()),
            float((mu.tanh() - r.tanh()).square().mean().sqrt()),
        ]

    response = []
    curves = []
    response_windows = []
    sensory_windows = []
    noise_windows = []

    def window_details(mu, r):
        sq = (mu - r).square()
        return (
            torch.stack(
                [
                    (0.5 * sq / variance).sum(-1).mean(1),
                    0.5 * sq.sum(-1).mean(1),
                    (mu.tanh() - r.tanh()).square().mean((1, 2)).sqrt(),
                ],
                1,
            )
            .cpu()
            .numpy()
        )

    for name in ["target", "measured"]:
        shifted = []
        for sign in [-1, 1]:
            cf = windows.clone()
            target = windows[:, :, 46] * 20
            if name == "target":
                target = target + sign
            measured = raw + (sign if name == "measured" else 0)
            cf[:, :, 46] = torch.clamp(target / 20, 0, 1)
            cf[:, :, 47] = torch.clamp(measured / 20, -1, 1)
            cf[:, :, 48] = torch.clamp((target - measured) / 20, -1, 1)
            mu = policy_mean_on_windows(p, cf, h)
            shifted.append(mu.tanh())
        # Central finite difference per +1 mm/s, with the explicit error channel updated.
        derivative = (shifted[1] - shifted[0]) / 2
        for wi, values in enumerate(derivative.cpu().numpy()):
            response_windows.append(
                dict(
                    seed=seed,
                    method=method,
                    input=name,
                    window=wi,
                    history_method=METHODS[(wi % 4) // 2],
                    kind=CASES[wi % 2]["kind"],
                    start=STARTS[wi // 4],
                    drive_response=float(values[:, 0].mean()),
                )
            )
        for step in range(16):
            curves.append(
                dict(
                    seed=seed,
                    method=method,
                    input=name,
                    time_ms=40 * (step + 1),
                    drive_derivative=float(derivative[:, step, 0].mean()),
                    turn_derivative=float(derivative[:, step, 1].mean()),
                )
            )
        response.append(
            dict(
                seed=seed,
                method=method,
                input=name,
                drive_response=float(derivative[:, :, 0].mean()),
                drive_response_40ms=float(derivative[:, 0, 0].mean()),
                drive_response_160ms=float(derivative[:, :4, 0].mean()),
                drive_response_640ms=float(derivative[:, -1, 0].mean()),
                turn_response=float(derivative[:, :, 1].mean()),
                expected_sign_fraction=float(
                    (
                        (derivative[:, :, 0] > 0)
                        if name == "target"
                        else (derivative[:, :, 0] < 0)
                    )
                    .float()
                    .mean()
                ),
            )
        )
    pd.DataFrame(response).to_csv(folder / "responses.csv", index=False)
    pd.DataFrame(curves).to_csv(folder / "response_curves.csv", index=False)
    pd.DataFrame(response_windows).to_csv(folder / "response_windows.csv", index=False)
    w = windows[:, :4]
    r = ref[:, :4]
    sd = x[:, 25:].reshape(-1, 51).std(dim=0, unbiased=False)
    sensory = []
    for name, (lo, hi) in BLOCKS.items():
        for direction in range(4):
            rng = np.random.default_rng(20261010 + direction)
            noise = np.zeros(51, dtype=np.float32)
            noise[lo:hi] = rng.standard_normal(hi - lo)
            delta = torch.as_tensor(noise, device=device) * sd * 0.1
            if name == "speed":
                delta[48] = delta[46] - delta[47]
            for sign in [-1, 1]:
                cf = w + sign * delta
                # Current raw values can be outside the clipped sensor range; preserve coherent speed/error.
                if name == "speed":
                    target = w[:, :, 46] * 20 + 20 * sign * delta[46]
                    measured = raw[:, :4] + 20 * sign * delta[47]
                    cf[:, :, 46] = torch.clamp(target / 20, 0, 1)
                    cf[:, :, 47] = torch.clamp(measured / 20, -1, 1)
                    cf[:, :, 48] = torch.clamp((target - measured) / 20, -1, 1)
                mu = policy_mean_on_windows(p, cf, h)
                sensory.append(
                    dict(
                        seed=seed,
                        method=method,
                        block=name,
                        direction=direction,
                        sign=sign,
                        policy_kl=metrics(mu, r)[0],
                        unit_kl=metrics(mu, r)[1],
                        action_rms=metrics(mu, r)[2],
                    )
                )
                for wi, v in enumerate(window_details(mu, r)):
                    sensory_windows.append(
                        dict(
                            seed=seed,
                            method=method,
                            block=name,
                            direction=direction,
                            sign=sign,
                            window=wi,
                            history_method=METHODS[(wi % 4) // 2],
                            policy_kl=float(v[0]),
                            unit_kl=float(v[1]),
                            action_rms=float(v[2]),
                        )
                    )
    pd.DataFrame(sensory).to_csv(folder / "sensory.csv", index=False)
    pd.DataFrame(sensory_windows).to_csv(folder / "sensory_windows.csv", index=False)
    original = {n: v.detach().clone() for n, v in p.actor.named_parameters()}
    directions = [
        actor_filter_relative_direction(p, seed=20261010 + i) for i in range(8)
    ]
    noise_rows = []
    for di, direction in enumerate(directions):
        for radius in [0.05, 0.1, 0.2]:
            for sign in [-1, 1]:
                with multiply_perturbed_actor(p, ((direction, radius * sign),)):
                    mu = policy_mean_on_windows(p, w, h)
                    v = metrics(mu, r)
                noise_rows.append(
                    dict(
                        seed=seed,
                        method=method,
                        direction=di,
                        radius=radius,
                        sign=sign,
                        policy_kl=v[0],
                        unit_kl=v[1],
                        action_rms=v[2],
                    )
                )
                for wi, values in enumerate(window_details(mu, r)):
                    noise_windows.append(
                        dict(
                            seed=seed,
                            method=method,
                            direction=di,
                            radius=radius,
                            sign=sign,
                            window=wi,
                            history_method=METHODS[(wi % 4) // 2],
                            policy_kl=float(values[0]),
                            unit_kl=float(values[1]),
                            action_rms=float(values[2]),
                        )
                    )
    pd.DataFrame(noise_rows).to_csv(folder / "parameter_noise.csv", index=False)
    pd.DataFrame(noise_windows).to_csv(
        folder / "parameter_noise_windows.csv", index=False
    )
    dose = np.linspace(-0.2, 0.2, 21)
    grid = np.empty((21, 21, 3))
    for j, y in enumerate(dose):
        for i, xx in enumerate(dose):
            with multiply_perturbed_actor(
                p, ((directions[0], float(xx)), (directions[1], float(y)))
            ):
                mu = policy_mean_on_windows(p, w, h)
                grid[j, i] = metrics(mu, r)
    assert np.max(grid[10, 10]) == 0
    np.savez_compressed(folder / "surface.npz", dose=dose, metrics=grid)
    assert all(torch.equal(v, original[n]) for n, v in p.actor.named_parameters())
    assert sha(cp) == cph
    write_json(
        folder / "COMPLETE.json",
        dict(
            seed=seed,
            method=method,
            training_decisions=15872,
            checkpoint_sha256=cph,
            observation_banks={str(n.relative_to(ROOT)): sha(n) for n in names},
            baseline_replay_max_abs_error=replay_error,
            windows=24,
            response_horizon_steps=16,
            flatness_horizon_steps=4,
            directions=8,
            grid_size=21,
            parameter_count=sum(v.numel() for v in p.actor.parameters()),
            log_std=p.log_std.cpu().tolist(),
            checkpoint_and_actor_restored=True,
            wall_s=time.time() - start,
            device=device,
            outputs_sha256={
                f.name: sha(f) for f in folder.iterdir() if f.suffix in [".npz", ".csv"]
            },
        ),
    )
    print("COMPLETE", method, seed, round(time.time() - start, 1), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--device", required=True)
    a = ap.parse_args()
    run(a.method, a.seed, a.device)
