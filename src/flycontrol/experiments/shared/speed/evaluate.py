"""Simulate all supplemental speed waveforms from fresh checkpoints."""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from flycontrol.experiments.shared.speed import policy as legacy
from flycontrol.paths import run_root
O = run_root() / 'configs/supplemental'
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def target(profile, t):
    if profile['protocol'] == 'legacy':
        return legacy.target_at(profile['task'], profile['direction'], t, profile['target'])
    family = profile['family']
    if family == 'step':
        return profile['v0'] if t < profile['onset'] - 1e-9 else profile['v1']
    if family == 'ramp':
        onset = profile['onset']; d = abs(profile['v1']-profile['v0'])/profile['slope']
        return float(np.interp(t, [0, onset, onset+d, onset+d+.5, onset+2*d+.5, profile['duration']],
                              [profile['v0'], profile['v0'], profile['v1'], profile['v1'], profile['v0'], profile['v0']]))
    if family == 'sine':
        return float(profile['center'] + (profile['amplitude']*np.sin(2*np.pi*profile['frequency']*(t-profile['onset'])) if t >= profile['onset'] else 0))
    if family == 'multistep':
        i = int(np.searchsorted(profile['knots'], t+1e-9, side='right')-1)
        return float(profile['values'][i])
    raise ValueError(family)

class EvalEnv(legacy.AxialEnv):
    def _apply_external_perturbation(self, elapsed_s):
        self._clear_external_forces()
        if self._sim is not None and self.push_onset <= elapsed_s < self.push_end:
            self._sim.physics.bind(self._fly.thorax).xfrc_applied = np.array([self.axial_force, 0., 0., 0., 0., 0.])
        if self._sim is not None:
            self.impulse += float(self._sim.physics.data.xfrc_applied[:, 0].sum())*self.timestep

def physics_digest(env):
    arrays = [env._sim.physics.data.qpos, env._sim.physics.data.qvel, env._controller.cpg.curr_phases]
    return hashlib.sha256(b''.join(np.ascontiguousarray(x).tobytes() for x in arrays)).hexdigest()

def rollout(model, env, profile, seed, out, run):
    start = time.time(); horizon = profile['duration']; env.max_time = horizon
    env.axial_force = 0.; env.impulse = 0.
    env.push_onset = profile.get('push_onset', 1.5)
    env.push_end = profile.get('push_end', 1.54)
    obs, _ = env.reset(seed=seed, options={'target_speed_mm_s': target(profile, 0.)})
    env.axial_force = profile.get('force', 0.); env.impulse = 0.
    hidden = np.zeros((model['N'], 1), np.float32)
    initial_sha = physics_digest(env); initial_full = legacy.digest(env, obs, hidden)
    prefix = None; rows = []; observations = []; numerical = False; flipped = False; last_heartbeat = 0.
    for _ in range(round(horizon/.01)):
        before = env._physics_steps*env.timestep
        command = target(profile, before); env._target_speed_mm_s = command
        obs[46:] = legacy.speed_target_observation(command, 20., env._filtered_forward_velocity,
                    env._last_heading_rate, env.speed_heading_rate_scale_rad_s)[0]
        if abs(before-env.push_onset) < 1e-8:
            prefix = legacy.digest(env, obs, hidden)
        observations.append(obs.copy())
        hidden, action = legacy.step(model, hidden, obs)
        obs, reward, terminated, truncated, info = env.step(action)
        after = env._physics_steps*env.timestep
        numerical |= bool(info.get('nan_detected', False) or info.get('physics_error', False))
        numerical |= not np.isfinite(obs).all() or not np.isfinite(hidden).all()
        flipped |= bool(info.get('flip', False))
        velocity = float(info.get('filtered_forward_velocity_mm_s', np.nan))
        raw = float(info.get('forward_velocity_mm_s', np.nan))
        numerical |= not np.isfinite(velocity) or not np.isfinite(raw)
        raw_obs = env._last_raw_obs
        rows.append(dict(time_s=after, dt_s=after-before, target_mm_s=command,
            speed_mm_s=velocity, raw_speed_mm_s=raw, error_mm_s=abs(velocity-command),
            drive=float(action[0]), turn=float(action[1]), reward=float(reward),
            heading_rate_rad_s=float(info.get('heading_rate_rad_s', env._last_heading_rate)),
            x_mm=float(raw_obs['fly'][0, 0]), y_mm=float(raw_obs['fly'][0, 1]),
            flip=flipped, numerical_failure=numerical, terminated=bool(terminated), truncated=bool(truncated),
            impulse=float(env.impulse), frequency_hz=float(info.get('cpg_frequency_mean_hz', np.nan)),
            duty_factor=float(info.get('duty_factor_command', np.nan)),
            phase_lag=float(info.get('ipsilateral_lag_cycles_command', np.nan))))
        if time.time()-last_heartbeat >= 30:
            write(out/'HEARTBEAT.json',dict(timestamp=time.time(),case_id=profile['case_id'],simulation_time_s=after,
                horizon_s=horizon,decisions=len(rows),wall_s=time.time()-start))
            last_heartbeat=time.time()
        if terminated or truncated or numerical:
            break
    duration = env._physics_steps*env.timestep
    expected_impulse = env.axial_force*max(0., min(duration, env.push_end)-env.push_onset)
    assert np.isclose(env.impulse, expected_impulse, rtol=.003, atol=.002), (env.impulse, expected_impulse)
    trace = out/(profile['case_id']+'.csv.gz')
    pd.DataFrame(rows).to_csv(trace, index=False, compression='gzip')
    obs_path = out/(profile['case_id']+'.npz')
    np.savez_compressed(obs_path, obs=np.asarray(observations, np.float32),
                        time_pre_s=np.cumsum([0]+[x['dt_s'] for x in rows[:-1]]))
    return dict(run_id=run['run_id'], model=run['model'], train_seed=run['train_seed'], eval_seed=seed,
        case_id=profile['case_id'], family=profile['family'], protocol=profile['protocol'],
        checkpoint=model['path'], checkpoint_sha256=model['sha'], profile_sha256=sha(O/'speed_profiles.json'),
        trace=str(trace), trace_sha256=sha(trace), observations=str(obs_path),
        planned_duration_s=horizon, duration_s=duration, steps=len(rows), flip=flipped,
        numerical_failure=numerical, full_duration=bool(duration >= horizon-1e-8 and not flipped and not numerical),
        physics_initial_sha256=initial_sha, initial_sha256=initial_full, prepush_sha256=prefix,
        impulse=float(env.impulse), expected_impulse=expected_impulse, wall_s=time.time()-start,
        provenance='new_simulation')
