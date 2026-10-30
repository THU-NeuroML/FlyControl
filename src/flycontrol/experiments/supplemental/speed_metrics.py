"""Time-weighted control metrics, with explicit failure and censoring fields."""
import numpy as np
from scipy.signal import savgol_filter

def circular_mean_deg(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if not len(values):return np.nan
    z=np.mean(np.exp(1j*np.deg2rad(values)))
    return float(np.rad2deg(np.angle(z))) if abs(z)>1e-8 else np.nan

def circular_sd_deg(values):
    values=np.asarray(values,float);values=values[np.isfinite(values)]
    if len(values)<2:return np.nan
    strength=abs(np.mean(np.exp(1j*np.deg2rad(values))))
    return float(np.rad2deg(np.sqrt(-2*np.log(np.clip(strength,1e-15,1.)))))

def weights(frame, lo, hi):
    t = frame.time_s.to_numpy(float); dt = frame.dt_s.to_numpy(float)
    return np.maximum(0., np.minimum(t, hi)-np.maximum(t-dt, lo))

def errors(frame, lo, hi, velocity='speed_mm_s'):
    w = weights(frame, lo, hi); v = frame[velocity].to_numpy(float); r = frame.target_mm_s.to_numpy(float)
    valid = np.isfinite(v) & np.isfinite(r); w = np.where(valid, w, 0.)
    covered = float(w.sum()); missing = max(0., hi-lo-covered)
    e = np.where(valid, v-r, 0.)
    return dict(mae=(float(np.sum(w*np.abs(e)))+20*missing)/(hi-lo),
        rmse=np.sqrt((float(np.sum(w*e**2))+400*missing)/(hi-lo)),
        iae=float(np.sum(w*np.abs(e)))+20*missing,
        observed_bias=float(np.sum(w*e)/covered) if covered else np.nan,
        observed_mean=float(np.sum(w*np.where(valid, v, 0))/covered) if covered else np.nan,
        observed_mae=float(np.sum(w*np.abs(e))/covered) if covered else np.nan,
        observed_seconds=covered, missing_seconds=missing)

def weighted_sd(frame, lo, hi, field='speed_mm_s'):
    w = weights(frame, lo, hi); x = frame[field].to_numpy(float)
    ok = (w > 0) & np.isfinite(x)
    if not ok.any(): return np.nan
    mean = np.average(x[ok], weights=w[ok])
    return float(np.sqrt(np.average((x[ok]-mean)**2, weights=w[ok])))

def derivative(frame, field='speed_mm_s', seconds=.11):
    x = frame[field].to_numpy(float); n = int(round(seconds/.01))
    if n % 2 == 0: n += 1
    out = np.full(len(x), np.nan)
    regular = np.isfinite(x) & (np.abs(frame.dt_s.to_numpy(float)-.01)<1e-6)
    bad = np.flatnonzero(~regular); end = int(bad[0]) if len(bad) else len(x)
    # A failed simulation may stop inside its final decision interval. Preserve
    # the case and score missing time, but never differentiate that partial step
    # as if it were a full 10-ms sample.
    if end >= n: out[:end] = savgol_filter(x[:end], n, 2, deriv=1, delta=.01, mode='interp')
    return out

def cycle_mean(frame, seconds):
    """Causal time-weighted speed over a fixed 100- or 200-ms window.

    This is an explicitly supplementary gait-oscillation sensitivity analysis;
    it does not replace the predeclared controller-filtered velocity metrics.
    It introduces a known averaging delay and never anticipates a command.
    """
    out=frame.copy();positive=frame.dt_s.to_numpy(float)>1e-12
    measured=frame.loc[positive];t=measured.time_s.to_numpy(float);dt=measured.dt_s.to_numpy(float)
    assert np.all(np.diff(t)>0)
    x=measured.speed_mm_s.to_numpy(float);valid=np.isfinite(x)
    integral=np.r_[0.,np.cumsum(np.where(valid,x,0.)*dt)]
    coverage=np.r_[0.,np.cumsum(valid*dt)];bounds=np.r_[0.,t]
    lo=np.maximum(0.,t-seconds)
    values=(integral[1:]-np.interp(lo,bounds,integral))/seconds
    covered=coverage[1:]-np.interp(lo,bounds,coverage)
    values[(t<seconds-1e-8)|(covered<seconds-1e-8)]=np.nan
    out['speed_mm_s']=np.nan;out.loc[positive,'speed_mm_s']=values
    return out

def settling(time_s, error, onset, end, full_duration, tolerance, hold=.2):
    mask = (time_s > onset+1e-9) & (time_s <= end+1e-8)
    t = time_s[mask]; e = error[mask]
    if not full_duration or not len(t):
        return dict(reached=False, time_s=np.nan, restricted_time_s=end-onset)
    outside = np.flatnonzero(~np.isfinite(e) | (np.abs(e) > tolerance))
    first = int(outside[-1]+1) if len(outside) else 0
    if first < len(t) and end-t[first] >= hold-1e-8:
        return dict(reached=True, time_s=float(t[first]-onset), restricted_time_s=float(t[first]-onset))
    return dict(reached=False, time_s=np.nan, restricted_time_s=end-onset)

def step_metrics(frame, onset, end, v0, v1, full_duration):
    out = {}; t = frame.time_s.to_numpy(float); v = frame.speed_mm_s.to_numpy(float)
    post = (t > onset+1e-9) & (t <= end+1e-8)
    before = (t > onset-.2+1e-9) & (t <= onset+1e-8)
    pre = errors(frame,onset-.2,onset)['observed_mean']
    out['pre_command_bias_mm_s'] = pre-v0
    out['pre_command_within_1_mm_s'] = bool(np.isfinite(pre) and abs(pre-v0) <= 1.)
    progress = (v-v0)/(v1-v0)
    pre_progress=(pre-v0)/(v1-v0)
    out['already_above_10_before_command']=bool(np.isfinite(pre_progress) and pre_progress>=.1)
    out['already_above_90_before_command']=bool(np.isfinite(pre_progress) and pre_progress>=.9)
    out['time_to_90_eligible']=bool(np.isfinite(pre_progress) and pre_progress<.9)
    times = []
    for fraction in [.1, .9]:
        found = np.flatnonzero(post & np.isfinite(progress) & (progress >= fraction))
        times.append(float(t[found[0]]-onset) if len(found) and np.isfinite(pre_progress) and pre_progress<fraction else np.nan)
    out['time_to_10_s'], out['time_to_90_s'] = times
    out['reached_90_fraction'] = bool(np.any(post & np.isfinite(progress) & (progress>=.9)))
    already_above_10 = np.isfinite(pre) and (pre-v0)/(v1-v0) >= .1
    out['rise_time_eligible'] = not already_above_10 and np.isfinite(pre)
    out['rise_10_90_s'] = times[1]-times[0] if out['rise_time_eligible'] and np.isfinite(times).all() else np.nan
    valid = post & np.isfinite(v)
    out['observed_peak_overshoot_mm_s'] = float(max(0., np.max(np.sign(v1-v0)*(v[valid]-v1)))) if valid.any() else np.nan
    out['overshoot_trace_complete'] = bool(full_duration)
    for tolerance in [1., 2.]:
        s = settling(t, v-v1, onset, end, full_duration, tolerance)
        for key, value in s.items(): out[f'settle_{tolerance:g}_{key}'] = value
    early_end = min(onset+.5, end)
    for name, (lo, hi) in {'post':(onset,end), 'early':(onset,early_end), 'tail':(max(onset,end-.5),end)}.items():
        for key,value in errors(frame,lo,hi).items(): out[f'{name}_{key}'] = value
    a = derivative(frame); valid = post & np.isfinite(a)
    out['observed_peak_signed_accel_mm_s2'] = float(np.max(np.sign(v1-v0)*a[valid])) if valid.any() else np.nan
    return out

def ramp_segments(profile):
    if profile['protocol'] == 'legacy':
        starts = [.5, 2.5]; ends = [2., 4.]
    else:
        d = abs(profile['v1']-profile['v0'])/profile['slope']
        starts = [profile['onset'],profile['onset']+d+.5]; ends = [starts[0]+d,starts[1]+d]
    sign = 1 if profile['v1'] > profile['v0'] else -1
    return [(starts[i],ends[i],profile['slope']*sign*(1 if i==0 else -1)) for i in range(2)]

def ramp_metrics(frame, profile):
    t = frame.time_s.to_numpy(float); v = frame.speed_mm_s.to_numpy(float)
    r = frame.target_mm_s.to_numpy(float); a = derivative(frame); a_sensitivity = derivative(frame,seconds=.21)
    rows = []; bins = {}
    for lo, hi, requested in ramp_segments(profile):
        direction = 'up' if requested > 0 else 'down'
        inner = (t > lo+.1) & (t <= hi-.1+1e-8) & np.isfinite(v)
        complete = len(t) > 0 and t[-1] >= hi-1e-8 and int(inner.sum()) >= 5
        row = dict(direction=direction, requested_accel_mm_s2=requested, start_s=lo, end_s=hi,
                   segment_complete=complete)
        row.update(errors(frame,lo,hi))
        row['fitted_accel_mm_s2'] = float(np.polyfit(t[inner],v[inner],1)[0]) if complete else np.nan
        row['fitted_accel_error_mm_s2'] = row['fitted_accel_mm_s2']-requested
        for label, values in [('accel',a),('accel_sensitivity',a_sensitivity)]:
            ok = inner & np.isfinite(values)
            row[f'{label}_rmse_mm_s2'] = float(np.sqrt(np.mean((values[ok]-requested)**2))) if complete and ok.any() else np.nan
        rows.append(row)
        select = (t > lo+1e-9) & (t <= hi+1e-8) & np.isfinite(v)
        centers = np.arange(8.2,16.,.4)
        bins[direction] = [float(np.mean(v[select & (abs(r-center)<.2)])) if np.any(select & (abs(r-center)<.2)) else np.nan for center in centers]
    bup = np.asarray(bins['up']); bdown = np.asarray(bins['down']); ok = np.isfinite(bup)&np.isfinite(bdown)
    hysteresis = float(np.mean(np.abs(bup[ok]-bdown[ok]))) if ok.all() and all(x['segment_complete'] for x in rows) else np.nan
    return rows, hysteresis

def sine_metrics(frame, profile, full_duration):
    t = frame.time_s.to_numpy(float); v = frame.speed_mm_s.to_numpy(float); r = frame.target_mm_s.to_numpy(float)
    ok = (t > profile['onset']+1e-9)&np.isfinite(v)&np.isfinite(r)
    if not full_duration or ok.sum() < 100:
        return dict(frequency_hz=profile['frequency'], frequency_fit_valid=False, gain=np.nan, phase_lag_deg=np.nan, phase_delay_s=np.nan, response_offset_mm_s=np.nan, harmonic_residual_rms_mm_s=np.nan, harmonic_r_squared=np.nan)
    w = 2*np.pi*profile['frequency']; phase = w*(t[ok]-profile['onset'])
    X = np.column_stack([np.ones(ok.sum()),np.sin(phase),np.cos(phase)])
    bv = np.linalg.lstsq(X,v[ok],rcond=None)[0]; br = np.linalg.lstsq(X,r[ok],rcond=None)[0]
    phase_difference = np.angle(np.exp(1j*(np.arctan2(bv[2],bv[1])-np.arctan2(br[2],br[1]))))
    return dict(frequency_hz=profile['frequency'], frequency_fit_valid=True,
        gain=float(np.hypot(bv[1],bv[2])/np.hypot(br[1],br[2])), phase_lag_deg=float(-np.degrees(phase_difference)),
        phase_delay_s=float(-phase_difference/w), response_offset_mm_s=float(bv[0]-br[0]),
        harmonic_residual_rms_mm_s=float(np.sqrt(np.mean((v[ok]-X@bv)**2))),
        harmonic_r_squared=float(1-np.sum((v[ok]-X@bv)**2)/np.sum((v[ok]-np.mean(v[ok]))**2)) if np.std(v[ok])>1e-10 else np.nan)

def heading_diagnostics(frame, start, end):
    """Post-hoc diagnosis of heading drift, without redefining target speed.

    Frozen environment heading_rate is accumulated yaw / nominal 10 ms,
    including a short terminal interval. Numerical returns may contain a stale
    rate and are excluded. Planar chord speed uses consecutive position samples;
    it is neither body-axis speed nor proof of correct task performance.
    """
    if not {'heading_rate_rad_s','x_mm','y_mm'} <= set(frame):return {}
    f=frame.copy();rate=f.heading_rate_rad_s.to_numpy(float)
    good=(f.dt_s.to_numpy()>1e-12)&np.isfinite(rate)&~f.numerical_failure.to_numpy(bool)
    # Only a contiguous observed prefix can support integrated heading.
    bad=np.flatnonzero(~good)
    if len(bad):good[int(bad[0]):]=False
    yaw=np.cumsum(np.where(good,rate*.01,0.));yaw[~good]=np.nan
    wrapped=np.abs(np.rad2deg(np.angle(np.exp(1j*yaw))))
    w=weights(f,start,end);w=np.where(np.isfinite(yaw),w,0.);covered=float(w.sum())
    out={'heading_observed_seconds':covered,
         'heading_abs_mean_deg':float(np.sum(w*np.nan_to_num(wrapped))/covered) if covered else np.nan,
         'heading_final_unwrapped_deg':float(np.rad2deg(yaw[good][-1])) if good.any() else np.nan,
         'heading_abs_gt30_fraction':float(np.sum(w*(wrapped>30))/covered) if covered else np.nan,
         'heading_ever_over90':bool(np.any((w>0)&(wrapped>90))) if covered else np.nan}
    xy=f[['x_mm','y_mm']].to_numpy(float)
    planar=np.r_[np.nan,np.linalg.norm(np.diff(xy,axis=0),axis=1)/f.dt_s.to_numpy(float)[1:].clip(1e-12)]
    valid=good&np.isfinite(planar)&np.isfinite(f.raw_speed_mm_s)&np.isclose(f.dt_s,.01,atol=1e-8)
    common=np.where(valid,weights(f,start,end),0.);seconds=float(common.sum())
    out['planar_comparison_observed_seconds']=seconds
    for name,v in [('planar_chord',planar),('initial_axis',f.raw_speed_mm_s.to_numpy(float))]:
        v=np.where(valid,v,0.);target=f.target_mm_s.to_numpy(float)
        out[name+'_raw_observed_mean']=float(np.sum(common*v)/seconds) if seconds else np.nan
        out[name+'_raw_observed_mae']=float(np.sum(common*np.abs(v-target))/seconds) if seconds else np.nan
    return out

def observed_orientation_diagnostics(frame, observations, start, end):
    """Common old/new heading diagnostic from saved 51-D policy inputs.

    Canonical proprioception indices 24:27 contain the world-frame body
    orientation; observation noise and delay are zero in every matched config.
    Angles are sampled at decision START, whereas the trace rows end there.
    Weight the left-endpoint samples by their actual following intervals and
    keep this convention explicit rather than inventing a final observation.
    """
    observations=np.asarray(observations)
    assert observations.shape==(len(frame),51)
    xy=observations[:,24:26].astype(float);norm=np.linalg.norm(xy,axis=1)
    yaw=np.arctan2(xy[:,1],xy[:,0]);relative=np.angle(np.exp(1j*(yaw-yaw[0])))
    valid=np.isfinite(relative)&(norm>1e-8)&(frame.dt_s.to_numpy()>1e-12)&~frame.numerical_failure.to_numpy(bool)
    w=np.where(valid,weights(frame,start,end),0.);covered=float(w.sum());absolute=np.abs(np.rad2deg(relative))
    out={'pre_heading_observed_seconds':covered,
         'pre_heading_abs_mean_deg':float(np.sum(w*np.nan_to_num(absolute))/covered) if covered else np.nan,
         'pre_heading_abs_gt30_fraction':float(np.sum(w*(absolute>30))/covered) if covered else np.nan}
    selected=np.flatnonzero(valid&(w>0))
    out['pre_heading_last_abs_deg']=float(absolute[selected[-1]]) if len(selected) else np.nan
    out['pre_heading_last_sample_time_s']=float((frame.time_s-frame.dt_s).iloc[selected[-1]]) if len(selected) else np.nan
    return out

def score(frame, profile, metadata):
    start = 1. if profile['family']=='static' else profile.get('onset',0.)
    row = {key:metadata[key] for key in ['run_id','model','train_seed','eval_seed','case_id','family','protocol']}
    row.update(full_duration=bool(metadata['full_duration']),flip=bool(metadata['flip']),
               numerical_failure=bool(metadata['numerical_failure']),duration_s=float(metadata['duration_s']))
    row['zero_duration_terminal_events']=int((frame.dt_s<=1e-12).sum())
    row['partial_decision_intervals']=int(((frame.dt_s>1e-12)&(frame.dt_s<.01-1e-8)).sum())
    for key,value in errors(frame,start,profile['duration']).items(): row[f'tracking_{key}'] = value
    for key,value in errors(frame,start,profile['duration'],velocity='raw_speed_mm_s').items(): row[f'raw_tracking_{key}'] = value
    row['late_speed_sd_mm_s'] = weighted_sd(frame,max(start,profile['duration']-.5),profile['duration'])
    row['drive_saturation_fraction'] = float(np.mean(abs(frame.drive.to_numpy(float))>=.95))
    row['turn_abs_mean'] = float(np.mean(abs(frame.turn.to_numpy(float))))
    row['drive_total_variation_per_s'] = float(np.abs(np.diff(frame.drive.to_numpy(float))).sum()/max(float(metadata['duration_s']),.01))
    row.update(heading_diagnostics(frame,start,profile['duration']))
    details = []
    if profile['family']=='static':
        row['target_mm_s'] = profile['target']
        row['late_drift_mm_s'] = errors(frame,2.,3.)['observed_mean']-errors(frame,1.,2.)['observed_mean']
    if profile['family'] in ['step','push']:
        row['v0']=profile['v0'];row['v1']=profile['v1'];row['direction']='up' if profile['v1']>profile['v0'] else 'down'
        row.update(step_metrics(frame,profile['onset'],profile['duration'],profile['v0'],profile['v1'],metadata['full_duration']))
    for seconds in [.1,.2]:
        averaged=cycle_mean(frame,seconds);prefix=f'cycle_mean_{round(seconds*1000)}ms_'
        for key,value in errors(averaged,start,profile['duration']).items():row[prefix+'tracking_'+key]=value
        if profile['family'] in ['step','push']:
            for band in [1.,2.]:
                s=settling(averaged.time_s.to_numpy(),averaged.speed_mm_s.to_numpy()-profile['v1'],
                    profile['onset'],profile['duration'],metadata['full_duration'],band)
                for key,value in s.items():row[prefix+f'settle_{band:g}_'+key]=value
    if profile['family']=='ramp':
        details,hysteresis=ramp_metrics(frame,profile);row['ramp_hysteresis_mm_s']=hysteresis;row['slope_abs_mm_s2']=profile['slope']
    if profile['family']=='sine':row.update(sine_metrics(frame,profile,metadata['full_duration']))
    if profile['family']=='multistep':
        for i in range(1,len(profile['knots'])):
            lo=float(profile['knots'][i]);hi=float(profile['knots'][i+1] if i+1<len(profile['knots']) else profile['duration'])
            v0=profile['values'][i-1];v1=profile['values'][i]
            survived_event = metadata['duration_s']>=hi-1e-8 and (metadata['duration_s']>hi+1e-8 or not (metadata['numerical_failure'] or metadata['flip']))
            details.append(dict(event=i,start_s=lo,end_s=hi,v0=v0,v1=v1,
                **step_metrics(frame,lo,hi,v0,v1,survived_event)))
    return row,[dict(**{k:row[k] for k in ['run_id','model','train_seed','eval_seed','case_id','family','protocol']},**d) for d in details]

def selftest():
    import pandas as pd
    t=np.arange(1,301)*.01
    f=pd.DataFrame(dict(time_s=t,dt_s=np.full(300,.01),target_mm_s=np.full(300,12.),speed_mm_s=np.full(300,10.)))
    assert abs(errors(f,1.,3.)['mae']-2)<1e-12
    assert abs(errors(f.iloc[:150],1.,3.)['mae']-15.5)<1e-12
    e=np.where(t<1.5,3.,0.)
    s=settling(t,e,1.,3.,True,1.);assert s['reached'] and abs(s['time_s']-.5)<1e-12
    e[(t>2.99)]=3.;assert not settling(t,e,1.,3.,True,1.)['reached']
    assert not settling(t,np.zeros(300),1.,3.,False,1.)['reached']
    f['speed_mm_s']=8+4*t;assert np.max(abs(derivative(f)[20:-20]-4))<1e-10
    short=f.iloc[:151].copy();short.loc[short.index[-1],'dt_s']=.003
    assert np.isnan(derivative(short)[-1]) and np.isfinite(derivative(short)[:-1]).all()
    for seconds in [.1,.2]:
        avg=cycle_mean(f,seconds);n=round(seconds/.01)
        assert np.allclose(avg.speed_mm_s.iloc[n:],8+4*(t[n:]-(n-1)*.005),atol=1e-10)
    f['speed_mm_s']=np.full(300,8.);f['target_mm_s']=np.where(t<=1.,16.,8.)
    s=step_metrics(f,1.,3.,16.,8.,True)
    assert s['already_above_90_before_command'] and not s['rise_time_eligible'] and np.isnan(s['time_to_90_s'])
    zero_event=f.iloc[[-1]].copy();zero_event['dt_s']=0.;zero_event['speed_mm_s']=np.nan
    with_failure_event=pd.concat([f,zero_event],ignore_index=True)
    assert errors(with_failure_event,1.,3.)==errors(f,1.,3.)
    assert np.isnan(cycle_mean(with_failure_event,.1).speed_mm_s.iloc[-1])
    empty_time=pd.DataFrame(dict(time_s=[0.],dt_s=[0.],target_mm_s=[12.],speed_mm_s=[np.nan]))
    assert errors(empty_time,1.,3.)['mae']==20. and np.isnan(cycle_mean(empty_time,.1).speed_mm_s.iloc[0])
    t=np.arange(1,901)*.01;freq=.75;delay=.1
    f=pd.DataFrame(dict(time_s=t,target_mm_s=12+4*np.sin(2*np.pi*freq*(t-1)),speed_mm_s=11+2*np.sin(2*np.pi*freq*(t-1-delay))))
    s=sine_metrics(f,dict(onset=1.,frequency=freq),True)
    assert abs(s['gain']-.5)<1e-10 and abs(s['phase_delay_s']-delay)<1e-10 and abs(s['response_offset_mm_s']+1)<1e-10
    assert abs(abs(circular_mean_deg([179.,-179.]))-180.)<1e-10
    assert circular_sd_deg([179.,-179.])<1.01 and np.isnan(circular_mean_deg([0.,180.]))
    # Synthetic half-circle: integrate heading exactly, preserve travel speed,
    # and distinguish direction loss from a motionless body.
    t=np.arange(1,301)*.01;omega=np.pi/3;radius=12/omega
    f=pd.DataFrame(dict(time_s=t,dt_s=np.full(300,.01),target_mm_s=np.full(300,12.),
        x_mm=radius*np.sin(omega*t),y_mm=radius*(1-np.cos(omega*t)),
        raw_speed_mm_s=12*np.cos(omega*(t-.005)),heading_rate_rad_s=np.full(300,omega),numerical_failure=False))
    d=heading_diagnostics(f,1.,3.)
    assert abs(d['heading_final_unwrapped_deg']-180)<1e-9 and d['heading_ever_over90']
    assert abs(d['planar_chord_raw_observed_mean']-12)<1e-3 and d['initial_axis_raw_observed_mean']<0
    assert d['planar_chord_raw_observed_mae']<1e-3 and d['initial_axis_raw_observed_mae']>12
    assert heading_diagnostics(f.drop(columns='heading_rate_rad_s'),1.,3.)=={}
    observations=np.zeros((len(f),51),np.float32)
    observations[:,24]=np.cos(omega*(t-.01));observations[:,25]=np.sin(omega*(t-.01))
    d=observed_orientation_diagnostics(f,observations,1.,3.)
    assert abs(d['pre_heading_abs_mean_deg']-119.7)<1e-5
    assert abs(d['pre_heading_last_abs_deg']-179.4)<1e-5
    assert abs(d['pre_heading_last_sample_time_s']-2.99)<1e-10
    print('PASS: time-weighted windows, failure penalty, sustained settling, analytic acceleration, sinusoidal gain/phase/offset')

if __name__=='__main__':selftest()
