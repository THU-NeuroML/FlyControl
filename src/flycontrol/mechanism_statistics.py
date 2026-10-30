"""Fixed neural comparison families; never select tests by their outcomes."""
import numpy as np
import pandas as pd
from .analysis import signflip,holm


def terrain_comparisons(out):
    adult=pd.read_csv(out/'atlas.csv');er=pd.read_csv(out/'er_atlas.csv')
    keys=['seed','height','group']
    d=adult.merge(er,on=keys,suffixes=('_adult','_er'),validate='one_to_one')
    rows=[]
    for (height,group),q in d.groupby(['height','group']):
        x=(q.enrichment_adult-q.enrichment_er).to_numpy()
        if len(x)!=20 or not np.isfinite(x).all():raise ValueError('Incomplete Adult–ER neural cohort')
        draws=np.random.default_rng(20261021).choice(x,(20000,len(x))).mean(1)
        lo,hi=np.quantile(draws,[.025,.975])
        rows.append(dict(height=height,group=group,adult_mean=q.enrichment_adult.mean(),er_mean=q.enrichment_er.mean(),difference=x.mean(),ci95_low=lo,ci95_high=hi,p_raw=signflip(x),n=len(x)))
    q=pd.DataFrame(rows)
    if len(q)!=45:raise ValueError('Expected all 45 Adult–ER contrasts')
    q['p_holm45']=holm(q.p_raw);q.to_csv(out/'adult_er_45_contrasts.csv',index=False)


def comparison(p):
 a=p.connectome.to_numpy();b=p.mlp.to_numpy();assert (a>0).all() and (b>0).all()
 d=np.log2(a/b);n=len(d);draw=np.random.default_rng(20261021).integers(0,n,(50000,n));lo,hi=2**np.quantile(d[draw].mean(axis=1),[.025,.975])
 return dict(n_pairs=n,adult_mean=float(a.mean()),mlp_mean=float(b.mean()),geometric_ratio=float(2**d.mean()),ratio_ci_low=float(lo),ratio_ci_high=float(hi),adult_lower_pairs=int((d<0).sum()),p_exact=signflip(d))


def terrain_gain_comparisons(out):
    metrics=['policy_kl','unit_kl','action_mse'];screen=pd.read_csv(out/'input_screen.csv');surface=pd.read_csv(out/'input_gain.csv')
    rows=[]
    units=screen.groupby(['training_seed','method','axis'])[metrics].mean().reset_index()
    for cohort,predicate in [('all20',units.training_seed>0),('discovery10',units.training_seed<=2020),('validation10',units.training_seed>=2021)]:
        for axis,g in units[predicate].groupby('axis'):
            for metric in metrics:rows.append(dict(cohort=cohort,axis=axis,metric=metric,**comparison(g.pivot(index='training_seed',columns='method',values=metric).sort_index())))
    q=pd.DataFrame(rows)
    for _,idx in q.groupby('cohort').groups.items():q.loc[idx,'p_holm30']=holm(q.loc[idx,'p_exact'])
    selected=q.cohort.eq('validation10')&q.axis.isin(['Velocity','Joint velocity'])&q.metric.isin(metrics[:2]);assert selected.sum()==4
    q.loc[selected,'p_holm4']=holm(q.loc[selected,'p_exact']);q.to_csv(out/'gain_axis_contrasts.csv',index=False)
    # The registered held-out inference used an 11 x 11 grid. The 21 x 21 display
    # refines that same plane; retain the original grid for these tests.
    d=surface[np.isclose(surface.x*50,np.round(surface.x*50))&np.isclose(surface.y*50,np.round(surface.y*50))]
    d=d[(abs(d.x)>1e-10)|(abs(d.y)>1e-10)]
    units=d.groupby(['training_seed','method'])[metrics].mean().reset_index();rows=[]
    for cohort,predicate in [('all20',units.training_seed>0),('discovery10',units.training_seed<=2020),('validation10',units.training_seed>=2021)]:
        for metric in metrics:rows.append(dict(cohort=cohort,metric=metric,**comparison(units[predicate].pivot(index='training_seed',columns='method',values=metric).sort_index())))
    q=pd.DataFrame(rows)
    for cohort in q.cohort.unique():
        mask=q.cohort.eq(cohort)&q.metric.isin(metrics[:2]);q.loc[mask,'p_holm2']=holm(q.loc[mask,'p_exact'])
    q['p_bonferroni90']=np.minimum(90*q.p_exact,1);q.to_csv(out/'gain_surface_contrasts.csv',index=False)


def turning_intervention_comparisons(raw,out):
    """All ten intact comparisons and the separate four specificity contrasts."""
    from scipy.stats import t,ttest_1samp
    modes=['AN_off','AN_half','AN_restore','matched_off','mech_off','ocellar_off',
           'AN_DN_cut','AN_I_cut','AN_DN_matchcut','steering_off']
    wide=raw[raw.force.eq(0)].pivot(index=['seed','phase','angle'],columns='mode',values='command_auc_rad_s')
    expected=pd.MultiIndex.from_product([range(4301,4325),[790101,790102],[-60,60]],names=['seed','phase','angle'])
    if set(wide.columns)!=set(modes+['intact']) or not wide.index.sort_values().equals(expected.sort_values()):
        raise ValueError('The turning intervention family requires every registered model, state and mode')
    if not np.isfinite(wide.to_numpy()).all():raise ValueError('Nonfinite turning intervention outcomes')
    delta=wide[modes].subtract(wide.intact,axis=0).groupby('seed').mean()
    delta.to_csv(out/'intervention_seed_effects.csv')
    def summary(x):
        x=np.asarray(x);mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
        margin=float(t.ppf(.975,len(x)-1)*se)
        p=1. if np.all(x==0) else float(ttest_1samp(x,0).pvalue)
        return dict(difference=mean,ci95_low=mean-margin,ci95_high=mean+margin,p_raw=p,n=len(x))
    q=pd.DataFrame([dict(mode=mode,**summary(delta[mode])) for mode in modes])
    q['p_holm10']=holm(q.p_raw);q.to_csv(out/'intervention_vs_intact_statistics.csv',index=False)
    pairs=[('AN_off','matched_off'),('AN_DN_cut','AN_I_cut'),('AN_DN_cut','AN_DN_matchcut'),('AN_off','mech_off')]
    q=pd.DataFrame([dict(a=a,b=b,**summary(delta[a]-delta[b])) for a,b in pairs])
    q['p_holm4']=holm(q.p_raw);q.to_csv(out/'specificity_statistics.csv',index=False)
