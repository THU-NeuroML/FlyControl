"""Anatomy-defined edge controls, generated before measuring interventions."""
import numpy as np
import pandas as pd
from scipy import sparse

def prepare_interventions(root, m, g):
    O = root / 'data/figure3/mechanisms/interventions'
    (O / 'source_data').mkdir(parents=True, exist_ok=True)
    pre, post, w = [g[k] for k in ['edge_pre', 'edge_post', 'edge_weight']]
    N, S, D = len(m), int(g['sensory_dim']), int(g['output_dim'])
    cells=dict(np.load(root / 'data/figure3/mechanisms/anatomy/source_data/masks.npz'))
    for name in ['ocellar_retinula_cell','PFL2','PFL3','LAL010','DNa01','DNa02','DNa03','DNg13','AN_GNG_172','AN_GNG_175','AN_GNG_196','AN_multi_1']:
        cells[name]=np.flatnonzero(m.primary_type==name)
    cells['steering']=np.unique(np.concatenate([cells[k] for k in ['PFL2','PFL3','LAL010','DNa01','DNa02','DNa03','DNg13']]))
    an=np.isin(pre,cells['true_AN']);edges={k:np.flatnonzero(np.isin(pre,cells[v])) for k,v in [('AN_off','true_AN'),('matched_off','matched_nonAN'),('mech_off','mechanosensory'),('ocellar_off','ocellar_retinula_cell'),('steering_off','steering')]}
    edges['AN_DN_cut']=np.flatnonzero(an&(post>=N-D));edges['AN_I_cut']=np.flatnonzero(an&(post>=S)&(post<N-D))
    # Match one non-AN afferent edge to every AN->DN edge, same postsynaptic DN
    # and weight sign, closest log(abs(weight)), without replacement. A source
    # internal neuron is allowed only if the afferent pool is exhausted.
    matches=[];pairs=[];used=set()
    for target in np.unique(post[edges['AN_DN_cut']]):
        for sign in [-1,1]:
            orig=edges['AN_DN_cut'][(post[edges['AN_DN_cut']]==target)&(np.sign(w[edges['AN_DN_cut']])==sign)]
            pool=np.flatnonzero((post==target)&(~an)&(np.sign(w)==sign))
            for i in orig[np.argsort(-np.abs(w[orig]))]:
                avail=pool[np.array([int(j) not in used for j in pool],dtype=bool)]
                aff=avail[pre[avail]<S];avail=aff if len(aff) else avail
                fallback=False
                if not len(avail):
                    fallback=True
                    other=np.flatnonzero((post>=N-D)&(~an)&(np.sign(w)==sign))
                    other=other[np.array([int(j) not in used for j in other],dtype=bool)]
                    for keep in [(m.primary_type.iloc[post[other]].to_numpy()==m.primary_type.iloc[target])&(m.side.iloc[post[other]].to_numpy()==m.side.iloc[target]),m.side.iloc[post[other]].to_numpy()==m.side.iloc[target],np.ones(len(other),bool)]:
                        if keep.any():avail=other[keep];break
                j=int(avail[np.argmin(abs(np.log(abs(w[avail]))-np.log(abs(w[i]))))]);used.add(j);matches.append(j)
                pairs.append(dict(original_edge=int(i),control_edge=j,post=int(target),control_post=int(post[j]),same_target=bool(post[j]==target),original_pre=int(pre[i]),control_pre=int(pre[j]),original_weight=float(w[i]),control_weight=float(w[j]),control_afferent=bool(pre[j]<S)))
    edges['AN_DN_matchcut']=np.array(matches,dtype=np.int64)
    edges['AN_half']=edges['AN_off'];edges['AN_restore']=edges['AN_off'];edges['intact']=np.array([],dtype=np.int64)
    np.savez_compressed(O/'source_data/cells.npz',**cells);np.savez_compressed(O/'source_data/edges.npz',**edges)
    pd.DataFrame(pairs).to_csv(O/'source_data/edge_matching_pairs.csv.gz',index=False)
    summary=[]
    for name,ids in edges.items():summary.append(dict(mode=name,n_edges=len(ids),n_presynaptic_cells=len(np.unique(pre[ids])),n_postsynaptic_cells=len(np.unique(post[ids])),abs_weight_sum=float(abs(w[ids]).sum()),positive_fraction=float(np.mean(w[ids]>0)) if len(ids) else None))
    pd.DataFrame(summary).to_csv(O/'source_data/intervention_sizes.csv',index=False)
    A=sparse.csr_matrix((abs(w),(post,pre)),shape=(N,N));ain=np.asarray(A[:,cells['true_AN']].sum(1)).ravel();dout=np.asarray(A[-D:].sum(0)).ravel();score=ain*dout;score[:S]=0;score[-D:]=0
    ix=np.flatnonzero(score);x=m.iloc[ix].copy();x['AN_to_relay_abs_weight']=ain[ix];x['relay_to_DN_abs_weight']=dout[ix];x['twohop_abs_product']=score[ix];x.sort_values('twohop_abs_product',ascending=False).to_csv(O/'source_data/AN_twohop_relays.csv.gz',index=False)
    cells['AN_top50_relays']=np.argsort(score)[-50:];np.savez_compressed(O/'source_data/cells.npz',**cells)
    named=np.isin(pre,cells['steering'])&np.isin(post,cells['steering']);ii=np.flatnonzero(named)
    pd.DataFrame(dict(edge_id=ii,pre_state_index=pre[ii],post_state_index=post[ii],pre_root_id=m.root_id.iloc[pre[ii]].to_numpy(),post_root_id=m.root_id.iloc[post[ii]].to_numpy(),pre_type=m.primary_type.iloc[pre[ii]].to_numpy(),post_type=m.primary_type.iloc[post[ii]].to_numpy(),pre_side=m.side.iloc[pre[ii]].to_numpy(),post_side=m.side.iloc[post[ii]].to_numpy(),weight=w[ii])).to_csv(O/'source_data/named_steering_edges.csv',index=False)
