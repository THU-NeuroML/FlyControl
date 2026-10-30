from flycontrol.experiments.supp_s5.common import *
import gc
FEATURES={'body':list(range(28)),'feet':list(range(18)),'contact':list(range(18,24)),'orientation':list(range(24,27)),'height':[27],'CPG':list(range(28,46)),'phase':list(range(28,40)),'amplitude':list(range(40,46)),'target':[46],'current_speed':[47],'speed_error':[48],'speed_fields':[46,47,48],'heading_rate':[49],'mode':[50]}
def prepare():
    sources=[]
    def record(p):sources.append(dict(path=str(Path(p).resolve()),sha256=sha(p)))
    from flycontrol.connectomes.anatomy import adult_nodes
    adult,_=adult_nodes(ROOT, ROOT/'inputs/graphs/speed_adult_None.npz')
    adult['root_id']=adult.root_id.astype(str)
    if 'partition' not in adult: raise ValueError('Missing adult partitions')
    bp=ROOT/'inputs/banc626/neurons.csv.gz';banc=pd.read_csv(bp,dtype=str).fillna('unassigned');record(bp)
    banc=banc.rename(columns={'Root ID':'root_id','Super Class':'super_class','Class':'class','Sub Class':'sub_class','Primary Cell Type':'primary_type','Soma side':'side','Body Part':'body_part','Function':'function'})
    graphs=[]
    for model in [m for m in MODELS if m!='er']:
        run=next(r for r in RUNS if r['model']==model);m=load(run,'initial');g=m['graph']
        ids=[str(x) for x in (*g.sensory_ids,*g.internal_ids,*g.output_ids)]
        if model=='adult':
            d=adult.copy();assert d.root_id.tolist()==ids
        else:
            d=pd.DataFrame(dict(state_index=np.arange(g.N),root_id=ids)).merge(banc,on='root_id',how='left',validate='one_to_one').fillna('unassigned')
            d['partition']=np.where(d.state_index<g.sensory_dim,'sensory',np.where(d.state_index>=g.N-g.output_dim,'output','internal'))
        d['refined_group']=d['class']
        d.loc[(d['class']=='AN')&(d.super_class=='ascending'),'refined_group']='true_AN'
        d.loc[(d['class']=='AN')&(d.super_class=='sensory_ascending'),'refined_group']='SA'
        w=m['W'];wd=w[-m['D']:,:];wc=w.tocsc()
        d['out_degree']=np.diff(wc.indptr);d['direct_output_degree']=np.diff(wd.tocsc().indptr)
        d['out_l1']=np.asarray(abs(w).sum(0)).ravel();d['out_l2_squared']=np.asarray(w.multiply(w).sum(0)).ravel()
        d['direct_output_l1']=np.asarray(abs(wd).sum(0)).ravel();d['direct_output_l2_squared']=np.asarray(wd.multiply(wd).sum(0)).ravel()
        d.to_csv(O/'source_data'/f'{model}_nodes.csv.gz',index=False)
        for (part,group),v in d.groupby(['partition','refined_group']):graphs.append(dict(model=model,partition=part,group=group,n_cells=len(v),direct_output_edges=int(v.direct_output_degree.sum()),direct_output_l2_squared=v.direct_output_l2_squared.sum()))
        print('Mapping',model,d.iloc[:g.sensory_dim].groupby('refined_group').size().to_dict(),flush=True)
        del m;gc.collect()
    pd.DataFrame(graphs).to_csv(O/'source_data/atlas_group_inventory.csv',index=False)
    histories=[];windows=[];means=[];seconds=[]
    for model in MODELS:
        for es in [101003,101004]:
            for task in ['step','ramp']:
                for direction in ['up','down']:
                    case=f'legacy_{task}_{direction}'+('_+0' if task=='step' else '')
                    q=ROOT/'evaluation/speed'/f'{model}_seed0'/f'eval_seed{es}'/(case+'.npz')
                    assert (q.parents[1]/'complete.json').exists()
                    x=np.load(q)['obs'];assert x.shape[1]==51 and np.isfinite(x).all()
                    row=dict(history=len(histories),source_model=model,source_train_seed=0,eval_seed=es,task=task,direction=direction,path=str(q),sha256=sha(q),steps=len(x));histories.append(row)
                    xx=x.astype(float);means.append(xx.mean(0));seconds.append(xx.T@xx/len(xx))
                    for onset in ([90,105] if task=='step' else [75,275]):
                        phase='before_switch' if task=='step' and onset==90 else ('acceleration' if (direction=='up')==(task=='step' or onset==75) else 'deceleration')
                        windows.append(dict(**row,onset_step=onset,phase=phase,window_updates=30,included=bool(len(x)>=onset+30),reason='' if len(x)>=onset+30 else 'observed history ends before full window'))
    write(O/'source_data/history_manifest.json',histories);pd.DataFrame(windows).to_csv(O/'source_data/window_manifest.csv',index=False)
    mean=np.mean(means,0);second=np.mean(seconds,0);cov=second-np.outer(mean,mean);np.savez_compressed(O/'source_data/common_input_statistics.npz',mean=mean,covariance=cov,std=np.sqrt(np.maximum(np.diag(cov),0)))
    rows=[];subrows=[];nodes=[]
    for run in RUNS:
        ini=load(run,'initial');fin=load(run,'final');assert ini['edge_sha']==fin['edge_sha']
        for m in [ini,fin]:sources.append(dict(path=m['path'],sha256=m['sha']))
        mp=mapping(run['model']).iloc[:ini['S']];ids=[str(x) for x in ini['graph'].sensory_ids];assert mp.root_id.tolist()==ids
        for stage,m in [('initial',ini),('final',fin)]:
            e=m['E'].astype(float);delta=e-ini['E'].astype(float)
            for feature,cols in FEATURES.items():
                ee=np.sum(e[:,cols]**2,axis=1);de=np.sum(delta[:,cols]**2,axis=1);vv=np.maximum(np.einsum('ij,jk,ik->i',e[:,cols],cov[np.ix_(cols,cols)],e[:,cols]),0)
                for group,g in mp.groupby('refined_group'):
                    ii=g.index.to_numpy();fraction=len(ii)/m['S']
                    rows.append(dict(run_id=run['run_id'],model=run['model'],train_seed=run['train_seed'],stage=stage,group=group,feature=feature,n_cells=len(ii),node_fraction=fraction,weight_fraction=ee[ii].sum()/ee.sum(),weight_enrichment=ee[ii].sum()/ee.sum()/fraction,displacement_fraction=de[ii].sum()/de.sum() if de.sum()>0 else np.nan,linear_variance_fraction=vv[ii].sum()/vv.sum() if vv.sum()>0 else np.nan))
                if feature in ['body','CPG','speed_fields']:
                    for grouping in ['sub_class','primary_type']:
                        for (cl,typ),g in mp.groupby(['refined_group',grouping]):
                            ii=g.index.to_numpy();subrows.append(dict(run_id=run['run_id'],model=run['model'],train_seed=run['train_seed'],stage=stage,feature=feature,group=cl,grouping=grouping,label=typ,n_cells=len(ii),node_fraction=len(ii)/m['S'],weight_fraction=ee[ii].sum()/ee.sum(),displacement_fraction=de[ii].sum()/de.sum() if de.sum()>0 else np.nan))
            if stage=='final':
                n=mp[['state_index','root_id','refined_group','sub_class','primary_type']].copy();n['run_id']=run['run_id']
                for feature in ['body','CPG','speed_fields']:
                    cols=FEATURES[feature];ee=(e[:,cols]**2).sum(1);dd=(delta[:,cols]**2).sum(1);n[feature+'_weight_fraction']=ee/ee.sum();n[feature+'_displacement_fraction']=dd/dd.sum()
                nodes.append(n)
        print('Static',run['run_id'],flush=True);del ini,fin;gc.collect()
    pd.DataFrame(rows).to_csv(O/'source_data/projection_group_metrics.csv',index=False);pd.DataFrame(subrows).to_csv(O/'source_data/projection_subtype_metrics.csv.gz',index=False);pd.concat(nodes).to_csv(O/'source_data/projection_node_metrics.csv.gz',index=False)
    pd.DataFrame(sources).drop_duplicates('path').to_csv(O/'source_data/SOURCE_SHA256.csv',index=False)
    write(O/'qa/PREPARE_COMPLETE.json',dict(status='PASS',models=len(RUNS),checkpoints=2*len(RUNS),histories=len(histories),windows=len(windows),included_windows=sum(r['included'] for r in windows),code_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json')))
if __name__=='__main__':prepare()
