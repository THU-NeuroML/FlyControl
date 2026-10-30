"""Exact state and one-pulse input derivatives on common speed histories."""
from .common import *
import argparse,time,gc

class Kernel:
    def __init__(self,m,ini,fin,device):
        self.m=m;self.dev=device;self.N=m['N'];self.S=m['S'];self.D=m['D'];self.g=m['gate']
        self.W=sparse(m['W'],device);self.WT=sparse(m['W'].T,device)
        self.E=self.tensor(m['E']);self.eb=self.tensor(m['eb'])
        self.C=self.tensor(np.concatenate([ini['C'],fin['C']],axis=0));self.cb=self.tensor(np.concatenate([ini['cb'],fin['cb']]))
    def tensor(self,x):return torch.as_tensor(x,device=self.dev,dtype=torch.float32)
    def step(self,h,o):
        r=h.tanh();z=torch.sparse.mm(self.W,r.T.contiguous()).T
        z[:,:self.S]+=o@self.E.T+self.eb;u=z.tanh();hn=(1-self.g)*h+self.g*u
        raw=hn[:,-self.D:].tanh()@self.C.T+self.cb
        return hn,raw.tanh(),1-r.square(),self.g*(1-u.square())
    def cache(self):
        histories=pool();windows=pd.read_csv(O/'source_data/window_manifest.csv');windows=windows[windows.included].reset_index(drop=True)
        obs=[];h0=[None]*len(windows);events={}
        for j,r in windows.iterrows():
            start=int(r.onset_step);hist=int(r.history);obs.append(histories[hist][1][start:start+30]);events.setdefault(start,[]).append((j,hist))
        padded=np.zeros((max(events)+1,len(histories),51),np.float32)
        for i,(_,x) in enumerate(histories):padded[:min(len(x),len(padded)),i]=x[:len(padded)]
        h=torch.zeros(len(histories),self.N,device=self.dev);x=self.tensor(padded)
        for t,o in enumerate(x):
            for j,i in events.get(t,[]):h0[j]=h[i].clone()
            h,_,_,_=self.step(h,o)
        self.obs=self.tensor(np.stack(obs,axis=1));self.h0=torch.stack(h0);self.windows=windows;h=self.h0.clone()
        RR=[];BB=[];DD=[];AA=[]
        for o in self.obs:
            h,a,r,b=self.step(h,o);RR.append(r);BB.append(b);DD.append(1-h[:,-self.D:].tanh().square());AA.append(a)
        self.R=torch.stack(RR);self.B=torch.stack(BB);self.DN=torch.stack(DD);self.act=torch.stack(AA)
    def back(self,x,t):
        sh=x.shape
        return torch.sparse.mm(self.WT,(x*self.B[t]).reshape(-1,self.N).T.contiguous()).T.reshape(sh)*self.R[t]
    def gradient(self,horizon):
        f=horizon-1;readout=(1-self.act[f].T[:,:,None].square())*self.C[:,None,:]*self.DN[f][None,:,:]
        lam=torch.zeros((4,len(self.windows),self.N),device=self.dev);lam[:,:,-self.D:]=readout
        zero=lam.clone() if horizon==10 else None;one=torch.zeros_like(lam) if horizon==10 else None
        for t in range(f,-1,-1):
            if t==0:inputadj=(lam*self.B[0])[:,:,:self.S].clone()
            if horizon==10:
                one=(1-self.g)*one+self.back(zero,t);zero=(1-self.g)*zero
            lam=(1-self.g)*lam+self.back(lam,t)
        return lam,inputadj,one,readout
    def pulse(self,du,horizon):
        dh=torch.zeros_like(self.h0);dh[:,:self.S]=self.B[0,:,:self.S]*(du@self.E.T)
        for t in range(1,horizon):dh=(1-self.g)*dh+self.B[t]*torch.sparse.mm(self.W,(dh*self.R[t]).T.contiguous()).T
        return dh

def make_base(run,stage,windows,horizon):
    b=len(windows);base=pd.concat([windows[['history','source_model','eval_seed','task','direction','phase','onset_step']]]*4,ignore_index=True)
    base['run_id']=run['run_id'];base['model']=run['model'];base['train_seed']=run['train_seed'];base['E_stage']=stage
    base['C_stage']=np.repeat(['initial','initial','final','final'],b);base['channel']=np.repeat(['drive','turn','drive','turn'],b);base['updates']=horizon
    return base

def features(obs):
    b=len(obs);std=np.load(O/'source_data/common_input_statistics.npz')['std'];out={}
    def vec():return np.zeros((b,51),np.float32)
    u=vec();u[:,46]=(abs(obs[:,46])<1)/20;u[:,48]=(abs(obs[:,48])<1)/20;out['target_speed']=u
    u=vec();u[:,47]=(abs(obs[:,47])<1)/20;u[:,48]=-(abs(obs[:,48])<1).astype(float)/20;out['current_speed']=u
    u=vec();u[:,48]=(abs(obs[:,48])<1)/20;out['error_field']=u
    u=vec();u[:,28:34]=obs[:,34:40];u[:,34:40]=-obs[:,28:34];out['coherent_cpg_phase']=u
    for name,ids in {'feet_std':list(range(18)),'contact_std':list(range(18,24)),'orientation_std':list(range(24,27)),'height_std':[27],'CPG_std':list(range(28,46))}.items():out[name]=(ids,std[ids])
    return out

def transform(j,feature):
    if isinstance(feature,tuple):
        ids,scale=feature;return j[:,:,ids]*scale
    return (j*feature[None,:,:]).sum(-1,keepdims=True)

def finite_check(k,m,lam,inputadj):
    checks=[];c=2 if m['stage']=='final' else 0;window=0;h0=k.h0[window].cpu().numpy().astype(float)[:,None]
    x=k.obs[:10,window].cpu().numpy().astype(float)
    mm={key:(v.astype(float) if hasattr(v,'astype') else v) for key,v in m.items()}
    g=lam[c,window].cpu().numpy().astype(float);d=np.zeros(k.N);d[:k.S]=g[:k.S];d/=np.linalg.norm(d)
    vals=[];eps=.001
    for sign in [-1,1]:
        h=h0+sign*eps*d[:,None]
        for obs in x:h,a=cpu_step(mm,h,obs[:,None])
        vals.append(a[0,0])
    fd=(vals[1]-vals[0])/(2*eps);pred=float(g@d);err=abs(fd-pred)/max(abs(pred),1e-10);assert err<.005,(fd,pred,err)
    checks.append(dict(check='state_float64_central_difference',stage=m['stage'],analytic=pred,finite=fd,relative_error=err))
    du=features(x[:1])['target_speed'][0];vals=[]
    for sign in [-1,1]:
        h=h0.copy()
        for t,obs in enumerate(x):h,a=cpu_step(mm,h,(obs+sign*eps*du if t==0 else obs)[:,None])
        vals.append(a[0,0])
    fd=(vals[1]-vals[0])/(2*eps);pred=float(inputadj[c,window].cpu().numpy().astype(float)@(mm['E']@du));err=abs(fd-pred)/max(abs(pred),1e-10);assert err<.005,(fd,pred,err)
    checks.append(dict(check='target_float64_central_difference',stage=m['stage'],analytic=pred,finite=fd,relative_error=err))
    h=h0.copy()
    for obs in x:h,a=cpu_step(mm,h,obs[:,None])
    err=float(abs(a[:,0]-k.act[9,0,c:c+2].cpu().numpy()).max());assert err<2e-5
    checks.append(dict(check='independent_cpu_forward',stage=m['stage'],max_abs_error=err))
    return checks

@torch.no_grad()
def run_one(run,device):
    out=O/'models'/run['run_id'];out.mkdir(exist_ok=True)
    if (out/'COMPLETE.json').exists():return
    start=time.time();ini=load(run,'initial');fin=load(run,'final');assert ini['edge_sha']==fin['edge_sha']
    groups=masks(run['model'],ini['S']);states=[];inputs=[];paths=[];checks=[];cells={};dnterms={}
    for stage,m in [('initial',ini),('final',fin)]:
        m=dict(m,stage=stage);k=Kernel(m,ini,fin,device);k.cache();b=len(k.windows);ff=features(k.obs[0].cpu().numpy());native_c=0 if stage=='initial' else 2
        print(run['run_id'],stage,'cache',round(time.time()-start,1),flush=True)
        for horizon in [1,3,10,30]:
            base=make_base(run,stage,k.windows,horizon);lam,iadj,one,readout=k.gradient(horizon)
            energy=lam[:,:,:k.S].double().square().cpu().numpy();total=energy.sum(-1);assert np.all(total>0)
            jtotal=(iadj@k.E).cpu().numpy().astype(float);jreconstructed=np.zeros_like(jtotal)
            for group,ids in groups.items():
                en=energy[:,:,ids].sum(-1);share=en/total
                d=base.copy();d['group']=group;d['n_cells']=len(ids);d['node_fraction']=len(ids)/k.S;d['energy_fraction']=share.ravel();d['enrichment']=(share/(len(ids)/k.S)).ravel();d['mean_squared_derivative']=(en/len(ids)).ravel();states.append(d)
                j=(iadj[:,:,ids]@k.E[ids]).cpu().numpy().astype(float);jreconstructed+=j
                for feature,spec in ff.items():
                    v=transform(j,spec);allv=transform(jtotal,spec);den=(allv**2).sum(-1);num=(v*allv).sum(-1)
                    signed=np.divide(num,den,out=np.full_like(num,np.nan),where=den>1e-24)
                    f=base.copy();f['group']=group;f['feature']=feature;f['gain_norm']=np.sqrt((v*v).sum(-1)).ravel();f['total_gain_norm']=np.sqrt(den).ravel();f['signed_share']=signed.ravel();f['signed_derivative']=v[:,:,0].ravel() if v.shape[-1]==1 else np.nan;f['total_signed_derivative']=allv[:,:,0].ravel() if v.shape[-1]==1 else np.nan;inputs.append(f)
                if horizon==10:
                    v=one[:,:,ids].double().cpu().numpy();g=lam[:,:,ids].double().cpu().numpy();den=(g*g).sum(-1);direct=(v*g).sum(-1)/den
                    f=base.copy();f['group']=group;f['one_edge_signed_share']=direct.ravel();f['multi_edge_signed_share']=(1-direct).ravel();f['one_edge_squared_norm']=(v*v).sum(-1).ravel();f['total_squared_norm']=den.ravel();paths.append(f)
            rel=float(np.linalg.norm(jreconstructed-jtotal)/max(np.linalg.norm(jtotal),1e-20));assert rel<3e-5,rel
            checks.append(dict(check='input_group_additivity',stage=stage,updates=horizon,relative_error=rel))
            if horizon==1:assert float(abs(jtotal).max())==0
            if horizon==10:
                if run['train_seed']==0:checks.extend(finite_check(k,m,lam,iadj))
                for (task,phase),g in k.windows.groupby(['task','phase']):
                    ii=g.index.to_numpy()
                    for ch,name in enumerate(['drive','turn']):cells[f'{stage}__{name}__{task}__{phase}']=lam[native_c+ch,ii].square().mean(0).cpu().numpy()
                for feature in ['target_speed','current_speed']:
                    dh=k.pulse(k.tensor(ff[feature]),10);terms=readout*dh[:,-k.D:][None,:,:]
                    expected=transform(jtotal,ff[feature])[:,:,0];actual=terms.sum(-1).cpu().numpy();rel=float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1e-20));assert rel<3e-5,rel
                    checks.append(dict(check='forward_backward_pulse',stage=stage,feature=feature,relative_error=rel))
                    dnterms[f'{stage}__{feature}']=terms[native_c:native_c+2].cpu().numpy()
            del lam,iadj,one,energy
            print(run['run_id'],stage,'horizon',horizon,'seconds',round(time.time()-start,1),flush=True)
        k.windows.to_csv(out/'windows.csv',index=False);del k;gc.collect();torch.cuda.empty_cache()
    pd.concat(states,ignore_index=True).to_csv(out/'state_sensitivity.csv.gz',index=False);pd.concat(inputs,ignore_index=True).to_csv(out/'input_routes.csv.gz',index=False);pd.concat(paths,ignore_index=True).to_csv(out/'path_decomposition.csv.gz',index=False)
    np.savez_compressed(out/'cell_energy.npz',**cells);np.savez_compressed(out/'output_pulse_terms.npz',**dnterms);pd.DataFrame(checks).to_csv(out/'numerical_checks.csv',index=False)
    write(out/'COMPLETE.json',dict(status='PASS',run_id=run['run_id'],model=run['model'],train_seed=run['train_seed'],initial_checkpoint_sha256=ini['sha'],final_checkpoint_sha256=fin['sha'],edge_sha256=fin['edge_sha'],windows=b,E_C_combinations=4,code_sha256=sha(__file__),common_code_sha256=sha(Path(__file__).with_name('common.py')),seconds=time.time()-start))
    print('COMPLETE',run['run_id'],round(time.time()-start,1),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();run_one(next(r for r in RUNS if r['run_id']==a.run_id),torch.device(a.device))
