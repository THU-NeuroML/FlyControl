"""Reuse mj_step1 position/velocity cache; validated by EXP53 full trajectories."""
import types,time,argparse
import numpy as np
import mujoco

def install(physics):
    assert physics.legacy_step
    assert int(physics.model.opt.integrator)!=int(mujoco.mjtIntegrator.mjINT_RK4)
    original_step=physics.step; original_forward=physics.forward
    state=dict(cache_valid=False,skipped=0,full=0)
    def step(this,nstep=1):
        state['cache_valid']=False
        result=original_step(nstep)
        state['cache_valid']=True
        return result
    def forward(this):
        if state['cache_valid']:
            with this.check_invalid_state():
                mujoco.mj_forwardSkip(this.model.ptr,this.data.ptr,mujoco.mjtStage.mjSTAGE_VEL,0)
            this._dirty=False
            state['skipped']+=1
        else:
            original_forward();state['full']+=1
        state['cache_valid']=False
    physics.step=types.MethodType(step,physics)
    physics.forward=types.MethodType(forward,physics)
    return state
