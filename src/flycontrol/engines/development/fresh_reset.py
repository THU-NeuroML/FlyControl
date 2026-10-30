"""Isolated training coverage repair. No change to actions, reward, CPG or graph."""
import json
import sys
from pathlib import Path
import numpy as np
import gymnasium as gym

BASE = __import__("flycontrol.paths",fromlist=["run_root"]).run_root()
from experiment import ManagedLegacyEnv


class EpisodeSeedStream:
    def __init__(self,training_seed,rank,n_envs=8):
        if not 0<=rank<n_envs:raise ValueError('Invalid worker rank')
        self.training_seed=int(training_seed);self.rank=rank;self.n_envs=n_envs
        self.restart()

    def restart(self):
        self.rng=np.random.default_rng(np.random.SeedSequence([self.training_seed,self.rank,20260916]))
        self.used=set();self.episode_index=0

    def next(self):
        # Distinct residue classes make different workers disjoint within a run.
        # Keep all new seeds above the fixed legacy/diagnostic evaluation seeds.
        while True:
            code=int(self.rng.integers(1_000_000//self.n_envs+1,(2**31-1)//self.n_envs))
            seed=code*self.n_envs+self.rank
            if seed not in self.used:break
        self.used.add(seed)
        result=(self.episode_index,seed)
        self.episode_index+=1
        return result


class RecordedLegacyEnv(ManagedLegacyEnv):
    def _build_sim(self,terrain_name,seed):
        self.actual_terrain=terrain_name
        super()._build_sim(terrain_name,seed)


class FreshEpisodeReset(gym.Wrapper):
    def __init__(self,env,training_seed,rank,n_envs=8,log_path=None):
        super().__init__(env)
        self.stream=EpisodeSeedStream(training_seed,rank,n_envs)
        self.log_path=Path(log_path) if log_path else None
        self.episode_metadata={}

    def reset(self,*,seed=None,options=None):
        if seed is not None:
            # SB3 VecEnv.seed(s) supplies s+rank on the next reset.
            if seed!=self.stream.training_seed+self.stream.rank:
                raise ValueError('Explicit root seed must equal training_seed + worker rank')
            self.stream.restart()
        index,episode_seed=self.stream.next()
        obs,info=self.env.reset(seed=episode_seed,options=options)
        self.episode_metadata=dict(episode_seed=episode_seed,episode_index=index,
            terrain=self.env.unwrapped.actual_terrain,training_seed=self.stream.training_seed,
            environment_rank=self.stream.rank)
        if self.log_path:
            self.log_path.parent.mkdir(parents=True,exist_ok=True)
            with self.log_path.open('a') as f:f.write(json.dumps(self.episode_metadata)+'\n')
        return obs,{**info,**self.episode_metadata}

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        return obs,reward,terminated,truncated,{**info,**self.episode_metadata}


def make_env(rank,training_seed,monitor_dir=None,n_envs=8):
    """Drop-in factory for a *new* controlled training run, not activated globally."""
    from stable_baselines3.common.monitor import Monitor
    def factory():
        path=Path(monitor_dir) if monitor_dir else None
        env=RecordedLegacyEnv(terrain='mixed',terrain_seed=training_seed+rank,
            randomize_terrain=True,max_time=1.5,decision_interval=100,timestep=1e-4,
            max_correction=10.,flip_penalty=10.,correction_energy_penalty=1e-3)
        env=FreshEpisodeReset(env,training_seed,rank,n_envs,
            log_path=(path/f'env_{rank}_resets.jsonl' if path else None))
        return Monitor(env,filename=str(path/f'env_{rank}') if path else None,
            info_keywords=('executed_physics_steps_total','episode_seed','episode_index','terrain'))
    return factory
