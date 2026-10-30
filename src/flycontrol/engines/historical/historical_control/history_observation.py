"""Four causal 10-ms sensor frames for each 40-ms policy decision.

The body, micro reward and held action are unchanged. The recurrent actor
consumes frames in order; the feedforward comparator receives all four.
"""
import numpy as np
import gymnasium as gym

def history_task(base, enabled_getter=lambda: True):
    class HistoryTask(base):
        def __init__(self,*args,**kwargs):
            self.history_enabled=bool(enabled_getter())
            self._history_tick=None
            self._history_frames=None
            super().__init__(*args,**kwargs)
            if self.history_enabled:
                self.observation_space=gym.spaces.Box(-np.inf,np.inf,(688,),dtype=np.float32)

        def reset(self,*args,**kwargs):
            self._history_tick=None
            self._history_frames=None
            return super().reset(*args,**kwargs)

        def observation(self):
            current=super().observation()
            if not self.history_enabled:return current
            assert current.shape==(172,)
            tick=int(self._physics_steps)
            if self._history_frames is None:
                self._history_frames=np.repeat(current[None],4,axis=0)
            elif tick!=self._history_tick:
                self._history_frames=np.concatenate((self._history_frames[1:],current[None]),axis=0)
            else:
                self._history_frames[-1]=current
            self._history_tick=tick
            return self._history_frames.reshape(-1).copy()
    return HistoryTask
