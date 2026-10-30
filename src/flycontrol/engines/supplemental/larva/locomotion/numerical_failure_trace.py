"""Bounded action traces for diagnosing finite-course numerical failures."""
from collections import deque
import copy
import math

import numpy as np


class NumericalFailureTrace:
    def __init__(self, metadata, initial_physics_state, initial_fly, max_time, control_period):
        self.metadata = copy.deepcopy(metadata)
        self.initial_physics_state = np.asarray(initial_physics_state).copy()
        self.initial_fly = np.asarray(initial_fly).copy()
        self.capacity = (min(4096, max(1, math.ceil(max_time / control_period) + 2))
                         if math.isfinite(max_time) else 4096)
        self.records = deque(maxlen=self.capacity)
        self.decisions_seen = 0

    def before_step(self, action, physics_steps, fly):
        self.records.append((np.asarray(action, dtype=np.float32).copy(),
                             int(physics_steps), np.asarray(fly).copy()))
        self.decisions_seen += 1

    def payload(self, last_observed_fly, physics_steps):
        return {
            "schema": "finite_course_numerical_failure_actions_v1",
            "metadata": copy.deepcopy(self.metadata),
            "initial_physics_state": self.initial_physics_state.tolist(),
            "initial_fly": self.initial_fly.tolist(),
            "capacity": self.capacity,
            "decisions_seen": self.decisions_seen,
            "complete_from_reset": self.decisions_seen <= self.capacity,
            "actions": [r[0].tolist() for r in self.records],
            "before_decision_physics_steps": [r[1] for r in self.records],
            "before_decision_fly": [r[2].tolist() for r in self.records],
            "all_actions_finite": all(bool(np.isfinite(r[0]).all()) for r in self.records),
            "last_observed_fly": np.asarray(last_observed_fly).tolist(),
            "last_observed_fly_finite": bool(np.isfinite(last_observed_fly).all()),
            "completed_physics_steps": int(physics_steps),
            "scope": "replay actual environment input actions; does not reconstruct the changing stochastic policy or repair historical failures",
        }
