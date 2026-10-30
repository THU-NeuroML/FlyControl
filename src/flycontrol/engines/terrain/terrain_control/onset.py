import numpy as np


def map_action(action, body_x, rule):
    if not rule:
        return action, 1.0
    u = float(
        np.clip(
            (body_x - rule["start_x_mm"]) / (rule["full_x_mm"] - rule["start_x_mm"]),
            0.0,
            1.0,
        )
    )
    gate = rule["floor"] + (1.0 - rule["floor"]) * u * u * (3.0 - 2.0 * u)
    return (
        action
        if gate == 1.0
        else np.arctanh(gate * np.tanh(np.asarray(action, dtype=np.float32)))
    ), gate
