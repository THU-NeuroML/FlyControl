"""A finite-horizon traversal objective with an auditable episode return."""
import math


TRAVERSAL_OBJECTIVE_MODE = "traversal_success_potential_v1"


def validate_terrain_objective_configuration(config):
    if config.get("env", {}).get("reward_mode") != TRAVERSAL_OBJECTIVE_MODE:
        return
    ppo = config.get("ppo", {})
    if ppo.get("gamma", .99) != 1. or ppo.get("normalize_reward", True) is not False:
        raise ValueError("traversal_success_potential_v1 requires gamma=1 and normalize_reward=false")


def traversal_objective_transition(previous_potential, progress, elapsed_s, max_time,
                                   *, success, terminal):
    if not (math.isfinite(previous_potential) and 0. <= previous_potential <= 1.
            and math.isfinite(elapsed_s) and elapsed_s >= 0.
            and math.isfinite(max_time) and max_time > 0.):
        raise ValueError("invalid traversal objective state or time")
    if success and not terminal:
        raise ValueError("success must end this finite-horizon task")
    if not terminal and not math.isfinite(progress):
        raise ValueError("nonterminal progress must be finite")
    next_potential = 0. if terminal else min(1., max(0., progress))
    base = (1. - .1 * min(1., elapsed_s / max_time) if success else -1.) if terminal else 0.
    shaping = next_potential - previous_potential
    return base + shaping, {
        "reward_schema": TRAVERSAL_OBJECTIVE_MODE,
        "reward_terminal_base": base,
        "reward_shaping": shaping,
        "reward_potential_before": previous_potential,
        "reward_potential_after": next_potential,
    }
