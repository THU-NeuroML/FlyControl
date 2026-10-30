"""Pure traversal-state and outcome classification.

The environment emits raw progress. Evaluation code can fold samples through
these functions without coupling metric policy to MuJoCo or Gym wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Sequence

from .terrain import TerrainSpec


class TraversalFailure(str, Enum):
    """Mutually exclusive terminal failure, in classification priority order."""

    NONE = "none"
    PHYSICS_ERROR = "physics_error"
    NAN_DETECTED = "nan"
    FLIPPED = "flip"
    FALL_INTO_GAP = "fall_into_gap"
    BYPASS = "bypass"
    SLIP_BACK = "slip_back"
    STUCK = "stuck"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class TraversalCriteria:
    checkpoints_x: tuple[float, ...]
    stable_time_s: float = 0.0
    lateral_limit_abs_y: float | None = None
    fall_below_z: float | None = None
    stuck_distance_mm: float = 1.0
    continuous_valid_stability: bool = False

    def __post_init__(self) -> None:
        checkpoints = tuple(float(x) for x in self.checkpoints_x)
        if any(b <= a for a, b in zip(checkpoints, checkpoints[1:])):
            raise ValueError("checkpoints_x must be strictly increasing")
        if self.stable_time_s < 0:
            raise ValueError("stable_time_s must be non-negative")
        if self.stuck_distance_mm < 0:
            raise ValueError("stuck_distance_mm must be non-negative")
        object.__setattr__(self, "checkpoints_x", checkpoints)


@dataclass(frozen=True)
class TraversalSample:
    x: float
    y: float = 0.0
    body_z: float | None = None
    elapsed_s: float = 0.0
    flipped: bool = False
    physics_error: bool = False
    nan_detected: bool = False
    upright: bool = True


@dataclass(frozen=True)
class TraversalState:
    next_checkpoint: int = 0
    stable_time_s: float = 0.0
    max_x: float = float("-inf")
    start_x: float | None = None
    bypassed: bool = False
    fallen: bool = False
    last_sample_stable: bool = False


@dataclass(frozen=True)
class TraversalOutcome:
    survived: bool
    success: bool
    failure: TraversalFailure
    checkpoints_passed: int
    stable_time_s: float
    max_x: float


def traversal_criteria(spec: TerrainSpec) -> TraversalCriteria:
    """Return geometry-aware default success criteria for built-in arenas."""
    if spec.layout_mode == "course_v1":
        return TraversalCriteria(
            (8., 13.5, 20.), stable_time_s=.2,
            lateral_limit_abs_y=5., fall_below_z=0.,
            continuous_valid_stability=True,
        )
    if spec.name == "gapped":
        gap = 0.3 if spec.gap_width is None else spec.gap_width
        block = 1.0
        starts = []
        x = -10.0 + block + gap
        while x < 25.0:
            if x > 0.0:
                starts.append(x)
            x += block + gap
        # The predefined test section ends at x=20 mm, leaving 5 mm of
        # supported run-out. Requiring stability at the arena's x=25 edge
        # would make success impossible for a normally moving animal.
        required_platforms = [start for start in starts if start <= 20.0]
        return TraversalCriteria(
            (required_platforms[-1],), stable_time_s=0.2, fall_below_z=0.0
        )
    if spec.name == "blocks":
        return TraversalCriteria((20.0,), lateral_limit_abs_y=20.0)
    if spec.name == "mixed":
        return TraversalCriteria((5.0, 14.0, 23.0))
    return TraversalCriteria((25.0,))


def update_traversal_state(
    state: TraversalState,
    sample: TraversalSample,
    criteria: TraversalCriteria,
    dt: float,
) -> TraversalState:
    """Advance ordered checkpoints and stability using one raw sample."""
    if dt < 0:
        raise ValueError("dt must be non-negative")
    next_checkpoint = state.next_checkpoint
    while (
        next_checkpoint < len(criteria.checkpoints_x)
        and sample.x >= criteria.checkpoints_x[next_checkpoint]
    ):
        next_checkpoint += 1

    all_passed = next_checkpoint == len(criteria.checkpoints_x)
    stable_time = state.stable_time_s + dt if all_passed else 0.0
    if all_passed and criteria.checkpoints_x and sample.x < criteria.checkpoints_x[-1]:
        stable_time = 0.0
    bypassed = state.bypassed
    if criteria.lateral_limit_abs_y is not None:
        bypassed = bypassed or abs(sample.y) > criteria.lateral_limit_abs_y
    below_ground = (criteria.fall_below_z is not None and sample.body_z is not None
                    and sample.body_z < criteria.fall_below_z)
    fallen = state.fallen or (criteria.continuous_valid_stability and below_ground)
    stable_sample = (all_passed and not bypassed and not fallen and not below_ground
                     and sample.upright
                     and not (sample.flipped or sample.nan_detected or sample.physics_error)
                     and (not criteria.checkpoints_x or sample.x >= criteria.checkpoints_x[-1]))
    if criteria.continuous_valid_stability:
        # The first valid sample starts the window: its preceding interval
        # was not observed to be beyond the finish line. Invalid samples reset it.
        stable_time = (state.stable_time_s + dt
                       if stable_sample and state.last_sample_stable else 0.)
    return replace(
        state,
        next_checkpoint=next_checkpoint,
        stable_time_s=stable_time,
        max_x=max(state.max_x, float(sample.x)),
        start_x=float(sample.x) if state.start_x is None else state.start_x,
        bypassed=bypassed,
        fallen=fallen,
        last_sample_stable=stable_sample,
    )


def classify_traversal_outcome(
    state: TraversalState,
    sample: TraversalSample,
    criteria: TraversalCriteria,
    *,
    episode_done: bool = True,
) -> TraversalOutcome:
    """Classify with one failure enum and deterministic failure priority.

    Physics errors outrank NaNs, which outrank flips, bypass and ordinary
    lack of progress. A stuck/time-limited animal survived but did not
    succeed; this distinction is intentional.
    """
    all_passed = state.next_checkpoint == len(criteria.checkpoints_x)
    success = (
        all_passed
        and state.stable_time_s + 1e-12 >= criteria.stable_time_s
        and not state.bypassed
        and not (sample.physics_error or sample.nan_detected or sample.flipped)
    )
    if sample.physics_error:
        failure = TraversalFailure.PHYSICS_ERROR
    elif sample.nan_detected:
        failure = TraversalFailure.NAN_DETECTED
    elif sample.flipped:
        failure = TraversalFailure.FLIPPED
    elif state.bypassed:
        failure = TraversalFailure.BYPASS
    elif state.fallen or (
        criteria.fall_below_z is not None
        and sample.body_z is not None
        and sample.body_z < criteria.fall_below_z
    ):
        failure = TraversalFailure.FALL_INTO_GAP
    elif (
        all_passed
        and criteria.checkpoints_x
        and sample.x < criteria.checkpoints_x[-1]
    ):
        failure = TraversalFailure.SLIP_BACK
    elif episode_done and not success:
        start_x = sample.x if state.start_x is None else state.start_x
        progress = state.max_x - start_x
        failure = (
            TraversalFailure.STUCK
            if progress <= criteria.stuck_distance_mm
            else TraversalFailure.TIMEOUT
        )
    else:
        failure = TraversalFailure.NONE
    # Reaching a checkpoint never overrides physical or functional failure.
    success = success and failure is TraversalFailure.NONE
    survived = failure not in {
        TraversalFailure.PHYSICS_ERROR,
        TraversalFailure.NAN_DETECTED,
        TraversalFailure.FLIPPED,
        TraversalFailure.FALL_INTO_GAP,
    }
    return TraversalOutcome(
        survived=survived,
        success=success,
        failure=failure,
        checkpoints_passed=state.next_checkpoint,
        stable_time_s=state.stable_time_s,
        max_x=state.max_x,
    )


def fold_traversal(
    samples: Sequence[TraversalSample],
    criteria: TraversalCriteria,
) -> TraversalOutcome:
    """Convenience pure fold for a completed sampled trajectory."""
    state = TraversalState()
    previous_time = samples[0].elapsed_s if samples else 0.0
    for sample in samples:
        dt = max(0.0, sample.elapsed_s - previous_time)
        state = update_traversal_state(state, sample, criteria, dt)
        previous_time = sample.elapsed_s
    final = samples[-1] if samples else TraversalSample(0.0)
    return classify_traversal_outcome(state, final, criteria)
