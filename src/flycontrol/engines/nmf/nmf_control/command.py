"""Low-dimensional command-to-CPG modulation.

This module is intentionally deterministic. It maps policy commands to
CPG gains and does not read reward, terrain, task, or model internals.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CPGModulation:
    frequency_gain: float
    left_amplitude_gain: float
    right_amplitude_gain: float
    step_gain: float = 1.0
    gait_coordinate: float | None = None
    duty_factor: float | None = None
    ipsilateral_lag_cycles: float | None = None
    gait_smoothing_alpha: float = 1.0


@dataclass(frozen=True)
class CommandToCPGAdapter:
    k_drive: float = 0.25
    k_turn: float = 0.25
    k_step: float = 0.25
    k_drive_step: float = 0.0
    frequency_bounds: tuple[float, float] = (0.6, 1.4)
    amplitude_bounds: tuple[float, float] = (0.6, 1.4)
    step_bounds: tuple[float, float] = (0.75, 1.5)
    continuous_gait: bool = False
    slow_duty_factor: float = 0.72
    fast_duty_factor: float = 0.58
    slow_ipsilateral_lag_cycles: float = 1.0 / 3.0
    fast_ipsilateral_lag_cycles: float = 0.5
    swing_duration_s: float = 0.035
    base_frequency_hz: float = 12.0
    gait_smoothing_alpha: float = 0.005

    def __post_init__(self) -> None:
        if not 0.0 < self.fast_duty_factor < 1.0:
            raise ValueError("fast_duty_factor must be in (0, 1)")
        if not 0.0 < self.slow_duty_factor < 1.0:
            raise ValueError("slow_duty_factor must be in (0, 1)")
        if self.fast_duty_factor >= self.slow_duty_factor:
            raise ValueError("fast_duty_factor must be smaller than slow_duty_factor")
        if self.swing_duration_s <= 0.0 or self.base_frequency_hz <= 0.0:
            raise ValueError("swing_duration_s and base_frequency_hz must be positive")
        if not 0.0 < self.gait_smoothing_alpha <= 1.0:
            raise ValueError("gait_smoothing_alpha must be in (0, 1]")

    def continuous_gait_parameters(
        self, gait_coordinate: float
    ) -> tuple[float, float, float]:
        """Return frequency gain, duty factor and ipsilateral lag for ``z``."""
        z = float(np.clip(gait_coordinate, 0.0, 1.0))
        duty_factor = (
            self.slow_duty_factor
            + z * (self.fast_duty_factor - self.slow_duty_factor)
        )
        ipsilateral_lag = (
            self.slow_ipsilateral_lag_cycles
            + z * (
                self.fast_ipsilateral_lag_cycles
                - self.slow_ipsilateral_lag_cycles
            )
        )
        frequency_hz = (1.0 - duty_factor) / self.swing_duration_s
        frequency_gain = float(np.clip(
            frequency_hz / self.base_frequency_hz,
            *self.frequency_bounds,
        ))
        return frequency_gain, duty_factor, ipsilateral_lag

    def map(self, command: np.ndarray) -> CPGModulation:
        command = np.asarray(command, dtype=np.float32)
        if command.shape not in ((2,), (3,)):
            raise ValueError("command must have shape (2,) or (3,)")

        forward_drive = float(np.clip(command[0], -1.0, 1.0))
        turn_bias = float(np.clip(command[1], -1.0, 1.0))
        step_command = float(np.clip(command[2], -1.0, 1.0)) if command.shape == (3,) else 0.0
        stride_command = step_command + self.k_drive_step * forward_drive

        gait_coordinate = 0.5 * (forward_drive + 1.0)
        if self.continuous_gait:
            frequency_gain, duty_factor, ipsilateral_lag = (
                self.continuous_gait_parameters(gait_coordinate)
            )
        else:
            frequency_gain = np.clip(
                1.0 + self.k_drive * forward_drive,
                *self.frequency_bounds,
            )
            duty_factor = None
            ipsilateral_lag = None
        left_amplitude_gain = np.clip(
            1.0 - self.k_turn * turn_bias,
            *self.amplitude_bounds,
        )
        right_amplitude_gain = np.clip(
            1.0 + self.k_turn * turn_bias,
            *self.amplitude_bounds,
        )
        step_gain = np.clip(
            1.0 + self.k_step * stride_command,
            *self.step_bounds,
        )

        return CPGModulation(
            frequency_gain=float(frequency_gain),
            left_amplitude_gain=float(left_amplitude_gain),
            right_amplitude_gain=float(right_amplitude_gain),
            step_gain=float(step_gain),
            gait_coordinate=(gait_coordinate if self.continuous_gait else None),
            duty_factor=duty_factor,
            ipsilateral_lag_cycles=ipsilateral_lag,
            gait_smoothing_alpha=self.gait_smoothing_alpha,
        )
