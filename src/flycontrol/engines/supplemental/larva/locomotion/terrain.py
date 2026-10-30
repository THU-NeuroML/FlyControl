"""Serializable terrain specifications for locomotion OOD experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping

from flygym.arena import (
    BlocksTerrain,
    FlatTerrain,
    GappedTerrain,
    MixedTerrain,
    RandomMixedTerrain,
)


TERRAIN_NAMES = ("flat", "gapped", "blocks", "mixed", "random_mixed")
DEFAULT_FRICTION = (1.0, 0.005, 0.0001)


def _finite_positive(value: float, field: str) -> float:
    value = float(value)
    if not 0.0 < value < float("inf"):
        raise ValueError(f"{field} must be finite and positive")
    return value


@dataclass(frozen=True)
class TerrainSpec:
    """Complete, immutable description of one terrain instance.

    ``None`` means "use FlyGym's legacy constructor default". This matters:
    ``TerrainSpec(name, seed)`` must remain exactly equivalent to the old
    ``make_arena(name, seed)`` call.
    """

    name: str
    seed: int = 0
    gap_width: float | None = None
    block_height: float | None = None
    friction: tuple[float, float, float] | None = None
    observation_noise_std: float | tuple[float, ...] | None = None
    observation_delay: int | None = None
    action_delay: int | None = None
    body_mass_scale: float | None = None
    adhesion_force_scale: float | None = None
    adhesion_dropout_probability: float | None = None
    layout_mode: str = "legacy"

    def __post_init__(self) -> None:
        name = str(self.name).lower()
        if name not in TERRAIN_NAMES:
            raise ValueError(f"Unknown terrain: {self.name}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "seed", int(self.seed))
        if self.layout_mode not in {"legacy", "course_v1"}:
            raise ValueError("layout_mode must be legacy or course_v1")
        if self.layout_mode == "course_v1" and name == "random_mixed":
            raise ValueError("course_v1 uses explicit flat/gapped/blocks/mixed families")
        if self.gap_width is not None:
            object.__setattr__(
                self, "gap_width", _finite_positive(self.gap_width, "gap_width")
            )
        if self.block_height is not None:
            block_height = float(self.block_height)
            if block_height < 0.0 or not math.isfinite(block_height):
                raise ValueError("block_height must be finite and non-negative")
            object.__setattr__(
                self,
                "block_height",
                block_height,
            )
        if self.friction is not None:
            friction = tuple(float(x) for x in self.friction)
            if len(friction) != 3 or any(
                x < 0.0 or not math.isfinite(x) for x in friction
            ):
                raise ValueError(
                    "friction must contain three finite non-negative values"
                )
            object.__setattr__(self, "friction", friction)
        if self.observation_noise_std is not None:
            noise = self.observation_noise_std
            if isinstance(noise, (list, tuple)):
                noise = tuple(float(x) for x in noise)
                if any(x < 0.0 or not math.isfinite(x) for x in noise):
                    raise ValueError(
                        "observation_noise_std must be finite and non-negative"
                    )
            else:
                noise = float(noise)
                if noise < 0.0 or not math.isfinite(noise):
                    raise ValueError(
                        "observation_noise_std must be finite and non-negative"
                    )
            object.__setattr__(self, "observation_noise_std", noise)
        for field in ("observation_delay", "action_delay"):
            value = getattr(self, field)
            if value is not None:
                value = int(value)
                if value < 0:
                    raise ValueError(f"{field} must be non-negative")
                object.__setattr__(self, field, value)
        for field in ("body_mass_scale", "adhesion_force_scale"):
            value = getattr(self, field)
            if value is not None:
                value = float(value)
                lower_ok = value > 0.0 if field == "body_mass_scale" else value >= 0.0
                if not lower_ok or not math.isfinite(value):
                    qualifier = (
                        "positive" if field == "body_mass_scale" else "non-negative"
                    )
                    raise ValueError(f"{field} must be finite and {qualifier}")
                object.__setattr__(self, field, value)
        if self.adhesion_dropout_probability is not None:
            dropout = float(self.adhesion_dropout_probability)
            if not 0.0 <= dropout <= 1.0:
                raise ValueError("adhesion_dropout_probability must be in [0, 1]")
            object.__setattr__(self, "adhesion_dropout_probability", dropout)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "TerrainSpec":
        """Build from JSON/YAML data, accepting ``terrain`` as a name alias."""
        data = dict(value)
        if "name" not in data and "terrain" in data:
            data["name"] = data.pop("terrain")
        if "name" not in data and "family" in data:
            data["name"] = data.pop("family")
        if "layout_seed" in data and "seed" not in data:
            data["seed"] = data.pop("layout_seed")
        aliases = {
            "gap_width_mm": "gap_width",
            "block_height_mm": "block_height",
            "obs_noise_std": "observation_noise_std",
            "obs_delay_steps": "observation_delay",
            "action_delay_steps": "action_delay",
            "adhesion_dropout": "adhesion_dropout_probability",
        }
        for alias, canonical in aliases.items():
            if alias in data and canonical not in data:
                data[canonical] = data.pop(alias)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete effective JSON/YAML-safe representation."""
        gap_width = self.gap_width
        if gap_width is None and self.name in {"gapped", "mixed"}:
            gap_width = 0.3
        if self.name in {"flat", "blocks"}:
            gap_width = None
        block_height = self.block_height
        if block_height is None and self.name in {"blocks", "mixed"}:
            block_height = 0.35
        if self.name in {"flat", "gapped"}:
            block_height = None
        friction = self.friction or DEFAULT_FRICTION
        noise = self.observation_noise_std
        if isinstance(noise, tuple):
            noise = list(noise)
        result = {
            "name": self.name,
            "seed": self.seed,
            "gap_width": gap_width,
            "block_height": block_height,
            "friction": list(friction),
            "observation_noise_std": noise,
            "observation_delay": self.observation_delay,
            "action_delay": self.action_delay,
            "body_mass_scale": self.body_mass_scale,
            "adhesion_force_scale": self.adhesion_force_scale,
            "adhesion_dropout_probability": self.adhesion_dropout_probability,
        }
        # Preserve canonical IDs of all existing legacy protocols.
        if self.layout_mode != "legacy":
            result["layout_mode"] = self.layout_mode
        return result

    @property
    def cache_key(self) -> tuple[Any, ...]:
        """Normalized, hashable key containing every geometry parameter."""
        data = self.to_dict()
        key = (
            data["name"],
            data["seed"],
            data["gap_width"],
            data["block_height"],
            tuple(data["friction"]),
            (
                tuple(data["observation_noise_std"])
                if isinstance(data["observation_noise_std"], list)
                else data["observation_noise_std"]
            ),
            data["observation_delay"],
            data["action_delay"],
            data["body_mass_scale"],
            data["adhesion_force_scale"],
            data["adhesion_dropout_probability"],
        )
        return key if self.layout_mode == "legacy" else (*key, self.layout_mode)

    @property
    def terrain_spec_id(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return f"{self.name}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def terrain_spec_id(spec: TerrainSpec | Mapping[str, Any]) -> str:
    """Functional form convenient for tabular evaluation code."""
    if not isinstance(spec, TerrainSpec):
        spec = TerrainSpec.from_mapping(spec)
    return spec.terrain_spec_id


def make_arena_from_spec(spec: TerrainSpec):
    """Construct an arena without changing legacy constructor defaults."""
    if spec.layout_mode == "course_v1":
        from .terrain_course import TerrainCourse
        data = spec.to_dict()
        return TerrainCourse(
            family=spec.name, seed=spec.seed,
            gap_width=.3 if data["gap_width"] is None else data["gap_width"],
            block_height=.35 if data["block_height"] is None else data["block_height"],
            friction=tuple(data["friction"]),
        )
    common = {}
    if spec.friction is not None:
        common["friction"] = spec.friction

    if spec.name == "flat":
        arena = FlatTerrain(**common)
        return _set_all_friction(arena, spec.friction)
    if spec.name == "gapped":
        if spec.gap_width is not None:
            common["gap_width"] = spec.gap_width
        arena = GappedTerrain(**common)
        return _set_all_friction(arena, spec.friction)
    if spec.name == "blocks":
        if spec.block_height is not None:
            common["height_range"] = (spec.block_height, spec.block_height)
        arena = BlocksTerrain(rand_seed=spec.seed, **common)
        return _set_all_friction(arena, spec.friction)
    if spec.name == "mixed":
        if spec.gap_width is not None:
            common["gap_width"] = spec.gap_width
        if spec.block_height is not None:
            common["height_range"] = (spec.block_height, spec.block_height)
        arena = MixedTerrain(rand_seed=spec.seed, **common)
        return _set_all_friction(arena, spec.friction)

    # RandomMixedTerrain exposes ranges. A scalar OOD level deliberately
    # becomes a degenerate range, so the requested level is exact.
    if spec.gap_width is not None:
        common["gap_width_range"] = (spec.gap_width, spec.gap_width)
    if spec.block_height is not None:
        common["block_height_range"] = (spec.block_height, spec.block_height)
    arena = RandomMixedTerrain(rand_seed=spec.seed, **common)
    return _set_all_friction(arena, spec.friction)


def _set_all_friction(arena, friction):
    """Apply custom friction to every surface, including gap base planes."""
    if friction is not None:
        for geom in arena.root_element.find_all("geom"):
            geom.friction = friction
    return arena
