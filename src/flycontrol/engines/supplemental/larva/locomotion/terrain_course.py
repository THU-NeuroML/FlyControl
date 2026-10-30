"""Versioned obstacle courses with real layout variation and a supported finish.

This is an extended benchmark, separate from FlyGym's legacy fixed terrains.
Nominal gap width and block height remain constant across geometry seeds.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from flygym.arena.base import BaseArena
from flygym.arena.complex_terrain import RandomMixedTerrain


class TerrainCourse(RandomMixedTerrain):
    """Three obstacle sections, shared start/finish, seeded local arrangement."""

    def __init__(self, *, family: str, seed: int, gap_width: float = .3,
                 block_height: float = .35,
                 friction: tuple[float, float, float] = (1., .005, .0001)):
        if family not in {"flat", "gapped", "blocks", "mixed"}:
            raise ValueError(f"course_v1 does not support {family}")
        BaseArena.__init__(self)
        self.x_range = (-10., 25.)
        self.y_range = (-6., 6.)
        self.friction = friction
        self._max_floor_height = 0.
        rng = np.random.RandomState(seed)
        common = dict(friction=friction, ground_alpha=1.)
        self._add_prism("course_start", -3.75, 0., 12.5, 12., 0., -2., **common)
        self._add_prism("course_finish", 22.25, 0., 5.5, 12., 0., -2., **common)
        boundaries = (2.5, 8., 13.5, 19.5)
        families = [family] * 3
        if family == "mixed":
            # Every mixed course includes both families; seed changes order
            # and local geometry without choosing an easier family subset.
            families = ["gapped", "blocks", "gapped"]
            if rng.randint(2):
                families = ["blocks", "gapped", "blocks"]
        self.course_sections = []
        for index, kind in enumerate(families):
            start, end = boundaries[index:index + 2]
            section = dict(index=index, family=kind, x_start=start, x_end=end)
            self.course_sections.append(section)
            if kind == "flat":
                self._add_prism(f"course_flat_{index}", (start + end) / 2, 0.,
                                end - start, 12., 0., -2., **common)
            elif kind == "gapped":
                self._build_gapped_segment(
                    rand_state=rng, segment_idx=index, x_start=start, x_end=end,
                    gap_width_range=(gap_width, gap_width),
                    block_width_range=(.7, 1.5), gap_depth_range=(2., 2.), **common)
            else:
                self._build_blocks_segment(
                    rand_state=rng, segment_idx=index, x_start=start, x_end=end,
                    block_size_range=(.9, 1.6),
                    block_height_range=(block_height, block_height),
                    block_spacing=0., **common)
        self.geometry_sha256 = geometry_sha256(self)

    def _add_plane(self, name, x_start, x_end, z_pos, ground_alpha):
        # MuJoCo planes are infinite collision surfaces regardless of visual
        # size. Use finite slabs so a block floor cannot fill another gap.
        self._add_prism(name, (x_start + x_end) / 2, 0., x_end - x_start,
                        12., z_pos, z_pos - .1, self.friction, ground_alpha)


def geometry_sha256(arena) -> str:
    """Hash physical geometry, excluding spec labels and seed identifiers."""
    def values(value):
        return None if value is None else np.asarray(value).tolist()

    rows = [dict(type=str(g.type), pos=values(g.pos), size=values(g.size),
                 friction=values(g.friction))
            for g in arena.root_element.find_all("geom")]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()
