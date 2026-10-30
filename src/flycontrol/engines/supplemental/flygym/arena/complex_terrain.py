import numpy as np
from typing import Optional
from dm_control import mjcf

from .base import BaseArena


class GappedTerrain(BaseArena):
    """Terrain with horizontal gaps.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : tuple[float, float, float]
        The sliding, torsional, and rolling friction coefficients of the
        ground, by default (1, 0.005, 0.0001).
    x_range : tuple[float, float]
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20).
    y_range : tuple[float, float]Í
        Same as above in y, by default (-10, 10)
    gap_width : float
        Width of each gap, by default 0.4
    block_width : float
        Width of each block (piece of floor), by default 1
    gap_depth : float
        Height of the gaps, by default 2

    Parameters
    ----------
    x_range : tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20)
    y_range : tuple[float, float], optional
        Same as above in y, by default (-10, 10).
    friction : tuple[float, float, float], optional
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).
    gap_width : float, optional
        Width of each gap, by default 0.3.
    block_width : float, optional
        Width of each block (piece of floor), by default 1.
    gap_depth : float, optional
        Height of the gaps, by default 2.
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    scale_bar_pos : tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        gap_width: float = 0.3,
        block_width: float = 1.0,
        gap_depth: float = 2.0,
        ground_alpha: float = 1.0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self.gap_width = gap_width
        self.block_width = block_width
        self.gap_depth = gap_depth

        # add blocks
        block_centers = np.arange(
            x_range[0] + block_width / 2, x_range[1], block_width + gap_width
        )
        box_size = (block_width / 2, (y_range[1] - y_range[0]) / 2, gap_depth / 2)
        for x_pos in block_centers:
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_x{x_pos}",
                size=box_size,
                pos=(x_pos, 0, 0),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )

        # add floor underneath
        ground_size = ((self.x_range[1] - self.x_range[0]) / 2, max(self.y_range), 1)
        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name="ground_base",
            pos=(np.mean(x_range), 0, -gap_depth / 2),
            rgba=(0.3, 0.3, 0.3, ground_alpha),
            size=ground_size,
        )

        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.05, 0.5),
                pos=scale_bar_pos + np.array([0, 0, self.gap_depth / 2]),
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, self.gap_depth / 2])
        return adj_pos, rel_angle

    def _get_max_floor_height(self):
        return self.gap_depth / 2


class BlocksTerrain(BaseArena):
    """Terrain formed by blocks at random heights.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    x_range : tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20).
    y_range : tuple[float, float], optional
        Same as above in y, by default (-10, 10).
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).
    block_size : float
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : tuple[float, float]
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.35, 0.35).

    Parameters
    ----------
    x_range : tuple[float, float], optional
        Range of the arena in the x direction (anterior-posterior axis of
        the fly) over which the block-gap pattern should span, by default
        (-10, 20).
    y_range : tuple[float, float], optional
        Same as above in y, by default (-10, 10).
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.35, 0.35).
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    rand_seed : int, optional
        Seed for generating random block heights, by default 0.
    scale_bar_pos : tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ):
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self.block_size = block_size
        self.height_range = height_range
        rand_state = np.random.RandomState(rand_seed)

        self.max_height = -np.inf

        x_centers = np.arange(x_range[0] + block_size / 2, x_range[1], block_size)
        y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
        for i, x_pos in enumerate(x_centers):
            for j, y_pos in enumerate(y_centers):
                is_i_odd = i % 2 == 1
                is_j_odd = j % 2 == 1

                if is_i_odd != is_j_odd:
                    height = 0.1
                else:
                    height = 0.1 + rand_state.uniform(*height_range)

                self.max_height = max(self.max_height, height)

                self.root_element.worldbody.add(
                    "geom",
                    type="box",
                    name=f"ground_element_x{x_pos}_y{y_pos}",
                    size=(
                        block_size / 2 + 0.1 * block_size / 2,
                        block_size / 2 + 0.1 * block_size / 2,
                        height / 2 + block_size / 2,
                    ),
                    pos=(
                        x_pos,
                        y_pos,
                        height / 2 - block_size / 2,
                    ),
                    rgba=(0.3, 0.3, 0.3, ground_alpha),
                    friction=friction,
                )

        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.1, 0.5),
                pos=scale_bar_pos + np.array([0, 0, 0.1]),
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, 0.1])
        return adj_pos, rel_angle

    def _get_max_floor_height(self):
        return self.max_height


class MixedTerrain(BaseArena):
    """A mixture of flat, blocks, and gaps terrains.

    Attributes
    ----------
    root_element : mjcf.RootElement
        The root MJCF element of the arena.
    friction : tuple[float, float, float]
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001).

    Parameters
    ----------
    friction : tuple[float, float, float], optional
        Sliding, torsional, and rolling friction coefficients, by default
        (1, 0.005, 0.0001)
    gap_width : float, optional
        Width of each gap, by default 0.4.
    gapped_block_width : float, optional
        Width of each block (piece of floor), by default 1.
    gap_depth : float, optional
        Height of the gaps, by default 2.
    block_size : float, optional
        The side length of the rectangular blocks forming the terrain in
        mm, by default 1.3.
    height_range : tuple[float, float], optional
        Range from which the height of the extruding blocks should be
        sampled. Only half of the blocks arranged in a diagonal pattern are
        extruded, by default (0.35, 0.35).
    ground_alpha : float, optional
        Opacity of the ground, by default 1 (fully opaque).
    rand_seed : int, optional
        Seed for generating random block heights, by default 0.
    scale_bar_pos : tuple[float, float, float], optional
        If supplied, a 1 mm scale bar will be placed at this location.
    """

    def __init__(
        self,
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        gap_width: float = 0.3,
        gapped_block_width: float = 1.0,
        gap_depth: float = 2,
        block_size: float = 1.3,
        height_range: tuple[float, float] = (0.35, 0.35),
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ):
        super().__init__()

        self.friction = friction
        y_range = (-20, 20)
        rand_state = np.random.RandomState(rand_seed)

        self._height_expected_value = np.mean([*height_range])

        self._max_block_height = -np.inf

        # 3 repetitions, each consisting of a block part, 2 gaps, and a flat part
        for x_range in [(-4, 5), (5, 14), (14, 23)]:
            # block part
            x_centers = np.arange(
                x_range[0] + block_size / 2, x_range[0] + block_size * 3, block_size
            )
            y_centers = np.arange(y_range[0] + block_size / 2, y_range[1], block_size)
            for i, x_pos in enumerate(x_centers):
                for j, y_pos in enumerate(y_centers):
                    is_i_odd = i % 2 == 1
                    is_j_odd = j % 2 == 1

                    if is_i_odd != is_j_odd:
                        height = 0.1
                    else:
                        height = 0.1 + rand_state.uniform(*height_range)

                    box_size = (
                        block_size / 2 + 0.1 * block_size / 2,
                        block_size / 2 + 0.1 * block_size / 2,
                        height / 2 + block_size / 2,
                    )
                    box_pos = (
                        x_pos,
                        y_pos,
                        height / 2 - block_size / 2 - self._height_expected_value - 0.1,
                    )
                    self._max_block_height = max(
                        self._max_block_height,
                        height - self._height_expected_value - 0.1,
                    )
                    self.root_element.worldbody.add(
                        "geom",
                        type="box",
                        name=f"ground_element_block_x{x_pos}_x{y_pos}",
                        size=box_size,
                        pos=box_pos,
                        rgba=(0.3, 0.3, 0.3, ground_alpha),
                        friction=friction,
                    )

            # gap part
            curr_x_pos = x_range[0] + block_size * 3
            arena_width = y_range[1] - y_range[0]
            # first flat bit
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_gap_x{curr_x_pos}",
                size=(gapped_block_width / 4, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 4, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )
            # second flat bit
            curr_x_pos += gapped_block_width / 2 + gap_width
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_gap_x{curr_x_pos}",
                size=(gapped_block_width / 2, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + gapped_block_width / 2, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )

            # flat part
            curr_x_pos += gapped_block_width + gap_width
            remaining_space = x_range[1] - curr_x_pos
            assert remaining_space > 0, "remaining space for flat part is negative"
            self.root_element.worldbody.add(
                "geom",
                type="box",
                name=f"ground_element_flat_x{curr_x_pos}",
                size=(remaining_space / 2, arena_width / 2, gap_depth / 2),
                pos=(curr_x_pos + remaining_space / 2, 0, -gap_depth / 2),
                friction=friction,
                rgba=(0.3, 0.3, 0.3, ground_alpha),
            )

            # add floor underneath
            ground_size = ((x_range[1] - x_range[0]) / 2, max(y_range), 1)
            self.root_element.worldbody.add(
                "geom",
                type="plane",
                name=f"ground_base_{x_range[0]}",
                pos=(np.mean(x_range), 0, -gap_depth / 2),
                rgba=(0.3, 0.3, 0.3, ground_alpha),
                size=ground_size,
            )

        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.05, 0.5),
                pos=scale_bar_pos,
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        adj_pos = rel_pos + np.array([0, 0, -1 * self._height_expected_value])
        return adj_pos, rel_angle

    def _get_max_floor_height(self):
        # The floor and gap tops are at z=0
        return max(0, self._max_block_height)


class RandomMixedTerrain(BaseArena):
    """A randomized terrain assembled from flat, gapped, and block segments."""

    def __init__(
        self,
        x_range: tuple[float, float] = (-10, 25),
        y_range: tuple[float, float] = (-20, 20),
        friction: tuple[float, float, float] = (1, 0.005, 0.0001),
        terrain_probabilities: tuple[float, float, float] = (0.15, 0.4, 0.45),
        segment_length_range: tuple[float, float] = (3.0, 6.0),
        spawn_flat_x_end: float = 2.5,
        gap_width_range: tuple[float, float] = (0.15, 0.8),
        gapped_block_width_range: tuple[float, float] = (0.5, 1.8),
        gap_depth_range: tuple[float, float] = (1.0, 3.0),
        block_size_range: tuple[float, float] = (0.9, 1.8),
        block_height_range: tuple[float, float] = (0.15, 0.75),
        block_spacing: float = 0.08,
        ground_alpha: float = 1.0,
        rand_seed: int = 0,
        scale_bar_pos: Optional[tuple[float, float, float]] = None,
    ) -> None:
        super().__init__()

        self.x_range = x_range
        self.y_range = y_range
        self.friction = friction
        self._max_floor_height = 0.0
        rand_state = np.random.RandomState(rand_seed)

        terrain_probabilities = np.asarray(terrain_probabilities, dtype=float)
        if terrain_probabilities.shape != (3,) or terrain_probabilities.sum() <= 0:
            raise ValueError(
                "terrain_probabilities must contain 3 positive weights: "
                "(flat, gapped, blocks)."
            )
        terrain_probabilities /= terrain_probabilities.sum()
        terrain_types = np.array(["flat", "gapped", "blocks"])

        arena_width = y_range[1] - y_range[0]
        safe_x_end = min(max(spawn_flat_x_end, x_range[0]), x_range[1])
        self._add_prism(
            name="ground_spawn_flat",
            x_center=(x_range[0] + safe_x_end) / 2,
            y_center=0,
            x_size=safe_x_end - x_range[0],
            y_size=arena_width,
            top_height=0.0,
            bottom_height=-1.0,
            friction=friction,
            ground_alpha=ground_alpha,
        )

        curr_x = safe_x_end
        segment_idx = 0
        while curr_x < x_range[1] - 1e-6:
            remaining = x_range[1] - curr_x
            segment_length = min(rand_state.uniform(*segment_length_range), remaining)
            x_end = curr_x + segment_length
            terrain_type = rand_state.choice(terrain_types, p=terrain_probabilities)

            if terrain_type == "flat":
                self._add_prism(
                    name=f"ground_flat_{segment_idx}",
                    x_center=(curr_x + x_end) / 2,
                    y_center=0,
                    x_size=x_end - curr_x,
                    y_size=arena_width,
                    top_height=0.0,
                    bottom_height=-1.0,
                    friction=friction,
                    ground_alpha=ground_alpha,
                )
            elif terrain_type == "gapped":
                self._build_gapped_segment(
                    rand_state=rand_state,
                    segment_idx=segment_idx,
                    x_start=curr_x,
                    x_end=x_end,
                    gap_width_range=gap_width_range,
                    block_width_range=gapped_block_width_range,
                    gap_depth_range=gap_depth_range,
                    friction=friction,
                    ground_alpha=ground_alpha,
                )
            else:
                self._build_blocks_segment(
                    rand_state=rand_state,
                    segment_idx=segment_idx,
                    x_start=curr_x,
                    x_end=x_end,
                    block_size_range=block_size_range,
                    block_height_range=block_height_range,
                    block_spacing=block_spacing,
                    friction=friction,
                    ground_alpha=ground_alpha,
                )

            curr_x = x_end
            segment_idx += 1

        if scale_bar_pos:
            self.root_element.worldbody.add(
                "geom",
                type="cylinder",
                size=(0.05, 0.5),
                pos=scale_bar_pos,
                rgba=(0, 0, 0, 1),
                euler=(0, np.pi / 2, 0),
            )

    def _add_prism(
        self,
        name: str,
        x_center: float,
        y_center: float,
        x_size: float,
        y_size: float,
        top_height: float,
        bottom_height: float,
        friction: tuple[float, float, float],
        ground_alpha: float,
    ) -> None:
        if x_size <= 1e-6 or y_size <= 1e-6 or top_height <= bottom_height:
            return

        self.root_element.worldbody.add(
            "geom",
            type="box",
            name=name,
            size=(x_size / 2, y_size / 2, (top_height - bottom_height) / 2),
            pos=(x_center, y_center, (top_height + bottom_height) / 2),
            friction=friction,
            rgba=(0.3, 0.3, 0.3, ground_alpha),
        )
        self._max_floor_height = max(self._max_floor_height, top_height)

    def _add_plane(
        self, name: str, x_start: float, x_end: float, z_pos: float, ground_alpha: float
    ) -> None:
        if x_end - x_start <= 1e-6:
            return

        self.root_element.worldbody.add(
            "geom",
            type="plane",
            name=name,
            pos=((x_start + x_end) / 2, 0, z_pos),
            rgba=(0.3, 0.3, 0.3, ground_alpha),
            size=((x_end - x_start) / 2, max(map(abs, self.y_range)), 1),
        )

    def _build_gapped_segment(
        self,
        rand_state: np.random.RandomState,
        segment_idx: int,
        x_start: float,
        x_end: float,
        gap_width_range: tuple[float, float],
        block_width_range: tuple[float, float],
        gap_depth_range: tuple[float, float],
        friction: tuple[float, float, float],
        ground_alpha: float,
    ) -> None:
        gap_depth = rand_state.uniform(*gap_depth_range)
        self._add_plane(
            name=f"ground_gap_base_{segment_idx}",
            x_start=x_start,
            x_end=x_end,
            z_pos=-gap_depth,
            ground_alpha=ground_alpha,
        )

        arena_width = self.y_range[1] - self.y_range[0]
        curr_x = x_start
        piece_idx = 0
        while curr_x < x_end - 1e-6:
            remaining = x_end - curr_x
            block_width = min(rand_state.uniform(*block_width_range), remaining)
            self._add_prism(
                name=f"ground_gap_block_{segment_idx}_{piece_idx}",
                x_center=curr_x + block_width / 2,
                y_center=0,
                x_size=block_width,
                y_size=arena_width,
                top_height=0.0,
                bottom_height=-gap_depth,
                friction=friction,
                ground_alpha=ground_alpha,
            )
            curr_x += block_width
            remaining = x_end - curr_x
            if remaining <= gap_width_range[0]:
                if remaining > 1e-6:
                    self._add_prism(
                        name=f"ground_gap_tail_{segment_idx}_{piece_idx}",
                        x_center=curr_x + remaining / 2,
                        y_center=0,
                        x_size=remaining,
                        y_size=arena_width,
                        top_height=0.0,
                        bottom_height=-gap_depth,
                        friction=friction,
                        ground_alpha=ground_alpha,
                    )
                break
            curr_x += min(rand_state.uniform(*gap_width_range), remaining)
            piece_idx += 1

    def _build_blocks_segment(
        self,
        rand_state: np.random.RandomState,
        segment_idx: int,
        x_start: float,
        x_end: float,
        block_size_range: tuple[float, float],
        block_height_range: tuple[float, float],
        block_spacing: float,
        friction: tuple[float, float, float],
        ground_alpha: float,
    ) -> None:
        block_size = rand_state.uniform(*block_size_range)
        base_depth = max(block_height_range[1], 0.6)
        raise_prob = rand_state.uniform(0.35, 0.7)
        self._add_plane(
            name=f"ground_blocks_base_{segment_idx}",
            x_start=x_start,
            x_end=x_end,
            z_pos=-base_depth,
            ground_alpha=ground_alpha,
        )

        x_curr = x_start
        x_idx = 0
        while x_curr < x_end - 1e-6:
            x_size = min(block_size, x_end - x_curr)
            y_curr = self.y_range[0]
            y_idx = 0
            while y_curr < self.y_range[1] - 1e-6:
                y_size = min(block_size, self.y_range[1] - y_curr)
                if rand_state.rand() < raise_prob:
                    top_height = rand_state.uniform(*block_height_range)
                else:
                    top_height = 0.0
                self._add_prism(
                    name=f"ground_block_{segment_idx}_{x_idx}_{y_idx}",
                    x_center=x_curr + x_size / 2,
                    y_center=y_curr + y_size / 2,
                    x_size=x_size * (1 + block_spacing),
                    y_size=y_size * (1 + block_spacing),
                    top_height=top_height,
                    bottom_height=-base_depth,
                    friction=friction,
                    ground_alpha=ground_alpha,
                )
                y_curr += y_size
                y_idx += 1
            x_curr += x_size
            x_idx += 1

    def get_spawn_position(
        self, rel_pos: np.ndarray, rel_angle: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return rel_pos, rel_angle

    def _get_max_floor_height(self):
        return self._max_floor_height
