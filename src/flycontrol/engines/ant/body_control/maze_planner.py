"""Deterministic grid waypoints for registered Gymnasium AntMaze layouts."""

from __future__ import annotations

from collections import deque

import numpy as np


class OfficialAntMazeWaypointPlanner:
    """Shortest-path waypoints on the maze cells exposed by GoalEnv.

    Q iteration on the deterministic four-neighbour grid and breadth-first
    search induce the same shortest cell paths.  The planner never modifies
    MuJoCo state, rewards, termination, walls, or final desired goals.
    """

    def __init__(self, maze) -> None:
        self.maze = maze
        self.maze_map = np.asarray(maze.maze_map, dtype=object)
        self.walkable = frozenset(
            (row, column)
            for row in range(self.maze_map.shape[0])
            for column in range(self.maze_map.shape[1])
            if self.maze_map[row, column] != 1
        )
        if not self.walkable:
            raise ValueError("maze contains no walkable cells")

    def _cell(self, xy: np.ndarray) -> tuple[int, int]:
        proposed = tuple(
            int(value)
            for value in self.maze.cell_xy_to_rowcol(np.asarray(xy, dtype=np.float64))
        )
        if proposed in self.walkable:
            return proposed
        # Contacts can place the torso infinitesimally across a cell boundary.
        # Fall back to the closest valid cell center without changing physics.
        return min(
            self.walkable,
            key=lambda cell: float(
                np.linalg.norm(self.maze.cell_rowcol_to_xy(np.asarray(cell)) - xy)
            ),
        )

    def shortest_path(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
    ) -> list[tuple[int, int]]:
        start = self._cell(start_xy)
        goal = self._cell(goal_xy)
        frontier = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        while frontier:
            current = frontier.popleft()
            if current == goal:
                break
            row, column = current
            for candidate in (
                (row - 1, column),
                (row + 1, column),
                (row, column - 1),
                (row, column + 1),
            ):
                if candidate in self.walkable and candidate not in parent:
                    parent[candidate] = current
                    frontier.append(candidate)
        if goal not in parent:
            raise RuntimeError(f"no maze route from {start} to {goal}")
        path = []
        cursor: tuple[int, int] | None = goal
        while cursor is not None:
            path.append(cursor)
            cursor = parent[cursor]
        return list(reversed(path))

    def waypoint(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        path = self.shortest_path(achieved_goal, desired_goal)
        if len(path) <= 2:
            # The final target is directly reachable within the current or
            # adjacent cell, so preserve its official continuous coordinates.
            return np.asarray(desired_goal, dtype=np.float32).copy()
        return np.asarray(
            self.maze.cell_rowcol_to_xy(np.asarray(path[1])),
            dtype=np.float32,
        )

    def tracker(
        self,
        success_radius: float = 0.45,
    ) -> "OfficialAntMazeWaypointTracker":
        return OfficialAntMazeWaypointTracker(self, success_radius)


class OfficialAntMazeWaypointTracker:
    """Latch Q-iteration waypoints until the low-level skill reaches them.

    Replanning solely from the current cell advances a waypoint as soon as the
    torso crosses a cell boundary, roughly two metres before it reaches the
    cell centre.  That supplies no local sparse-reward successes.  This tracker
    instead materializes the shortest official-grid route at reset and advances
    only inside the same 0.45-m radius used by AntMaze-v5.
    """

    def __init__(
        self,
        planner: OfficialAntMazeWaypointPlanner,
        success_radius: float = 0.45,
    ) -> None:
        if success_radius <= 0:
            raise ValueError("success_radius must be positive")
        self.planner = planner
        self.success_radius = float(success_radius)
        self._waypoints: list[np.ndarray] = []
        self._index = 0
        self._final_reached = False

    @property
    def current(self) -> np.ndarray:
        if not self._waypoints:
            raise RuntimeError("waypoint tracker must be reset before use")
        return self._waypoints[self._index].copy()

    @property
    def index(self) -> int:
        return self._index

    @property
    def total_waypoints(self) -> int:
        return len(self._waypoints)

    def reset(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
    ) -> np.ndarray:
        path = self.planner.shortest_path(achieved_goal, desired_goal)
        intermediate = [
            np.asarray(
                self.planner.maze.cell_rowcol_to_xy(np.asarray(cell)),
                dtype=np.float32,
            )
            for cell in path[1:-1]
        ]
        self._waypoints = [
            *intermediate,
            np.asarray(desired_goal, dtype=np.float32).copy(),
        ]
        self._index = 0
        self._final_reached = False
        return self.current

    def advance(
        self,
        achieved_goal: np.ndarray,
    ) -> tuple[np.ndarray, bool, bool]:
        """Return ``(current_goal, switched, newly_reached)``."""
        if not self._waypoints:
            raise RuntimeError("waypoint tracker must be reset before use")
        within = bool(
            np.linalg.norm(
                np.asarray(achieved_goal, dtype=np.float32)
                - self._waypoints[self._index]
            )
            <= self.success_radius
        )
        switched = False
        newly_reached = False
        if within and self._index < len(self._waypoints) - 1:
            self._index += 1
            switched = True
            newly_reached = True
        elif within and not self._final_reached:
            self._final_reached = True
            newly_reached = True
        return self.current, switched, newly_reached
