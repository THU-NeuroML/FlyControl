"""Continuous flat run-in: move a non-contacting obstacle, never the body state."""

from flycontrol.experiments.fig2.current.steady_state import task, np, raw
from terrain_control import physical_interface as physical
from terrain_control.physics_cache import install
import mujoco


class SteadyObstacleTask(task.ObstacleTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._canonical_onset_rule = dict(self.onset_rule)

    def _arena(self):
        arena = task.SingleObstacleArena("blocks", 0.0)
        # Extend the compiled flat floor for the continuous 1 s run-in.
        ground = arena.root_element.find("geom", "ground")
        ground.pos = (35.0, 0.0, -1.0)
        ground.size = (45.0, 40.0, 1.0)  # x [-10,80], y [-40,40], top z=0
        height = 0.2 if self.value == 0 else float(self.value)
        body = arena.root_element.worldbody.add(
            "body", name="moving_obstacle", mocap=True, pos=(100.0, 0.0, height / 2)
        )
        body.add(
            "geom",
            name="single_obstacle",
            type="box",
            size=(0.65, 20.0, height / 2),
            friction=arena.friction,
            rgba=(0.35, 0.35, 0.35, 1.0),
        )
        return arena

    def _build_sim(self, terrain_name, seed):
        if self._sim is not None:
            self._sim.close()
        original = physical._make_arena
        # Identical model and contact ordering for every height, including Flat.
        physical._make_arena = lambda name, sd: self._arena()
        try:
            physical.CPGRLEnv._build_sim(self, terrain_name, seed)
        finally:
            physical._make_arena = original
        if self.cache_physics:
            self.forward_cache_stats = install(self._sim.physics)

    def reset(self, *, seed=None, options=None):
        self.near = 3.0
        self.onset_rule = dict(self._canonical_onset_rule)
        if hasattr(self, "obstacle_geom_id"):
            del self.obstacle_geom_id
        _, info = super().reset(seed=seed, options=options)
        original_body = raw(self)["fly"][0].copy()
        self.far = 1e6
        self.max_time = 3.0
        samples = []
        times = []
        # Stability checks require micro-frames even when the caller does not record
        # task trajectories, as in training. Restore the caller's tracing preference.
        capture_requested = self.capture_micro_trace
        self.capture_micro_trace = True
        try:
            while self.elapsed < 1.0 - 1e-8:
                _, _, _, _, step_info = self.step(np.zeros(6, dtype=np.float32))
                for frame in self.last_micro_trace:
                    o = frame["observation"].reshape(-1, 172)[-1]
                    samples.append(
                        [
                            *frame["position"],
                            float(frame["up"][2]),
                            *o[46:49] * 10.0,
                            float(np.arctan2(o[25], o[24])),
                        ]
                    )
                    times.append(float(frame["time_s"]))
                if step_info["physical_termination"] or step_info["numerical_error"]:
                    raise RuntimeError(
                        "Run-in failed; retain and report, never replace phase"
                    )
        finally:
            self.capture_micro_trace = capture_requested
        a = np.asarray(samples)
        t = np.asarray(times)
        late = a[t > 0.7500001]
        prev = a[(t > 0.5000001) & (t <= 0.7500001)]
        yaw = np.unwrap(a[:, 7])
        speed = late[:, 4].mean()
        previous_speed = prev[:, 4].mean()
        self.runin_diagnostics = dict(
            duration_s=self.elapsed,
            min_up_last_250ms=float(late[:, 3].min()),
            forward_speed_mm_s=float(speed),
            speed_window_change_fraction=float(
                abs(speed - previous_speed) / max(abs(previous_speed), 1.0)
            ),
            body_height_window_change_mm=float(
                abs(late[:, 2].mean() - prev[:, 2].mean())
            ),
            heading_window_change_rad=float(
                abs(
                    yaw[t > 0.7500001].mean()
                    - yaw[(t > 0.5000001) & (t <= 0.7500001)].mean()
                )
            ),
            cpg_min_magnitude=float(self._cpg.curr_magnitudes.min()),
        )
        d = self.runin_diagnostics
        d["stable"] = bool(
            d["min_up_last_250ms"] >= 0.95
            and speed >= 1.0
            and d["speed_window_change_fraction"] <= 0.15
            and d["body_height_window_change_mm"] <= 0.03
            and d["heading_window_change_rad"] <= 0.05
            and d["cpg_min_magnitude"] >= 0.99
        )
        assert d["stable"], d
        self.runin_time = t
        self.runin_samples = a
        self.warmup_sim_time = float(self._sim.curr_time)
        p = self._sim.physics
        r = raw(self)
        self.coordinate_offset_xy = r["fly"][0, :2] - original_body[:2]
        self.near = 3.0 + float(self.coordinate_offset_xy[0])
        self.far = 4.3 + float(self.coordinate_offset_xy[0])
        geom = [
            i
            for i in range(p.model.ngeom)
            if (p.model.id2name(i, "geom") or "").endswith("single_obstacle")
        ]
        assert len(geom) == 1
        g = geom[0]
        self.obstacle_geom_id = g
        self.obstacle_contact_macro_samples = 0
        before_qpos = p.data.qpos.copy()
        before_qvel = p.data.qvel.copy()
        # A mocap obstacle updates collision bounds through the supported kinematic path.
        # Geometry sizes are compiled once; no static-BVH or geom-size mutation is used.
        if self.value > 0:
            body = int(p.model.geom_bodyid[g])
            mocap = int(p.model.body_mocapid[body])
            assert mocap >= 0
            p.data.mocap_pos[mocap] = [
                (self.near + self.far) / 2,
                float(self.coordinate_offset_xy[1]),
                self.value / 2,
            ]
        if self.cache_physics:
            self.forward_cache_stats["cache_valid"] = False
        p.forward()
        if self.value > 0:
            np.testing.assert_allclose(
                p.data.geom_xpos[g],
                [
                    (self.near + self.far) / 2,
                    float(self.coordinate_offset_xy[1]),
                    self.value / 2,
                ],
                atol=1e-9,
                rtol=0,
            )
        hit = np.array([-1], dtype=np.int32)
        point = np.array(
            [
                (self.near + self.far) / 2,
                float(self.coordinate_offset_xy[1]),
                float(self.value) + 1.0,
            ]
        )
        distance = mujoco.mj_ray(
            p.model.ptr, p.data.ptr, point, np.array([0.0, 0.0, -1.0]), None, 1, -1, hit
        )
        assert abs(distance - 1.0) < 1e-8, (distance, self.value, int(hit[0]))
        if self.value > 0:
            assert int(hit[0]) == g
        self.geometry_audit = dict(
            ray_distance=float(distance),
            ray_hit_geom=int(hit[0]),
            obstacle_geom=g,
            nominal_height=float(self.value),
            relative_near=float(self.near - r["fly"][0, 0]),
        )
        assert np.array_equal(before_qpos, p.data.qpos) and np.array_equal(
            before_qvel, p.data.qvel
        )
        # Keep simulator time and every physical state continuous; only task bookkeeping resets.
        self._physics_steps = 0
        self.elapsed = 0.0
        self.max_time = 1.2
        self.onset_rule = dict(self.onset_rule)
        self.onset_rule["start_x_mm"] += float(self.coordinate_offset_xy[0])
        self.onset_rule["full_x_mm"] += float(self.coordinate_offset_xy[0])
        self.start = r["fly"][0].copy()
        self._prev_fly_pos = self.start.copy()
        heading = r["fly_orientation"][:2].copy()
        self._initial_heading = heading / np.linalg.norm(heading)
        self.start_min_foot_x = float(r["end_effectors"][:, 0].min())
        self.previous_action = np.zeros(6)
        self.previous_progress = 0.0
        self.previous_heading_potential = 0.0
        self.max_lateral = 0.0
        self.clear_duration = 0.0
        self.episode_return = 0.0
        self.success = False
        self.last_info = {}
        self.last_micro_trace = []
        self.last_micro_infos = []
        self._history_tick = None
        self._history_frames = None
        assert r["end_effectors"][:, 0].max() < self.near
        assert abs((self.near - r["fly"][0, 0]) - (3.0 - original_body[0])) < 2e-6
        return self.observation(), dict(info, runin_diagnostics=d)

    def step(self, action):
        result = super().step(action)
        if hasattr(self, "obstacle_geom_id"):
            c = self._sim.physics.data.contact
            if np.any(
                (np.asarray(c.geom1) == self.obstacle_geom_id)
                | (np.asarray(c.geom2) == self.obstacle_geom_id)
            ):
                self.obstacle_contact_macro_samples += 1
        return result


class ColdMocapObstacleTask(SteadyObstacleTask):
    """Cold control using exactly the new obstacle representation and no run-in."""

    def _arena(self):
        arena = super()._arena()
        if self.value > 0:
            arena.root_element.find("body", "moving_obstacle").pos = (
                3.65,
                0.0,
                self.value / 2,
            )
        return arena

    def reset(self, *, seed=None, options=None):
        return task.ObstacleTask.reset(self, seed=seed, options=options)
