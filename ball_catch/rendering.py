"""Home-made matplotlib visualisation of an episode.

Four panels show the same instant from the viewpoints that matter for this task:

    side view        the throw in the x-z plane: where the ball really is (filled) and where
                     the controller last saw it (hollow ring). The gap between the two is the
                     effect of the delay.
    net plane        the plane the net moves in, seen from the camera: net opening, workspace,
                     and the point where the ball will cross.
    camera image     the latest frame that has reached the controller.
    frame delivery   one line per camera frame, from its capture time (bottom) to its arrival
                     time (top). Slanted lines are delayed frames; crossing lines are frames that
                     overtook each other, and the overtaken ones are discarded.

Colour encodes identity (orange = ball, blue = net); green/red are reserved for frame and episode
status and always come with a label.
"""

from typing import Optional

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Rectangle

from ball_catch.camera import PinholeCamera
from ball_catch.config import EnvConfig

# Chrome and ink.
SURFACE = "#fcfcfb"
IMAGE_BACKGROUND = "#f0efec"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
# Identity colours (validated as a colour-blind-safe pair).
BALL = "#eb6834"
NET = "#2a78d6"
NET_FILL = (0.165, 0.471, 0.839, 0.15)  # the net colour, translucent, for the opening
# Status colours.
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

FIGSIZE = (12.8, 7.2)
DPI = 75  # 960 x 540 pixels

_FRAME_STYLE = {
    "in_transit": (MUTED, "solid"),
    "accepted": (GOOD, "solid"),
    "discarded": (CRITICAL, "dashed"),
}


class Renderer:
    def __init__(self, cfg: EnvConfig, camera: PinholeCamera, mode: str):
        self.cfg, self.camera, self.mode = cfg, camera, mode
        self._plt = None
        if mode == "human":
            import matplotlib.pyplot as plt

            self._plt = plt
            plt.ion()
            self.fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
        else:
            # An off-screen canvas: works without a display and leaves pyplot's state alone.
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure

            self.fig = Figure(figsize=FIGSIZE, dpi=DPI)
            FigureCanvasAgg(self.fig)
        self.fig.set_facecolor(SURFACE)
        self._launch = None
        self._build()

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        cfg, fig = self.cfg, self.fig
        grid = fig.add_gridspec(
            2, 3, left=0.055, right=0.985, top=0.83, bottom=0.085,
            height_ratios=[2.2, 1.0], width_ratios=[1.35, 1.0, 1.15], hspace=0.45, wspace=0.2,
        )  # fmt: skip
        fig.text(0.055, 0.95, "Ball catching under visual delay", fontsize=13, fontweight="bold", color=INK)
        self.status_text = fig.text(0.055, 0.9, "", fontsize=10, color=INK_SECONDARY)
        self.outcome_marker = fig.text(0.755, 0.9, "", fontsize=11, ha="right")
        self.outcome_text = fig.text(0.762, 0.9, "", fontsize=11, fontweight="bold", color=INK)

        y_lo, y_hi = cfg.workspace_y
        z_lo, z_hi = cfg.workspace_z
        cam_x, _, cam_z = cfg.camera_position
        radius = cfg.net_radius

        # --- side view -----------------------------------------------------------------
        ax = fig.add_subplot(grid[0, 0])
        self._style(ax, "Side view")
        ax.set_xlim(-cfg.launch_distance_range[1] - 0.6, cam_x + 0.6)
        ax.set_ylim(0.0, 4.0)
        ax.set_xlabel("x (m)", fontsize=8, color=MUTED)
        ax.set_ylabel("height z (m)", fontsize=8, color=MUTED)
        ax.plot([0.0, 0.0], [z_lo - radius, z_hi + radius], color=AXIS, lw=1)
        ax.text(0.0, z_hi + radius + 0.08, "net plane", fontsize=8, color=MUTED, ha="center")
        ax.plot([cam_x], [cam_z], marker="s", ms=7, color=MUTED, ls="none")
        ax.text(cam_x, cam_z + 0.18, "camera", fontsize=8, color=MUTED, ha="center")
        (self.side_path,) = ax.plot([], [], color=AXIS, lw=1, ls=(0, (4, 3)))
        (self.side_trail,) = ax.plot([], [], color=BALL, lw=2, solid_capstyle="round")
        (self.side_net,) = ax.plot([], [], color=NET, lw=5, solid_capstyle="round")
        (self.side_seen,) = ax.plot([], [], marker="o", ms=10, mfc="none", mec=BALL, mew=2, ls="none")
        (self.side_ball,) = ax.plot([], [], marker="o", ms=9, mfc=BALL, mec=SURFACE, mew=1.5, ls="none")

        # --- net plane -----------------------------------------------------------------
        ax = fig.add_subplot(grid[0, 1])
        self._style(ax, "Net plane, seen from the camera")
        ax.set_xlim(y_lo - 0.6, y_hi + 0.6)
        ax.set_ylim(0.0, z_hi + 0.8)
        ax.set_aspect("equal", adjustable="box", anchor="N")
        ax.set_xlabel("y (m)", fontsize=8, color=MUTED)
        ax.add_patch(
            Rectangle((y_lo, z_lo), y_hi - y_lo, z_hi - z_lo, fill=False, ec=MUTED, lw=1, ls=(0, (4, 3)))
        )
        ax.text(y_lo, z_hi + 0.06, "reachable net centres", fontsize=7.5, color=MUTED)
        self.front_net = ax.add_patch(Circle((0, 0), radius, fc=NET_FILL, ec=NET, lw=2.5))
        (self.front_cross,) = ax.plot([], [], marker="+", ms=11, mew=1.6, color=INK_SECONDARY, ls="none")
        self.front_seen = ax.add_patch(Circle((0, 0), cfg.ball_radius, fc="none", ec=BALL, lw=1.8))
        self.front_ball = ax.add_patch(Circle((0, 0), cfg.ball_radius, fc=BALL, ec=SURFACE, lw=1))

        # --- camera image --------------------------------------------------------------
        ax = self.ax_cam = fig.add_subplot(grid[0, 2])
        self._style(ax, "Latest frame delivered to the controller")
        ax.set_facecolor(IMAGE_BACKGROUND)
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)
        ax.set_aspect(cfg.image_height / cfg.image_width, adjustable="box", anchor="N")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
        self.cam_net = ax.add_patch(Ellipse((0, 0), 0.1, 0.1, fc="none", ec=NET, lw=1.5))
        self.cam_blob = ax.add_patch(Ellipse((0, 0), 0.0, 0.0, fc=BALL, ec="none", alpha=0.35))
        (self.cam_seen,) = ax.plot([], [], marker="o", ms=11, mfc="none", mec=BALL, mew=2, ls="none")
        self.cam_note = ax.text(0.0, 0.0, "", fontsize=9, color=MUTED, ha="center", va="center")

        # --- frame delivery timeline ---------------------------------------------------
        ax = self.ax_time = fig.add_subplot(grid[1, :])
        self._style(ax, "Frame delivery: one line per camera frame, from capture to arrival")
        ax.set_ylim(-0.2, 1.2)
        ax.set_yticks([0.0, 1.0])
        ax.set_yticklabels(["captured", "arrived"])
        ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
        self.time_lines = LineCollection([], linewidths=1.6, capstyle="round")
        ax.add_collection(self.time_lines)
        self.time_now = ax.axvline(0.0, color=INK_SECONDARY, lw=1)
        ax.legend(
            handles=[
                Line2D([], [], color=MUTED, lw=1.6, label="in transit"),
                Line2D([], [], color=GOOD, lw=1.6, label="accepted"),
                Line2D([], [], color=CRITICAL, lw=1.6, ls="dashed", label="discarded (overtaken)"),
            ],
            loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False, fontsize=8,
            labelcolor=INK_SECONDARY, borderaxespad=0.2,
        )  # fmt: skip

        fig.legend(
            handles=[
                Line2D([], [], marker="o", ms=8, mfc=BALL, mec=SURFACE, ls="none", label="ball now (truth)"),
                Line2D([], [], marker="o", ms=8, mfc="none", mec=BALL, mew=2, ls="none",
                       label="ball as last seen"),
                Line2D([], [], color=NET, lw=4, label="net"),
                Line2D([], [], marker="+", ms=9, mew=1.6, color=INK_SECONDARY, ls="none",
                       label="true crossing point"),
            ],
            loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=4, frameon=False, fontsize=9,
            labelcolor=INK_SECONDARY, handletextpad=0.4, columnspacing=1.4,
        )  # fmt: skip

    @staticmethod
    def _style(ax, title: str) -> None:
        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc="left", fontsize=10, color=INK, pad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color(AXIS)
        ax.tick_params(colors=MUTED, labelsize=8, length=3)

    # ------------------------------------------------------------------ drawing
    def _begin_episode(self, env) -> None:
        """Redraw what is fixed during one throw: the true trajectory and the time axis."""
        self._launch = launch = env.launch
        path = np.array([launch.position_at(t) for t in np.linspace(0.0, launch.flight_time, 40)])
        self.side_path.set_data(path[:, 0], path[:, 2])
        self.front_cross.set_data([launch.intercept[0]], [launch.intercept[1]])
        self.ax_time.set_xlim(0.0, min(launch.flight_time, self.cfg.episode_time_limit) + 0.02)

    def draw(self, env) -> Optional[np.ndarray]:
        """Draw the current state of `env`. Returns an RGB array in "rgb_array" mode."""
        if env.launch is not self._launch:
            self._begin_episode(env)
        cfg, launch, t = self.cfg, env.launch, env.time
        latest = env.frames.latest
        ball = launch.position_at(t)
        seen = launch.position_at(latest.capture_time) if latest is not None and latest.valid else None

        # side view
        trail = np.array([launch.position_at(s) for s in np.linspace(0.0, t, 30)])
        self.side_trail.set_data(trail[:, 0], trail[:, 2])
        self.side_ball.set_data([ball[0]], [ball[2]])
        self.side_net.set_data([0.0, 0.0], [env.net_pos[1] - cfg.net_radius, env.net_pos[1] + cfg.net_radius])
        self.side_seen.set_data(*(([seen[0]], [seen[2]]) if seen is not None else ([], [])))

        # net plane
        self.front_net.set_center(env.net_pos)
        self.front_ball.set_center(ball[1:])
        self.front_seen.set_visible(seen is not None)
        if seen is not None:
            self.front_seen.set_center(seen[1:])

        # camera image
        net_u, net_v, _, depth = self.camera.project(np.array([0.0, env.net_pos[0], env.net_pos[1]]))
        net_radius_px = self.camera.focal_px * cfg.net_radius / depth
        self.cam_net.set_center((net_u / self.camera.half_width, net_v / self.camera.half_height))
        self.cam_net.set_width(2.0 * net_radius_px / self.camera.half_width)
        self.cam_net.set_height(2.0 * net_radius_px / self.camera.half_height)
        detected = latest is not None and latest.valid
        self.cam_blob.set_visible(detected)
        if detected:
            u, v, area = latest.features
            ball_radius_px = self.camera.reference_radius_px * np.sqrt(area)
            self.cam_blob.set_center((u, v))
            self.cam_blob.set_width(2.0 * ball_radius_px / self.camera.half_width)
            self.cam_blob.set_height(2.0 * ball_radius_px / self.camera.half_height)
            self.cam_seen.set_data([u], [v])
            self.cam_note.set_text("")
        else:
            self.cam_seen.set_data([], [])
            self.cam_note.set_text("no frame has arrived yet" if latest is None else "ball not in view")
        age = "" if latest is None else f"captured {1000.0 * env.observation_age:.0f} ms ago"
        self.ax_cam.set_xlabel(age, fontsize=9, color=INK_SECONDARY)

        # frame delivery timeline
        frames = env.frames.frames
        self.time_lines.set_segments([[(f.capture_time, 0.0), (f.arrival_time, 1.0)] for f in frames])
        self.time_lines.set_color([_FRAME_STYLE[f.status][0] for f in frames])
        self.time_lines.set_linestyle([_FRAME_STYLE[f.status][1] for f in frames])
        self.time_now.set_xdata([t, t])

        # header
        self.status_text.set_text(
            f"t = {t:.3f} s of {launch.flight_time:.3f} s      "
            f"observation age {1000.0 * env.observation_age:.0f} ms      "
            f"capture interval {1000.0 * env.frames.capture_interval:.0f} ms      "
            f"frames: {env.frames.accepted_count} accepted, {env.frames.discarded_count} discarded"
        )
        self._draw_outcome(env.outcome)

        if self.mode == "human":
            self.fig.canvas.draw_idle()
            self._plt.pause(0.001)
            return None
        self.fig.canvas.draw()
        return np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()

    def _draw_outcome(self, outcome: Optional[dict]) -> None:
        if outcome is None:
            self.outcome_marker.set_text("")
            self.outcome_text.set_text("")
            return
        if outcome["caught"]:
            colour = GOOD
            label = f"CAUGHT  {outcome['miss_distance']:.2f} m from the net centre"
        elif outcome["timeout"]:
            colour, label = CRITICAL, "TIMEOUT"
        else:
            outside = outcome["miss_distance"] - self.cfg.catch_radius
            colour, label = CRITICAL, f"MISSED  by {outside:.2f} m"
        self.outcome_marker.set_text("■")
        self.outcome_marker.set_color(colour)
        self.outcome_text.set_text(label)

    def close(self) -> None:
        if self._plt is not None:
            self._plt.close(self.fig)
