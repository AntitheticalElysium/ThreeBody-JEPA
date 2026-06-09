# manim_scene.py — hidden-body 3D orbit (cairo). The liked design, with polished (un-gridded) balls.
#   MANIM_N=3 uv run manim -qm --disable_caching manim_scene.py HiddenOrbit3D
import os
import numpy as np
from manim import *

N = int(os.environ.get("MANIM_N", "3"))
D = np.load(os.environ.get("MANIM_NPZ", f"results/manim_N{N}.npz"))
VIS, TRUE, PRED, ERR = D["vis"], D["true_hidden"], D["pred_hidden"], D["err"]
F_, NV = VIS.shape[0], VIS.shape[1]
S = 2.6 / (np.percentile(np.abs(np.concatenate([VIS.reshape(-1, 3), TRUE, PRED])), 95) + 1e-6)
VIS_S, TRUE_S, PRED_S = VIS * S, TRUE * S, PRED * S
BG = "#070710"
GOLD, TRUTH_C, JEPA_C = GOLD_B, "#dfe6f0", BLUE_B
TITLE = f"JEPA predicts the missing planet  ·  {NV+1}-body system"


def _p(traj, idx):
    i = int(np.clip(np.floor(idx), 0, len(traj) - 1)); j = min(i + 1, len(traj) - 1); f = float(idx - i)
    p = (1 - f) * traj[i] + f * traj[j]
    return np.array([p[0], p[1], p[2]])


def _dir(traj, idx):
    a, b = _p(traj, idx), _p(traj, min(idx + 3, len(traj) - 1))
    d = b - a; nrm = np.linalg.norm(d)
    return d / nrm if nrm > 1e-6 else RIGHT


def _err(idx):
    return ERR[int(np.clip(idx, 0, F_ - 1))]


def _legend():
    items = [(Dot(color=GOLD), "visible planets", GREY_A),
             (Circle(radius=0.09, color=TRUTH_C, stroke_width=2.5).set_fill(opacity=0), "missing planet (truth)", GREY_A),
             (Dot(color=JEPA_C), "JEPA prediction", JEPA_C)]
    rows = [VGroup(m, Text(lbl, font_size=20, color=c)).arrange(RIGHT, buff=0.22) for m, lbl, c in items]
    return VGroup(*rows).arrange(DOWN, aligned_edge=LEFT, buff=0.18).scale(0.8).to_corner(DL)


def _ball(color, radius, op):
    # polished: no checkerboard, no mesh stroke -> clean shaded ball
    return Sphere(radius=radius, resolution=(24, 24), checkerboard_colors=False).set_color(color).set_stroke(width=0).set_opacity(op)


class HiddenOrbit3D(ThreeDScene):
    def construct(self):
        self.camera.background_color = BG
        self.set_camera_orientation(phi=64 * DEGREES, theta=-50 * DEGREES, zoom=1.4)
        self.add(NumberPlane(x_range=[-4, 4, 1], y_range=[-4, 4, 1],
                             background_line_style={"stroke_color": BLUE_E, "stroke_width": 1, "stroke_opacity": 0.28}))
        title = Text(TITLE, font_size=25); title.to_edge(UP)
        self.add_fixed_in_frame_mobjects(title, _legend())
        t = ValueTracker(0)

        def trail(getter, color, w, op, dt):
            self.add(TracedPath(getter, stroke_color=color, stroke_width=w, stroke_opacity=op, dissipating_time=dt))

        def pointer(traj, ball, color):
            p = Line(ORIGIN, RIGHT, stroke_width=5, color=color)
            p.add_updater(lambda m, tr=traj, bl=ball: m.put_start_and_end_on(bl.get_center(), bl.get_center() + 0.28 * _dir(tr, t.get_value())))

            def tip(tr=traj, bl=ball, col=color):
                d = _dir(tr, t.get_value())
                c = Cone(base_radius=0.05, height=0.11, direction=d, checkerboard_colors=False)
                return c.set_color(col).set_stroke(width=0).move_to(bl.get_center() + 0.28 * d)  # base overlaps line tip
            self.add(p, always_redraw(tip))

        for b in range(NV):
            s = _ball(GOLD, 0.12, 0.85)  # brighter context
            s.add_updater(lambda m, b=b: m.move_to(_p(VIS_S[:, b], t.get_value())))
            trail(s.get_center, GOLD, 2, 0.4, 0.7); self.add(s)

        truth = _ball(TRUTH_C, 0.16, 0.35)
        truth.add_updater(lambda m: m.move_to(_p(TRUE_S, t.get_value())))
        self.add(truth)  # no trail for the missing planet -> declutters

        jepa = _ball(JEPA_C, 0.17, 1.0)
        jepa.add_updater(lambda m: m.move_to(_p(PRED_S, t.get_value())))
        trail(jepa.get_center, JEPA_C, 4, 0.9, 0.9); self.add(jepa)

        pointer(TRUE_S, truth, TRUTH_C); pointer(PRED_S, jepa, JEPA_C)
        conn = Line(ORIGIN, RIGHT, stroke_width=2.5, color=GREY_B).set_opacity(0.55)
        conn.add_updater(lambda m: m.put_start_and_end_on(truth.get_center(), jepa.get_center())
                         if np.linalg.norm(truth.get_center() - jepa.get_center()) > 1e-3 else None)
        self.add(conn)
        err = always_redraw(lambda: Text(f"error: {_err(t.get_value()):.2f}", font_size=24, color=GREY_A).to_corner(UR))
        self.add_fixed_in_frame_mobjects(err)

        self.begin_ambient_camera_rotation(rate=0.018)
        self.wait(0.3)
        self.play(t.animate.set_value(F_ - 1), run_time=26, rate_func=linear)
        self.wait(1.0)
