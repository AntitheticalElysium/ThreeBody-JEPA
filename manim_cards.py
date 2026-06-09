# manim_cards.py — title / transition cards (white text, solid black) for the JEPA reel.
#   Write reveal, then ApplyWave on the highlighted phrase. Standalone cuts to glue between orbits.
#   CARD=intro uv run manim -r 1280,720 --fps 60 --disable_caching -o card_intro manim_cards.py TitleCard
import os
from manim import *

WHITE_T = "#f2f5fa"
GAP = "__gap__"

# each line: (text, font_size, wave_target)
#   wave_target: None | "all" | (start, end) glyph slice (spaces are not glyphs, so index visible chars)
CARDS = {
    "intro": [("Let's watch JEPA predict the missing", 40, None),
              ("planet in the 3 Body Problem...", 40, (11, 23))],   # "3 Body Problem"
    "add1":  [("We'll make things more interesting...", 40, None),
              ("let's add another planet.", 40, None)],
    "add2":  [("One more?", 52, None)],
    "outro": [("The system is very chaotic now, but JEPA still", 36, None),
              ("manages to extrapolate well from 3 to 5 bodies.", 36, None),
              (GAP, 22, None),
              ("Read the full story below!", 40, "all")],
}


def _line(t, s):
    # invisible spacer needs a real bounding box, else arrange() collapses the whole group
    if t == GAP:
        return Rectangle(width=0.01, height=0.45, stroke_opacity=0, fill_opacity=0)
    return Text(t, font_size=s, color=WHITE_T)


class TitleCard(Scene):
    def construct(self):
        self.camera.background_color = "#000000"
        spec = CARDS[os.environ.get("CARD", "intro")]
        lines = [_line(t, s) for t, s, _ in spec]
        group = VGroup(*lines).arrange(DOWN, buff=0.34).move_to(ORIGIN)

        writeable = VGroup(*[m for (t, _, _), m in zip(spec, lines) if t != GAP])
        waves = []
        for (t, _, wt), m in zip(spec, lines):
            if wt == "all":
                waves.append(ApplyWave(m, amplitude=0.18))
            elif isinstance(wt, tuple):
                waves.append(ApplyWave(m[wt[0]:wt[1]], amplitude=0.18))

        self.wait(0.4)
        self.play(Write(writeable), run_time=2.6)
        self.wait(0.5)
        if waves:
            self.play(*waves, run_time=1.5)
        self.wait(1.4)
        self.play(FadeOut(group), run_time=0.8)
        self.wait(0.3)
