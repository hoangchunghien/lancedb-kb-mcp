"""Draw the model comparison charts from bench/results/models.json as SVG.

The SVGs follow the reader's light or dark theme (prefers-color-scheme), so they work
inside a blog post's <img> and in the GitHub README.

    uv run --group bench python bench/charts.py
"""

import json
import math
import os
from xml.sax.saxutils import escape

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results", "models.json")
OUT = os.path.join(HERE, "results")
KEYWORD = "keyword only, no model"

SHORT = {
    "sentence-transformers/all-MiniLM-L6-v2": "all-MiniLM-L6",
    "BAAI/bge-small-en-v1.5": "bge-small",
    "Snowflake/snowflake-arctic-embed-s": "arctic-embed-s",
    "intfloat/e5-small-v2": "e5-small",
    "BAAI/bge-base-en-v1.5": "bge-base",
    "intfloat/multilingual-e5-small": "multilingual-e5-small",
    "nomic-ai/nomic-embed-text-v1.5": "nomic-embed-v1.5",
    "mixedbread-ai/mxbai-embed-large-v1": "mxbai-large",
    "Qwen/Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B",
    "Qwen3-Embedding-8B (Q6_K, llama.cpp, 1536 dims)": "Qwen3-Embedding-8B",
    "clip-ViT-B-32": "CLIP ViT-B/32",
    "clip-ViT-B-16": "CLIP ViT-B/16",
    "clip-ViT-L-14": "CLIP ViT-L/14",
}
DEFAULTS = {"BAAI/bge-small-en-v1.5", "clip-ViT-B-32"}

STYLE = """<style>
svg{--bg:#ffffff;--ink:#1d2733;--sub:#5a6875;--grid:#e3e8ee;--axis:#9aa6b2;--bar:#2f6fb2;--hi:#2e9e6a;--ref:#c9665f}
@media (prefers-color-scheme:dark){svg{--bg:#121820;--ink:#e6ebf1;--sub:#a9b6c3;--grid:#26303b;--axis:#5b6876;--bar:#7fb0e6;--hi:#82c99e;--ref:#e08a83}}
.bg{fill:var(--bg)} .title{fill:var(--ink);font:700 22px Helvetica,Arial,sans-serif}
.sub{fill:var(--sub);font:15px Helvetica,Arial,sans-serif} .lab{fill:var(--ink);font:15px Helvetica,Arial,sans-serif}
.labb{fill:var(--ink);font:700 15px Helvetica,Arial,sans-serif}
.lab,.labb{paint-order:stroke;stroke:var(--bg);stroke-width:5px;stroke-linejoin:round} .tick{fill:var(--sub);font:14px Helvetica,Arial,sans-serif}
.val{fill:var(--ink);font:14px Helvetica,Arial,sans-serif} .grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1.2} .model{fill:var(--bar)} .default{fill:var(--hi)} .keyword{fill:var(--ref)}
.ref{stroke:var(--ref);stroke-width:2;stroke-dasharray:6 5} .reft{fill:var(--ref);font:700 14px Helvetica,Arial,sans-serif}
.lift{stroke:var(--axis);stroke-width:1.5} .hollow{fill:var(--bg);stroke:var(--bar);stroke-width:2.5}
.dot{fill:var(--bar);stroke:var(--bg);stroke-width:2} .doth{fill:var(--hi);stroke:var(--bg);stroke-width:2}
</style>"""


class Chart:
    def __init__(self, w, h, title, sub, aria):
        self.w, self.h = w, h
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'role="img" aria-label="{escape(aria)}">'
        )
        self.out = [
            svg,
            STYLE,
            f'<rect class="bg" width="{w}" height="{h}" rx="10"/>',
            f'<text class="title" x="28" y="40">{escape(title)}</text>',
            f'<text class="sub" x="28" y="66">{escape(sub)}</text>',
        ]

    def text(self, x, y, s, cls="lab", anchor="start"):
        self.out.append(
            f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}">{escape(s)}</text>'
        )

    def line(self, x1, y1, x2, y2, cls):
        self.out.append(
            f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
        )

    def rect(self, x, y, w, h, cls):
        self.out.append(
            f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{h:.1f}" rx="3"/>'
        )

    def circle(self, x, y, cls, r=7):
        self.out.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>')

    def save(self, name):
        path = os.path.join(OUT, name)
        with open(path, "w") as f:
            f.write("\n".join([*self.out, "</svg>"]))
        return path


def nice_step(span, most=6):
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 500):
        if span / step <= most:
            return step
    return span / most


def hbars(name, title, sub, rows, fmt, lo, hi, foot=""):
    """Horizontal bars. rows: (label, value, kind) with kind "model", "default" or "keyword"."""
    left, right, top, row_h = 250, 90, 92, 40
    feet = foot.split("\n") if foot else []
    h = top + row_h * len(rows) + 50 + 22 * len(feet)
    aria = f"{title}. {sub}. " + "; ".join(f"{lab}: {fmt(v)}" for lab, v, _ in rows)
    c = Chart(800, h, title, sub, aria)
    width = 800 - left - right

    def xs(v):
        return left + (v - lo) / (hi - lo) * width

    step = nice_step(hi - lo)
    hi = lo + math.ceil(round((hi - lo) / step, 6)) * step
    for k in range(round((hi - lo) / step) + 1):
        v = lo + step * k
        c.line(xs(v), top - 6, xs(v), top + row_h * len(rows), "grid")
        c.text(xs(v), top + row_h * len(rows) + 22, fmt(v), "tick", "middle")
    for i, (label, value, kind) in enumerate(rows):
        y = top + i * row_h
        shown = label + (" (default)" if kind == "default" else "")
        c.text(left - 12, y + 25, shown, "labb" if kind != "model" else "lab", "end")
        c.rect(left, y + 8, xs(value) - left, 24, kind)
        c.text(xs(value) + 8, y + 25, fmt(value), "val")
    for i, line in enumerate(feet):
        c.text(28, h - 18 - 22 * (len(feet) - 1 - i), line, "tick")
    return c.save(name)


def scatter(name, title, sub, points, ylo, yhi, ref, foot):
    """x: parameters in millions (log); each point has a vector-only and a hybrid score."""
    left, right, top, bottom = 80, 40, 100, 86
    w, h = 800, 540
    aria = f"{title}. {sub}. {KEYWORD}: {ref:.3f}. " + "; ".join(
        f"{lab} ({x}M): vector {v:.3f}, hybrid {hy:.3f}" for lab, x, v, hy, _ in points
    )
    c = Chart(w, h, title, sub, aria)
    pw, ph = w - left - right, h - top - bottom
    xlo, xhi = math.log10(15), math.log10(12000)

    def xs(v):
        return left + (math.log10(v) - xlo) / (xhi - xlo) * pw

    def ys(v):
        return top + (yhi - v) / (yhi - ylo) * ph

    for v, lab in (
        (20, "20M"),
        (50, "50M"),
        (100, "100M"),
        (300, "300M"),
        (1000, "1B"),
        (8000, "8B"),
    ):
        c.line(xs(v), top, xs(v), top + ph, "grid")
        c.text(xs(v), top + ph + 22, lab, "tick", "middle")
    n = round((yhi - ylo) / 0.02)
    for k in range(n + 1):
        v = ylo + 0.02 * k
        c.line(left, ys(v), left + pw, ys(v), "grid")
        c.text(left - 10, ys(v) + 5, f"{v:.2f}", "tick", "end")
    c.line(left, top + ph, left + pw, top + ph, "axis")
    c.line(left, top, left, top + ph, "axis")
    c.text(
        left + pw / 2, h - 40, "model size, parameters (log scale)", "tick", "middle"
    )
    c.line(left, ys(ref), left + pw, ys(ref), "ref")
    c.text(left + pw - 4, ys(ref) + 20, f"{KEYWORD}: {ref:.3f}", "reft", "end")
    for _, x, vec, hyb, default in sorted(
        points, key=lambda p: p[4]
    ):  # default drawn last, on top
        c.line(xs(x), ys(vec), xs(x), ys(hyb), "lift")
        c.circle(xs(x), ys(vec), "hollow", 6)
        c.circle(xs(x), ys(hyb), "doth" if default else "dot")
    # Labels: models of the same size share an x, so stack their labels to the right of the
    # column with leader lines instead of letting them overprint.
    by_x = {}
    for p in points:
        by_x.setdefault(p[1], []).append(p)
    for x, group in by_x.items():
        group.sort(key=lambda p: -p[3])
        offset = 11 if len(group) == 1 else 56
        if xs(x) > left + pw * 0.85:  # near the right edge: label on the left
            offset = -14
        last = -1e9
        for label, _, _, hyb, default in group:
            y = max(ys(hyb) + 5, last + 21)
            last = y
            if offset > 11:
                c.line(xs(x) + 8, ys(hyb), xs(x) + offset - 4, y - 5, "lift")
            anchor = "end" if offset < 0 else "start"
            c.text(xs(x) + offset, y, label, "labb" if default else "lab", anchor)
    c.circle(left + 18, top - 14, "hollow", 6)
    c.text(left + 30, top - 9, "vector only", "tick")
    c.circle(left + 140, top - 14, "dot")
    c.text(left + 152, top - 9, "hybrid (vector + keyword)", "tick")
    c.text(28, h - 12, foot, "tick")
    return c.save(name)


def main():
    with open(RESULTS) as f:
        r = json.load(f)
    text, image, kw = r["text"], r["image"], r["keyword"]

    def kind(m):
        return "default" if m in DEFAULTS else "model"

    written = []

    pts = [
        (
            SHORT[t["model"]],
            t["params_m"],
            t["scifact"]["vector"]["ndcg10"],
            t["scifact"]["hybrid"]["ndcg10"],
            t["model"] in DEFAULTS,
        )
        for t in text
    ]
    vals = [v for p in pts for v in p[2:4]] + [kw["scifact"]["ndcg10"]]
    ylo, yhi = math.floor(min(vals) * 50) / 50, math.ceil(max(vals) * 50) / 50
    written.append(
        scatter(
            "models-scifact.svg",
            "Text search quality vs model size",
            "BEIR SciFact, 5,183 abstracts, 300 claims: nDCG@10 (higher is better)",
            pts,
            ylo,
            yhi,
            kw["scifact"]["ndcg10"],
            "Each model uses its recommended prompts. Line: the change from adding keyword search.",
        )
    )

    rows = [
        (SHORT[t["model"]], t["vidore"]["hybrid"]["hit1"], kind(t["model"]))
        for t in text
    ]
    rows.append((KEYWORD, kw["vidore"]["hit1"], "keyword"))
    rows.sort(key=lambda x: -x[1])
    lo = math.floor(min(v for _, v, _ in rows) * 10) / 10
    written.append(
        hbars(
            "models-vidore.svg",
            "Finding the right PDF page",
            "ViDoRe V3, 945-page textbook PDF, 132 questions: right page first (hybrid)",
            rows,
            lambda v: f"{v:.0%}",
            lo,
            1.0,
            foot="\n".join(
                f"{SHORT[t['model']]} does better with vector search alone: "
                f"{t['vidore']['vector']['hit1']:.0%}."
                for t in text
                if t["vidore"]["vector"]["hit1"] > t["vidore"]["hybrid"]["hit1"] + 0.02
            ),
        )
    )

    cpu = [t for t in text if "cpu_query_ms" in t]
    rows = [
        (
            SHORT[t["model"]],
            t["cpu_query_ms"] + kw["hybrid_search_ms"],
            kind(t["model"]),
        )
        for t in cpu
    ]
    rows.append((KEYWORD, kw["search_ms"], "keyword"))
    rows.sort(key=lambda x: x[1])
    hi = math.ceil(max(v for _, v, _ in rows) / 10) * 10
    gpu = [t for t in text if "gpu_server_query_ms" in t]
    foot = "Embedding the question plus a hybrid search over 5,183 abstracts. Intel i7-14700F, GPU off."
    if gpu:
        foot += (
            f"\n{SHORT[gpu[0]['model']]} is too big for a laptop CPU; on a GPU server it took "
            f"{gpu[0]['gpu_server_query_ms']:.0f} ms to embed a question."
        )
    written.append(
        hbars(
            "models-cpu.svg",
            "Time per question on a CPU",
            "median of 100 questions, one at a time (lower is better)",
            rows,
            lambda v: f"{v:.0f} ms",
            0,
            hi,
            foot=foot,
        )
    )

    rows = [
        (SHORT[i["model"]] + f" ({i['params_m']}M)", i["recall1"], kind(i["model"]))
        for i in image
    ]
    written.append(
        hbars(
            "models-clip.svg",
            "Finding a photo from a sentence",
            "Flickr30k 1k test, 5,000 captions: right photo first (recall@1)",
            rows,
            lambda v: f"{v:.0%}",
            0.0,
            1.0,
            foot="CPU time to embed a question: "
            + ", ".join(
                f"{SHORT[i['model']]} {i['cpu_query_ms']:.0f} ms" for i in image
            )
            + ".\nPhotos have no text, so there is no keyword-only row.",
        )
    )
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
