"""Comparison report: pass rates with 95% Wilson intervals, paired exact McNemar tests, per-category tables.

    python3 eval_report.py            # prints markdown and writes eval/compare/RESULTS.md

Multi-step and hand-written runs are re-scored from their job folders with the current verify(), so every run is judged
by the same checker version. Single-step runs use the score stored by eval_agent (layer-1 task checks).
"""
import collections
import json
import math
import os

from teacher import verify

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval")
SETS = {"single": "data/sft/test_single.jsonl", "multi": "data/sft/test_multi.jsonl", "ood": "data/ood/test_ood.jsonl"}
MODELS = [  # label, size, run name per set (None = not run)
    ("Qwen3.5-0.8B base (no fine-tune)", "0.8B", {"single": "compare/base-single", "multi": "qwen08b-base", "ood": "compare/base-ood"}),
    ("V1: Qwen2.5-Coder-7B + burak29 (as published)", "7B", {"single": "compare/v1-single", "multi": "compare/v1-multi", "ood": "compare/v1-ood"}),
    ("Control: Qwen3.5-0.8B + burak29 data", "0.8B", {"single": "compare/ctrl-single", "multi": "compare/ctrl-multi", "ood": "compare/ctrl-ood"}),
    ("V2: Qwen3.5-0.8B + verified data", "0.8B", {"single": "compare/v2-single", "multi": "compare/v2-multi", "ood": "compare/v2-ood"}),
]
ONE_SHOT = [("V2, first command only", "compare/v2-single-1shot")]


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired pass/fail dicts (same ids). Returns (a-only wins, b-only wins, p)."""
    ids = a.keys() & b.keys()
    x = sum(1 for i in ids if a[i] and not b[i])
    y = sum(1 for i in ids if b[i] and not a[i])
    n = x + y
    if n == 0:
        return x, y, 1.0
    p = 2 * sum(math.comb(n, k) for k in range(min(x, y) + 1)) / 2 ** n
    return x, y, min(1.0, p)


def load(name, rows_by_id, rescore):
    path = os.path.join(EVAL, f"{name}.jsonl")
    if not os.path.exists(path):
        return None
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r["id"] not in rows_by_id:
            continue
        if rescore:
            wd = os.path.join(EVAL, name, "jobs", r["id"])
            ok = r["status"] == "done" and not verify(wd, rows_by_id[r["id"]]["verify"]["spec"])
        else:
            ok = r["pass"]
        out[r["id"]] = ok
    return out


def ffmpeg_only(row):
    """Single-step rows whose reference solution uses only ffmpeg/ffprobe (V1 was only trained on ffmpeg)."""
    cmds = [json.loads(tc["function"]["arguments"])["cmd"] if isinstance(tc["function"]["arguments"], str)
            else tc["function"]["arguments"]["cmd"]
            for m in row["messages"] if m.get("weight") != 0 for tc in m.get("tool_calls") or []]
    return bool(cmds) and all(c.split()[0] in ("ffmpeg", "ffprobe") for c in cmds)


def fmt(res, ids=None):
    if res is None:
        return "not run"
    vals = [v for i, v in res.items() if ids is None or i in ids]
    k, n = sum(vals), len(vals)
    lo, hi = wilson(k, n)
    return f"**{k}/{n} ({k / n:.0%})** [{lo:.0%}–{hi:.0%}]" if n else "–"


def main():
    rows = {s: {r["id"]: r for r in map(json.loads, open(os.path.join(HERE, p)))} for s, p in SETS.items()}
    res = {(m, s): load(names[s], rows[s], rescore=s != "single") for m, _, names in MODELS for s in SETS}
    ff = {i for i, r in rows["single"].items() if ffmpeg_only(r)}
    L = ["# Clipwright comparison eval", "",
         "Every result below was produced by running the model's commands in the sandbox and checking the output files "
         "(duration, resolution, codecs, loudness, file size, metadata...). Greedy decoding. 95% Wilson intervals in brackets.", "",
         f"| Model | Size | Single-step ({len(rows['single'])}) | ffmpeg-only subset ({len(ff)}) | Multi-step ({len(rows['multi'])}) | Hand-written, unseen ({len(rows['ood'])}) |",
         "|---|---|---|---|---|---|"]
    for m, size, _ in MODELS:
        L.append(f"| {m} | {size} | {fmt(res[(m, 'single')])} | {fmt(res[(m, 'single')], ff)} | {fmt(res[(m, 'multi')])} | {fmt(res[(m, 'ood')])} |")
    L += ["", "## Same model, different data (the controlled comparison)", "",
          "Control and V2 share the base model, LoRA settings and trainer; only the training data differs. "
          "Exact McNemar test on the same tasks:", "", "| Set | V2 only passes | Control only passes | p |", "|---|---|---|---|"]
    ctrl, v2 = MODELS[2][0], MODELS[3][0]
    for s in SETS:
        if res[(ctrl, s)] and res[(v2, s)]:
            x, y, p = mcnemar(res[(v2, s)], res[(ctrl, s)])
            L.append(f"| {s} | {x} | {y} | {p:.1e} |")
    v1 = MODELS[1][0]
    L += ["", "## V1 (7B, as published) vs V2 (0.8B)", "", "| Set | V2 only passes | V1 only passes | p |", "|---|---|---|---|"]
    for s in SETS:
        if res[(v1, s)] and res[(v2, s)]:
            x, y, p = mcnemar(res[(v2, s)], res[(v1, s)])
            L.append(f"| {s} | {x} | {y} | {p:.1e} |")
    one = {lab: load(n, rows["single"], False) for lab, n in ONE_SHOT}
    if any(one.values()):
        L += ["", "## First command only (no feedback loop), single-step set", "",
              "Scores each 0.8B model on its first command alone, like V1's one-shot answer. Separates what the data "
              "taught from the benefit of seeing results and retrying.", "", "| Model | All | ffmpeg-only |", "|---|---|---|"]
        for lab, r in one.items():
            L.append(f"| {lab} | {fmt(r)} | {fmt(r, ff)} |")
    L += ["", "## Single-step by category", "", "| Category | n | " + " | ".join(m.split(":")[0] for m, _, _ in MODELS) + " |",
          "|---|---|" + "---|" * len(MODELS)]
    cats = collections.defaultdict(list)
    for i, r in rows["single"].items():
        cats[r["category"].split("/")[0]].append(i)
    for c, ids in sorted(cats.items(), key=lambda x: -len(x[1])):
        cells = []
        for m, _, _ in MODELS:
            r = res[(m, "single")]
            cells.append("–" if r is None else f"{sum(r.get(i, False) for i in ids) / len(ids):.0%}")
        L.append(f"| {c} | {len(ids)} | " + " | ".join(cells) + " |")
    L += ["", "## Hand-written set by kind", "", "| Kind | n | " + " | ".join(m.split(":")[0] for m, _, _ in MODELS) + " |",
          "|---|---|" + "---|" * len(MODELS)]
    kinds = collections.defaultdict(list)
    for i, r in rows["ood"].items():
        kinds[r["category"]].append(i)
    for c, ids in sorted(kinds.items()):
        cells = []
        for m, _, _ in MODELS:
            r = res[(m, "ood")]
            cells.append("–" if r is None else f"{sum(r.get(i, False) for i in ids)}/{len(ids)}")
        L.append(f"| {c} | {len(ids)} | " + " | ".join(cells) + " |")
    L += ["", "## Read before quoting", "",
          "- **Single-step and multi-step test sets come from the same generators as V2's training data** (held-out rows, "
          "never trained on, but the same task families and phrasing). That favors V2. The hand-written set is the fair "
          "test: written by hand, real-world phrasing, no model saw it.",
          "- **On the hand-written set, V2 (0.8B) and V1 (7B) are statistically tied** (67% vs 69%, p=1.0), and V2 vs "
          "Control (67% vs 47%) is suggestive but not significant at 45 tasks (p≈0.06).",
          "- V1 can only write ffmpeg commands; categories that need exiftool/ImageMagick/SoX or a question back "
          "(metadata, probe, batch, clarify) are out of its reach. The ffmpeg-only column is the fair single-step comparison.",
          "- Base model: single-step is a random 100 of the 850 (it rambles to the token cap, ~6 min/task); multi-step is "
          "the earlier run with a 2,048-token reply cap (all other runs: 1,024).",
          "- V1 gets one command, run once, no tools; its bare file names are mapped to in/ and out/ (same mapping used to "
          "convert its data for the Control run). The 0.8B models run as agents (up to 15 commands, they see results). "
          "The 'first command only' row shows V2 without that loop: 90% vs 91%, so the loop is not where V2's lead comes from.",
          "- Greedy decoding, one run per task. Checks are automatic (ffprobe/exiftool/magick); visual quality is not judged."]
    text = "\n".join(L) + "\n"
    os.makedirs(os.path.join(EVAL, "compare"), exist_ok=True)
    open(os.path.join(EVAL, "compare", "RESULTS.md"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
