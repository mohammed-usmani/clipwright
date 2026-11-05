"""Verify teacher trajectories against hidden specs, build the human review page, export SFT rows.

python3 teacher.py verify            -> teacher/verdicts.json + teacher/review/index.html (visual items for a human)
python3 teacher.py export            -> data/layer2/teacher.jsonl (auto-verified jobs, minus any you rejected)
Human verdicts: teacher/review/rejected.txt, one job id per line.
"""
import fnmatch
import glob
import html
import json
import os
import re
import subprocess
import sys

import checks
import sandbox
from clean_ebur128 import clean_result
from build import SYSTEM

HERE = os.path.dirname(os.path.abspath(__file__))
TEACH = os.path.join(HERE, "teacher")
OUT = None
RUN = TEACH            # set by --run NAME -> teacher/runs/NAME (jobs/, verdicts.json, review/)
REVIEW = os.path.join(TEACH, "review")


def lufs(path):
    r = subprocess.run(["ffmpeg", "-nostdin", "-i", path, "-af", "ebur128=framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True)
    m = re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)
    return float(m[-1]) if m else None


def first_frame_mean(path):
    r = subprocess.run(f"ffmpeg -v error -i '{path}' -frames:v 1 -f image2pipe -c:v png - | magick - -format '%[fx:mean]' info:",
                       shell=True, capture_output=True, text=True)
    return float(r.stdout) if r.stdout.strip() else None


def check_file(f, s):
    errs = []
    ext = os.path.splitext(f)[1].lower()
    if ext in (".csv", ".txt", ".srt", ".vtt", ".json", ".ass"):
        text = open(f, errors="replace").read()
        return [f"missing /{p}/" for p in s.get("contains", []) if not re.search(p, text, re.I | re.M)] or \
            ([] if text.strip() else ["empty file"])
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".pdf"):
        if ext == ".pdf":  # identify renders PDFs through ghostscript (slow, can hang): check the header only
            return [] if open(f, "rb").read(5) == b"%PDF-" else ["not a PDF"]
        r = subprocess.run(["magick", "identify", "-ping", "-format", "%m %w %h", f + "[0]"], capture_output=True, text=True)
        try:
            fmt, w, h = r.stdout.split()
            w, h = int(w), int(h)
        except ValueError:
            return ["unreadable"]
        want = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG", ".webp": "WEBP", ".heic": "HEIC", ".pdf": "PDF"}[ext]
        if fmt != want:
            errs.append(f"{ext} file is actually {fmt}")
        if "max_side" in s and max(w, h) > s["max_side"]:
            errs.append(f"{w}x{h} exceeds {s['max_side']}")
        if "wh" in s and [w, h] != s["wh"]:
            errs.append(f"{w}x{h}, expected {s['wh'][0]}x{s['wh'][1]}")
        if s.get("no_gps") or s.get("has_date"):
            ex = json.loads(subprocess.run(["exiftool", "-j", f], capture_output=True, text=True).stdout)[0]
            if s.get("no_gps") and any(k.startswith("GPS") for k in ex):
                errs.append("GPS still present")
            if s.get("has_date") and "DateTimeOriginal" not in ex:
                errs.append("DateTimeOriginal lost")
        return errs + checks.extra(f, s, os.path.dirname(os.path.dirname(f)))
    info = sandbox.probe(f)
    if not info:
        return ["unreadable"]
    st = info["streams"]
    v = next((x for x in st if x["codec_type"] == "video" and not x.get("disposition", {}).get("attached_pic")), None)
    a = next((x for x in st if x["codec_type"] == "audio"), None)
    d = float(info["format"].get("duration") or 0)
    if "dur" in s and not s["dur"][0] <= d <= s["dur"][1]:
        errs.append(f"duration {d:.1f}s not in {s['dur']}")
    if "has_video" in s and bool(v) != s["has_video"]:
        errs.append(f"has_video={bool(v)}")
    if "has_audio" in s and bool(a) != s["has_audio"]:
        errs.append(f"has_audio={bool(a)}")
    if v:
        if "wh" in s and [v["width"], v["height"]] != s["wh"]:
            errs.append(f"{v['width']}x{v['height']} != {s['wh']}")
        if "aspect" in s:
            aw, ah = s["aspect"]
            if abs(v["width"] / v["height"] - aw / ah) > 0.01 or min(v["width"], v["height"]) < s.get("min_short", 0):
                errs.append(f"{v['width']}x{v['height']} is not {aw}:{ah} (short side >= {s.get('min_short', 0)})")
        if "minside" in s and min(v["width"], v["height"]) != s["minside"]:
            errs.append(f"short side {min(v['width'], v['height'])} != {s['minside']}")
        if "max_w" in s and v["width"] > s["max_w"]:
            errs.append(f"width {v['width']} > {s['max_w']}")
        if "vcodec" in s and v["codec_name"] != s["vcodec"]:
            errs.append(f"vcodec {v['codec_name']}")
        if "pix_fmt" in s and v.get("pix_fmt") != s["pix_fmt"]:
            errs.append(f"pix_fmt {v.get('pix_fmt')}")
        if "fps" in s:
            n, dd = v["avg_frame_rate"].split("/")
            if abs(float(n) / float(dd) - s["fps"]) > 0.1:
                errs.append(f"fps {float(n) / float(dd):.2f}")
        if s.get("first_dark"):
            mean = first_frame_mean(f)
            if mean is None or mean > 0.12:
                errs.append(f"first frame not dark (mean {mean})")
    if a:
        if "acodec" in s and a["codec_name"] != s["acodec"]:
            errs.append(f"acodec {a['codec_name']}")
        if "ach" in s and a["channels"] != s["ach"]:
            errs.append(f"channels {a['channels']}")
        if "sr" in s and int(a["sample_rate"]) != s["sr"]:
            errs.append(f"sample rate {a['sample_rate']}")
        if "lufs" in s:
            l = lufs(f)
            if l is None or not s["lufs"][0] <= l <= s["lufs"][1]:
                errs.append(f"loudness {l} LUFS not in {s['lufs']}")
    if "max_mb" in s and os.path.getsize(f) > s["max_mb"] * 1e6:
        errs.append(f"size {os.path.getsize(f) / 1e6:.1f} MB > {s['max_mb']}")
    if s.get("no_meta"):
        tags = {k.lower() for k in info["format"].get("tags", {})}
        if tags & {"title", "creation_time", "comment"}:
            errs.append(f"metadata left: {sorted(tags & {'title', 'creation_time', 'comment'})}")
    return errs + checks.extra(f, s, os.path.dirname(os.path.dirname(f)), info)


def verify(wd, spec, final=None):
    """Spec check of out/ (+ the final reply, for question tasks). Returns error strings (empty = pass). Also the RL reward."""
    if final is None and os.path.exists(os.path.join(wd, "final.txt")):
        final = open(os.path.join(wd, "final.txt")).read()
    errs = checks.answer(final, spec, wd)
    def match(pat):
        out = os.path.join(wd, "out")  # case-insensitive: GX010123.MP4 / IMG_4821.MOV are valid video outputs
        return {os.path.join(out, f) for f in os.listdir(out) for m in pat.split("|")
                if fnmatch.fnmatch(f.lower(), m.lower()) and os.path.isfile(os.path.join(out, f))}
    for s in spec["outputs"]:
        # a file matched by a named pattern (contact*) belongs to it, not to a generic one (*.jpg)
        others = set().union(*[match(o["match"]) for o in spec["outputs"] if o is not s and not o["match"].startswith("*")])
        files = sorted(match(s["match"]) - others)
        want = s.get("count")
        if not files or (want and len(files) != want) or (not want and s is spec["outputs"][0] and len(files) != 1):
            errs.append(f"{s['match']}: found {len(files)} file(s)" + (f", expected {want}" if want else ""))
            continue
        for f in files:
            errs += [f"{os.path.basename(f)}: {e}" for e in check_file(f, s)]
    return errs


def review_image(wd, jid):
    os.makedirs(REVIEW, exist_ok=True)
    outs = sorted(p for p in glob.glob(os.path.join(wd, "out", "*")) if os.path.isfile(p))
    vids = [p for p in outs if os.path.splitext(p)[1].lower() in (".mp4", ".webm", ".mov", ".mkv")]
    dst = os.path.join(REVIEW, f"{jid}.jpg")
    if vids:
        d = float((sandbox.probe(vids[0]) or {}).get("format", {}).get("duration") or 0)
        tmp = []
        for i, t in enumerate([0.3, 1.5, 2.5, d * 0.35, d * 0.6, d * 0.9]):
            t = min(t, max(0, d - 0.2))
            fr = os.path.join(REVIEW, f".{jid}_{i}.jpg")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", vids[0], "-frames:v", "1", "-vf",
                            "scale=-2:360", fr], capture_output=True)
            if os.path.exists(fr):
                subprocess.run(["magick", fr, "-gravity", "north", "-background", "black", "-fill", "white", "-font", "DejaVu-Sans",
                                "-pointsize", "16", "-splice", "0x22", "-annotate", "+0+2", f"t={t:.1f}s", fr])
                tmp.append(fr)
        if tmp:
            subprocess.run(["magick", "montage", *tmp, "-tile", "3x", "-geometry", "+4+4", "-background", "#222", dst])
            for x in tmp:
                os.remove(x)
    else:
        imgs = [p for p in outs if os.path.splitext(p)[1].lower() in (".jpg", ".png", ".webp")][:6]
        if imgs:
            subprocess.run(["magick", "montage", *imgs, "-resize", "480x480>", "-tile", "3x", "-geometry", "+4+4",
                            "-background", "#222", dst])
    return os.path.exists(dst)


def cmd_verify():
    verdicts, rows = {}, []
    for wd in sorted(glob.glob(os.path.join(RUN, "jobs", "*"))):
        jid = os.path.basename(wd)
        spec = json.load(open(os.path.join(TEACH, "specs", f"{jid}.json")))
        traj = os.path.join(wd, "trajectory.jsonl")
        if not os.path.exists(os.path.join(wd, "final.txt")):
            verdicts[jid] = {"status": "unfinished" if os.path.exists(traj) else "not started"}
            continue
        errs = verify(wd, spec)
        steps = sum(1 for _ in open(traj)) if os.path.exists(traj) else 0
        v = {"status": "pass" if not errs else "fail", "errors": errs, "steps": steps, "visual": spec["visual"]}
        if not errs and spec["visual"]:
            v["review_image"] = review_image(wd, jid)
            rows.append((jid, spec, v))
        verdicts[jid] = v
    json.dump(verdicts, open(os.path.join(RUN, "verdicts.json"), "w"), indent=1)
    write_review(rows)
    counts = {}
    for v in verdicts.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(json.dumps(counts), f"| {len(rows)} need visual review -> {os.path.join(REVIEW, 'index.html')}")
    for jid, v in verdicts.items():
        if v["status"] == "fail":
            print(f"  FAIL {jid}: {'; '.join(v['errors'])}")


def write_review(rows):
    os.makedirs(REVIEW, exist_ok=True)
    parts = ["<!doctype html><meta charset=utf-8><title>Teacher review</title><style>body{font:14px system-ui;background:#111;"
             "color:#eee;max-width:1100px;margin:auto;padding:16px}section{border-top:1px solid #444;padding:12px 0}"
             "img{max-width:100%}pre{white-space:pre-wrap;background:#1b1b1b;padding:8px}li{margin:4px 0}</style>",
             "<h1>Visual review</h1><p>Each job below passed every automatic check. Look at the frames and judge the listed items. "
             "Write the ids of any bad jobs into <code>teacher/review/rejected.txt</code> (one per line).</p>"]
    for jid, spec, v in rows:
        parts.append(f"<section><h2>{jid}</h2><pre>{html.escape(spec['request'])}</pre><b>Check:</b><ul>" +
                     "".join(f"<li>{html.escape(x)}</li>" for x in spec["visual"]) + "</ul>" +
                     (f"<img src='{jid}.jpg'>" if v.get("review_image") else "<p>(no image)</p>") + "</section>")
    open(os.path.join(REVIEW, "index.html"), "w").write("\n".join(parts))


def cmd_export():
    verdicts = json.load(open(os.path.join(RUN, "verdicts.json")))
    rej_f = os.path.join(REVIEW, "rejected.txt")
    rejected = set(open(rej_f).read().split()) if os.path.exists(rej_f) else set()
    os.makedirs(os.path.join(HERE, "data", "layer2"), exist_ok=True)
    n = 0
    with open(OUT or os.path.join(HERE, "data", "layer2", "teacher.jsonl"), "w") as out:
        for jid, v in sorted(verdicts.items()):
            if v["status"] != "pass" or jid in rejected:
                continue
            wd = os.path.join(RUN, "jobs", jid)
            spec = json.load(open(os.path.join(TEACH, "specs", f"{jid}.json")))
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": open(os.path.join(wd, "request.txt")).read().strip()}]
            tf = os.path.join(wd, "trajectory.jsonl")
            for line in (open(tf) if os.path.exists(tf) else []):  # a clarifying question has no commands
                s = json.loads(line)
                args = {"cmd": s["cmd"], **({"foreach": s["foreach"]} if s["foreach"] else {})}
                turn = {"role": "assistant", "content": s["think"],
                        "tool_calls": [{"type": "function", "function": {"name": "run", "arguments": args}}]}
                if s["result"].get("exit_code") != 0:
                    turn["weight"] = 0  # keep the mistake + error as context, don't train on producing it
                msgs += [turn, {"role": "tool", "name": "run",
                                 "content": clean_result(json.dumps(s["result"], ensure_ascii=False))}]
            msgs.append({"role": "assistant", "content": open(os.path.join(wd, "final.txt")).read().strip()})
            out.write(json.dumps({"id": jid, "task": "teacher", "category": f"multistep/{spec['kind']}",
                                  "tools": [sandbox.TOOL_SCHEMA], "messages": msgs,
                                  "verify": {"spec": spec, "files": spec["files"]},
                                  "human_reviewed": bool(spec["visual"])}, ensure_ascii=False) + "\n")
            n += 1
    print(f"exported {n} trajectories ({len(rejected)} rejected by review)")


def cmd_compare():
    """Pass rate / steps / cost per model run, cheapest-per-pass first."""
    rows = []
    for d in sorted(glob.glob(os.path.join(TEACH, "runs", "*"))):
        vf = os.path.join(d, "verdicts.json")
        if not os.path.exists(vf):
            continue
        v = json.load(open(vf))
        u = json.load(open(os.path.join(d, "usage.json"))) if os.path.exists(os.path.join(d, "usage.json")) else {}
        n = len(v)
        ok = [x for x in v.values() if x["status"] == "pass"]
        steps = sum(x["steps"] for x in ok) / len(ok) if ok else 0
        cost = u.get("cost", 0)
        rows.append((os.path.basename(d), len(ok), n, steps, cost, cost / len(ok) if ok else float("inf")))
    print(f"{'run':40s} {'pass':>9s} {'steps/pass':>10s} {'cost':>8s} {'$/pass':>8s}")
    for r in sorted(rows, key=lambda r: (-r[1] / max(r[2], 1), r[5])):
        print(f"{r[0]:40s} {r[1]:>4d}/{r[2]:<4d} {r[3]:>10.1f} {r[4]:>8.3f} {r[5]:>8.4f}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--out" in args:
        i = args.index("--out")
        OUT = args[i + 1]
        del args[i:i + 2]
    if "--run" in args:
        i = args.index("--run")
        RUN = os.path.join(TEACH, "runs", args[i + 1])
        REVIEW = os.path.join(RUN, "review")
        del args[i:i + 2]
    {"verify": cmd_verify, "export": cmd_export, "compare": cmd_compare}.get(args[0] if args else "", lambda: sys.exit(__doc__))()
