"""Layer 1: render template tasks, execute each in a fresh job dir, verify, write chat-format JSONL.

python3 build.py --per-task 60            # full run
python3 build.py --per-task 2 --only v_   # smoke test a subset
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import sandbox
from fixtures import FIX, clone
from tasks import BATCH, POOLS, TASKS

HERE = os.path.dirname(os.path.abspath(__file__))
SYSTEM = (
    "You are a media-processing assistant working in a sandbox with ffmpeg, ffprobe, ImageMagick (magick), SoX and "
    "ExifTool. The user's files are in in/ (read-only) and are listed with their properties under 'Files:'. Write every "
    "result to out/; that folder is returned to the user. Use the run tool to execute commands. Before each command, "
    "briefly explain your reasoning, based on the actual file properties (e.g. copy streams instead of re-encoding when the "
    "codec already fits). Quote paths that contain spaces. If a command fails, read the error and fix it. When done, "
    "reply with a short summary of the output files. If a request is ambiguous or impossible with these tools, ask or "
    "explain instead of guessing."
)
VERBS = {"convert", "make", "compress", "trim", "cut", "resize", "extract", "remove", "add", "turn", "change", "create",
         "put", "rotate", "flip", "split", "join", "merge", "normalize", "crop", "speed", "slow", "reverse", "burn",
         "strip", "shrink", "set", "get", "save", "mix", "generate", "grab", "take", "loop", "fade", "encode", "apply"}


class Check:
    """Helpers the task checks call. All paths default to the first file in out/."""

    def __init__(self, wd, res, M):
        self.wd, self.res, self.M = wd, res, M
        self.outs = sorted(p for p in glob.glob(os.path.join(wd, "out", "*")) if os.path.isfile(p))
        self.o = self.outs[0] if self.outs else None
        self.stdout = res.get("stdout", "")
        self._info = {}

    def info(self, f=None):
        f = f or self.o
        if f not in self._info:
            self._info[f] = sandbox.probe(f) or {"streams": [], "format": {}}
        return self._info[f]

    def _st(self, t, f=None):
        return next((s for s in self.info(f)["streams"] if s["codec_type"] == t), None)

    def v(self, f=None):
        return self._st("video", f)

    def a(self, f=None):
        return self._st("audio", f)

    def sub(self, f=None):
        return self._st("subtitle", f)

    def wh(self, f=None):
        s = self.v(f)
        return (s["width"], s["height"])

    def dur(self, f=None):
        d = self.info(f)["format"].get("duration")
        return float(d) if d else None

    def fps(self, f=None):
        n, d = self.v(f)["avg_frame_rate"].split("/")
        return float(n) / float(d)

    def rotation(self, f=None):
        for sd in self.v(f).get("side_data_list", []):
            if "rotation" in sd:
                return int(sd["rotation"])
        return 0

    def tags(self, f=None):
        return {k.lower(): v for k, v in self.info(f)["format"].get("tags", {}).items()}

    def size(self, f=None):
        return os.path.getsize(f or self.o)

    def count(self, pat="*"):
        return len([p for p in glob.glob(os.path.join(self.wd, "out", pat)) if os.path.isfile(p)])

    def text(self, f=None):
        return open(f or self.o, errors="replace").read()

    def moov_first(self, f=None):
        b = open(f or self.o, "rb").read()
        return 0 <= b.find(b"moov") < b.find(b"mdat")

    def img(self, f=None):
        r = subprocess.run(["magick", "identify", "-format", "%m %w %h %A %x|", f or self.o], capture_output=True, text=True)
        frames = [x.split() for x in r.stdout.split("|") if x.strip()]
        m, w, h, a, x = frames[0][:5]
        return {"fmt": m, "w": int(w), "h": int(h), "alpha": a.lower() not in ("undefined", "false"),
                "xres": float(x), "n": len(frames)}

    def exif(self, f=None):
        r = subprocess.run(["exiftool", "-j", "-n", f or self.o], capture_output=True, text=True)
        return json.loads(r.stdout)[0] if r.returncode == 0 else {}


def aug(r, s):
    """Light surface variation; meaning never changes."""
    first = s.split()[0].lower() if s.split() else ""
    x = r.random()
    if first in VERBS and x < 0.12:
        s = r.choice(["can you ", "please ", "pls ", "hey, ", "I need to ", "how do I "]) + s[0].lower() + s[1:]
        if s.startswith(("can you", "how do I")) and not s.endswith("?"):
            s += "?"
    elif x < 0.2:
        s = s[0].upper() + s[1:]
    if r.random() < 0.06:
        s += r.choice([" thanks", " pls", " asap", " :)"])
    return s


def fmt(template, vars):
    return re.sub(r" {2,}", " ", template.format(**vars)).strip()


def render(task, r, meta):
    """Pick files/params/phrasing. Returns a job dict, or None if the combo doesn't apply."""
    files, M, disp = {}, {}, {}
    for slot, pool in (task.ins or {}).items():
        choices = [c for c in POOLS[pool] if c[0] not in files]
        name, fx = r.choice(choices)
        files[name] = fx
        disp[slot] = name
        M[slot] = {**meta[fx], "_name": name, "_fix": fx}
    base = {}
    if task.batch:
        B = BATCH[r.choice(task.batch)]
        for name, fx in B["files"]:
            files[name] = fx
        M["batch"] = {**meta.get(B["files"][0][1], {}), "_n": len(B["files"])}
        base.update(files=B["desc"], glob=B["glob"])
    try:
        p = task.gen(r, M) if task.gen else {}
    except (ValueError, IndexError, KeyError):
        return None
    if p is None:
        return None
    if task.batch:
        p["_n"] = M["batch"]["_n"]
    for slot, name in disp.items():
        base[f"{slot}_name"] = name
        m = M[slot]
        if "dur" in m:
            base[f"{slot}_dur"] = f"{m['dur']:.1f}"
        if "w" in m:
            base[f"{slot}_w"], base[f"{slot}_h"] = m["w"], m["h"]
    if "in" in disp:
        stem, ext = os.path.splitext(disp["in"])
        base.update(stem=stem, ext=ext[1:].lower(), stem_q=shlex.quote(stem))
        if M["in"].get("ach"):
            base["cs_from"] = "mono" if M["in"]["ach"] == 1 else "stereo"
    second = disp.get("in2") or disp.get("img")
    if second:
        base["stem2"] = os.path.splitext(second)[0]
    v = {**base, **p}
    out = fmt(task.out, v) if task.out else ""
    ask_v = {**v, **disp, "out": out}
    cmd_v = {**v, **{s: shlex.quote("in/" + n) for s, n in disp.items()},
             "out": shlex.quote("out/" + out) if out else "", "stem2": shlex.quote(base.get("stem2", ""))}
    ask = aug(r, fmt(r.choice(task.asks), ask_v))
    job = {"task": task.id, "cat": task.cat, "ask": ask, "files": files, "M": M, "p": p, "vars": {**v, **disp}}
    if task.cmd is None:
        job["reply"] = fmt(task.reply, {**v, **disp}) if isinstance(task.reply, str) else task.reply(p, M)
        return job
    job["cmd"] = fmt(task.cmd, cmd_v)
    job["foreach"] = BATCH_GLOB(task, base)
    if task.broken:
        job["broken"] = fmt(task.broken, cmd_v)
    job["note"] = fmt(task.note, {**v, **disp}) if isinstance(task.note, str) else None
    return job


def BATCH_GLOB(task, base):
    return base["glob"] if task.foreach else None


def setup(files):
    wd = tempfile.mkdtemp(prefix="job_", dir=os.environ.get("JOB_TMP"))
    os.makedirs(os.path.join(wd, "in"))
    os.makedirs(os.path.join(wd, "out"))
    for name, fx in files.items():
        clone(os.path.join(FIX, fx), os.path.join(wd, "in", name))
    return wd


def call(cmd, foreach=None, think=""):
    args = {"cmd": cmd}
    if foreach:
        args["foreach"] = foreach
    return {"role": "assistant", "content": think,
            "tool_calls": [{"type": "function", "function": {"name": "run", "arguments": args}}]}


def tool(res):
    return {"role": "tool", "name": "run", "content": json.dumps(res, ensure_ascii=False)}


def final_text(job, res, task):
    if job.get("note") is None:  # analysis task: the answer is derived from the real tool output
        return task.note(job["vars"], res)
    outs = res.get("outputs", {})
    lines = [f"- {k}: {v}" for k, v in outs.items()]
    return "Done." + ("\n" + "\n".join(lines) if lines else "")


def execute(job):
    """Runs in a worker process. Returns (row | None, failure | None)."""
    try:
        return _execute(job)
    except Exception as e:
        return None, {"task": job["task"], "why": "crash", "err": repr(e), "cmd": job.get("cmd")}


def _execute(job):
    task = TASKS[job["task"]]
    wd = setup(job["files"])
    try:
        man = sandbox.manifest(wd) or "(none)"
        user = {"role": "user", "content": f"{job['ask']}\n\nFiles:\n{man}"}
        msgs = [{"role": "system", "content": SYSTEM}, user]
        if "reply" in job:
            return {**_meta(job), "messages": msgs + [{"role": "assistant", "content": job["reply"]}]}, None
        if job.get("broken"):
            bres = sandbox.run(wd, job["broken"], job.get("foreach"))
            if bres["exit_code"] == 0:
                return None, {"task": job["task"], "why": "broken command did not fail", "cmd": job["broken"]}
            shutil.rmtree(os.path.join(wd, "out"))
            msgs += [{**call(job["broken"], job.get("foreach"), _pre_broken(job)), "weight": 0}, tool(bres)]
        res = sandbox.run(wd, job["cmd"], job.get("foreach"))
        if res["exit_code"] != 0:
            return None, {"task": job["task"], "why": "exit", "cmd": job["cmd"], "err": res.get("error")}
        try:
            ok = task.check(Check(wd, res, job["M"]), job["p"]) if task.check else True
        except Exception as e:  # a crashing check is a failed check
            ok, res["_check_error"] = False, repr(e)
        if not ok:
            return None, {"task": job["task"], "why": "check", "cmd": job["cmd"], "res": res}
        think = job["note"] or ""
        msgs += [call(job["cmd"], job.get("foreach"), think), tool(res),
                 {"role": "assistant", "content": final_text(job, res, task)}]
        return {**_meta(job), "messages": msgs}, None
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def _pre_broken(job):
    return ""  # the wrong attempt carries no reasoning; it's masked from the loss anyway (weight 0)


def _meta(job):
    return {"id": hashlib.sha1((job["task"] + job["ask"]).encode()).hexdigest()[:12], "task": job["task"],
            "category": job["cat"], "tools": [sandbox.TOOL_SCHEMA],
            "verify": {"task": job["task"], "files": job["files"], "params": job["p"], "M": job["M"]}}


def refresh_logs(out_dir):
    """Re-run rows whose tool result carries a stderr log on the current sandbox; rewrite tool result + answer."""
    n = 0
    for name in ("train.jsonl", "test.jsonl"):
        path = os.path.join(out_dir, name)
        rows = [json.loads(l) for l in open(path)]
        for row in rows:
            msgs = row["messages"]
            i = next((k for k, m in enumerate(msgs) if m["role"] == "tool" and '"log"' in m["content"]), None)
            if i is None:
                continue
            call = msgs[i - 1]["tool_calls"][0]["function"]["arguments"]
            wd = setup(row["verify"]["files"])
            try:
                res = sandbox.run(wd, call["cmd"], call.get("foreach"))
            finally:
                shutil.rmtree(wd, ignore_errors=True)
            if res["exit_code"] != 0:
                raise RuntimeError(f"{row['id']}: refresh failed: {res}")
            task = TASKS[row["task"]]
            msgs[i] = tool(res)
            msgs[-1] = {"role": "assistant", "content": task.note(row["verify"]["params"], res)}
            n += 1
        with open(path + ".tmp", "w") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(path + ".tmp", path)
    print(f"refreshed {n} rows")


def _key(job):
    return hashlib.sha1((job["task"] + job["ask"]).encode()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-task", type=int, default=60)
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--refresh-logs", action="store_true", help="re-run log-bearing rows in --out on the current sandbox")
    a = ap.parse_args()
    if a.refresh_logs:
        return refresh_logs(a.out)
    meta = json.load(open(os.path.join(FIX, "meta.json")))
    r = random.Random(a.seed)
    jobs, seen = [], set()
    for t in TASKS.values():
        if not t.id.startswith(tuple(a.only.split(","))):
            continue
        want = max(1, round(a.per_task * t.weight))
        got = 0
        for _ in range(want * 8):
            if got >= want:
                break
            j = render(t, r, meta)
            if j and j["ask"] not in seen:
                seen.add(j["ask"])
                jobs.append(j)
                got += 1
    os.makedirs(a.out, exist_ok=True)
    # every result is appended + fsynced as it lands; a rerun with the same args skips what's already on disk
    rows_f, fails_f = os.path.join(a.out, "rows.jsonl"), os.path.join(a.out, "failures.jsonl")
    rows = [json.loads(l) for l in open(rows_f)] if os.path.exists(rows_f) else []
    fails = [json.loads(l) for l in open(fails_f)] if os.path.exists(fails_f) else []
    done = {x["id"] for x in rows} | {x.get("key") for x in fails}
    jobs = [j for j in jobs if _key(j) not in done]
    print(f"{len(jobs)} jobs to run ({len(rows)} kept + {len(fails)} failed already on disk); {a.workers} workers")
    with ProcessPoolExecutor(a.workers) as ex, open(rows_f, "a") as rf, open(fails_f, "a") as ff:
        futs = {ex.submit(execute, j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            row, fail = f.result()
            if row:
                rows.append(row)
                fh, x = rf, row
            else:
                fail["key"] = _key(futs[f])
                fails.append(fail)
                fh, x = ff, fail
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if i % 250 == 0:
                print(f"  {i}/{len(jobs)} done, {len(fails)} failed", flush=True)
    rows.sort(key=lambda x: x["id"])
    test = [x for x in rows if int(x["id"], 16) % 10 == 0]
    train = [x for x in rows if int(x["id"], 16) % 10 != 0]
    for name, data in (("train.jsonl", train), ("test.jsonl", test)):
        with open(os.path.join(a.out, name), "w") as fh:
            for x in data:
                fh.write(json.dumps(x, ensure_ascii=False) + "\n")
    stats = {}
    for x in rows:
        stats.setdefault(x["task"], [0, 0])[0] += 1
    for x in fails:
        stats.setdefault(x["task"], [0, 0])[1] += 1
    json.dump({k: {"ok": v[0], "failed": v[1]} for k, v in sorted(stats.items())},
              open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(f"kept {len(rows)} ({len(train)} train / {len(test)} test), failed {len(fails)}")
    bad = {k: v for k, v in stats.items() if v[1]}
    if bad:
        print("failures by task:", json.dumps({k: v[1] for k, v in sorted(bad.items(), key=lambda kv: -kv[1][1])}))


if __name__ == "__main__":
    main()
