"""Post-run triage: sort failed teacher jobs into harness vs model failures, and prepare the cheapest fix.

    python3 triage.py --run main-deepseek            # analyze only (local, $0): writes teacher/runs/<run>/triage.json
    python3 triage.py --run main-deepseek --apply    # archive failures, then:
                                                     #   harness -> keep steps before the first changed observation,
                                                     #              swap in the corrected one, mark for resume (teacher_api continues)
                                                     #   model   -> reset for a full regeneration (max --max-regens per job)

A failure counts as harness-caused when replaying the model's own recorded commands on the CURRENT sandbox produces a
materially different observation at some step k (the model was reacting to something that's no longer true).
If every observation replays identically, the failure is the model's.
"""
import argparse
import glob
import json
import os
import re
import shutil
import tempfile

import sandbox
from fixtures import clone
from teacher import TEACH, verify

TERMINAL_FAIL = {"loop", "step limit", "empty response"}


TS = re.compile(r"\d{4}[:-]\d{2}[:-]\d{2}[ T]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?")


def _clean(s):
    s = re.sub(r"@ 0x[0-9a-f]+", "@", s)
    s = TS.sub("<ts>", s)  # file mtimes/atimes (exiftool) differ on every replay
    s = re.sub(r"\d+\.\d+u \d+:\d+\.\d+", "<t>", s)  # magick identify's own elapsed time
    s = re.sub(r"(?m)^\s*(Pixels per second|User time|Elapsed time):.*$", "", s)  # identify -verbose timing
    return "\n".join(l for l in s.replace("\r", "\n").split("\n") if l.strip() and not sandbox.NOISE.match(l)).strip()


def material(res):
    """The parts of a tool result the model actually acts on. Cosmetic changes (temp files now listed, nicer
    summaries, banner/progress noise, memory addresses) don't count; missing reports, errors, broken outputs do."""
    d = {"exit": res.get("exit_code"), "note": bool(res.get("note"))}
    if res.get("error"):
        err = "\n".join(_clean(res["error"]).splitlines()[-2:])  # e.g. magick compare scores jitter like stdout
        d["error"] = (re.sub(NUM, "#", err), [float(x) for x in re.findall(NUM, err)])
    if res.get("stdout"):
        raw = res["stdout"].split("\n...[truncated")[0]
        if len(raw) >= 3000:  # sandboxes capped stdout at different lengths: compare the common, complete lines
            raw = raw[:3000].rsplit("\n", 1)[0]
        txt = _clean(raw)
        d["stdout"] = (re.sub(NUM, "#", txt), [float(x) for x in re.findall(NUM, txt)])
    outs = {k: v for k, v in (res.get("outputs") or {}).items() if k.startswith("out/")}
    d["outputs"] = sorted((k, v.startswith("UNREADABLE")) for k, v in outs.items())
    d["analysis"] = [l for l in _clean(res.get("log", "")).splitlines() if sandbox.ANALYSIS.search(l)]
    return d


NUM = r"-?\d+(?:\.\d+)?"


def same(a, b):
    """Structural equality: exit code, notes, outputs, analysis lines and the text of stdout/errors with numbers
    masked. Numbers alone never count: replays re-encode video (x264 threading isn't bit-exact), so pixel means,
    compare scores, sizes and frame counts drift several % while the model's situation is unchanged."""
    skel = lambda d: {k: (v[0] if k in ("stdout", "error") else v) for k, v in d.items()}
    return skel(a) == skel(b)


def fresh_job(src_in):
    wd = tempfile.mkdtemp(prefix="triage_", dir=os.environ.get("JOB_TMP"))
    shutil.copytree(src_in, os.path.join(wd, "in"), copy_function=clone)
    os.makedirs(os.path.join(wd, "out"))
    return wd


def replay(wd, steps, upto=None):
    """Re-run recorded commands on the current sandbox. Returns (first divergent index or None, new results)."""
    first, new = None, []
    for i, s in enumerate(steps[:upto]):
        r = sandbox.run(wd, s["cmd"], s["foreach"]) if s["cmd"] else s["result"]
        new.append(r)
        if first is None and not same(material(r), material(s["result"])):
            first = i
    return first, new


def analyze(run_dir, also=()):
    report = {}
    for wd in sorted(glob.glob(os.path.join(run_dir, "jobs", "*"))):
        jid = os.path.basename(wd)
        st_f = os.path.join(wd, "status.txt")
        if not os.path.exists(st_f):
            continue  # unfinished: a normal resume handles it
        status = open(st_f).read().strip()
        spec = json.load(open(os.path.join(TEACH, "specs", f"{jid}.json")))
        errs = verify(wd, spec) if status == "done" else [status]
        if not errs and jid not in also:
            continue
        traj_f = os.path.join(wd, "trajectory.jsonl")
        steps = [json.loads(l) for l in open(traj_f)] if os.path.exists(traj_f) else []
        tmp = fresh_job(os.path.join(wd, "in"))
        try:
            k, new = replay(tmp, steps)
            replay_errs = verify(tmp, spec) if steps else ["no steps"]
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        report[jid] = {"status": status, "errors": errs, "steps": len(steps),
                       "bucket": "harness" if k is not None else "model", "diverged_at": k, "passed": not errs,
                       "replay_passes_as_is": not replay_errs,
                       "old_obs": material(steps[k]["result"]) if k is not None else None,
                       "new_obs": material(new[k]) if k is not None else None}
    return report


def archive(run_dir, wd):
    jid = os.path.basename(wd)
    adir = os.path.join(run_dir, "archive")
    os.makedirs(adir, exist_ok=True)
    n = len(glob.glob(os.path.join(adir, f"{jid}.*.json")))
    rec = {"job": jid}
    for f in ("trajectory.jsonl", "final.txt", "status.txt"):
        p = os.path.join(wd, f)
        if os.path.exists(p):
            rec[f] = open(p).read()
    json.dump(rec, open(os.path.join(adir, f"{jid}.{n}.json"), "w"))
    return n + 1  # attempts so far


def clear(wd):
    shutil.rmtree(os.path.join(wd, "out"), ignore_errors=True)
    os.makedirs(os.path.join(wd, "out"))
    for p in glob.glob(os.path.join(wd, "*")):
        if os.path.isfile(p) and os.path.basename(p) != "request.txt":
            os.remove(p)


def apply(run_dir, report, max_regens):
    counts = {"resume": 0, "regenerate": 0, "dropped": 0}
    for jid, r in report.items():
        if r.get("passed") and r["bucket"] == "model":
            continue  # force-checked job that passes and replays identically: keep it
        wd = os.path.join(run_dir, "jobs", jid)
        steps = [json.loads(l) for l in open(os.path.join(wd, "trajectory.jsonl"))] if r["steps"] else []
        attempts = archive(run_dir, wd)
        clear(wd)
        if r["bucket"] == "harness":
            k = r["diverged_at"]
            _, new = replay(wd, steps, upto=k + 1)  # rebuild the job dir state up to and including step k
            with open(os.path.join(wd, "trajectory.jsonl"), "w") as f:
                for i, s in enumerate(steps[:k + 1]):
                    f.write(json.dumps({**s, "result": new[i]}, ensure_ascii=False) + "\n")
            open(os.path.join(wd, "resume.txt"), "w").write(f"continue after step {k} (corrected observation)\n")
            counts["resume"] += 1
        elif attempts <= max_regens:
            counts["regenerate"] += 1  # cleared: teacher_api regenerates it from scratch
        else:
            open(os.path.join(wd, "status.txt"), "w").write("dropped")
            counts["dropped"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--also-grep", help="also triage jobs whose trajectory contains this text, even if they passed")
    ap.add_argument("--max-regens", type=int, default=1, help="full regenerations allowed per job for model failures")
    a = ap.parse_args()
    run_dir = os.path.join(TEACH, "runs", a.run)
    also = set()
    if a.also_grep:
        also = {os.path.basename(os.path.dirname(f)) for f in glob.glob(os.path.join(run_dir, "jobs", "*", "trajectory.jsonl"))
                if a.also_grep in open(f).read()}
    report = analyze(run_dir, also)
    report = {j: r for j, r in report.items() if not (r["passed"] and r["bucket"] == "model")}
    json.dump(report, open(os.path.join(run_dir, "triage.json"), "w"), indent=1)
    by = {}
    for r in report.values():
        by.setdefault(r["bucket"], []).append(r)
    print(f"{len(report)} failed jobs: " + ", ".join(f"{b} {len(v)}" for b, v in by.items()))
    for jid, r in report.items():
        line = f"  {jid} [{r['bucket']}] {r['status']}, {r['steps']} steps: {'; '.join(r['errors'])[:110]}"
        if r["bucket"] == "harness":
            line += f" | diverged at step {r['diverged_at']}"
        print(line)
    if a.apply:
        print("applied:", apply(run_dir, report, a.max_regens))


if __name__ == "__main__":
    main()
