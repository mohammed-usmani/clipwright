"""Run multi-step jobs (teacher/jobs/*) with an open model on any OpenAI-compatible API (Together, DeepInfra, Z.ai...), via tool calling against the sandbox.

Writes <job>/trajectory.jsonl + final.txt in the same format as step.py, so `teacher.py verify/export` work unchanged.

    export TOGETHER_API_KEY=...            # or: --env-file path/to/.env
    python3 jobs.py --n 10 --seed 1
    python3 teacher_api.py --model zai-org/GLM-5.3-Flash --run glm --env-file path/to/.env
    python3 teacher_api.py --base-url https://api.deepinfra.com/v1/openai --key-var DEEPINFRA_API_KEY --model <id> --run glm-di
    python3 teacher.py verify
"""
import argparse
import glob
import json
import socket
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import sandbox
from fixtures import clone
from build import SYSTEM

HERE = os.path.dirname(os.path.abspath(__file__))
API = "https://api.together.xyz/v1"  # any OpenAI-compatible base URL via --base-url
MAX_STEPS = 15

# Extra guidance for the generating model only; exported rows use the plain SYSTEM prompt.
GUIDE = """
Working rules:
- If the user asks a question about the files, probe with one or two commands and answer in words; don't create files.
- If the request is too vague to act on (which file? what result?), ask one short clarifying question instead of guessing.
- Allowed programs: ffmpeg, ffprobe, magick, sox, soxi, exiftool. Only && may chain commands (no pipes, redirects, ;, $ or backticks). One command line per tool call.
- Inputs are in in/. Only the deliverables the user asked for go in out/. Put intermediate files in the job root (e.g. tmp_trim.mp4).
- First message: state the deliverables, the relevant file properties (codec, resolution, duration, audio), and your plan.
- Prefer one encode for the main video (combine trim/scale/overlay/fades/audio in one filter_complex); stream-copy when nothing needs re-encoding. Derive GIFs, thumbnails and MP3s from the finished main output.
- Show the arithmetic for derived numbers (fade-out start, bitrate for a size cap, etc.). loudnorm needs -ar to restore the sample rate.
- Text: ffmpeg drawtext needs no fontfile; ImageMagick text needs -font DejaVu-Sans.
- After producing the deliverables, check them with ffprobe / magick identify / exiftool and fix anything that's off. If a command fails, explain the cause and fix it.
- To read a subtitle/text file use `ffmpeg -v error -i in/file.srt -f srt -` (cat/echo are not available). Never add -v error / -loglevel error to analysis commands (loudnorm, ebur128, volumedetect, silencedetect): it hides their report. Never repeat a command whose result you already have.
- Loudness targets apply to the FINAL deliverable: normalize after all mixing/concatenation (music, jingles, intros, downmix to mono), then measure the final file with ebur128 and correct it if it's off by more than 1 LU.
- "720p"/"1080p" for a vertical (portrait) video means the short side, e.g. 720x1280. Re-check the requested fps, resolution and size limits with ffprobe before finishing.
- out/ must contain exactly the requested deliverables. There is no rm/mv/ls/cp: you can't delete, rename or list files, so write each deliverable directly under its final name in out/ and keep intermediates in the job root (they need no cleanup). Use ffprobe/exiftool on a path to check a file. If a deliverable in out/ turns out wrong (e.g. loudness off), re-render it to the SAME name with -y from its sources, never as a second file (no _fixed/_norm/_final copies). When a result may need correcting, render it to the job root first (tmp_main.mp4), check it, then stream-copy it into out/ (-c copy).
- Keep each explanation to 1-5 sentences. When everything is verified, reply with a short summary and no tool call.
"""

usage = {"in": 0, "cached": 0, "out": 0}
lock = threading.Lock()
STOP = threading.Event()  # set on out-of-credit / auth errors: finish nothing new, keep everything on disk
FATAL_WORDS = ("insufficient", "balance", "credit", "quota", "payment", "billing", "exceeded your")


class Stopped(Exception):
    pass


def load_key(env_file, var="TOGETHER_API_KEY"):
    if os.environ.get(var):
        return os.environ[var]
    if env_file:
        for line in open(env_file):
            k, _, v = line.strip().partition("=")
            if k.strip() == var:
                return v.strip().strip("'\"")
    sys.exit(f"set {var} or pass --env-file")


def append_durable(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_durable(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


MAX_TOKENS = 8192
EXTRA = {}  # extra request fields for self-hosted vLLM: priority, chat_template_kwargs (set from the CLI)


WITH_TOOLS = True  # paraphrase.py turns this off


def chat(key, model, messages, temperature, base=API):
    tools = {"tools": [sandbox.TOOL_SCHEMA], "tool_choice": "auto"} if WITH_TOOLS else {}
    body = json.dumps({"model": model, "messages": messages, **tools,
                       "temperature": temperature, "max_tokens": MAX_TOKENS, **EXTRA}).encode()
    net_waits = 0
    for attempt in range(6):
        if STOP.is_set():
            raise Stopped("run stopped")
        req = urllib.request.Request(base.rstrip("/") + "/chat/completions", body,
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                                      "User-Agent": "clipwright/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = json.load(r)
            u = data.get("usage") or {}
            call = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0),
                    "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0}
            with lock:
                for k in usage:
                    usage[k] += call[k]
            return data["choices"][0]["message"], call
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            if e.code in (401, 402) or (e.code in (403, 429) and any(w in msg.lower() for w in FATAL_WORDS)):
                STOP.set()
                raise Stopped(f"HTTP {e.code}: {msg}")
            if e.code in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(2 ** attempt * 2)
                continue
            raise RuntimeError(f"HTTP {e.code}: {msg}")
        except OSError:  # URLError, timeouts, resets, "no route to host"
            # network down (DNS, wifi): free to wait, so ride out ~15 min instead of failing the job
            while net_waits < 30:
                net_waits += 1
                time.sleep(30)
                try:
                    socket.getaddrinfo(urllib.parse.urlparse(base).hostname, 443)
                    break
                except OSError:
                    continue
            else:
                raise
            if attempt < 5:
                continue
            raise


class JobBudget:
    """Per-job cost tracking. Hard cap stops the job; structural anomalies alert immediately.
    Cap default $0.05 = ~2.5x the worst legitimate job (15 steps, capped tool output); a typical job is ~$0.002."""

    def __init__(self, jid, prices, cap, alert):
        self.jid, self.prices, self.cap, self.alert = jid, prices, cap, alert
        self.cost = 0.0
        self.tokens = {"in": 0, "cached": 0, "out": 0}
        self.calls = 0

    def add(self, call):
        pi, pc, po = self.prices
        self.calls += 1
        for k in self.tokens:
            self.tokens[k] += call[k]
        self.cost += ((call["in"] - call["cached"]) * pi + call["cached"] * pc + call["out"] * po) / 1e6
        if call["out"] > 6000:
            self.alert(self.jid, "anomaly", f"one call produced {call['out']} output tokens (normal is <1500)")
        if call["in"] > 120_000:
            self.alert(self.jid, "anomaly", f"one call sent {call['in']} prompt tokens (normal is <40k)")

    def exceeded(self):
        return self.cost >= self.cap


def parse_args(raw):
    try:
        a = json.loads(raw) if isinstance(raw, str) else raw
        return a.get("cmd"), a.get("foreach") or None
    except (json.JSONDecodeError, AttributeError):
        return None, None


def run_job(wd, key, model, temperature, base=API, budget=None):
    """Returns (jid, steps, status). status.txt is written only for terminal outcomes; jobs without it are resumed."""
    jid = os.path.basename(wd)
    log, fin = os.path.join(wd, "trajectory.jsonl"), os.path.join(wd, "final.txt")
    request = open(os.path.join(wd, "request.txt")).read().strip()
    msgs = [{"role": "system", "content": SYSTEM + "\n" + GUIDE}, {"role": "user", "content": request}]
    nudged, status, step, history = False, "step limit", 0, []
    sp = os.path.join(HERE, "teacher", "specs", jid + ".json")
    wants_files = not os.path.exists(sp) or bool(json.load(open(sp)).get("outputs"))  # questions are answered in words
    start = 0
    if os.path.exists(os.path.join(wd, "resume.txt")) and os.path.exists(log):
        # continue a triaged trajectory: replay its recorded steps as context (the last one carries the corrected result)
        for i, l in enumerate(open(log)):
            s = json.loads(l)
            args = {"cmd": s["cmd"], **({"foreach": s["foreach"]} if s["foreach"] else {})}
            msgs += [{"role": "assistant", "content": s["think"], "tool_calls": [
                         {"id": f"call_{i}", "type": "function", "function": {"name": "run", "arguments": json.dumps(args)}}]},
                     {"role": "tool", "tool_call_id": f"call_{i}", "name": "run",
                      "content": json.dumps(s["result"], ensure_ascii=False)}]
            history.append((s["cmd"], s["foreach"], json.dumps(s["result"], sort_keys=True)))
            start = i + 1
    for step in range(start, MAX_STEPS + 1):
        if budget and budget.exceeded():
            status = "over budget"
            budget.alert(jid, "cap", f"job cost ${budget.cost:.4f} reached the ${budget.cap} cap after {budget.calls} calls; stopped")
            break
        for _ in range(3):  # reasoning models sometimes spend the budget thinking and return nothing
            m, call = chat(key, model, msgs, temperature, base)
            if budget:
                budget.add(call)
            calls = m.get("tool_calls") or []
            text = (m.get("content") or "").strip()
            if calls or text:
                break
        else:
            status = "empty response"
            break
        if not calls and not nudged and wants_files and not any(os.scandir(os.path.join(wd, "out"))):
            nudged = True  # stopped with a plan but no deliverables: one nudge, not logged as a training turn
            msgs += [{"role": "assistant", "content": text},
                     {"role": "user", "content": "Nothing has been written to out/ yet. Carry out the job now with the run tool."}]
            continue
        if not calls:
            write_durable(fin, text)
            status = "done"
            break
        if step == MAX_STEPS:
            break
        msgs.append({"role": "assistant", "content": text, "tool_calls": calls[:1]})
        call = calls[0]  # one command per turn keeps trajectories clean
        cmd, foreach = parse_args(call["function"].get("arguments"))
        res = sandbox.run(wd, cmd, foreach) if cmd else {"exit_code": 1, "error": "tool call needs a 'cmd' string"}
        append_durable(log, {"think": text, "cmd": cmd or "", "foreach": foreach, "result": res})
        history.append((cmd, foreach, json.dumps(res, sort_keys=True)))
        if len(history) >= 3 and len(set(history[-3:])) == 1:
            status = "loop"  # same command, same result 3x: stop paying for it (no final -> never exported)
            break
        msgs.append({"role": "tool", "tool_call_id": call.get("id", f"call_{step}"), "name": "run",
                     "content": json.dumps(res, ensure_ascii=False)})
    write_durable(os.path.join(wd, "status.txt"), status)
    if os.path.exists(os.path.join(wd, "resume.txt")):
        os.remove(os.path.join(wd, "resume.txt"))  # only once the job has a terminal status
    return jid, step, status


def reset(wd):
    shutil.rmtree(os.path.join(wd, "out"), ignore_errors=True)
    os.makedirs(os.path.join(wd, "out"))
    for p in glob.glob(os.path.join(wd, "*")):
        if os.path.isfile(p) and os.path.basename(p) != "request.txt":
            os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--env-file")
    ap.add_argument("--key-var", default="TOGETHER_API_KEY", help="env var holding the API key")
    ap.add_argument("--base-url", default=API, help="OpenAI-compatible base URL, e.g. https://api.deepinfra.com/v1/openai")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--only", default="", help="comma-separated job ids, or @file with one id per line")
    ap.add_argument("--run", required=True, help="run name: inputs copied to teacher/runs/NAME/jobs, results kept there")
    ap.add_argument("--redo", action="store_true", help="wipe and rerun every selected job, including finished ones")
    ap.add_argument("--price-in", type=float, default=0.2, help="$ per 1M input tokens, for the cost line")
    ap.add_argument("--price-out", type=float, default=0.6, help="$ per 1M output tokens")
    ap.add_argument("--price-cached", type=float, default=None, help="$ per 1M cached input tokens (default: same as input)")
    ap.add_argument("--job-cap", type=float, default=0.05, help="hard $ cap per single job; exceeding it stops that job and alerts")
    ap.add_argument("--priority", type=int, default=None, help="vLLM request priority (self-hosted server only)")
    ap.add_argument("--no-think", action="store_true", help="disable Qwen thinking (vLLM chat_template_kwargs)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    a = ap.parse_args()
    global MAX_TOKENS
    MAX_TOKENS = a.max_tokens
    if a.priority is not None:
        EXTRA["priority"] = a.priority
    if a.no_think:
        EXTRA["chat_template_kwargs"] = {"enable_thinking": False}
    key = load_key(a.env_file, a.key_var)
    run_dir = os.path.join(HERE, "teacher", "runs", a.run)
    root = os.path.join(run_dir, "jobs")
    if a.only.startswith("@"):
        a.only = ",".join(open(a.only[1:]).read().split())
    only = set(a.only.split(",")) if a.only else None
    for src in sorted(glob.glob(os.path.join(HERE, "teacher", "jobs", "*"))):
        if only and os.path.basename(src) not in only:
            continue  # don't copy all 1,700 jobs for a small run
        dst = os.path.join(root, os.path.basename(src))
        if not os.path.exists(dst):
            os.makedirs(os.path.join(dst, "out"))
            shutil.copytree(os.path.join(src, "in"), os.path.join(dst, "in"), copy_function=clone)
            shutil.copyfile(os.path.join(src, "request.txt"), os.path.join(dst, "request.txt"))
    jobs = sorted(glob.glob(os.path.join(root, "*")))
    if a.only:
        jobs = [j for j in jobs if os.path.basename(j) in a.only.split(",")]
    todo, skipped = [], 0
    for wd in jobs:
        if os.path.exists(os.path.join(wd, "status.txt")) and not a.redo:
            skipped += 1  # finished earlier (done / empty / step limit / loop / dropped): never pay for it twice
            continue
        if not os.path.exists(os.path.join(wd, "resume.txt")) or a.redo:
            reset(wd)  # unfinished or interrupted: restart cleanly (triaged jobs keep their prefix)
        todo.append(wd)
    uf = os.path.join(run_dir, "usage.json")
    prev = json.load(open(uf)) if os.path.exists(uf) else {}
    prev = {"in": 0, "cached": 0, "out": 0, "cost": 0, **prev}
    pc = a.price_in if a.price_cached is None else a.price_cached

    def save_usage():
        with lock:
            cost = ((usage["in"] - usage["cached"]) * a.price_in + usage["cached"] * pc + usage["out"] * a.price_out) / 1e6
            write_durable(uf, json.dumps({"model": a.model, "base_url": a.base_url, "in": prev["in"] + usage["in"],
                                          "cached": prev["cached"] + usage["cached"], "out": prev["out"] + usage["out"],
                                          "cost": round(prev["cost"] + cost, 4)}))
            return cost

    alerts_f = os.path.join(run_dir, "alerts.jsonl")

    def alert(jid, kind, msg):
        append_durable(alerts_f, {"job": jid, "kind": kind, "msg": msg, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"ALERT [{kind}] {jid}: {msg}", flush=True)

    prices = (a.price_in, pc, a.price_out)
    budgets = {wd: JobBudget(os.path.basename(wd), prices, a.job_cap, alert) for wd in todo}
    print(f"{len(todo)} jobs to run ({skipped} already finished), model {a.model}, {a.workers} workers, "
          f"job cap ${a.job_cap}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(run_job, wd, key, a.model, a.temperature, a.base_url, budgets[wd]): wd for wd in todo}
        for f in as_completed(futs):
            jid = os.path.basename(futs[f])
            try:
                _, steps, status = f.result()
            except Stopped as e:
                steps, status = None, f"stopped ({e})" if "run stopped" not in str(e) else "stopped"
            except Exception as e:
                steps, status = None, f"error: {e}"
            b = budgets[futs[f]]
            append_durable(os.path.join(run_dir, "progress.jsonl"),
                           {"job": jid, "status": status, "steps": steps, "cost": round(b.cost, 5), "calls": b.calls,
                            **b.tokens, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
            cost = save_usage()
            print(f"  {jid}: {status}" + (f" after {steps} steps" if steps is not None else "") +
                  f"  job ${b.cost:.4f} | run ${cost:.3f}", flush=True)
    cost = save_usage()
    print(f"tokens: {usage['in']:,} in ({usage['cached']:,} cached) / {usage['out']:,} out ≈ ${cost:.3f} this session")
    if STOP.is_set():
        print("STOPPED: provider refused further requests (credit/auth). All finished jobs are saved; "
              "rerun the same command after topping up to resume the rest.")


if __name__ == "__main__":
    main()
