"""Execution eval: run a local model (llama-server) as the agent on held-out multi-step jobs, score with verify().

    llama-server -m models/qwen35-08b-f16.gguf --lora models/lora-qwen08b-full.gguf --jinja -c 16384 -np 1 --port 8080
    python3 eval_agent.py --name qwen08b-ft   --lora-scale 1.0
    python3 eval_agent.py --name qwen08b-base --lora-scale 0.0
    python3 eval_agent.py --report qwen08b-base qwen08b-ft

Each job keeps its files in eval/<name>/jobs/<id>/ (in/, out/, transcript.json); one line per job in eval/<name>.jsonl.
Re-running the same --name skips jobs already in the .jsonl (resume after a crash).
"""
import argparse
import collections
import json
import os
import re
import shlex
import shutil
import time
import urllib.error
import urllib.request

import sandbox
from build import Check
from clean_ebur128 import clean_result
from fixtures import FIX, clone
from build import SYSTEM
from teacher import verify
from tasks import TASKS
from teacher_api import GUIDE

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL = os.path.join(HERE, "eval")
MAX_STEPS = 15  # same as teacher_api
MAX_TOKENS = 1024  # per reply; the longest assistant turn in the training data is ~600 tokens
# V1 (Ray1239/ffmpeg-model) was trained to answer with one raw ffmpeg command and no tools: its exact system prompt
V1_SYSTEM = ("You are an expert video engineer. Translate the user's natural language request into a valid FFmpeg command. "
             "Output ONLY the raw FFmpeg command. Do not provide markdown formatting, explanations, or quotes.")


API = {}  # set by --api-model: {"model", "key", "price_in", "price_out", "budget", "ledger"} for a paid API run


class OutOfCredit(Exception):
    """The API refused for billing reasons (or our budget is spent): stop the whole run, keep everything saved."""


def chat(url, msgs, tools, scale, temp=0.0, lora_id=0, n_lora=1):
    body = {"messages": msgs, "temperature": temp, "top_p": 0.8 if temp else 1.0, "max_tokens": MAX_TOKENS}
    headers = {"Content-Type": "application/json"}
    if API:
        if API["spent"] >= API["budget"]:
            raise OutOfCredit(f"budget ${API['budget']:.2f} reached")
        body["model"] = API["model"]
        body["max_tokens"] = 4096
        body.pop("top_p")
        body["messages"] = [{k: v for k, v in m.items() if k != "mapped_cmd"} for m in msgs]
        headers["Authorization"] = "Bearer " + API["key"]
        if API.get("no_think"):  # Qwen-style switch, same as the local runs
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if API.get("priority") is not None:  # the user's shared vLLM server: every call must send priority 100
            body["priority"] = API["priority"]
    else:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if tools:
        body["tools"] = tools
    if scale is not None:  # only when the server was started with --lora (one entry per adapter, in load order)
        body["lora"] = [{"id": i, "scale": scale if i == lora_id else 0.0} for i in range(n_lora)]
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    try:
        for attempt in range(6 if API else 1):  # API runs ride out short network or server blips (~5 min total)
            try:
                with urllib.request.urlopen(req, timeout=900) as r:
                    data = json.load(r)
                break
            except urllib.error.HTTPError as e:
                if not (API and e.code >= 500) or attempt == 5:
                    raise
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                if not API:
                    raise
                if attempt == 5:
                    raise OutOfCredit(f"endpoint unreachable after retries: {e}")
            time.sleep(10 * 2 ** min(attempt, 4))
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")[:300]
        if API and (e.code in (401, 402, 403) or re.search(r"balance|credit|quota|payment|insufficient", msg, re.I)):
            raise OutOfCredit(f"http {e.code}: {msg}")
        raise urllib.error.HTTPError(e.url, e.code, msg, e.headers, None)
    if API:  # running cost, saved after every call so a crash or an empty balance never loses the tally
        u = data.get("usage") or {}
        cost = u.get("estimated_cost")
        if cost is None:
            cost = u.get("prompt_tokens", 0) * API["price_in"] / 1e6 + u.get("completion_tokens", 0) * API["price_out"] / 1e6
        API["spent"] += cost
        API["tokens_in"] += u.get("prompt_tokens", 0)
        API["tokens_out"] += u.get("completion_tokens", 0)
        API["calls"] += 1
        json.dump({k: API[k] for k in ("model", "spent", "calls", "tokens_in", "tokens_out")},
                  open(API["ledger"], "w"), indent=1)
    return data["choices"][0]


def map_v1_paths(cmd, names):
    """V1 writes bare file names (it was trained on 'resize input.jpg ...'). Map them to the sandbox layout, the same
    mapping used to convert its training data for the control run: inputs -> in/, the final output -> out/."""
    for n in sorted(names, key=len, reverse=True):
        cmd = re.sub(rf"""(?<![\w/.-])(['"]?)(?:in/)?{re.escape(n)}\1(?![\w.-])""",
                     lambda m: shlex.quote("in/" + n), cmd)
    try:
        t = shlex.split(cmd)
    except ValueError:
        return cmd
    for i, tok in enumerate(t[1:], 1):  # every output file: a bare file name that isn't an option value or an input
        if t[i - 1] != "-i" and re.fullmatch(r"[\w%.-]+\.[A-Za-z0-9]{2,4}", tok) and not tok.startswith(("in/", "out/")):
            t[i] = "out/" + tok
    return shlex.join(t)


def v1_command(text):
    text = re.sub(r"```\w*", "", text).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return next((l for l in lines if l.startswith("ffmpeg")), lines[0] if lines else "")


def score(row, wd, status, results, msgs):
    """Multi-step rows: verify() against the hidden spec. Single-step rows: the layer-1 task check."""
    if "spec" in row["verify"]:
        final = next((m.get("content") or "" for m in reversed(msgs) if m["role"] == "assistant"), "")
        errs = verify(wd, row["verify"]["spec"], final)
        return not errs and status == "done", errs
    task = TASKS[row["task"]]
    if task.reply:  # ambiguous/impossible request: the right answer is to ask or explain, without running anything
        ok = status == "done" and not results
        return ok, [] if ok else ["ran a command instead of asking/explaining"]
    if status != "done":
        return False, [status]
    if not results:
        return False, ["no command run"]
    for res in reversed(results):  # analysis checks read a tool result's stdout; file checks read out/
        try:
            if task.check(Check(wd, res, row["verify"]["M"]), row["verify"]["params"]):
                return True, []
        except Exception as e:
            err = repr(e)[:120]
        else:
            err = "check failed"
    return False, ["no output file" if not any(os.scandir(os.path.join(wd, "out"))) else err]


def run_job(row, wd, url, scale, guide=False, temp=0.0, lora_id=0, n_lora=1, one_shot=False, v1=False):
    shutil.rmtree(wd, ignore_errors=True)
    os.makedirs(os.path.join(wd, "out"))
    os.makedirs(os.path.join(wd, "in"))
    for name, fx in row["verify"]["files"].items():
        os.makedirs(os.path.dirname(os.path.join(wd, "in", name)), exist_ok=True)
        clone(os.path.join(FIX, fx), os.path.join(wd, "in", name))
    msgs = [dict(m) for m in row["messages"][:2]]  # system + user; the rest is the teacher's answer
    if v1:  # one raw command, run once, no tools and no second chance
        msgs[0]["content"] = V1_SYSTEM
        try:
            text = (chat(url, msgs, None, None)["message"].get("content") or "").strip()
            status = "done"
        except (urllib.error.HTTPError, TimeoutError, urllib.error.URLError) as e:
            text, status = "", f"error: {e}"
        cmd = map_v1_paths(v1_command(text), row["verify"]["files"].keys())
        res = sandbox.run(wd, cmd) if cmd else {"exit_code": 1, "error": "no command"}
        msgs += [{"role": "assistant", "content": text, "mapped_cmd": cmd},
                 {"role": "tool", "content": json.dumps(res, ensure_ascii=False)}]
        json.dump(msgs, open(os.path.join(wd, "transcript.json"), "w"), ensure_ascii=False, indent=1)
        ok, errs = score(row, wd, status, [res] if cmd else [], msgs)
        return {"id": row["id"], "category": row["category"], "pass": ok, "status": status, "steps": 1,
                "errors": errs, "cmd": cmd}
    if guide:  # fair baseline for untuned models: the teacher's own prompt (SYSTEM + working rules)
        msgs[0]["content"] = SYSTEM + "\n" + GUIDE
    status, steps, history, results = "step limit", 0, [], []
    for steps in range(MAX_STEPS + 1):
        try:
            ch = chat(url, msgs, row["tools"], scale, temp, lora_id, n_lora)
        except urllib.error.HTTPError as e:  # e.g. prompt exceeds the context window
            status = f"http {e.code}: {str(e.msg)[:200]}"
            break
        except (TimeoutError, urllib.error.URLError) as e:  # server stalled (e.g. swapping): record it, rerun later
            status = f"timeout: {e}"
            break
        m = ch["message"]
        text, calls = (m.get("content") or "").strip(), m.get("tool_calls") or []
        if not calls:
            msgs.append({"role": "assistant", "content": text})
            status = "done" if ch.get("finish_reason") == "stop" else f"stopped: {ch.get('finish_reason')}"
            break
        if steps == MAX_STEPS:
            break
        call = calls[0]  # one command per turn, as in training
        msgs.append({"role": "assistant", "content": text, "tool_calls": [call]})
        try:
            args = json.loads(call["function"]["arguments"])
            cmd, foreach = args["cmd"], args.get("foreach") or None
            res = sandbox.run(wd, cmd, foreach)
        except (json.JSONDecodeError, KeyError, TypeError):
            cmd, res = None, {"exit_code": 1, "error": "tool call needs a 'cmd' string"}
        msgs.append({"role": "tool", "tool_call_id": call.get("id", f"call_{steps}"), "name": "run",
                     "content": clean_result(json.dumps(res, ensure_ascii=False))})
        results.append(res)
        if one_shot:  # first command only: what the model knows without seeing any result
            status = "done"
            break
        # memory addresses ("@ 0x72504c000") differ on every run: strip them so a repeated failure counts as a loop
        history.append((cmd, foreach if cmd else None, re.sub(r"0x[0-9a-f]+", "", json.dumps(res, sort_keys=True))))
        if len(history) >= 3 and len(set(history[-3:])) == 1:
            status = "loop"
            break
    json.dump(msgs, open(os.path.join(wd, "transcript.json"), "w"), ensure_ascii=False, indent=1)
    ok, errs = score(row, wd, status, results, msgs)
    return {"id": row["id"], "category": row["category"], "pass": ok, "status": status,
            "steps": len([x for x in msgs if x["role"] == "tool"]), "errors": errs}


def report(names):
    for name in names:
        rows = [json.loads(l) for l in open(os.path.join(EVAL, f"{name}.jsonl"))]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["category"]].append(r["pass"])
        cats = "  ".join(f"{c.split('/')[-1]} {sum(v)}/{len(v)}" for c, v in sorted(by.items()))
        st = collections.Counter(r["status"].split(":")[0] for r in rows)
        print(f"{name:16} pass {sum(r['pass'] for r in rows)}/{len(rows)}  |  {cats}  |  {dict(st)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="run name, e.g. qwen08b-ft")
    ap.add_argument("--lora-scale", type=float, help="0 = base model, 1 = fine-tuned (server started with --lora)")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    ap.add_argument("--data", default=os.path.join(HERE, "data", "sft", "test_multi.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", default="", help="comma-separated job ids")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; >0 samples (pass@k), top_p 0.8")
    ap.add_argument("--guide", action="store_true", help="use the teacher's SYSTEM + GUIDE prompt (fair baseline)")
    ap.add_argument("--lora-id", type=int, default=0, help="which loaded adapter to apply (server load order)")
    ap.add_argument("--n-lora", type=int, default=1, help="how many adapters the server loaded")
    ap.add_argument("--one-shot", action="store_true", help="score after the first command (no feedback, no retries)")
    ap.add_argument("--shard", help="K/N: run only rows K, K+N, K+2N... (split one run across processes)")
    ap.add_argument("--v1", action="store_true", help="V1 mode: V1's prompt, one raw command, paths mapped, run once")
    ap.add_argument("--api-model", help="paid OpenAI-compatible API model, e.g. anthropic/claude-sonnet-5-5 (with --url)")
    ap.add_argument("--api-key-env", default="DEEPINFRA_API_KEY", help="key name in --env-file (never printed)")
    ap.add_argument("--env-file", default=os.path.join(HERE, ".env"))
    ap.add_argument("--price", default="2,10", help="$ per 1M input,output tokens (used if the API reports no cost)")
    ap.add_argument("--no-think", action="store_true", help="API runs: ask the model not to think (Qwen chat template)")
    ap.add_argument("--priority", type=int, help="API runs: vLLM request priority (the user's server needs 100)")
    ap.add_argument("--env-url", help="take the API base URL from this key in --env-file (e.g. BASE_URL)")
    ap.add_argument("--env-model", help="take the API model name from this key in --env-file (e.g. MODEL_NAME)")
    ap.add_argument("--budget", type=float, default=1.5, help="stop before spending more than this many $ in total")
    ap.add_argument("--report", nargs="+", help="print pass rates for these run names and exit")
    a = ap.parse_args()
    if a.report:
        return report(a.report)
    rows = [json.loads(l) for l in open(a.data)]
    if a.only:
        rows = [r for r in rows if r["id"] in a.only.split(",")]
    rows = rows[:a.limit]
    if a.shard:  # "K/N": this process takes every N-th row starting at K (parallel workers, one output file)
        k, n = map(int, a.shard.split("/"))
        rows = rows[k::n]
    os.makedirs(EVAL, exist_ok=True)
    out_f = os.path.join(EVAL, f"{a.name}.jsonl")
    env = {l.split("=", 1)[0]: l.split("=", 1)[1].strip().strip("'\"") for l in open(a.env_file)
           if "=" in l and not l.lstrip().startswith("#")} if (a.api_model or a.env_model) else {}
    if a.env_model:
        a.api_model = env[a.env_model]
    if a.env_url:
        base = env[a.env_url].rstrip("/")
        a.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    if a.api_model:
        key = env.get(a.api_key_env)
        if not key:
            raise SystemExit(f"{a.api_key_env} not found in {a.env_file}")
        ledger = os.path.join(EVAL, f"{a.name}.cost.json")
        prev = json.load(open(ledger)) if os.path.exists(ledger) else {}
        pin, pout = map(float, a.price.split(","))
        API.update(no_think=a.no_think, priority=a.priority, model=a.api_model, key=key, price_in=pin, price_out=pout, budget=a.budget, ledger=ledger,
                   spent=prev.get("spent", 0.0), calls=prev.get("calls", 0),
                   tokens_in=prev.get("tokens_in", 0), tokens_out=prev.get("tokens_out", 0))
    done = {json.loads(l)["id"] for l in open(out_f)} if os.path.exists(out_f) else set()
    for i, row in enumerate(rows, 1):
        if row["id"] in done:
            continue
        t = time.time()
        try:
            r = run_job(row, os.path.join(EVAL, a.name, "jobs", row["id"]), a.url, a.lora_scale, a.guide,
                        a.temperature, a.lora_id, a.n_lora, a.one_shot, a.v1)
        except OutOfCredit as e:  # this job is not recorded, so a rerun with the same --name retries it
            print(f"STOPPED before finishing {row['id']}: {e}. {len(done)} earlier + finished jobs are saved; "
                  f"top up and rerun the same command to continue.", flush=True)
            break
        r["secs"] = round(time.time() - t)
        with open(out_f, "a") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        done.add(row["id"])
        spent = f" ${API['spent']:.3f}" if API else ""
        print(f"[{i}/{len(rows)}] {r['id']} {'PASS' if r['pass'] else 'fail'} {r['status']}, {r['steps']} steps, "
              f"{r['secs']}s{spent} {'; '.join(r['errors'])[:100]}", flush=True)
    report([a.name])


if __name__ == "__main__":
    main()
