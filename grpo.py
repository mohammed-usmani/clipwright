"""GRPO for Clipwright: each task is attempted G times in the real sandbox, reward.py scores every attempt, and the LoRA
is pushed toward the attempts that beat their group's average.

One process on one GPU (Colab A100): vLLM serves base + the current LoRA, rollouts go through eval_agent.run_job (the
same agent loop and verify() as the eval), the LoRA trains in torch next to it, and each new adapter is hot-loaded
into vLLM. Everything needed to resume (adapter, optimizer, step, task position) is saved to --out every --save-every.

    # Colab setup: media tools matching the Mac versions (apt has ImageMagick 6, no `magick`)
    micromamba install -y -c conda-forge ffmpeg imagemagick sox exiftool && pip install vllm peft causal-conv1d flash-linear-attention  # fast linear-attention kernels
    python3 gen2.py --per-type 40 --seed 101 --out data/rl/pool.jsonl     # verified tasks, train fixtures only
    python3 grpo.py --pool data/rl/pool.jsonl --adapter mohammed-usmani/clipwright-4b --out /content/drive/MyDrive/rl/run1

    # local smoke test, no vLLM: rollouts from a running llama-server, one update on CPU/MPS
    python3 grpo.py --pool x.jsonl --model Qwen/Qwen3.5-0.8B --adapter models/qwen08b-full/final \
        --url http://127.0.0.1:8081/v1/chat/completions --steps 1 --tasks 2 --group 4 --out /tmp/grpo-test

Algorithm: one update per batch of fresh rollouts (on-policy), so the PPO ratio is 1 and clipping is a no-op; loss is
token-level -A * log p over the model's own turns (tool output masked). No KL term (beta=0, as in DAPO / Dr. GRPO);
LoRA + a small LR keep it close to the SFT model. ponytail: add a reference-model KL if evals show drift elsewhere.
"""
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import eval_agent
import sandbox
from reward import reward

# same rendering and masking as the SFT run (notebooks/train_v3_4b.py), so RL trains exactly what SFT trained
ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant\n(?:<think>\n\n</think>\n\n)?(.*?<\|im_end\|>)", re.DOTALL)


def for_template(messages):
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None and k != "weight"}
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])
                                                   if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]}}
                               for tc in m["tool_calls"]]
        out.append(m)
    return out


def encode(tok, msgs, tools):
    """Token ids + a mask of the tokens the model itself wrote (assistant turns). None if it can't be rendered."""
    try:
        msgs = for_template(msgs)
        text = tok.apply_chat_template(msgs, tools=tools, tokenize=False, enable_thinking=False)
    except Exception:
        return None
    spans = [m.span(1) for m in ASSISTANT_TURN.finditer(text)]
    if len(spans) != sum(m["role"] == "assistant" for m in msgs):
        return None
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    mask = [any(a0 <= a and b <= b0 for a0, b0 in spans) for a, b in enc["offset_mapping"]]
    return enc["input_ids"], mask


def http(url, body=None, timeout=60):
    req = urllib.request.Request(url, json.dumps(body).encode() if body is not None else None,
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def start_vllm(a, adapter_dir, name):
    env = {**os.environ, "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True"}
    cmd = ["vllm", "serve", a.model, "--port", str(a.port), "--gpu-memory-utilization", str(a.vllm_mem),
           "--max-model-len", "16384", "--enable-prefix-caching", "--max-num-seqs", str(a.workers), "--enable-lora", "--max-lora-rank", "32",
           "--max-loras", "2", "--lora-modules", f"{name}={adapter_dir}",
           "--enable-auto-tool-choice", "--tool-call-parser", a.tool_parser]
    proc = subprocess.Popen(cmd, stdout=open(os.path.join(a.out, "vllm.log"), "a"), stderr=subprocess.STDOUT, env=env)
    for _ in range(900):
        if proc.poll() is not None:
            raise RuntimeError(f"vLLM exited ({proc.returncode}); see {a.out}/vllm.log")
        try:
            http(f"http://127.0.0.1:{a.port}/health", timeout=5)
            return proc
        except OSError:
            time.sleep(2)
    raise RuntimeError("vLLM did not come up in 30 min")


def rollouts(rows, a):
    """G attempts per task through the eval's agent loop. Returns (row, group index, result, transcript, workdir)."""
    shutil.rmtree(a.work, ignore_errors=True)

    def one(job):
        row, g = job
        wd = os.path.join(a.work, f"{row['id']}_{g}")
        try:
            res = eval_agent.run_job(row, wd, a.url, None, temp=a.temp)
        except Exception as e:  # server hiccup or a crashing check: zero reward, keep the run alive
            res = {"pass": False, "steps": 0, "status": f"crash: {e!r}"[:200], "errors": []}
        tr = os.path.join(wd, "transcript.json")
        return row, g, res, json.load(open(tr)) if os.path.exists(tr) else None, wd

    with ThreadPoolExecutor(a.workers) as ex:
        return list(ex.map(one, [(row, g) for row in rows for g in range(a.group)]))


def advantages(rewards):
    """Group-relative: (r - mean) / std. A group where every attempt scored the same carries no signal -> all zero."""
    mean = sum(rewards) / len(rewards)
    std = (sum((r - mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
    return [0.0] * len(rewards) if std < 1e-6 else [(r - mean) / std for r in rewards]


def token_logprobs(model, ids, mask, torch):
    """log p of each trainable token. Logits only at those positions, in chunks: full-vocab logits for an 8k-token
    transcript would not fit next to vLLM."""
    base = model.get_base_model()
    h = base.get_decoder()(input_ids=ids[None], use_cache=False).last_hidden_state[0]
    pos = mask[1:].nonzero().squeeze(1)  # hidden state t predicts token t+1
    tgt = ids[1:][pos]
    head = base.get_output_embeddings()
    return torch.cat([torch.log_softmax(head(h[pos[c:c + 1024]]).float(), -1).gather(1, tgt[c:c + 1024, None]).squeeze(1)
                      for c in range(0, len(pos), 1024)])


def pull(a):
    """Restore checkpoint/, log.jsonl and samples/ from the hub repo, if it has a run."""
    from huggingface_hub import create_repo, snapshot_download
    create_repo(a.hub_repo, private=True, exist_ok=True)
    snapshot_download(a.hub_repo, local_dir=a.out)
    print("pulled", a.hub_repo, open(os.path.join(a.out, "checkpoint", "state.json")).read()
          if os.path.exists(os.path.join(a.out, "checkpoint", "state.json")) else "(empty: fresh run)", flush=True)


def push(a, step):
    """Upload the run dir; a failed upload is retried at the next save instead of stopping training."""
    from huggingface_hub import upload_folder
    try:
        upload_folder(repo_id=a.hub_repo, folder_path=a.out, commit_message=f"step {step}",
                      ignore_patterns=["vllm.log", "stdout.log"])
    except Exception as e:
        print(f"hub push failed at step {step}: {e!r}"[:300], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="verified task rows (gen2.py output)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="mohammed-usmani/clipwright-4b", help="starting LoRA (the SFT model): HF id or dir")
    ap.add_argument("--out", required=True, help="run dir: log.jsonl, samples/, checkpoint (resumes from it)")
    ap.add_argument("--work", default="/tmp/grpo-jobs", help="scratch dir for rollout job folders")
    ap.add_argument("--url", help="use an already running OpenAI-compatible server (no vLLM, no adapter sync)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tool-parser", default="qwen3_coder")
    ap.add_argument("--vllm-mem", type=float, default=0.45, help="GPU share for vLLM; training uses the rest")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--tasks", type=int, default=8, help="tasks per step")
    ap.add_argument("--group", type=int, default=8, help="attempts per task (G)")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cmd-timeout", type=int, default=60,
                    help="per-command limit in rollouts; test clips are under ~75 s, so longer means a hung command")
    ap.add_argument("--hub-repo", help="private HF repo for checkpoints: pushed every save, pulled on start (resume on a new VM)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sandbox.TIMEOUT = a.cmd_timeout
    if a.hub_repo and not os.path.exists(os.path.join(a.out, "checkpoint", "state.json")):
        pull(a)

    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ckpt = os.path.join(a.out, "checkpoint")
    state = json.load(open(os.path.join(ckpt, "state.json"))) if os.path.exists(os.path.join(ckpt, "state.json")) else {"step": 0, "pos": 0}
    start = ckpt if state["step"] else a.adapter
    if not os.path.isdir(start):  # vLLM needs a local dir
        from huggingface_hub import snapshot_download
        start = snapshot_download(start, allow_patterns=["adapter_*"])
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map=dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    cfg = PeftConfig.from_pretrained(start)
    cfg.lora_dropout = 0.0  # dropout would make the trained policy differ from the one that sampled
    model = PeftModel.from_pretrained(model, start, config=cfg, is_trainable=True)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    if state["step"]:
        opt.load_state_dict(torch.load(os.path.join(ckpt, "optim.pt"), map_location=dev))

    rows = [json.loads(l) for l in open(a.pool)]
    random.Random(a.seed).shuffle(rows)
    adapters = os.path.join(a.work + "-adapters")
    name = f"p{state['step']}"
    vllm = None
    if not a.url:
        vllm = start_vllm(a, start, name)
        a.url = f"http://127.0.0.1:{a.port}/v1/chat/completions"
        eval_agent.EXTRA["model"] = name
    eval_agent.EXTRA["top_p"] = 1.0  # sample from the policy itself
    log = open(os.path.join(a.out, "log.jsonl"), "a", buffering=1)

    try:
        for step in range(state["step"] + 1, a.steps + 1):
            batch = [rows[(state["pos"] + i) % len(rows)] for i in range(a.tasks)]
            state["pos"] = (state["pos"] + a.tasks) % len(rows)
            t0 = time.time()
            outs = rollouts(batch, a)
            t_roll = time.time() - t0

            train, rewards, signal = [], [], 0
            for i, row in enumerate(batch):
                grp = outs[i * a.group:(i + 1) * a.group]
                rs = [reward(res, row) for _, _, res, _, _ in grp]
                rewards += rs
                adv = advantages(rs)
                signal += any(adv)
                for (_, _, _, msgs, _), A in zip(grp, adv):
                    if A and msgs:
                        enc = encode(tok, msgs, row["tools"])
                        if enc and len(enc[0]) <= a.max_len and any(enc[1]):
                            train.append((enc, A))

            loss_sum, n_tok = 0.0, sum(sum(m) for (_, m), _ in train)
            t1 = time.time()
            if train:
                opt.zero_grad()
                for (ids, mask), A in train:
                    ids_t = torch.tensor(ids, device=dev)
                    lp = token_logprobs(model, ids_t, torch.tensor(mask, device=dev), torch)
                    loss = -(A * lp.sum()) / n_tok
                    loss.backward()
                    loss_sum += loss.item()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            t_train = time.time() - t1

            if vllm:  # hot-swap the new adapter into vLLM
                prev, name = name, f"p{step}"
                path = os.path.join(adapters, name)
                model.save_pretrained(path)
                base = a.url.rsplit("/v1/", 1)[0]
                http(f"{base}/v1/load_lora_adapter", {"lora_name": name, "lora_path": path}, timeout=600)
                eval_agent.EXTRA["model"] = name
                http(f"{base}/v1/unload_lora_adapter", {"lora_name": prev})
                shutil.rmtree(os.path.join(adapters, prev), ignore_errors=True)

            passed = [o for o in outs if o[2]["pass"]]
            rec = {"step": step, "reward": round(sum(rewards) / len(rewards), 4),
                   "pass_rate": round(len(passed) / len(outs), 4), "groups_with_signal": f"{signal}/{len(batch)}",
                   "trained_seqs": len(train), "tokens": n_tok, "loss": round(loss_sum, 5),
                   "statuses": dict(sorted({s: sum(o[2]["status"] == s for o in outs)
                                            for s in {o[2]["status"] for o in outs}}.items())),
                   "t_rollout": round(t_roll), "t_train": round(t_train)}
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            # passing transcripts, to read for reward hacking (a pass the spec allows but a person wouldn't accept)
            sdir = os.path.join(a.out, "samples", f"step{step:04d}")
            os.makedirs(sdir, exist_ok=True)
            for row, g, res, msgs, _ in passed[:4]:
                json.dump({"id": row["id"], "category": row["category"], "result": res, "messages": msgs},
                          open(os.path.join(sdir, f"{row['id']}_{g}.json"), "w"), ensure_ascii=False, indent=1)

            state["step"] = step
            if step % a.save_every == 0 or step == a.steps:
                model.save_pretrained(ckpt)
                torch.save(opt.state_dict(), os.path.join(ckpt, "optim.pt"))
                json.dump(state, open(os.path.join(ckpt, "state.json"), "w"))
                if a.hub_repo:
                    push(a, step)
    finally:
        if vllm:
            vllm.terminate()


if __name__ == "__main__":
    main()
