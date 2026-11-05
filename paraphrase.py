"""Batch 5: reword user requests so the model sees many phrasings of the same job.

The teacher rewrites 20 requests per call (thinking off). A rewrite is kept only if every file name and number of the
original survives. Each row then keeps its original request or takes the rewrite at random (no duplicate rows).

    set -a && . ./.env && set +a
    python3 paraphrase.py data/gen2/*.jsonl --out data/gen2/paraphrased.jsonl --priority 100
"""
import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import teacher_api

PROMPT = """Rewrite each user request below the way a different real person might type it to a media assistant.
Vary it: casual or terse or wordy, typos sometimes, different word order, maybe some context about why they need it.
Keep the meaning exactly. Copy every file name, number, time, size and unit exactly as written. Don't add requirements.
Return only a JSON array of strings, one rewrite per request, same order.

Requests:
"""
TOKENS = re.compile(r"[\w.\-]+\.[A-Za-z0-9]{2,4}\b|\d+(?:\.\d+)?")


def keeps_facts(orig, new):
    return all(t in new for t in TOKENS.findall(orig)) and 0.4 < len(new) / max(1, len(orig)) < 3


def rewrite(batch, a):
    msgs = [{"role": "user", "content": PROMPT + "\n".join(f"{i + 1}. {r}" for i, r in enumerate(batch))}]
    try:
        m, _ = teacher_api.chat(a.key, a.model, msgs, 0.9, a.base_url)
        txt = (m.get("content") or "").strip()
        out = json.loads(re.sub(r",\s*\]", "]", txt[txt.index("["): txt.rindex("]") + 1]))
        return out if isinstance(out, list) and len(out) == len(batch) else [None] * len(batch)
    except Exception as e:
        print("batch failed:", repr(e)[:150], file=sys.stderr, flush=True)
        return [None] * len(batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-original", type=float, default=0.5, help="share of rows that keep their original wording")
    ap.add_argument("--priority", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    a.key, a.model, a.base_url = os.environ["API_KEY"], os.environ["MODEL_NAME"], os.environ["BASE_URL"]
    teacher_api.EXTRA.update(priority=a.priority, chat_template_kwargs={"enable_thinking": False})
    teacher_api.MAX_TOKENS = 2048
    teacher_api.WITH_TOOLS = False
    rows = [json.loads(l) for f in a.files for l in open(f)]
    reqs = []
    for r in rows:
        u = next(m for m in r["messages"] if m["role"] == "user")["content"]
        reqs.append(u.split("\n\nFiles:")[0])
    batches = [list(range(i, min(i + 20, len(rows)))) for i in range(0, len(rows), 20)]
    print(f"{len(rows)} rows, {len(batches)} calls", flush=True)
    new = [None] * len(rows)
    with ThreadPoolExecutor(a.workers) as ex:
        for done, (idx, outs) in enumerate(zip(batches, ex.map(lambda b: rewrite([reqs[i] for i in b], a), batches)), 1):
            for i, t in zip(idx, outs):
                if isinstance(t, str) and keeps_facts(reqs[i], t):
                    new[i] = t.strip()
            if done % 20 == 0:
                print(f"  {done}/{len(batches)} calls, {sum(x is not None for x in new)} usable rewrites", flush=True)
    rnd = random.Random(7)
    used = 0
    with open(a.out, "w") as f:
        for r, q, n in zip(rows, reqs, new):
            if n and rnd.random() >= a.keep_original:
                for m in r["messages"]:
                    if m["role"] == "user":
                        m["content"] = n + m["content"][len(q):]
                        break
                r["paraphrased"] = True
                used += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"usable rewrites {sum(x is not None for x in new)}/{len(rows)}; {used} rows reworded -> {a.out}")


if __name__ == "__main__":
    assert keeps_facts("cut 2s to 7.5s of talk.mp4", "pls trim talk.mp4 from 2s to 7.5s")
    assert not keeps_facts("cut 2s to 7.5s of talk.mp4", "trim talk.mp4 from 2 to 8 seconds")
    main()
