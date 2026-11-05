"""Drop ebur128 per-frame log lines (one every 100 ms) from tool results in exported datasets.

Same rule as sandbox._log: the log was already the last 60 lines, so removing the frame lines leaves what the
current sandbox returns. The model only ever used the Summary block, which is kept.

    python3 clean_ebur128.py data/sft/*.jsonl data/layer2/*.jsonl
"""
import json
import sys

from sandbox import EBUR128_FRAME


def clean_result(content):
    res = json.loads(content)
    log = res.get("log")
    if not log:
        return content
    kept = "\n".join(l for l in log.split("\n") if not EBUR128_FRAME.match(l))
    if kept == log:
        return content
    if kept:
        res["log"] = kept
    else:
        del res["log"]
    return json.dumps(res, ensure_ascii=False)


def clean_file(path):
    rows, changed, before, after = [], 0, 0, 0
    for line in open(path):
        r = json.loads(line)
        for m in r["messages"]:
            if m["role"] == "tool":
                new = clean_result(m["content"])
                before += len(m["content"])
                after += len(new)
                changed += new != m["content"]
                m["content"] = new
        rows.append(json.dumps(r, ensure_ascii=False))
    with open(path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"{path}: {changed} tool results cleaned, tool text {before:,} -> {after:,} chars")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        clean_file(p)
