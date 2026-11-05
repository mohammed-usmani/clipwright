"""Sanity check for eval_agent.score(): every test row's own reference solution must pass it.

    python3 eval_check_scorer.py data/sft/test_single.jsonl
"""
import json, os, shutil, sys, tempfile
from concurrent.futures import ProcessPoolExecutor

import sandbox
from eval_agent import score
from fixtures import FIX, clone


def ref_passes(row):
    wd = tempfile.mkdtemp(prefix="chk_", dir=os.environ.get("JOB_TMP"))
    try:
        os.makedirs(os.path.join(wd, "out"))
        for name, fx in row["verify"]["files"].items():
            os.makedirs(os.path.dirname(os.path.join(wd, "in", name)), exist_ok=True)
            clone(os.path.join(FIX, fx), os.path.join(wd, "in", name))
        results = []
        for m in row["messages"]:
            for tc in m.get("tool_calls") or []:
                if m.get("weight") == 0:  # the deliberately broken first attempt in recovery rows
                    continue
                a = tc["function"]["arguments"]
                a = json.loads(a) if isinstance(a, str) else a
                results.append(sandbox.run(wd, a["cmd"], a.get("foreach")))
        ok, errs = score(row, wd, "done", results, row["messages"])
        return row["id"], row.get("task", row["category"]), ok, errs
    finally:
        shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    with ProcessPoolExecutor(6) as ex:
        out = list(ex.map(ref_passes, rows))
    bad = [o for o in out if not o[2]]
    for b in bad[:20]:
        print("FAIL", b)
    print(f"{len(out) - len(bad)}/{len(out)} reference solutions pass the scorer")
