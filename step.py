"""The teacher's only tool. Every call is logged to <job>/trajectory.jsonl, which becomes training data.

Run a command (reasoning + command read from stdin):
    python3 step.py <jobdir> run <<'STEP'
    THINK: why this command, grounded in the file properties
    CMD: ffmpeg -i in/clip.mov ... out/result.mp4
    FOREACH: in/*.mov            (optional)
    STEP

Finish:
    python3 step.py <jobdir> final <<'STEP'
    short summary for the user
    STEP
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sandbox  # noqa: E402

MAX_STEPS = 15


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in ("run", "final"):
        sys.exit(__doc__)
    wd, mode = os.path.abspath(sys.argv[1]), sys.argv[2]
    log = os.path.join(wd, "trajectory.jsonl")
    if not os.path.isfile(os.path.join(wd, "request.txt")):
        sys.exit("not a job dir")
    if os.path.exists(os.path.join(wd, "final.txt")):
        sys.exit("job already finalized")
    text = sys.stdin.read().strip()
    if mode == "final":
        open(os.path.join(wd, "final.txt"), "w").write(text)
        print("finalized")
        return
    steps = sum(1 for _ in open(log)) if os.path.exists(log) else 0
    if steps >= MAX_STEPS:
        sys.exit(f"step limit ({MAX_STEPS}) reached; finalize the job")
    m = re.match(r"THINK:(.*?)\nCMD:(.*?)(?:\nFOREACH:(.*))?$", text, re.S)
    if not m:
        sys.exit("stdin must be 'THINK: ...' then 'CMD: ...' (and optionally 'FOREACH: glob') lines")
    think, cmd, foreach = m.group(1).strip(), m.group(2).strip(), (m.group(3) or "").strip() or None
    if "\n" in cmd:
        sys.exit("CMD must be a single line (chain steps with &&)")
    res = sandbox.run(wd, cmd, foreach)
    with open(log, "a") as f:
        f.write(json.dumps({"think": think, "cmd": cmd, "foreach": foreach, "result": res}, ensure_ascii=False) + "\n")
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
