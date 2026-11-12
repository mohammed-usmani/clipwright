"""Run grpo.py on a Colab A100 unattended: create the VM, set it up, launch, watch, recover, stop.

    caffeinate -i python3 grpo_colab.py --steps 100 --hub-repo Ray1239/clipwright-grpo-ckpt

Checkpoints go to a private HF repo every --save-every steps, so a VM that dies (or a Mac that sleeps or drops off the
network) costs at most that many steps: the next pass of the loop makes a new VM and grpo.py resumes from the repo.
The training runs detached on the VM (nohup), independent of this script; this script only supervises.
Progress: <out>/supervisor.log. Stops the VM when the run reaches --steps. The HF token is copied to the VM as a file
(never printed or put on a command line).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB_PY = os.path.join(os.path.dirname(os.path.realpath(shutil.which("colab"))), "python")
HF_PY = os.path.join(HERE, ".venv-merge", "bin", "python")  # has huggingface_hub
NAME = "grpo"

SETUP = r"""
set -x
cd /content
mkdir -p cw && cat part_* | tar xz -C cw 2>/dev/null
mkdir -p /root/.cache/huggingface && mv -f hf_token /root/.cache/huggingface/token
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar xj bin/micromamba
./bin/micromamba create -y -q -p /content/mm -c conda-forge ffmpeg imagemagick sox > mm.log 2>&1
rm -f /content/mm/bin/exiftool   # conda's exiftool picks up the system perl and breaks; use apt's
apt-get install -y -qq libimage-exiftool-perl > /dev/null 2>&1
pip install -q vllm peft flash-linear-attention > pip.log 2>&1
pip uninstall -y -q torchao      # Colab's old torchao makes peft refuse to load
export PATH=/content/mm/bin:$PATH
ffmpeg -version | head -1 && magick -version | head -1 && sox --version && exiftool -ver \
  && python -c "import vllm, peft" && touch /content/setup.done
"""

# Retries grpo.py (it resumes from its own checkpoint) so a crash doesn't need the Mac; writes run.status for us.
RUN = r"""
cd /content/cw
export PATH=/content/mm/bin:$PATH PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in 1 2 3; do
  echo "{\"state\": \"running\", \"attempt\": $attempt}" > /content/run.status
  python grpo.py {args} >> /content/run/stdout.log 2>&1 && { echo '{"state": "done"}' > /content/run.status; exit 0; }
  pkill -f "vllm serve"; sleep 30
done
echo '{"state": "failed"}' > /content/run.status
"""


def log(a, *msg):
    line = time.strftime("%m-%d %H:%M:%S ") + " ".join(str(m) for m in msg)
    print(line, flush=True)
    with open(os.path.join(a.out, "supervisor.log"), "a") as f:
        f.write(line + "\n")


def colab(*args, timeout=300):
    try:
        r = subprocess.run(["colab", *args], capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return 1, "timeout"


GONE = "GONE"


def vm_py(a, code, tries=4):
    """Run python on the VM: its stdout, GONE if the CLI has no such session, None if unreachable right now."""
    path = os.path.join(a.out, "_exec.py")
    open(path, "w").write(code)
    for _ in range(tries):
        rc, out = colab("exec", "-s", NAME, "-f", path, timeout=150)
        if "not found" in out:
            return GONE
        if rc == 0 and "Connection was lost" not in out and "Traceback (most recent" not in out:
            return "\n".join(l for l in out.splitlines() if not l.startswith("[colab]"))
        time.sleep(20)
    return None


def readopt():
    """The CLI forgets a live VM when a token refresh misses it in the list; put it back."""
    code = ("from colab_cli.common import State, _apply_proxy_info\nfrom colab_cli.state import SessionState\n"
            "st = State()\nfor x in st.client.list_assignments():\n"
            "    if x.endpoint.startswith('gpu-a100'):\n"
            f"        s = SessionState(name='{NAME}', token='', url='', endpoint=x.endpoint, variant='GPU', accelerator='A100')\n"
            "        _apply_proxy_info(s, x.runtime_proxy_info); st.store.add(s); print('readopted', x.endpoint)\n")
    r = subprocess.run([COLAB_PY, "-c", code], capture_output=True, text=True, timeout=120)
    return "readopted" in r.stdout


def hub_step(a):
    code = ("import json,sys\nfrom huggingface_hub import hf_hub_download\n"
            f"try: print(json.load(open(hf_hub_download('{a.hub_repo}', 'checkpoint/state.json')))['step'])\n"
            "except Exception: print(0)\n")
    r = subprocess.run([HF_PY, "-c", code], capture_output=True, text=True, timeout=300)
    try:
        return int(r.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return -1


def bundle(a):
    """Code + train fixtures (never fixtures/test) + the task pool, split for the CLI's upload size limit."""
    d = os.path.join(a.out, "bundle")
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    tgz = os.path.join(d, "bundle.tgz")
    with tarfile.open(tgz, "w:gz") as t:
        for f in glob.glob(os.path.join(HERE, "*.py")):
            t.add(f, os.path.basename(f))
        t.add(os.path.join(HERE, "fixtures"), "fixtures",
              filter=lambda ti: None if ti.name == "fixtures/test" or ti.name.startswith("fixtures/test/") else ti)
        t.add(a.pool, "data/rl/pool.jsonl")
    subprocess.run(["split", "-b", "40m", tgz, os.path.join(d, "part_")], check=True)
    os.remove(tgz)
    return sorted(glob.glob(os.path.join(d, "part_*")))


def provision(a, parts):
    log(a, "creating A100 VM")
    rc, out = colab("new", "-s", NAME, "--gpu", "A100", timeout=900)
    if rc != 0:
        log(a, "colab new failed:", out[-300:])
        return False
    files = [(p, "/content/" + os.path.basename(p)) for p in parts]
    files.append((os.path.expanduser("~/.cache/huggingface/token"), "/content/hf_token"))
    for local, remote in files:
        for _ in range(3):
            rc, out = colab("upload", "-s", NAME, local, remote, timeout=600)
            if "Uploaded" in out:
                break
        else:
            log(a, "upload failed:", os.path.basename(local))
            return False
    open(os.path.join(a.out, "setup.sh"), "w").write(SETUP)
    colab("upload", "-s", NAME, os.path.join(a.out, "setup.sh"), "/content/setup.sh")
    vm_py(a, "import subprocess\nsubprocess.Popen('nohup bash /content/setup.sh > /content/setup.log 2>&1 &', shell=True)\n")
    log(a, "setup started")
    for _ in range(60):  # ~30 min ceiling; normally ~10
        time.sleep(30)
        out = vm_py(a, "import os\nprint('SETUP_OK' if os.path.exists('/content/setup.done') else 'waiting')\n"
                       "print(open('/content/setup.log').read()[-400:] if not os.path.exists('/content/setup.done') else '')\n")
        if out and "SETUP_OK" in out:
            log(a, "setup done")
            return True
    log(a, "setup did not finish:", (out or "")[-400:])
    return False


def launch(a):
    args = (f"--pool data/rl/pool.jsonl --adapter {a.adapter} --out /content/run --work /content/jobs "
            f"--steps {a.steps} --tasks {a.tasks} --group {a.group} --save-every {a.save_every} "
            f"--workers 24 --hub-repo {a.hub_repo}")
    open(os.path.join(a.out, "run.sh"), "w").write(RUN.replace("{args}", args))
    colab("upload", "-s", NAME, os.path.join(a.out, "run.sh"), "/content/run.sh")
    vm_py(a, "import os, subprocess\nos.makedirs('/content/run', exist_ok=True)\n"
             "subprocess.Popen('nohup bash /content/run.sh > /content/run.log 2>&1 &', shell=True)\n")
    log(a, "training launched:", args)


STATUS = """import json, os
st = json.load(open('/content/run.status')) if os.path.exists('/content/run.status') else {'state': 'none'}
lines = open('/content/run/log.jsonl').read().splitlines() if os.path.exists('/content/run/log.jsonl') else []
st['last'] = json.loads(lines[-1]) if lines else None
print('STATUS ' + json.dumps(st))
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub-repo", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--adapter", default="Ray1239/clipwright-4b")
    ap.add_argument("--pool", default=os.path.join(HERE, "data", "rl", "pool.jsonl"))
    ap.add_argument("--out", default=os.path.join(HERE, "models", "grpo-colab"))
    ap.add_argument("--every", type=int, default=300, help="seconds between checks")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    parts = bundle(a)
    log(a, f"bundle ready: {len(parts)} parts")
    seen, unreachable = None, 0
    while True:
        out = vm_py(a, STATUS)
        if out is GONE and readopt():
            log(a, "re-adopted the VM after the CLI dropped it")
            out = vm_py(a, STATUS)
        if out is None:  # VM exists but didn't answer (busy CPUs, network blip): only give up on it after ~30 min
            unreachable += 1
            log(a, f"VM unreachable ({unreachable})")
            if unreachable < 6:
                time.sleep(a.every)
                continue
        unreachable = 0 if out not in (None, GONE) else unreachable
        if out in (None, GONE):
            unreachable = 0
            step = hub_step(a)
            if step >= a.steps:
                log(a, f"run complete on the hub (step {step})")
                break
            log(a, f"no reachable VM; hub checkpoint at step {step}; starting a new one")
            colab("stop", "-s", NAME)
            if provision(a, parts):
                launch(a)
            time.sleep(a.every)
            continue
        st = json.loads(out.split("STATUS ", 1)[1].splitlines()[0]) if "STATUS " in out else {"state": "?"}
        if st["state"] == "none":
            launch(a)
        elif st["state"] in ("done", "failed"):
            log(a, f"run {st['state']}; last step: {st.get('last')}")
            if st["state"] == "failed":
                tail = vm_py(a, "print(open('/content/run/stdout.log').read()[-3000:])")
                log(a, "stdout.log tail:\n" + (tail or "(unreadable)"))
            colab("stop", "-s", NAME)
            log(a, "VM stopped")
            break
        elif st.get("last") != seen:
            seen = st.get("last")
            log(a, "progress:", json.dumps(seen))
        time.sleep(a.every)


if __name__ == "__main__":
    sys.exit(main())
