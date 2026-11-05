"""Clipwright web app: upload files, describe the job, the model runs it in a sandbox, download the results.

    llama-server -m models/clipwright-4b-Q8_0.gguf --jinja -c 32768 --port 8080     # the model
    .venv-app/bin/uvicorn app.server:app --port 8000                                # this app
"""
import asyncio
import json
import os
import re
import shutil
import sys
import threading
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
import zipfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import sandbox  # noqa: E402
from build import SYSTEM  # noqa: E402
from clean_ebur128 import clean_result  # noqa: E402
from fixtures import clone  # noqa: E402

LLM = os.environ.get("CLIPWRIGHT_LLM", "http://127.0.0.1:8080/v1/chat/completions")
JOBS = os.environ.get("CLIPWRIGHT_JOBS", os.path.join(ROOT, "app", "jobs"))
MAX_STEPS = 15  # same budget as training and eval
app = FastAPI()
jobs = {}  # id -> Job; history on disk is reloaded lazily
lock = threading.Lock()


class Job:
    def __init__(self, jid, title):
        self.id, self.title, self.created = jid, title, time.time()
        self.wd = os.path.join(JOBS, jid)
        self.events, self.msgs, self.state = [], [], "running"
        self.use = None  # input files the next turn works on (names in in/); None = all of them
        self.clones = {}  # out/ result -> its copy in in/, so reusing a result keeps one stable name
        self.pending = []  # user texts since the last turn that produced files (questions, clarifications)
        self.stop = False  # set by the Stop button; checked between model calls and commands
        self.cond = threading.Condition()

    def emit(self, kind, **data):
        with self.cond:
            self.events.append({"kind": kind, "t": time.time(), **data})
            self.cond.notify_all()
        self.save()

    def save(self):
        meta = {"id": self.id, "title": self.title, "created": self.created, "state": self.state,
                "events": self.events, "msgs": self.msgs, "pending": self.pending, "clones": self.clones}
        tmp = os.path.join(self.wd, "job.json.tmp")
        json.dump(meta, open(tmp, "w"), ensure_ascii=False)
        os.replace(tmp, os.path.join(self.wd, "job.json"))

    @classmethod
    def load(cls, jid):
        meta = json.load(open(os.path.join(JOBS, jid, "job.json")))
        j = cls(jid, meta["title"])
        j.created, j.state, j.events, j.msgs = meta["created"], meta["state"], meta["events"], meta["msgs"]
        j.clones = meta.get("clones", {})
        j.pending = [p if isinstance(p, list) else ["User", p] for p in meta.get("pending", [])]
        if j.state == "running":  # server restarted mid-job
            j.state = "stopped"
        return j


def get_job(jid):
    if not re.fullmatch(r"[0-9a-f]{12}", jid):
        raise HTTPException(404)
    with lock:
        if jid not in jobs:
            try:
                jobs[jid] = Job.load(jid)
            except FileNotFoundError:
                raise HTTPException(404)
        return jobs[jid]


def chat(msgs):
    body = {"messages": msgs, "tools": [sandbox.TOOL_SCHEMA], "temperature": 0.0, "max_tokens": 1024,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(LLM, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["choices"][0]


def snapshot(job):
    return {p: os.path.getmtime(p) for p in sandbox._files(job.wd, "out")}


def out_files(job, before):
    """Files this turn created or changed, so earlier results don't repeat under every reply."""
    out = os.path.join(job.wd, "out")
    return [{"name": os.path.relpath(p, out), "size": os.path.getsize(p), "desc": sandbox.summarize(p)}
            for p in sorted(sandbox._files(job.wd, "out")) if before.get(p) != os.path.getmtime(p)]


def turn_messages(job):
    """The model was trained on one request per conversation, and gets lost in chat-style follow-ups (it even
    repeats its own earlier wording). So each turn is sent as a fresh request: the user's messages since the last
    turn that produced files, listing only the files this turn works on (the UI defaults to the latest result)."""
    ind = os.path.join(job.wd, "in")
    names = job.use if job.use is not None else [os.path.relpath(p, ind) for p in sandbox._files(job.wd, "in")]
    man = "\n".join(f"- in/{n} — {sandbox.summarize(os.path.join(ind, n))}" for n in names)
    (_, latest), earlier = job.pending[-1], job.pending[:-1]
    ctx = "\n\nEarlier in this conversation:\n" + "\n".join(f"{r}: {t}" for r, t in earlier) if earlier else ""
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{latest}{ctx}\n\nFiles:\n{man}"}]


class Stopped(Exception):
    pass


def agent(job):
    """The same loop the model was evaluated with: one command per turn, stop on a plain reply or a 3x repeat."""
    history = []
    before = snapshot(job)
    job.msgs = turn_messages(job)
    job.stop = False
    try:
        for step in range(MAX_STEPS + 1):
            if job.stop:
                raise Stopped
            job.emit("thinking")
            ch = chat(job.msgs)
            if job.stop:  # ponytail: a running command or model call finishes first; stop takes effect after it
                raise Stopped
            m = ch["message"]
            text, calls = (m.get("content") or "").strip(), m.get("tool_calls") or []
            if not calls:
                job.msgs.append({"role": "assistant", "content": text})
                files = out_files(job, before)
                if files:  # finished deliverables close the thread; a question, offer or refusal keeps it open
                    job.pending = []
                else:
                    job.pending.append(["Assistant", text])
                job.state = "done"
                job.emit("reply", text=text, files=files)
                return
            if step == MAX_STEPS:
                break
            call = calls[0]
            job.msgs.append({"role": "assistant", "content": text, "tool_calls": [call]})
            try:
                args = json.loads(call["function"]["arguments"])
                cmd, foreach = args["cmd"], args.get("foreach") or None
            except (json.JSONDecodeError, KeyError, TypeError):
                cmd, foreach = None, None
            job.emit("step", think=text, cmd=cmd, foreach=foreach)
            res = sandbox.run(job.wd, cmd, foreach) if cmd else {"exit_code": 1, "error": "tool call needs a 'cmd' string"}
            job.msgs.append({"role": "tool", "tool_call_id": call.get("id", f"call_{step}"), "name": "run",
                             "content": clean_result(json.dumps(res, ensure_ascii=False))})
            job.emit("result", ok=res.get("exit_code") == 0, error=res.get("error"),
                     outputs=res.get("outputs"), stdout=(res.get("stdout") or "")[:1500])
            history.append((cmd, foreach, re.sub(r"0x[0-9a-f]+", "", json.dumps(res, sort_keys=True))))
            if len(history) >= 3 and len(set(history[-3:])) == 1:
                job.state = "failed"
                job.emit("error", text="The model repeated the same failing command three times and was stopped.",
                         reason="loop", files=out_files(job, before))
                return
        job.state = "failed"
        job.emit("error", text=f"The job hit the {MAX_STEPS}-step limit before finishing.", reason="steps",
                 files=out_files(job, before))
        return
    except Stopped:
        job.state = "failed"
        job.emit("error", text="Stopped.", reason="stopped", files=out_files(job, before))
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        job.state = "failed"
        job.emit("error", text="Can't reach the model. Start it with app/run.sh and try again.",
                 reason="offline", files=out_files(job, before))
    except Exception as e:  # noqa: BLE001 - surface anything to the UI rather than dying silently
        job.state = "failed"
        job.emit("error", text=str(e), files=out_files(job, before))
    finally:
        if job.state == "failed":  # a failed request doesn't carry into the next one; "Try again" resends its text
            job.pending = []
            job.save()


def start(job):
    job.state = "running"
    threading.Thread(target=agent, args=(job,), daemon=True).start()


def unique(folder, name, taken=()):
    base, ext = os.path.splitext(name)
    n = 1
    while name.lower() in taken or os.path.exists(os.path.join(folder, name)):
        n += 1
        name = f"{base}_{n}{ext}"
    return name


def as_input(job, where, name):
    """Map a file the user picked (an original in in/, or a result in out/) to its name in in/. Results are
    copied into in/ so the model reads from in/ and writes to out/ as it was trained, even when it reuses the name."""
    root = os.path.realpath(os.path.join(job.wd, where)) if where in ("in", "out") else None
    path = os.path.realpath(os.path.join(root, name)) if root else ""
    if not root or not path.startswith(root + os.sep) or not os.path.isfile(path):
        raise HTTPException(400, f"Can't find {name}.")
    ind = os.path.join(job.wd, "in")
    if where == "in":
        return os.path.relpath(path, os.path.realpath(ind))
    key = os.path.relpath(path, root)
    dst_name = job.clones.get(key) or unique(ind, safe_name(key))
    job.clones[key] = dst_name
    dst = os.path.join(ind, dst_name)
    if os.path.exists(dst):  # refresh: a later turn may have rewritten this result
        os.remove(dst)
    clone(path, dst)
    return dst_name


def input_info(job, names):
    return [{"name": n, "desc": sandbox.summarize(os.path.join(job.wd, "in", n)),
             "size": os.path.getsize(os.path.join(job.wd, "in", n))} for n in names]


def save_uploads(job, files):
    names = []
    for f in files:
        name = unique(os.path.join(job.wd, "in"), safe_name(f.filename), {n.lower() for n in names})
        with open(os.path.join(job.wd, "in", name), "wb") as dst:
            shutil.copyfileobj(f.file, dst)
        names.append(name)
    return names


def safe_name(name):
    """Shell-safe names: the model often forgets to quote paths with spaces or brackets, so don't give it any."""
    base, ext = os.path.splitext(os.path.basename(name or ""))
    base = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode()  # résumé -> resume
    base = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9.\-]+", "_", base)).strip("._-")
    return (base or "file") + re.sub(r"[^A-Za-z0-9.]", "", ext).lower()


@app.post("/api/jobs")
async def create_job(prompt: str = Form(...), files: list[UploadFile] = File(...)):
    prompt = prompt.strip()
    if not prompt or not files:
        raise HTTPException(400, "Add at least one file and describe the job.")
    job = Job(uuid.uuid4().hex[:12], prompt[:80])
    os.makedirs(os.path.join(job.wd, "in"))
    os.makedirs(os.path.join(job.wd, "out"))
    names = save_uploads(job, files)
    job.pending = [["User", prompt]]
    job.emit("request", text=prompt, inputs=await asyncio.to_thread(input_info, job, names))
    with lock:
        jobs[job.id] = job
    start(job)
    return {"id": job.id}


@app.post("/api/jobs/{jid}/reply")
async def reply(jid: str, text: str = Form(...), use: str = Form("[]"),
                files: list[UploadFile] | None = File(None)):
    """`use` is a JSON list of {"where": "in"|"out", "name"} picked in the composer; `files` are new uploads."""
    job = get_job(jid)
    if job.state == "running":
        raise HTTPException(409, "The job is still running.")
    if not text.strip():
        raise HTTPException(400, "Write a reply first.")
    try:
        picked = [(u["where"], u["name"]) for u in json.loads(use)]
    except (json.JSONDecodeError, KeyError, TypeError):
        raise HTTPException(400, "Bad file list.")
    names = list(dict.fromkeys([as_input(job, w, n) for w, n in picked] + save_uploads(job, files or [])))
    if not names:
        raise HTTPException(400, "Pick a file to work on.")
    job.use = names
    job.pending.append(["User", text.strip()])
    job.emit("request", text=text.strip(), inputs=await asyncio.to_thread(input_info, job, names))
    start(job)
    return {"ok": True}


@app.post("/api/jobs/{jid}/stop")
def stop(jid: str):
    job = get_job(jid)
    job.stop = True
    return {"ok": True}


@app.get("/api/jobs")
def list_jobs():
    os.makedirs(JOBS, exist_ok=True)
    out = []
    for jid in os.listdir(JOBS):
        try:
            j = get_job(jid)
        except HTTPException:
            continue
        first = next((e.get("inputs") or [] for e in j.events if e["kind"] == "request"), [])
        out.append({"id": j.id, "title": j.title, "created": j.created, "state": j.state,
                    "file": first[0]["name"] if first else ""})
    return sorted(out, key=lambda j: -j["created"])[:50]


@app.get("/api/jobs/{jid}/events")
async def events(jid: str):
    job = get_job(jid)

    def wait(i):
        with job.cond:
            job.cond.wait_for(lambda: len(job.events) > i, timeout=15)
            return job.events[i:]

    async def stream():
        i = 0
        while True:
            new = await asyncio.to_thread(wait, i)
            for e in new:
                yield f"data: {json.dumps(e, ensure_ascii=False)}\n\n"
            i += len(new)
            if not new:
                yield ": keepalive\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


@app.get("/api/jobs/{jid}/file/{where}/{name:path}")
def file(jid: str, where: str, name: str, download: int = 0):
    job = get_job(jid)
    if where not in ("in", "out"):
        raise HTTPException(404)
    base = os.path.realpath(os.path.join(job.wd, where))
    path = os.path.realpath(os.path.join(base, name))
    if not path.startswith(base + os.sep) or not os.path.isfile(path):
        raise HTTPException(404)
    return FileResponse(path, filename=os.path.basename(path) if download else None,
                        content_disposition_type="attachment" if download else "inline")


@app.get("/api/jobs/{jid}/zip")
def zip_outputs(jid: str):
    job = get_job(jid)
    files = sandbox._files(job.wd, "out")
    if not files:
        raise HTTPException(404, "No results yet.")
    path = os.path.join(job.wd, "results.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:  # media is already compressed
        for p in files:
            z.write(p, os.path.relpath(p, os.path.join(job.wd, "out")))
    return FileResponse(path, filename=f"clipwright-{jid}.zip")


@app.get("/api/health")
def health():
    try:
        with urllib.request.urlopen(LLM.replace("/v1/chat/completions", "/health"), timeout=2) as r:
            return {"model": r.status == 200}
    except Exception:  # noqa: BLE001
        return {"model": False}


app.mount("/", StaticFiles(directory=os.path.join(ROOT, "app", "static"), html=True), name="static")
