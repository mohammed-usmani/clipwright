"""Executes the model's `run` tool calls inside a job dir (in/ = inputs, out/ = results).

This layer enforces the command allowlist and path rules. Real isolation (no network,
CPU/mem/disk limits, non-root, read-only in/) is the container's job, not this file's.
Shared by dataset generation, the GRPO reward, and the production runtime.
"""
import glob
import json
import os
import re
import shlex
import signal
import socket
import subprocess

RUNNER = os.environ.get("CLIPWRIGHT_RUNNER")  # unix socket of the isolated runner container (compose.yaml); unset = run here
ALLOWED = {"ffmpeg", "ffprobe", "magick", "sox", "soxi", "exiftool"}
TIMEOUT = 300
MEDIA_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".gif", ".mp3", ".wav", ".m4a", ".m4r", ".flac",
             ".opus", ".ogg", ".aac", ".ts", ".jpg", ".jpeg", ".png", ".webp", ".heic", ".ico", ".tif", ".tiff"}

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run",
        "description": "Run one media CLI command (ffmpeg, ffprobe, magick, sox, soxi, exiftool) in the job "
                       "directory. Inputs are in in/ (read-only), write results to out/. Chain steps with &&. "
                       "Set foreach to a glob to run the command once per matching file, with {in} = the file "
                       "path and {stem} = its name without extension.",
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {"type": "string", "description": "The command line."},
                "foreach": {"type": "string", "description": "Optional glob, e.g. in/*.mov"},
            },
            "required": ["cmd"],
        },
    },
}


class Rejected(Exception):
    pass


def split_segments(cmd):
    """Split on && outside quotes; reject every other shell operator."""
    segs, cur, q, i = [], "", None, 0
    while i < len(cmd):
        ch = cmd[i]
        if q:
            if q == '"' and ch in "$`":
                raise Rejected(f"{ch!r} inside double quotes is expanded by the shell; use single quotes or remove it")
            cur += ch
            if ch == "\\" and q == '"' and i + 1 < len(cmd):
                cur += cmd[i + 1]
                i += 1
            elif ch == q:
                q = None
        elif ch == "\\" and i + 1 < len(cmd):
            cur += ch + cmd[i + 1]
            i += 1
        elif ch in "'\"":
            q = ch
            cur += ch
        elif cmd.startswith("&&", i):
            segs.append(cur)
            cur = ""
            i += 1
        elif ch in ";|&<>`$\n":
            raise Rejected(f"shell operator {ch!r} outside quotes; only && is allowed, quote arguments containing it")
        else:
            cur += ch
        i += 1
    if q:
        raise Rejected("unbalanced quotes")
    segs.append(cur)
    out = []
    for s in segs:
        argv = shlex.split(s)
        if not argv:
            raise Rejected("empty command")
        if argv[0] not in ALLOWED:
            raise Rejected(f"'{argv[0]}' is not allowed; use one of: {', '.join(sorted(ALLOWED))}")
        for a in argv[1:]:
            if a.startswith("/") and a not in ("/dev/null",):
                raise Rejected(f"absolute path {a!r} not allowed; read from in/ and write to out/")
            if "../" in a or a.startswith("~"):
                raise Rejected(f"path {a!r} escapes the job directory; write to out/")
        out.append(argv)
    return out


NOISE = re.compile(r"^\s*(frame|size)=.*time=|^Press \[q\]|^\s*\[q\] command received|^ffmpeg version |"
                   r"^\s+(built with|configuration:|lib(avutil|avcodec|avformat|avdevice|avfilter|swscale|swresample|postproc)\s)")
# lines an analysis filter/tool reports (loudnorm JSON, ebur128, volumedetect, silencedetect, cropdetect, sox stats...)
ANALYSIS = re.compile(r"^\s*\[Parsed_(loudnorm|ebur128|volumedetect|silencedetect|cropdetect|astats|aspectralstats|"
                      r"blackdetect|freezedetect|silencedetect|idet|scdet|signalstats|showinfo|ashowinfo)|^\s*[{}]\s*$|^\s*\"[a-z_]+\"\s*:|LUFS|LRA|True peak|mean_volume|max_volume|"
                      r"silence_(start|end)|crop=|^(DC offset|Min level|Max level|Pk lev|RMS|Bit-depth|Length s|Samples)")


def _tail(text, n=15):
    """Last n meaningful lines: ffmpeg progress/status noise is dropped so analysis reports aren't pushed out."""
    lines = [l for l in text.replace("\r", "\n").split("\n") if l.strip() and not NOISE.match(l)]
    return "\n".join(lines[-n:])


def _files(wd, sub):
    return sorted(p for p in glob.glob(os.path.join(wd, sub, "**", "*"), recursive=True) if os.path.isfile(p))


def _work_files(wd):
    """Every file the job can produce: out/ plus intermediates in the root (in/ is read-only input)."""
    inp = os.path.join(wd, "in") + os.sep
    return [p for p in sorted(glob.glob(os.path.join(wd, "**", "*"), recursive=True))
            if os.path.isfile(p) and not p.startswith(inp) and os.path.basename(p) not in HIDDEN]


HIDDEN = {"request.txt", "trajectory.jsonl", "final.txt", "status.txt"}


def _exec(argv, cwd=None, timeout=TIMEOUT):
    """Run argv, in the runner container when CLIPWRIGHT_RUNNER is set. Returns (returncode, stdout, stderr);
    returncode is None on timeout."""
    if not RUNNER:
        return _local(argv, cwd, timeout)
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(timeout + 30)
        s.connect(RUNNER)
        s.sendall(json.dumps({"argv": argv, "cwd": cwd, "timeout": timeout}).encode() + b"\n")
        r = json.loads(s.makefile().readline())
    return r["rc"], r["out"], r["err"]


def _local(argv, cwd, timeout):
    p = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True, errors="replace", start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)  # kills ffmpeg too, not just bash
        p.communicate()
        return None, "", ""
    return p.returncode, out, err


def probe(path):
    rc, out, _ = _exec(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", path], timeout=60)
    return json.loads(out) if rc == 0 and out.strip() else None


def _size(n):
    return f"{n / 1e6:.1f} MB" if n >= 1e6 else f"{max(1, round(n / 1e3))} KB"


def summarize(path):
    """One-line description of a file, used for the input manifest and for tool results. Never raises."""
    try:
        return _summarize(path)
    except Exception:
        return f"UNREADABLE: could not be parsed, {_size(os.path.getsize(path))}"


def _summarize(path):
    ext = os.path.splitext(path)[1].lower()
    size = _size(os.path.getsize(path))
    if ext == ".pdf":
        data = open(path, "rb").read()
        if not data.startswith(b"%PDF-"):
            return f"UNREADABLE: not a valid PDF, {size}"
        return f"pdf document, {len(re.findall(rb'/Type\s*/Page[^s]', data))} page(s), {size}"
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".heic", ".ico", ".tif", ".tiff", ".svg"):
        rc, out, _ = _exec(["magick", "identify", "-format", "%m %w %h %A|", path], timeout=60)
        if rc == 0:
            first = out.split("|")[0].split()
            n = out.count("|")
            alpha = ", alpha" if first[3].lower() not in ("undefined", "false") else ""
            extra = f", {n} sizes" if n > 1 else ""
            return f"image {first[0]} {first[1]}x{first[2]}{alpha}{extra}, {size}"
        return f"UNREADABLE: not a valid {ext[1:]} image (corrupt or empty output), {size}"
    if ext in MEDIA_EXT:
        info = probe(path)
        if info:
            parts = []
            for s in info.get("streams", []):
                t = s.get("codec_type")
                if t == "video" and s.get("disposition", {}).get("attached_pic"):
                    parts.append(f"cover art {s['codec_name']} {s['width']}x{s['height']}")
                elif t == "video":
                    if not s.get("width") or not s.get("height"):
                        return f"UNREADABLE: video stream has no picture (corrupt or empty output), {size}"
                    num, den = (s.get("avg_frame_rate") or "0/1").split("/")
                    fps = float(num) / float(den) if float(den) else 0
                    fps_s = f" {fps:.3g}fps" if fps and ext != ".gif" else ""
                    parts.append(f"video {s['codec_name']} {s['width']}x{s['height']}{fps_s}")
                elif t == "audio":
                    ch = {1: "mono", 2: "stereo"}.get(s.get("channels"), f"{s.get('channels')}ch")
                    parts.append(f"audio {s['codec_name']} {ch} {int(s.get('sample_rate', 0)) / 1000:g}kHz")
                elif t == "subtitle":
                    parts.append(f"subtitles {s['codec_name']}")
            dur = float(info.get("format", {}).get("duration") or 0)
            if dur and ext not in (".jpg", ".jpeg", ".png", ".webp"):
                parts.append(f"{dur:.1f}s")
            return ", ".join(parts + [size])
    if ext in MEDIA_EXT:
        return f"UNREADABLE: not valid {ext[1:]} media (corrupt or empty output), {size}"
    if ext in (".srt", ".vtt", ".ass", ".txt", ".m3u8", ".csv", ".trf", ".prof"):
        with open(path, errors="replace") as f:
            n = sum(1 for _ in f)
        return f"text, {n} lines, {size}"
    return f"file, {size}"


def manifest(wd):
    return "\n".join(f"- {os.path.relpath(p, wd)} — {summarize(p)}" for p in _files(wd, "in"))


def _summarize_outputs(wd, paths, cap=6):
    rel = {os.path.relpath(p, wd): p for p in paths}
    keys = list(rel)
    shown = {k: summarize(rel[k]) for k in keys[:cap]}
    if len(keys) > cap:
        total = sum(os.path.getsize(rel[k]) for k in keys)
        shown[f"... and {len(keys) - cap} more"] = f"{len(keys)} files total, {_size(total)}"
    return shown


def run(wd, cmd, foreach=None, timeout=TIMEOUT):
    """Execute one tool call. Returns the dict the model sees as the tool result."""
    os.makedirs(os.path.join(wd, "out"), exist_ok=True)
    before = {p: os.path.getmtime(p) for p in _work_files(wd)}
    if foreach:
        if foreach.startswith("/") or ".." in foreach:
            return {"exit_code": 1, "error": "rejected: foreach glob must be relative, e.g. in/*.mov"}
        matches = sorted(glob.glob(os.path.join(wd, foreach)))
        if not matches:
            return {"exit_code": 1, "error": f"foreach glob {foreach!r} matched no files"}
        cmds = [cmd.replace("{in}", shlex.quote(os.path.relpath(m, wd)))
                   .replace("{stem}", shlex.quote(os.path.splitext(os.path.basename(m))[0])) for m in matches]
    else:
        cmds = [cmd]
    stdout, failures, stderr_all = "", [], ""
    for c in cmds:
        try:
            split_segments(c)  # validation only; bash runs it so globs and quoting behave like a terminal
        except Rejected as e:
            return {"exit_code": 1, "error": f"rejected: {e}"}
        rc, out, err = _exec(["bash", "-c", c], cwd=wd, timeout=timeout)
        if rc is None:
            failures.append(f"timed out after {timeout}s (killed); use a faster preset, a shorter clip, or a lower resolution")
            continue
        stdout += out
        stderr_all += err
        if rc != 0:
            failures.append(_tail(err) or f"exited {rc}")
    new = [p for p in _work_files(wd) if before.get(p) != os.path.getmtime(p)]
    res = {"exit_code": 1 if failures else 0}
    if foreach:
        res["ran"] = len(cmds)
        res["failed"] = len(failures)
    if failures:
        res["error"] = failures[0]
    out_s = stdout.strip()
    if out_s:
        res["stdout"] = out_s if len(out_s) <= STDOUT_CAP else (
            out_s[:STDOUT_CAP] + f"\n...[truncated {len(out_s) - STDOUT_CAP} chars; narrow the query, e.g. -show_entries]")
    if new:
        res["outputs"] = _summarize_outputs(wd, new)
    if not failures:
        log = _log(stderr_all, analysis_only=bool(new or out_s))
        if log:
            res["log"] = log
        elif not new and not out_s:
            note = "succeeded but printed nothing and wrote no files"
            if re.search(r"(^|\s)-(v|loglevel)\s+(error|quiet|fatal|panic)", " ".join(cmds)):
                note += "; -v/-loglevel error hides filter reports (loudnorm, ebur128, volumedetect, silencedetect) - drop it"
            res["note"] = note
    return res


STDOUT_CAP = 4000


def _log(stderr, analysis_only, n=60):
    """What a command reported on stderr. With outputs/stdout present only analysis lines are kept (filter reports,
    loudnorm JSON, sox stats); otherwise the last n meaningful lines, noise removed."""
    lines = [l for l in stderr.replace("\r", "\n").split("\n") if l.strip() and not NOISE.match(l)]
    if analysis_only:
        lines = [l for l in lines if ANALYSIS.search(l)]
    # ebur128 logs a line every 100 ms; drop them AFTER taking the tail so the window still ends at the Summary
    # (dropping first would refill it with the stream header). Same rule clean_ebur128.py applied to the dataset.
    return "\n".join(l for l in lines[-n:] if not EBUR128_FRAME.match(l))


EBUR128_FRAME = re.compile(r"^\[Parsed_ebur128_\d+ @ [^\]]+\] t: ")
