"""Command runner for the isolated container in compose.yaml (no network, read-only root, no capabilities).

The app sends one JSON line per connection over a unix socket: {"argv", "cwd", "timeout"} -> {"rc", "out", "err"}.
Requests are re-checked here so a compromised app can't use the runner for anything but the media tools in /jobs.
"""
import json
import os
import socketserver

import sandbox

JOBS = os.path.realpath(os.environ.get("CLIPWRIGHT_JOBS", "/jobs"))


def inside(path):
    return os.path.commonpath([os.path.realpath(path), JOBS]) == JOBS


def check(argv, cwd):
    if not (cwd is None or inside(cwd)):
        raise sandbox.Rejected("cwd outside /jobs")
    if len(argv) == 3 and argv[:2] == ["bash", "-c"]:
        sandbox.split_segments(argv[2])
    elif not (argv and argv[0] in sandbox.ALLOWED and all(inside(a) for a in argv[1:] if a.startswith("/"))):
        raise sandbox.Rejected(f"not allowed: {argv[:1]}")


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline())
            check(req["argv"], req.get("cwd"))
            rc, out, err = sandbox._local(req["argv"], req.get("cwd"), min(req.get("timeout", 60), sandbox.TIMEOUT))
        except Exception as e:
            rc, out, err = 1, "", f"runner: {e}"
        self.wfile.write(json.dumps({"rc": rc, "out": out, "err": err}).encode() + b"\n")


if __name__ == "__main__":
    sock = os.environ["CLIPWRIGHT_RUNNER"]
    if os.path.exists(sock):
        os.unlink(sock)
    with socketserver.ThreadingUnixStreamServer(sock, Handler) as srv:
        print(f"runner listening on {sock}", flush=True)
        srv.serve_forever()
