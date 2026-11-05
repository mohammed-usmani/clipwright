"""Sandbox regression checks against real tools. Run: python3 test_sandbox.py (takes ~20s)."""
import os
import shutil
import subprocess
import tempfile
import time

import sandbox
from fixtures import FIX, clone


def job(*files):
    wd = tempfile.mkdtemp(prefix="sbtest_")
    os.makedirs(os.path.join(wd, "in"))
    for name, fx in files:
        clone(os.path.join(FIX, fx), os.path.join(wd, "in", name))
    return wd


def rejected(cmd):
    r = sandbox.run(job(), cmd)
    return r["exit_code"] == 1 and r["error"].startswith("rejected")


def main():
    # --- allowlist / operators / paths ---
    assert rejected("rm -rf in")
    assert rejected("ffmpeg -i in/a.wav out/a.mp3; rm x")
    assert rejected("ffmpeg -i in/a.wav out/a.mp3 | cat")
    assert rejected("ffmpeg -i in/a.wav > out/log.txt")
    assert rejected('ffmpeg -i "$(rm -rf in)" out/x.mp3'), "$() inside double quotes must be rejected"
    assert rejected('ffmpeg -i "`id`" out/x.mp3'), "backticks inside double quotes must be rejected"
    assert rejected("ffmpeg -i in/a.wav /tmp/x.mp3")
    assert rejected("ffmpeg -i ../secret.wav out/x.mp3")
    assert rejected("ffmpeg -i in/a.wav ~/x.mp3")
    assert not rejected("magick in/x.png -resize '800x800>' out/x.png"), "quoted > is fine"
    assert not rejected("ffmpeg -i in/a.wav -af 'volume=enable=between(t,1,2):volume=0' out/a.wav"), "single-quoted $-free filters fine"

    # --- analysis-only command: loudnorm JSON reaches the model, progress noise removed ---
    wd = job(("voice.wav", "a.wav"))
    r = sandbox.run(wd, "ffmpeg -i in/voice.wav -af loudnorm=print_format=json -f null -")
    assert r["exit_code"] == 0 and '"input_i"' in r["log"] and "time=" not in r["log"], r
    assert "ffmpeg version" not in r["log"] and "configuration:" not in r["log"]

    # --- analysis + output in one chain: the measurement is NOT dropped ---
    r = sandbox.run(wd, "ffmpeg -i in/voice.wav -af ebur128=framelog=quiet -f null - && ffmpeg -i in/voice.wav -c:a libmp3lame out/v.mp3")
    assert "out/v.mp3" in r["outputs"] and "LUFS" in r.get("log", ""), r
    r = sandbox.run(wd, "ffmpeg -y -i in/voice.wav -af loudnorm=I=-16:print_format=json out/n.wav")
    assert "out/n.wav" in r["outputs"] and '"output_i"' in r.get("log", ""), r

    # --- filter chatter (palettegen) is not reported as analysis ---
    vd0 = job(("c.mp4", "short.mp4"))
    r = sandbox.run(vd0, 'ffmpeg -t 1 -i in/c.mp4 -vf "fps=5,scale=160:-1,split[a][b];[a]palettegen[p];[b][p]paletteuse" out/c.gif')
    assert "out/c.gif" in r["outputs"] and "palettegen" not in r.get("log", ""), r
    shutil.rmtree(vd0)

    # --- plain encode: no noisy log attached ---
    r = sandbox.run(wd, "ffmpeg -i in/voice.wav -c:a aac out/v.m4a")
    assert "out/v.m4a" in r["outputs"] and "log" not in r, r

    # --- intermediates in the job root are reported; in/ and bookkeeping files are not ---
    open(os.path.join(wd, "request.txt"), "w").write("x")
    r = sandbox.run(wd, "ffmpeg -i in/voice.wav -t 3 tmp_cut.wav")
    assert list(r["outputs"]) == ["tmp_cut.wav"], r

    # --- silent success explains itself (e.g. -v error hiding a loudness report) ---
    r = sandbox.run(wd, "ffmpeg -v error -i in/voice.wav -af loudnorm=print_format=summary -f null -")
    assert r["exit_code"] == 0 and "-v/-loglevel error hides" in r.get("note", ""), r

    # --- ebur128: per-100ms frame lines dropped, the Summary block kept ---
    r = sandbox.run(wd, "ffmpeg -i in/voice.wav -af ebur128 -f null -")
    assert "Integrated loudness" in r.get("log", "") and " t: " not in r.get("log", ""), r

    # --- sox stats (stderr report) ---
    r = sandbox.run(wd, "sox in/voice.wav -n stats")
    assert "RMS" in r.get("log", ""), r

    # --- failure: real error tail, no progress noise ---
    r = sandbox.run(wd, "ffmpeg -i in/voice.wav -c:a libnotreal out/x.mp3")
    assert r["exit_code"] == 1 and "libnotreal" in r["error"] and "time=" not in r["error"], r

    # --- stdout truncation is explicit ---
    vd = job(("clip.mp4", "land.mp4"))
    r = sandbox.run(vd, "ffprobe -v error -show_frames -select_streams v -of json in/clip.mp4")
    assert "[truncated" in r["stdout"] and len(r["stdout"]) < sandbox.STDOUT_CAP + 200, len(r["stdout"])

    # --- foreach with spaces in names ---
    bd = job(("memo 1.m4a", "a.m4a"), ("memo 2.m4a", "a.m4a"))
    r = sandbox.run(bd, "ffmpeg -i {in} -t 2 out/{stem}.mp3", foreach="in/*.m4a")
    assert r["exit_code"] == 0 and r["ran"] == 2 and "out/memo 1.mp3" in r["outputs"], r

    # --- shell glob expansion for tools that take many files ---
    r = sandbox.run(bd, "exiftool -csv in/*.m4a")
    assert r["exit_code"] == 0 and "SourceFile" in r["stdout"], r

    # --- timeout kills the whole process group, including ffmpeg behind && ---
    t = time.time()
    r = sandbox.run(wd, "ffprobe -v error in/voice.wav && ffmpeg -re -f lavfi -i anullsrc -t 600 -f null -", timeout=3)
    assert r["exit_code"] == 1 and "timed out" in r["error"] and time.time() - t < 10, r
    time.sleep(0.5)
    left = subprocess.run(["pgrep", "-f", "anullsrc -t 600"], capture_output=True, text=True).stdout.strip()
    assert not left, f"orphaned ffmpeg still running: {left}"

    # --- summarize never crashes on garbage ---
    open(os.path.join(wd, "out", "broken.jpg"), "wb").write(b"not an image")
    assert "UNREADABLE" in sandbox.summarize(os.path.join(wd, "out", "broken.jpg"))

    for d in (wd, vd, bd):
        shutil.rmtree(d)

    # --- triage: replay noise is not a "changed observation", real changes are ---
    import triage
    m = lambda out: triage.material({"exit_code": 0, "stdout": out})
    assert triage.same(m("size=8153232\nwidth=1280"), m("size=8162196\nwidth=1280")), "encoder size jitter"
    assert not triage.same(m("size=8153232\nwidth=1280"), m("size=8153232")), "missing field"
    assert triage.same(m("x.gif[0] GIF 0.000u 0:00.007"), m("x.gif[0] GIF 0.000u 0:00.001")), "identify timing"
    assert triage.same(m("File Access Date/Time : 2026:10:15 13:29:35+05:30"),
                       m("File Access Date/Time : 2026:10:15 13:47:02+05:30")), "exiftool timestamps"
    assert triage.same(m("mean=0.494774"), m("mean=0.494807")), "float jitter"
    assert not triage.same(m("I: -16.0 LUFS"), m("")), "loudness report vanished"
    assert not triage.same(m("x"), triage.material({"exit_code": 1, "stdout": "x"})), "exit code"
    long_old = "\n".join(f"line {i} 0:00.007" for i in range(400))
    assert triage.same(m(long_old[:3000]), m(long_old[:4000] + "\n...[truncated 9 chars; x]")), "different stdout caps"
    e = lambda err: triage.material({"exit_code": 1, "error": err})
    assert triage.same(e("739.63 (0.011286)"), e("778.833 (0.0118842)")), "magick compare score jitter"
    assert not triage.same(e("No such file: out/a.mp4"), e("No such file: out/b.mp4")), "different error"
    assert triage.same(m("Pixels per second: 58.3MP\n  User time: 1.003u\nDepth: 8-bit"),
                       m("Pixels per second: 40.5MP\n  User time: 1.005u\nDepth: 8-bit")), "identify -verbose timing"
    print("sandbox + triage: all checks passed")


if __name__ == "__main__":
    main()
