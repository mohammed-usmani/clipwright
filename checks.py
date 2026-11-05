"""Content checks for output specs, beyond what ffprobe metadata shows: frames, loudness over time, pitch, freezes,
interlacing, frame-rate regularity, brightness, A/V sync, silences, image pixels. Called from teacher.check_file().

Spec keys (all optional; times in seconds, a negative time counts back from the end; "in/x" paths are inside the job):
  frame_like      [[t, "in/x.png"], ...]       frame at t looks like the image (or like the input video at t2: "in/x.mp4@t2")
  frame_unlike    [[t, "in/x.png"], ...]       frame at t does NOT look like it (catches an overlay over the whole video)
  loud_at         [[t0, t1, min_db], ...]      mean volume in the window >= min_db (dBFS)
  quiet_at        [[t0, t1, max_db], ...]      mean volume in the window <= max_db
  band_drop       [hz, [t0, t1], [t2, t3], db] the hz band is >= db quieter in window 1 than window 2 (ducking)
  band_max        [hz, max_db]                 the hz band is at most max_db over the whole file (hum removed)
  contains        [regex, ...]                 text outputs (.csv .txt .srt .json): every pattern is present
  (frame_like / frame_unlike take an optional 3rd item, the difference threshold; default 0.08)
  pitch_ratio     ["in/x", lo, hi]             median voice pitch of the output / of the input is in [lo, hi]
  freeze          [t0, t1]                     a frozen stretch covers t0..t1
  no_freeze       true                         no frozen stretch >= 1 s
  progressive     true                         not interlaced (idet)
  cfr             true                         frame intervals are regular
  sar             [n, d]                       sample aspect ratio
  yavg            [lo, hi]                     mean brightness (0-255)
  av_sync_ms      [lo, hi]                     first flash vs first beep, in ms (fixtures with a flash + beep)
  max_silence     sec                          no silence (< -35 dB) longer than sec
  audio_streams   n                            number of audio streams
  audio_lang      "eng"                        language tag of the first audio stream
  corner_alpha    true                         image corner pixel is transparent (rounded corners)
  corner_color    [r, g, b]                    image corner pixel colour, +-12 per channel
  gray            true                         image has (almost) no colour
  pdf_pages       n                            PDF page count
  exif_eq         {tag: value}                 exiftool tag values (e.g. a shifted DateTimeOriginal)
  exif_absent     [tag, ...]                   tags that must be gone (Artist, GPSLatitude...)
"""
import json
import os
import re
import subprocess

W, H = 64, 36  # frames are compared as small grayscale thumbnails


def _run(args):
    return subprocess.run(args, capture_output=True)


def duration(path):
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path])
    try:
        return float(r.stdout.decode().strip())
    except ValueError:
        return 0.0


def _t(path, t):
    return max(0.0, duration(path) + t) if t < 0 else t


def thumb(path, t=None):
    """Grayscale WxH pixels of an image, or of a video frame at time t."""
    seek = ["-ss", f"{_t(path, t):.3f}"] if t else []  # images: no seeking
    r = _run(["ffmpeg", "-v", "error", *seek, "-i", path, "-frames:v", "1",
              "-vf", f"scale={W}:{H}:flags=area,format=gray", "-f", "rawvideo", "-"])
    return r.stdout if len(r.stdout) == W * H else None


def diff(a, b):
    return sum(abs(x - y) for x, y in zip(a, b)) / (255 * W * H)


def mean_db(path, t0, t1, band=None):
    t0, t1 = _t(path, t0), _t(path, t1)
    af = (f"bandpass=f={band}:width_type=q:w=30,bandpass=f={band}:width_type=q:w=30," if band else "") + "volumedetect"
    r = _run(["ffmpeg", "-v", "info", "-ss", f"{t0:.3f}", "-t", f"{max(0.05, t1 - t0):.3f}", "-i", path,
              "-vn", "-af", af, "-f", "null", "-"])
    m = re.search(r"mean_volume: (-?[\d.]+|-inf) dB", r.stderr.decode(errors="replace"))
    return None if not m else (-120.0 if m.group(1) == "-inf" else float(m.group(1)))


def pitch(path, sr=8000):
    """Median fundamental frequency (Hz) over voiced 40 ms frames, by autocorrelation. Pure Python, ~1-3 s per minute."""
    r = _run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(sr), "-af", "highpass=f=60,lowpass=f=1000",
              "-t", "60", "-f", "s16le", "-"])
    import array
    x = array.array("h", r.stdout[: len(r.stdout) // 2 * 2])
    n, lo, hi = int(0.04 * sr), sr // 500, sr // 60  # 60-500 Hz
    f0s = []
    for s in range(0, len(x) - n - hi, n):
        fr = x[s:s + n + hi]
        energy = sum(v * v for v in fr[:n])
        if energy < n * 300 ** 2:  # skip silence / quiet frames
            continue
        cs = [sum(fr[i] * fr[i + L] for i in range(0, n, 2)) for L in range(lo, hi)]
        best = max(cs)
        # first local peak close to the best one; the best alone can be 2x the period (octave error)
        lag = next((lo + i for i in range(1, len(cs) - 1)
                    if cs[i] >= 0.85 * best and cs[i] >= cs[i - 1] and cs[i] >= cs[i + 1]), 0)
        if lag and best > 0.3 * energy / 2:
            f0s.append(sr / lag)
    f0s.sort()
    return f0s[len(f0s) // 2] if len(f0s) >= 5 else None


def freezes(path):
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-an", "-vf", "freezedetect=n=0.003:d=0.5", "-f", "null", "-"])
    log = r.stderr.decode(errors="replace")
    starts = [float(x) for x in re.findall(r"freeze_start: ([\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"freeze_end: ([\d.]+)", log)]
    ends += [duration(path)] * (len(starts) - len(ends))
    return list(zip(starts, ends))


def interlaced(path):
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-an", "-vf", "idet", "-frames:v", "300", "-f", "null", "-"])
    m = re.findall(r"Multi frame detection: TFF:\s*(\d+) BFF:\s*(\d+) Progressive:\s*(\d+)", r.stderr.decode(errors="replace"))
    if not m:
        return None
    tff, bff, prog = map(int, m[-1])
    # real interlacing is consistent (one field order dominates); a progressive video gives a noisy TFF/BFF/progressive mix
    return max(tff, bff) > 0.6 * max(1, tff + bff + prog)


def frame_times(path):
    r = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0", path])
    return [float(x) for x in r.stdout.decode().split() if x.replace(".", "", 1).isdigit()]


def is_cfr(path):
    t = frame_times(path)
    d = [b - a for a, b in zip(t, t[1:])]
    if len(d) < 10:
        return False
    m = sorted(d)[len(d) // 2]
    return sum(abs(x - m) > 0.25 * m for x in d) <= 0.02 * len(d)


def yavg(path):
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-an", "-vf", "fps=2,signalstats,metadata=print:key=lavfi.signalstats.YAVG",
              "-f", "null", "-"])
    v = [float(x) for x in re.findall(r"YAVG=([\d.]+)", r.stderr.decode(errors="replace"))]
    return sum(v) / len(v) if v else None


def first_flash(path):
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-an", "-vf", "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
              "-f", "null", "-"])
    log = r.stderr.decode(errors="replace")
    for t, y in re.findall(r"pts_time:([\d.]+)\s*\n.*?YAVG=([\d.]+)", log):
        if float(y) > 180:
            return float(t)
    return None


def first_beep(path):
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "silencedetect=n=-30dB:d=0.05", "-f", "null", "-"])
    m = re.search(r"silence_end: ([\d.]+)", r.stderr.decode(errors="replace"))
    return float(m.group(1)) if m else None


def silences(path, db=-35, d=0.3):
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", f"silencedetect=n={db}dB:d={d}", "-f", "null", "-"])
    return [float(x) for x in re.findall(r"silence_duration: ([\d.]+)", r.stderr.decode(errors="replace"))]


def speech_segments(path, db=-35, d=0.3):
    """[(start, end)] of non-silent stretches."""
    r = _run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", f"silencedetect=n={db}dB:d={d}", "-f", "null", "-"])
    log = r.stderr.decode(errors="replace")
    st = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    edges, t = [], 0.0
    for a, b in zip(st, en + [None] * (len(st) - len(en))):
        if a > t + 0.05:
            edges.append((round(t, 2), round(a, 2)))
        t = b if b is not None else duration(path)
    if duration(path) > t + 0.05:
        edges.append((round(t, 2), round(duration(path), 2)))
    return edges


def magick_fx(path, fmt):
    r = _run(["magick", path + "[0]", "-format", fmt, "info:"])
    return r.stdout.decode().strip()


def chroma(path):
    """(mean saturation, mean hue) over the first 2 s, from signalstats."""
    r = _run(["ffmpeg", "-v", "info", "-t", "2", "-i", path, "-vf", "signalstats,metadata=print:key=lavfi.signalstats.SATAVG,"
              "metadata=print:key=lavfi.signalstats.HUEAVG", "-f", "null", "-"])
    txt = r.stderr.decode(errors="replace")
    sat = [float(x) for x in re.findall(r"SATAVG=([\d.]+)", txt)]
    hue = [float(x) for x in re.findall(r"HUEAVG=([\d.]+)", txt)]
    return (sum(sat) / len(sat), sum(hue) / len(hue)) if sat and hue else None


def _src(wd, p):
    """'in/x.png' or 'in/x.mp4@3.5' -> (abs path, time or None)."""
    t = None
    if "@" in p:
        p, t = p.rsplit("@", 1)
        t = float(t)
    return os.path.join(wd, p), t


def extra(f, s, wd, info=None):
    """Errors for the content keys of spec s on output file f (job folder wd)."""
    errs = []
    for key, want in (("frame_like", True), ("frame_unlike", False)):
        for t, ref, *thr in s.get(key, []):
            thr = thr[0] if thr else 0.08
            a = thumb(f, t)
            rp, rt = _src(wd, ref)
            b = thumb(rp, rt)
            if a is None or b is None:
                errs.append(f"frame at {t}s unreadable")
                continue
            d = diff(a, b)
            if (d < thr) != want:
                errs.append(f"frame at {t}s {'does not look' if want else 'looks'} like {ref} (diff {d:.3f})")
    for t0, t1, lim in s.get("loud_at", []):
        v = mean_db(f, t0, t1)
        if v is None or v < lim:
            errs.append(f"audio {t0}..{t1}s at {v} dB, expected >= {lim}")
    for t0, t1, lim in s.get("quiet_at", []):
        v = mean_db(f, t0, t1)
        if v is None or v > lim:
            errs.append(f"audio {t0}..{t1}s at {v} dB, expected <= {lim}")
    if "band_drop" in s:
        hz, w1, w2, db = s["band_drop"]
        a, b = mean_db(f, *w1, band=hz), mean_db(f, *w2, band=hz)
        if a is None or b is None or b - a < db:
            errs.append(f"{hz} Hz band not ducked: {a} dB during speech vs {b} dB without")
    if "chroma_unlike" in s:  # colour-only effects: average saturation or hue moved vs the input
        ref, lim = s["chroma_unlike"]
        a_, b_ = chroma(f), chroma(os.path.join(wd, ref))
        if not a_ or not b_ or max(abs(x - y) for x, y in zip(a_, b_)) < lim:
            errs.append(f"colours look like {ref}: sat/hue {a_} vs {b_}")
    if "audio_unlike" in s:  # an effect was applied: overall level or some band moved by at least min_db vs the input
        ref, lim = s["audio_unlike"]
        rp = os.path.join(wd, ref)
        moved = [abs((mean_db(f, 0, -0.01, band=b) or -120) - (mean_db(rp, 0, -0.01, band=b) or -120)) for b in (None, 150, 1000, 4000)]
        if max(moved) < lim:
            errs.append(f"audio sounds like {ref}: level/bands differ by at most {max(moved):.1f} dB")
    if "band_max" in s:  # e.g. mains hum removed: the hum band must be at most max_db
        hz, lim = s["band_max"]
        v = mean_db(f, 0, -0.01, band=hz)
        if v is None or v > lim:
            errs.append(f"{hz} Hz band at {v} dB, expected <= {lim}")
    if "pitch_ratio" in s:
        ref, lo, hi = s["pitch_ratio"]
        p_in, p_out = pitch(os.path.join(wd, ref)), pitch(f)
        if not p_in or not p_out or not lo <= p_out / p_in <= hi:
            errs.append(f"pitch {p_out and round(p_out)} Hz vs input {p_in and round(p_in)} Hz, ratio not in [{lo}, {hi}]")
    if "freeze" in s:
        t0, t1 = s["freeze"]
        if not any(a <= t0 + 0.3 and b >= t1 - 0.3 for a, b in freezes(f)):
            errs.append(f"no freeze covering {t0}..{t1}s (found {[(round(a, 1), round(b, 1)) for a, b in freezes(f)]})")
    if s.get("no_freeze") and any(b - a >= 1 for a, b in freezes(f)):
        errs.append("video has a frozen stretch")
    if s.get("progressive") and interlaced(f) is not False:
        errs.append("still interlaced")
    if s.get("cfr") and not is_cfr(f):
        errs.append("frame rate is not constant")
    if "sar" in s and info:
        v = next((x for x in info["streams"] if x["codec_type"] == "video"), {})
        sar = v.get("sample_aspect_ratio", "1:1")
        if sar in ("0:1", "N/A"):
            sar = "1:1"
        n, d = map(int, sar.split(":"))
        if abs(n / d - s["sar"][0] / s["sar"][1]) > 0.01:
            errs.append(f"SAR {sar}")
    if "yavg" in s:
        y = yavg(f)
        if y is None or not s["yavg"][0] <= y <= s["yavg"][1]:
            errs.append(f"brightness {y and round(y)} not in {s['yavg']}")
    if "av_sync_ms" in s:
        fl, bp = first_flash(f), first_beep(f)
        if fl is None or bp is None:
            errs.append("flash/beep not found")
        else:
            off = (bp - fl) * 1000
            if not s["av_sync_ms"][0] <= off <= s["av_sync_ms"][1]:
                errs.append(f"audio is {off:.0f} ms off the picture")
    if "max_silence" in s:
        long = [x for x in silences(f) if x > s["max_silence"]]
        if long:
            errs.append(f"{len(long)} silence(s) longer than {s['max_silence']}s (longest {max(long):.1f}s)")
    if info and ("audio_streams" in s or "audio_lang" in s):
        aud = [x for x in info["streams"] if x["codec_type"] == "audio"]
        if "audio_streams" in s and len(aud) != s["audio_streams"]:
            errs.append(f"{len(aud)} audio streams, expected {s['audio_streams']}")
        if "audio_lang" in s and (not aud or aud[0].get("tags", {}).get("language") != s["audio_lang"]):
            errs.append(f"first audio track is not {s['audio_lang']}")
    if s.get("corner_alpha"):
        a = magick_fx(f, "%[fx:p{1,1}.a]")
        if not a or float(a) > 0.1:
            errs.append("corners are not transparent")
    if "corner_color" in s:
        r = magick_fx(f, "%[fx:int(255*p{2,2}.r)],%[fx:int(255*p{2,2}.g)],%[fx:int(255*p{2,2}.b)]")
        try:
            rgb = list(map(int, r.split(",")))
        except ValueError:
            rgb = None
        if not rgb or any(abs(x - y) > 12 for x, y in zip(rgb, s["corner_color"])):
            errs.append(f"corner colour {rgb}, expected {s['corner_color']}")
    if s.get("gray"):
        sat = _run(["magick", f + "[0]", "-colorspace", "HSL", "-channel", "G", "-separate", "-format", "%[fx:mean]",
                    "info:"]).stdout.decode()
        if not sat or float(sat) > 0.03:
            errs.append(f"image still has colour (saturation {sat})")
    if "exif_eq" in s or "exif_absent" in s:
        r = _run(["exiftool", "-j", "-s", f])
        try:
            ex = json.loads(r.stdout.decode())[0]
        except (ValueError, IndexError):
            ex = {}
        for k, v in s.get("exif_eq", {}).items():
            if str(ex.get(k, "")).strip() != str(v):
                errs.append(f"{k} is {ex.get(k)!r}, expected {v!r}")
        for k in s.get("exif_absent", []):
            if k in ex:
                errs.append(f"{k} still present")
    if "pdf_pages" in s:
        n = len(re.findall(rb"/Type\s*/Page[^s]", open(f, "rb").read()))
        if n != s["pdf_pages"]:
            errs.append(f"{n} PDF pages, expected {s['pdf_pages']}")
    return errs


def answer(final, spec, wd):
    """Question tasks: the reply must mention every `must` pattern; with no_files, nothing may be written to out/."""
    a = spec.get("answer")
    if not a:
        return []
    out = os.path.join(wd, "out")
    text = final or ""
    if os.path.isdir(out):  # a report may be written to a text file instead of the reply
        for n in sorted(os.listdir(out)):
            if os.path.splitext(n)[1].lower() in (".csv", ".txt", ".json", ".tsv", ".md"):
                text += "\n" + open(os.path.join(out, n), errors="replace").read()
    errs = [f"answer misses /{p}/" for p in a.get("must", []) if not re.search(p, text, re.I)]
    if a.get("no_files") and os.path.isdir(out) and os.listdir(out):
        errs.append("wrote files for a question")
    return errs


if __name__ == "__main__":  # self-check on generated media: every check must pass its positive and fail its negative
    import tempfile
    d = tempfile.mkdtemp()
    sh = lambda c: subprocess.run(c, shell=True, cwd=d, check=True, capture_output=True)
    sh("ffmpeg -f lavfi -i color=red:s=320x180:d=4 -f lavfi -i color=blue:s=320x180:d=3 -filter_complex '[0][1]concat' -r 30 rb.mp4")
    sh("magick -size 320x180 xc:blue blue.png")
    sh("ffmpeg -f lavfi -i 'sine=f=200:d=4' -f lavfi -i 'sine=f=300:d=4' -filter_complex '[0][1]concat=v=0:a=1' tones.wav")
    sh("ffmpeg -i tones.wav -af asetrate=44100*1.5,aresample=44100 hi.wav")
    os.makedirs(f"{d}/in")
    sh("cp blue.png in/ && cp tones.wav in/")
    assert not extra(f"{d}/rb.mp4", {"frame_like": [[-0.5, "in/blue.png"]], "frame_unlike": [[1, "in/blue.png"]]}, d)
    assert extra(f"{d}/rb.mp4", {"frame_like": [[1, "in/blue.png"]]}, d)
    p = pitch(f"{d}/tones.wav")
    assert p and 190 < p < 310, p
    assert not extra(f"{d}/hi.wav", {"pitch_ratio": ["in/tones.wav", 1.4, 1.6]}, d)
    assert extra(f"{d}/tones.wav", {"pitch_ratio": ["in/tones.wav", 1.4, 1.6]}, d)
    assert not extra(f"{d}/tones.wav", {"loud_at": [[0.5, 3, -30]]}, d)
    print("checks self-test ok")
