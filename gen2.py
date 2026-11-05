"""Training data for the 46 new task types (A-H), from the TRAIN fixtures only (fixtures/, never fixtures/test/).

Each generator returns a job: files, request, hidden spec, the steps (reasoning + command), and optionally a fixed final
reply (questions) or a wrong first attempt (recovery rows). Every job is executed in the sandbox and kept only if its
output passes verify() against the spec, the same checker the test set and the RL reward use.

    python3 gen2.py --per-type 45 --out data/gen2/train.jsonl     # all types
    python3 gen2.py --only A1,C21 --per-type 5 --out /tmp/x.jsonl  # a few, for checking
Rows have the same format as data/sft/train.jsonl.
"""
import argparse
import collections
import hashlib
import json
import os
import random
import shlex
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor

import sandbox
from build import SYSTEM, aug, call, tool
from fixtures import FIX, clone
from teacher import verify

META = json.load(open(os.path.join(FIX, "meta.json")))
TRUTH = json.load(open(os.path.join(FIX, "truth.json")))
GENS = {}


def gen(tid, weight=1.0):
    def deco(f):
        GENS[tid] = (f, weight)
        return f
    return deco


# ── input pools (train fixtures) ──
VID_A = ["talk.mp4", "cuts.mp4", "webcam.mp4", "version_a.mp4", "land.mp4", "short.mp4", "small.mp4", "lowres.mp4"]
VID_SPEECH = ["talk.mp4", "webcam.mp4"]          # speech at TRUTH["speech"]
VID_TONE = {"cuts.mp4": 440, "land.mp4": 440, "short.mp4": 600, "small.mp4": 300, "version_a.mp4": 410, "lowres.mp4": 420}
VID_SILENT = ["screenrec.mp4", "screen.mp4"]
CARDS = ["thanks.png", "title.png", "cover.jpg", "photo.jpg", "shot.png"]
VOICES = ["voice_m.m4a", "voice_f.m4a"]
MUSIC = ["music_bed.mp3", "music.mp3"]
VNAMES = ["vlog", "talk", "demo", "clip", "lecture", "interview", "promo", "recording", "meeting", "tutorial", "reel", "video",
          "final_cut", "IMG_4821", "GX010233", "webinar", "presentation", "update", "episode_3", "review", "launch", "keynote",
          "my video", "class recording", "trailer", "standup", "podcast_ep", "event"]
INAMES = ["endcard", "thank_you", "outro", "slide", "card", "closing", "end_slide", "logo_card", "credits", "final_slide"]
ANAMES = ["voiceover", "vo", "outro_vo", "narration", "my_voice", "signoff", "line", "thanks", "closing_line", "speech"]
MNAMES = ["music", "bgm", "song", "track", "bed", "background", "beat", "lofi"]


def m(fx):
    return META[fx]


def name(r, pool, ext, avoid=()):
    for _ in range(20):
        n = r.choice(pool) + ext
        if n not in avoid:
            return n
    return f"file{r.randint(10, 99)}{ext}"


def ext_of(fx):
    return os.path.splitext(fx)[1]


def q(n):
    return shlex.quote("in/" + n)


def o(n):
    return shlex.quote("out/" + n)


def stem(n):
    return os.path.splitext(n)[0].replace(" ", "_")


def even(x):
    return int(x) // 2 * 2


def ask(r, *options, **kw):
    return aug(r, r.choice(options).format(**kw))


ENC = "-c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p"
AENC = "-c:a aac -b:a 160k"
FIT = "scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={F},format=yuv420p"
AN = "aresample=48000,aformat=channel_layouts=stereo"


def fit(W, H, F):
    return FIT.format(W=W, H=H, F=F)


def r1(x):
    return round(x + 1e-9, 2)


# ═════════════════════════ A. editing & assembly ═════════════════════════

@gen("A1")
def end_card(r):
    fx = r.choice(VID_A)
    M = m(fx)
    card = r.choice(CARDS)
    vn, cn = name(r, VNAMES, ext_of(fx)), name(r, INAMES, ext_of(card))
    W, H, F, D = M["w"], M["h"], int(M["fps"]), M["dur"]
    files = {vn: fx, cn: card}
    if r.random() < 0.65:  # with a voice recording
        vo = r.choice(VOICES)
        an = name(r, ANAMES, ".m4a")
        files[an] = vo
        cd = round(m(vo)["dur"] + r.choice([0, 0.5, 1.0]), 2)
        extra = f" plus {r1(cd - m(vo)['dur'])}s of air" if cd > m(vo)["dur"] else ""
        req = ask(r, "add {c} at the end of {v}, shown while {a} plays as the voiceover",
                  "finish {v} with the slide {c} and {a} as the voice", "put {c} as a closing card after {v}, with {a} as the audio",
                  "end {v} on {c} with my outro recording {a}", "thank you slide at the end: {c}, voice: {a}. video is {v}",
                  vthe=vn, v=vn, c=cn, a=an)
        think = (f"Concat the video with a {cd}s still of {cn} (the length of {an}{extra}); the card is scaled/padded to "
                 f"{W}x{H} at {F} fps and the voice is padded to the same length so both streams line up.")
        cmd = (f"ffmpeg -i {q(vn)} -loop 1 -framerate {F} -i {q(cn)} -i {q(an)} -filter_complex "
               f"\"[0:v]setsar=1,fps={F},format=yuv420p[m];[0:a]{AN}[ma];[1:v]{fit(W, H, F)},trim=duration={cd}[s];"
               f"[2:a]{AN},apad=whole_dur={cd}[sa];[m][ma][s][sa]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" "
               f"{ENC} {AENC} {o(stem(vn) + '_end.mp4')}")
        spec = {"match": "*.mp4", "wh": [W, H], "dur": [D + cd - 0.4, D + cd + 0.5], "frame_unlike": [[1, f"in/{cn}"]],
                "loud_at": [[D + 0.25, D + m(vo)["dur"] - 0.3, -50]]}
    else:  # silent hold
        cd = r.choice([2, 3, 4, 5])
        req = ask(r, "hold {c} on screen for {n} seconds after {v} ends", "add a {n} second end card ({c}) to {v}, no sound needed",
                  "after {v}, show {c} for {n}s", v=vn, c=cn, n=cd)
        think = (f"Concat a {cd}s still of {cn} (scaled/padded to {W}x{H}) after the video, with {cd}s of silence so the "
                 f"audio stream stays continuous.")
        cmd = (f"ffmpeg -i {q(vn)} -loop 1 -framerate {F} -t {cd} -i {q(cn)} -f lavfi -t {cd} -i anullsrc=r=48000:cl=stereo "
               f"-filter_complex \"[0:v]setsar=1,fps={F},format=yuv420p[m];[0:a]{AN}[ma];[1:v]{fit(W, H, F)}[s];"
               f"[m][ma][s][2:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {ENC} {AENC} {o(stem(vn) + '_end.mp4')}")
        spec = {"match": "*.mp4", "wh": [W, H], "dur": [D + cd - 0.4, D + cd + 0.5], "frame_unlike": [[1, f"in/{cn}"]],
                "has_audio": True}
    return dict(files=files, ask=req, spec=[spec], steps=[(think, cmd)])


TITLES = ["Product Demo", "Episode 4", "Quarterly Update", "Welcome", "Chapter 1", "Behind the Scenes", "Team Offsite 2024",
          "How It Works", "Day 1", "Launch Recap", "Q&A", "Part 2", "Getting Started"]
COLORS = [("black", "black"), ("dark blue", "0x1e3a8a"), ("white", "white"), ("dark gray", "0x1f2937"), ("navy", "navy"),
          ("deep purple", "0x4c1d95")]


@gen("A2")
def text_intro(r):
    fx = r.choice(VID_A + VID_SILENT)
    M = m(fx)
    vn = name(r, VNAMES, ext_of(fx))
    W, H, F, D = M["w"], M["h"], int(M["fps"]), M["dur"]
    title, (cname, col) = r.choice(TITLES), r.choice(COLORS)
    fg = "black" if col == "white" else "white"
    d = r.choice([2, 3, 3, 4])
    fade = r.random() < 0.3
    has_a = M.get("has_audio")
    req = ask(r, "start {v} with a {d} second {c} title card that says '{t}'", "add an intro card saying \"{t}\" ({d}s, {c} background) before {v}",
              "put a title '{t}' on a {c} screen for {d} seconds at the beginning of {v}" + (", fading into the video" if fade else ""),
              v=vn, d=d, c=cname, t=title)
    if fade:
        req += "" if "fading" in req else " and fade into the video"
    fs = max(24, H // 12)
    dt = f"drawtext=text='{title.replace(':', chr(92) + ':')}':fontcolor={fg}:fontsize={fs}:x=(w-text_w)/2:y=(h-text_h)/2"
    src = f"-f lavfi -i color={col}:s={W}x{H}:r={F}:d={d}"
    if fade:
        think = f"A {d}s {cname} card with drawtext, then xfade (1s) into the video; offset = {d}-1 so the card shows fully first."
        a = (f"-f lavfi -t {d} -i anullsrc=r=48000:cl=stereo ", f";[1:a][2:a]{AN.split(',')[0]},acrossfade=d=1[a]", "-map \"[a]\" ") if has_a else ("", "", "")
        cmd = (f"ffmpeg {src} {a[0]}-i {q(vn)} -filter_complex \"[0:v]{dt},format=yuv420p,setsar=1,settb=AVTB,fps={F}[t];"
               f"[{2 if has_a else 1}:v]format=yuv420p,setsar=1,settb=AVTB,fps={F}[m];[t][m]xfade=transition=fade:duration=1:offset={d - 1}[v]"
               f"{a[1]}\" -map \"[v]\" {a[2]}{ENC} {AENC if has_a else ''} {o(stem(vn) + '_intro.mp4')}")
        dur = [D + d - 1.4, D + d - 0.5]
        fl = [[-1, f"in/{vn}@{r1(D - 1)}"]]
    else:
        think = f"Generate a {d}s {cname} card with drawtext at the video's {W}x{H} {F}fps, then concat it before the video" + (
            " with matching silence so the audio stays in sync." if has_a else ".")
        if has_a:
            cmd = (f"ffmpeg {src} -f lavfi -t {d} -i anullsrc=r=48000:cl=stereo -i {q(vn)} -filter_complex "
                   f"\"[0:v]{dt},format=yuv420p,setsar=1[t];[2:v]setsar=1,fps={F},format=yuv420p[m];[2:a]{AN}[ma];"
                   f"[t][1:a][m][ma]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {ENC} {AENC} {o(stem(vn) + '_intro.mp4')}")
        else:
            cmd = (f"ffmpeg {src} -i {q(vn)} -filter_complex \"[0:v]{dt},format=yuv420p,setsar=1[t];[1:v]setsar=1,fps={F},"
                   f"format=yuv420p[m];[t][m]concat=n=2:v=1:a=0[v]\" -map \"[v]\" {ENC} {o(stem(vn) + '_intro.mp4')}")
        dur = [D + d - 0.4, D + d + 0.5]
        fl = [[d + 2, f"in/{vn}@2"]]
    return dict(files={vn: fx}, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "dur": dur, "frame_like": fl}],
                steps=[(think, cmd)])


def norm_v(i, W, H, F):
    return f"[{i}:v]{fit(W, H, F)}[n{i}]"


@gen("A3")
def insert_at(r):
    fx = r.choice(["talk.mp4", "cuts.mp4", "land.mp4", "small.mp4", "webcam.mp4"])
    M = m(fx)
    W, H, F, D = M["w"], M["h"], int(M["fps"]), M["dur"]
    vn = name(r, VNAMES, ext_of(fx))
    T = r1(r.choice([2, 3, 4, 5, 6, 7.5, 8, 9.5, 10]) if D > 12 else r.choice([2, 3, 4]))
    T = min(T, D - 2)
    if r.random() < 0.6:  # a clip
        ix = r.choice([x for x in VID_A + VID_SILENT if x != fx and m(x)["dur"] <= 20])
        iN = name(r, ["ad", "sponsor", "broll", "insert", "cutaway", "clip2", "bumper", "selfie"], ext_of(ix), avoid={vn})
        L = m(ix)["dur"]
        has_ia = m(ix).get("has_audio")
        req = ask(r, "insert {i} into {v} at {t}s, then continue with the rest of {v}", "cut {i} into {v} at the {t} second mark",
                  "drop {i} in the middle of {v} at {t} seconds", v=vn, i=iN, t=T)
        ia = f"[1:a]{AN}[ia]" if has_ia else f"anullsrc=r=48000:cl=stereo,atrim=0:{L}[ia]"
        think = (f"Split {vn} at {T}s with trim/atrim, normalize {iN} to {W}x{H} {F}fps" +
                 ("" if has_ia else " (it has no audio, so add silence of its length)") + ", then concat the three parts.")
        cmd = (f"ffmpeg -i {q(vn)} -i {q(iN)} -filter_complex \"[0:v]trim=0:{T},setpts=PTS-STARTPTS,{fit(W, H, F)}[a];"
               f"[0:a]atrim=0:{T},asetpts=PTS-STARTPTS,{AN}[aa];[0:v]trim={T},setpts=PTS-STARTPTS,{fit(W, H, F)}[b];"
               f"[0:a]atrim={T},asetpts=PTS-STARTPTS,{AN}[ba];[1:v]{fit(W, H, F)}[i];{ia};"
               f"[a][aa][i][ia][b][ba]concat=n=3:v=1:a=1[v][o]\" -map \"[v]\" -map \"[o]\" {ENC} {AENC} {o(stem(vn) + '_edit.mp4')}")
        files = {vn: fx, iN: ix}
        chk = [[max(0.5, T - 1), f"in/{vn}@{max(0.5, T - 1)}"], [r1(T + L + 1), f"in/{vn}@{r1(T + 1)}"]] if D - T > 1.5 else []
    else:  # a still image for N seconds
        ix = r.choice(CARDS)
        L = r.choice([2, 3, 4, 5])
        iN = name(r, ["sponsor", "slide", "chart", "ad", "title"], ext_of(ix), avoid={vn})
        req = ask(r, "at {t}s in {v}, show {i} for {n} seconds (no sound), then continue", "put {i} into {v} at {t} seconds for {n}s",
                  v=vn, i=iN, t=T, n=L)
        think = f"Split at {T}s, insert a {L}s still of {iN} fitted to {W}x{H} with {L}s of silence, concat the three parts."
        cmd = (f"ffmpeg -i {q(vn)} -loop 1 -framerate {F} -t {L} -i {q(iN)} -f lavfi -t {L} -i anullsrc=r=48000:cl=stereo "
               f"-filter_complex \"[0:v]trim=0:{T},setpts=PTS-STARTPTS,{fit(W, H, F)}[a];[0:a]atrim=0:{T},asetpts=PTS-STARTPTS,{AN}[aa];"
               f"[0:v]trim={T},setpts=PTS-STARTPTS,{fit(W, H, F)}[b];[0:a]atrim={T},asetpts=PTS-STARTPTS,{AN}[ba];[1:v]{fit(W, H, F)}[i];"
               f"[a][aa][i][2:a][b][ba]concat=n=3:v=1:a=1[v][o]\" -map \"[v]\" -map \"[o]\" {ENC} {AENC} {o(stem(vn) + '_edit.mp4')}")
        files = {vn: fx, iN: ix}
        chk = [[r1(T + L + 1), f"in/{vn}@{r1(T + 1)}"]] if D - T > 1.5 else []
    return dict(files=files, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "dur": [D + L - 0.4, D + L + 0.5], "frame_like": chk}],
                steps=[(think, cmd)])


def ranges_expr(rs):
    return "+".join(f"between(t,{a},{b})" for a, b in rs)


@gen("A4")
def remove_sections(r):
    fx = r.choice(["talk.mp4", "cuts.mp4", "land.mp4", "small.mp4", "webcam.mp4", "short.mp4"])
    M = m(fx)
    D, W, H = M["dur"], M["w"], M["h"]
    vn = name(r, VNAMES, ext_of(fx))
    n = r.choice([1, 2, 2, 3])
    pts = sorted(r.sample(range(1, int(D) - 1), 2 * n))
    cut = [(pts[2 * i], pts[2 * i + 1]) for i in range(n)]
    keep = r.random() < 0.3
    if keep:
        rs, total = cut, sum(b - a for a, b in cut)
        desc = " and ".join(f"{a}-{b}s" for a, b in cut)
        req = ask(r, "keep only {d} of {v}, joined together", "I only want the parts {d} from {v}", v=vn, d=desc)
        sel = ranges_expr(rs)
        think = f"select/aselect keep frames inside {desc}; setpts/asetpts renumber them so the kept parts play back to back."
    else:
        total = D - sum(b - a for a, b in cut)
        desc = " and ".join(f"{a}s to {b}s" for a, b in cut)
        req = ask(r, "remove {d} from {v}", "cut out {d} of {v} and join the rest", "delete the bits {d} in {v}", v=vn, d=desc)
        sel = f"not({ranges_expr(cut)})"
        think = f"select/aselect drop frames in {desc}; setpts/asetpts close the gaps so audio and video stay in sync."
    vf = f"select='{sel}',setpts=N/FRAME_RATE/TB"
    af = f"aselect='{sel}',asetpts=N/SR/TB"
    cmd = f"ffmpeg -i {q(vn)} -vf \"{vf}\" -af \"{af}\" {ENC} {AENC} {o(stem(vn) + '_cut.mp4')}"
    return dict(files={vn: fx}, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "dur": [total - 0.35, total + 0.35]}],
                steps=[(think, cmd)])


@gen("A5")
def crossfade(r):
    if r.random() < 0.2:  # audio only
        a, b = r.sample(["narration.m4a", "music_bed.mp3", "music.mp3", "a.wav"], 2)
        an, bn = name(r, ANAMES, ext_of(a)), name(r, MNAMES, ext_of(b))
        d = r.choice([2, 3, 4])
        req = ask(r, "join {a} and {b} into one track with a {d} second crossfade, mp3", "crossfade {a} into {b} ({d}s), export as mp3",
                  a=an, b=bn, d=d)
        cmd = (f"ffmpeg -i {q(an)} -i {q(bn)} -filter_complex \"[0:a]{AN}[x];[1:a]{AN}[y];[x][y]acrossfade=d={d}\" "
               f"-c:a libmp3lame -b:a 192k {o('joined.mp3')}")
        T = m(a)["dur"] + m(b)["dur"] - d
        return dict(files={an: a, bn: b}, ask=req, spec=[{"match": "*.mp3", "dur": [T - 0.5, T + 0.5]}],
                    steps=[(f"acrossfade overlaps the end of {an} with the start of {bn} for {d}s.", cmd)])
    k = r.choice([2, 2, 3])
    clips = r.sample(["talk.mp4", "cuts.mp4", "version_a.mp4", "short.mp4", "webcam.mp4", "lowres.mp4"], k)
    names, files = [], {}
    for c in clips:
        nm = name(r, ["part1", "part2", "part3", "take1", "take2", "a", "b", "c", "scene1", "scene2", "intro", "main"], ext_of(c), avoid=set(names))
        names.append(nm)
        files[nm] = c
    W, H, F = m(clips[0])["w"], m(clips[0])["h"], int(m(clips[0])["fps"])
    d = r.choice([0.5, 1, 1, 1.5])
    tr = r.choice(["fade", "fade", "dissolve", "fadeblack", "wipeleft"])
    req = ask(r, "join {l} with a {d}s crossfade between each", "stitch {l} together, {d} second {t} transitions",
              "dissolve from one clip to the next: {l} ({d}s)", l=", ".join(names), d=d, t=tr)
    chain, off, acc, prev = [], 0.0, [], "n0"
    for i in range(k):
        chain.append(f"[{i}:v]{fit(W, H, F)},settb=AVTB[n{i}]")
        chain.append(f"[{i}:a]{AN}[a{i}]")
    ap = "a0"
    for i in range(1, k):
        off = round(off + m(clips[i - 1])["dur"] - d, 3)
        chain.append(f"[{prev}][n{i}]xfade=transition={tr}:duration={d}:offset={off}[x{i}]")
        chain.append(f"[{ap}][a{i}]acrossfade=d={d}[y{i}]")
        prev, ap = f"x{i}", f"y{i}"
    T = sum(m(c)["dur"] for c in clips) - d * (k - 1)
    inputs = " ".join(f"-i {q(n)}" for n in names)
    cmd = (f"ffmpeg {inputs} -filter_complex \"{';'.join(chain)}\" -map \"[{prev}]\" -map \"[{ap}]\" {ENC} {AENC} "
           f"{o('joined.mp4')}")
    think = (f"Normalize all clips to {W}x{H} {F}fps (xfade needs matching size/fps/timebase), then chain xfade; each offset is "
             f"the running length minus {d}s. acrossfade does the same for the audio.")
    return dict(files=files, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "dur": [T - 0.4, T + 0.5]}], steps=[(think, cmd)])


@gen("A6")
def voiceover_at(r):
    vo = r.choice(VOICES)
    an = name(r, ANAMES, ".m4a")
    L = m(vo)["dur"]
    mode = r.choice(["silent", "duck", "replace", "mix"])
    fx = r.choice(VID_SILENT) if mode == "silent" else r.choice(list(VID_TONE))
    M = m(fx)
    D = M["dur"]
    vn = name(r, VNAMES, ext_of(fx))
    T = r1(min(r.choice([1, 2, 3, 4, 5, 6, 8]), D - L - 0.5))
    ms = int(T * 1000)
    spec = {"match": "*.mp4", "dur": [D - 0.3, D + 0.4], "loud_at": [[T + 0.3, T + L - 0.4, -45]]}
    if mode == "silent":
        req = ask(r, "add {a} as a voiceover starting at {t}s in {v} (it has no sound)", "put my narration {a} on {v} at {t} seconds",
                  a=an, v=vn, t=T)
        think = f"{vn} has no audio stream: delay {an} by {ms} ms with adelay and pad it to the video length."
        cmd = f"ffmpeg -i {q(vn)} -i {q(an)} -filter_complex \"[1:a]adelay={ms}:all=1,apad[a]\" -map 0:v -map \"[a]\" -c:v copy {AENC} -shortest {o(stem(vn) + '_vo.mp4')}"
        spec["quiet_at"] = [[0.2, max(0.3, T - 0.3), -60]] if T > 0.8 else []
    else:
        hz = VID_TONE[fx]
        if mode == "duck":
            g = r.choice([0.1, 0.15, 0.2, 0.25])
            req = ask(r, "put {a} over {v} starting at {t}s and turn the original sound down while it plays",
                      "voiceover {a} at {t} seconds in {v}, lower the background under my voice", a=an, v=vn, t=T)
            think = f"Lower the original to {g}x only between {T}s and {r1(T + L)}s with a volume expression, delay {an} by {ms} ms, mix without normalizing."
            vol = f"volume='if(between(t,{T},{r1(T + L + 0.3)}),{g},1)':eval=frame"
            spec["band_drop"] = [hz, [T + 0.3, T + L - 0.3], [max(0, T - 1.5), max(0.5, T - 0.3)] if T > 1 else [T + L + 0.6, min(D, T + L + 2)], 8]
        elif mode == "replace":
            req = ask(r, "replace the audio from {t}s to {e}s of {v} with {a}, keep the rest", "swap the sound in {v} between {t} and {e} seconds for {a}",
                      a=an, v=vn, t=T, e=r1(T + L))
            think = f"Mute the original between {T}s and {r1(T + L)}s, delay {an} by {ms} ms into that gap, mix the two."
            vol = f"volume='if(between(t,{T},{r1(T + L)}),0,1)':eval=frame"
            spec["band_drop"] = [hz, [T + 0.3, T + L - 0.3], [max(0, T - 1.5), max(0.5, T - 0.3)] if T > 1 else [T + L + 0.6, min(D, T + L + 2)], 20]
        else:
            req = ask(r, "add {a} on top of the existing sound of {v} at {t}s", "mix my line {a} into {v} at {t} seconds", a=an, v=vn, t=T)
            think = f"Delay {an} by {ms} ms and mix it over the original audio (normalize=0 keeps the original level)."
            vol = "anull"
        cmd = (f"ffmpeg -i {q(vn)} -i {q(an)} -filter_complex \"[0:a]{AN},{vol}[bg];[1:a]{AN},adelay={ms}:all=1[v];"
               f"[bg][v]amix=inputs=2:duration=first:normalize=0[a]\" -map 0:v -map \"[a]\" -c:v copy {AENC} {o(stem(vn) + '_vo.mp4')}")
    return dict(files={vn: fx, an: vo}, ask=req, spec=[spec], steps=[(think, cmd)])


def atempo(f):
    parts = []
    while f > 2:
        parts.append(2.0)
        f /= 2
    while f < 0.5:
        parts.append(0.5)
        f /= 0.5
    parts.append(round(f, 4))
    return ",".join(f"atempo={p:g}" for p in parts)


@gen("A7")
def partial_speed(r):
    fx = r.choice(["cuts.mp4", "talk.mp4", "land.mp4", "short.mp4", "small.mp4"])
    M = m(fx)
    D, W, H = M["dur"], M["w"], M["h"]
    vn = name(r, VNAMES, ext_of(fx))
    a = r1(r.choice([1, 2, 3, 4, 5, 6]))
    b = r1(r.choice([x for x in (D, a + 3, a + 4, a + 6) if a + 1 < x <= D]))
    f = r.choice([2, 3, 4, 0.5, 1.5])
    with_audio = r.random() < 0.5
    word = f"{f}x faster" if f > 1 else f"{f}x speed (slow motion)"
    span = f"from {a}s onwards" if b == D else f"between {a}s and {b}s"
    req = ask(r, "make the part of {v} {s} play at {w}, rest normal" + ("" if with_audio else ", video only is fine"),
              "speed ramp: {v} {s} at {w}" + (", keep the sound in sync" if with_audio else ", no audio needed"), v=vn, s=span, w=word)
    segs = [(0, a, 1), (a, b, f)] + ([(b, D, 1)] if b < D else [])
    vparts = ";".join(f"[0:v]trim={s}:{e},setpts=(PTS-STARTPTS)/{k}[v{i}]" for i, (s, e, k) in enumerate(segs))
    T = sum((e - s) / k for s, e, k in segs)
    if with_audio:
        aparts = ";".join(f"[0:a]atrim={s}:{e},asetpts=PTS-STARTPTS{',' + atempo(k) if k != 1 else ''}[a{i}]" for i, (s, e, k) in enumerate(segs))
        cat = "".join(f"[v{i}][a{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=1[v][a]"
        cmd = f"ffmpeg -i {q(vn)} -filter_complex \"{vparts};{aparts};{cat}\" -map \"[v]\" -map \"[a]\" {ENC} {AENC} {o(stem(vn) + '_ramp.mp4')}"
    else:
        cat = "".join(f"[v{i}]" for i in range(len(segs))) + f"concat=n={len(segs)}:v=1:a=0[v]"
        cmd = f"ffmpeg -i {q(vn)} -filter_complex \"{vparts};{cat}\" -map \"[v]\" -an {ENC} {o(stem(vn) + '_ramp.mp4')}"
    think = (f"Split into segments with trim, divide the timestamps of the {a}-{b}s part by {f} (setpts), " +
             ("atempo the same part (chained, each step within 0.5-2), " if with_audio else "") + "then concat.")
    return dict(files={vn: fx}, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "dur": [T - 0.35, T + 0.4]}], steps=[(think, cmd)])


@gen("A8")
def combined(r):
    kind = r.choice(["speed_pitch", "slow_deep", "trim_speed_gif", "vertical_trim_loud", "speed_only_keep_pitch"])
    fx = r.choice(VID_SPEECH)
    M = m(fx)
    D, W, H, F = M["dur"], M["w"], M["h"], int(M["fps"])
    vn = name(r, VNAMES, ext_of(fx))
    if kind in ("speed_pitch", "slow_deep", "speed_only_keep_pitch"):
        if kind == "speed_pitch":
            f, p = r.choice([1.25, 1.5, 2.0]), r.choice([1.3, 1.4, 1.5])
            req = ask(r, "speed {v} up to {f}x and make the voice higher, like a female voice", "{f}x speed and a higher pitched voice for {v}",
                      "make {v} {f} times faster with a chipmunk-ish voice", v=vn, f=f)
            spec_p, why = [p * 0.82, p * 1.2], f"rubberband changes tempo ({f}) and pitch ({p}) independently"
        elif kind == "slow_deep":
            f, p = r.choice([0.75, 0.8, 0.9]), r.choice([0.7, 0.75, 0.8])
            req = ask(r, "slow {v} down to {f}x and make the voice deeper", "{f}x speed, deeper voice like a giant, for {v}", v=vn, f=f)
            spec_p, why = [p * 0.82, p * 1.15], f"rubberband slows the tempo to {f} and lowers the pitch to {p}"
        else:
            f, p = r.choice([1.25, 1.5, 2.0]), 1.0
            req = ask(r, "speed up {v} {f}x but keep the voice sounding normal (not chipmunk)", "{f}x faster {v}, same voice pitch", v=vn, f=f)
            spec_p, why = [0.88, 1.13], "atempo changes speed without changing pitch"
        af = f"rubberband=tempo={f}:pitch={p}" if p != 1.0 else atempo(f)
        cmd = (f"ffmpeg -i {q(vn)} -filter_complex \"[0:v]setpts=PTS/{f}[v];[0:a]{af}[a]\" -map \"[v]\" -map \"[a]\" -r {F} {ENC} {AENC} "
               f"{o(stem(vn) + '_edit.mp4')}")
        think = f"setpts=PTS/{f} for the video at a constant {F} fps (no fps filter on the speed); {why}."
        T = D / f
        spec = {"match": "*.mp4", "wh": [W, H], "dur": [T - 0.35, T + 0.4], "fps": F, "cfr": True, "pitch_ratio": [f"in/{vn}", *spec_p]}
    elif kind == "trim_speed_gif":
        a = r.choice([0, 1, 2, 3])
        b = a + r.choice([4, 6, 8])
        f = r.choice([2, 1.5])
        req = ask(r, "take {a}s to {b}s of {v}, play it at {f}x and make it a gif", "gif from {v}: {a}-{b} seconds, {f}x speed", v=vn, a=a, b=b, f=f)
        cmd = f"ffmpeg -ss {a} -to {b} -i {q(vn)} -vf \"setpts=PTS/{f},fps=12,scale=480:-1:flags=lanczos\" {o(stem(vn) + '.gif')}"
        think = f"-ss/-to cut {a}-{b}s, setpts=PTS/{f} speeds it up, fps=12 and 480px wide keep the gif small."
        T = (b - a) / f
        spec = {"match": "*.gif", "dur": [T - 0.4, T + 0.5]}
    else:
        n = r.choice([5, 8, 10])
        L = r.choice([-14, -16])
        cw = even(H * 9 / 16)
        req = ask(r, "vertical 9:16 version of the first {n} seconds of {v} at {l} LUFS", "first {n}s of {v}, vertical, loudness {l} LUFS",
                  v=vn, n=n, l=L)
        cmd = (f"ffmpeg -t {n} -i {q(vn)} -vf crop={cw}:{H},scale=1080:1920 -af loudnorm=I={L}:TP=-1.5 -ar 48000 {ENC} {AENC} "
               f"{o(stem(vn) + '_vertical.mp4')}")
        think = f"-t {n} trims, crop={cw}:{H} takes a centered 9:16 slice, scale to 1080x1920; loudnorm to {L} LUFS, -ar 48000 undoes its upsampling."
        spec = {"match": "*.mp4", "aspect": [9, 16], "min_short": 600, "dur": [n - 0.3, n + 0.3], "lufs": [L - 2, L + 2]}
    return dict(files={vn: fx}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("A9")
def pip(r):
    main = r.choice(["screenrec.mp4", "cuts.mp4", "talk.mp4", "land.mp4"])
    cam = r.choice([x for x in ["webcam.mp4", "lowres.mp4", "version_a.mp4", "small.mp4"] if x != main])
    M = m(main)
    W, H, D = M["w"], M["h"], M["dur"]
    mn = name(r, ["screen", "gameplay", "slides", "lecture", "recording", "main"], ext_of(main))
    cn = name(r, ["cam", "webcam", "facecam", "me", "face", "selfie"], ext_of(cam), avoid={mn})
    corner = r.choice([("bottom right", "W-w-{p}:H-h-{p}"), ("bottom left", "{p}:H-h-{p}"), ("top right", "W-w-{p}:{p}"), ("top left", "{p}:{p}")])
    frac = r.choice([(0.25, "a quarter of the width"), (0.3, "small"), (0.33, "a third of the width"), (0.2, "tiny")])
    cw, pad = even(W * frac[0]), r.choice([10, 20, 30])
    a_src = r.choice(["main", "cam"]) if M.get("has_audio") and m(cam).get("has_audio") else ("main" if M.get("has_audio") else "cam")
    req = ask(r, "put {c} in the {k} corner of {m}, {f}, with the {a} audio", "picture in picture: {m} full screen, {c} {f} in the {k}",
              "overlay my {c} on {m} ({k}, {f})", c=cn, m=mn, k=corner[0], f=frac[1], a="webcam's" if a_src == "cam" else "main video's")
    pos = corner[1].format(p=pad)
    amap = "1:a" if a_src == "cam" else "0:a"
    sh = ":shortest=1" if m(cam)["dur"] < D else ""
    cmd = (f"ffmpeg -i {q(mn)} -i {q(cn)} -filter_complex \"[1:v]scale={cw}:-2[c];[0:v][c]overlay={pos}{sh}[v]\" -map \"[v]\" "
           f"-map {amap} {ENC} {AENC} {o(stem(mn) + '_pip.mp4')}")
    think = f"Scale {cn} to {cw}px wide and overlay it {corner[0]} with a {pad}px margin; audio from {amap.split(':')[0] == '1' and cn or mn}."
    T = min(D, m(cam)["dur"]) if sh else D
    spec = {"match": "*.mp4", "wh": [W, H], "dur": [T - 0.4, max(D, T) + 0.4], "frame_like": [[2, f"in/{mn}@2", 0.35]],
            "frame_unlike": [[2, f"in/{mn}@2", 0.004]], "has_audio": True}
    return dict(files={mn: main, cn: cam}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("A10")
def side_by_side(r):
    a, b = r.sample(["version_a.mp4", "version_b.mp4", "dark.mp4", "talk.mp4", "cuts.mp4", "lowres.mp4"], 2)
    an, bn = r.choice([("before.mp4", "after.mp4"), ("old.mp4", "new.mp4"), ("raw.mp4", "graded.mp4"), ("v1.mp4", "v2.mp4"), ("left.mp4", "right.mp4")])
    vert = r.random() < 0.3
    labels = r.random() < 0.4
    h = r.choice([540, 720]) if not vert else r.choice([360, 540])
    w = even(h * 16 / 9)
    req = ask(r, ("stack {a} on top of {b}" if vert else "show {a} and {b} side by side") + (", labelled" if labels else "") + " so I can compare",
              ("{a} above {b} in one video" if vert else "split screen: {a} left, {b} right") + (" with labels" if labels else ""), a=an, b=bn)
    lab = lambda i, t: f",drawtext=text={t}:fontcolor=white:fontsize={h // 10}:x=20:y=20:box=1:boxcolor=black@0.5" if labels else ""
    st = "vstack" if vert else "hstack"
    cmd = (f"ffmpeg -i {q(an)} -i {q(bn)} -filter_complex \"[0:v]scale={w}:{h},setsar=1{lab(0, an.split('.')[0].upper())}[a];"
           f"[1:v]scale={w}:{h},setsar=1{lab(1, bn.split('.')[0].upper())}[b];[a][b]{st}=shortest=1[v]\" -map \"[v]\" {ENC} {o('compare.mp4')}")
    think = f"Scale both to {w}x{h} (stacking needs equal {'widths' if vert else 'heights'}), " + ("label them with drawtext, " if labels else "") + f"{st}; shortest=1 stops at the shorter one."
    T = min(m(a)["dur"], m(b)["dur"])
    wh = [w, 2 * h] if vert else [2 * w, h]
    return dict(files={an: a, bn: b}, ask=req, spec=[{"match": "*.mp4", "wh": wh, "dur": [T - 0.4, T + 0.4]}], steps=[(think, cmd)])


@gen("A11")
def freeze(r):
    fx = r.choice(["talk.mp4", "cuts.mp4", "land.mp4", "short.mp4", "webcam.mp4"])
    M = m(fx)
    D, W, H, F = M["dur"], M["w"], M["h"], int(M["fps"])
    vn = name(r, VNAMES, ext_of(fx))
    n = r.choice([1.5, 2, 3])
    where = r.choice(["mid", "mid", "end", "start"])
    if where == "end":
        req = ask(r, "hold the last frame of {v} for {n} more seconds", "freeze the ending of {v} for {n}s", v=vn, n=n)
        cmd = f"ffmpeg -i {q(vn)} -vf tpad=stop_duration={n}:stop_mode=clone -af apad=pad_dur={n} {ENC} {AENC} {o(stem(vn) + '_hold.mp4')}"
        think, fr = f"tpad clones the last frame for {n}s; apad adds matching silence.", [D + 0.15, D + n - 0.15]
    elif where == "start":
        req = ask(r, "start {v} frozen on its first frame for {n} seconds", "hold the first frame of {v} for {n}s before it plays", v=vn, n=n)
        cmd = f"ffmpeg -i {q(vn)} -vf tpad=start_duration={n}:start_mode=clone -af adelay={int(n * 1000)}:all=1 {ENC} {AENC} {o(stem(vn) + '_hold.mp4')}"
        think, fr = f"tpad clones the first frame for {n}s; adelay shifts the audio by the same {int(n * 1000)} ms.", [0.15, n - 0.15]
    else:
        T = r1(r.choice([2, 3, 5, 6, 8]) if D > 10 else r.choice([2, 3, 4]))
        quiet = r.random() < 0.5
        req = ask(r, "freeze the picture at {t}s of {v} for {n} seconds, then carry on" + (", no sound during the freeze" if quiet else ""),
                  "dramatic pause: at {t} seconds in {v}, freeze for {n}s", v=vn, t=T, n=n)
        cmd = (f"ffmpeg -i {q(vn)} -filter_complex \"[0:v]trim=0:{T},setpts=PTS-STARTPTS[a];[0:v]trim={T},setpts=PTS-STARTPTS,"
               f"tpad=start_duration={n}:start_mode=clone[b];[a][b]concat=n=2:v=1:a=0[v];[0:a]atrim=0:{T},asetpts=PTS-STARTPTS[x];"
               f"[0:a]atrim={T},asetpts=PTS-STARTPTS,adelay={int(n * 1000)}:all=1[y];[x][y]concat=n=2:v=0:a=1[o]\" -map \"[v]\" -map \"[o]\" "
               f"{ENC} {AENC} {o(stem(vn) + '_freeze.mp4')}")
        think = f"Split at {T}s; the second half starts with its first frame cloned for {n}s (tpad) and its audio delayed {int(n * 1000)} ms, so sound resumes with the picture."
        fr = [T + 0.15, T + n - 0.15]
    return dict(files={vn: fx}, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "dur": [D + n - 0.35, D + n + 0.4], "freeze": fr}],
                steps=[(think, cmd)])


# ═════════════════════════ B. social deliverables ═════════════════════════

ASPECTS = {"16:9": (16, 9), "9:16": (9, 16), "1:1": (1, 1), "4:5": (4, 5)}
PLAT = {"youtube": "16:9", "reels": "9:16", "tiktok": "9:16", "shorts": "9:16", "story": "9:16", "square": "1:1", "feed": "4:5", "instagram post": "1:1"}


def crop_to(W, H, aw, ah):
    if W / H > aw / ah:
        return even(H * aw / ah), even(H)
    return even(W), even(W * ah / aw)


@gen("B12")
def multi_platform(r):
    fx = r.choice(["talk.mp4", "land.mp4", "cuts.mp4", "phone.mov", "webcam.mp4"])
    M = m(fx)
    W, H, D = M["w"], M["h"], M["dur"]
    vn = name(r, VNAMES, ext_of(fx))
    plats = r.sample(sorted(PLAT), r.choice([2, 3]))
    while len({PLAT[p] for p in plats}) < len(plats):
        plats = r.sample(sorted(PLAT), r.choice([2, 3]))
    outs = {p: p.replace(" ", "_") + ".mp4" for p in plats}
    req = ask(r, "make {l} versions of {v}", "I need {v} for {l}", "export {v} for {l}", v=vn,
              l=", ".join(f"{p} ({PLAT[p]}, {outs[p]})" for p in plats))
    steps, specs = [], []
    for p in plats:
        aw, ah = ASPECTS[PLAT[p]]
        cw, ch = crop_to(W, H, aw, ah)
        tw = 1080 if aw <= ah else 1920
        th = even(tw * ah / aw)
        vf = f"crop={cw}:{ch},scale={tw}:{th},setsar=1"
        steps.append((f"{p}: center-crop {cw}x{ch} ({PLAT[p]}) from {W}x{H} and scale to {tw}x{th}.",
                      f"ffmpeg -i {q(vn)} -vf {vf} {ENC} {AENC} {o(outs[p])}"))
        specs.append({"match": outs[p], "aspect": [aw, ah], "min_short": 600, "dur": [D - 0.3, D + 0.3]})
    return dict(files={vn: fx}, ask=req, spec=specs, steps=steps)


@gen("B13")
def size_limit(r):
    fx = r.choice(["talk.mp4", "land.mp4", "big.mov", "interlaced.mp4", "version_a.mp4", "phone.mov", "small.mp4"])
    M = m(fx)
    W, H, D = M["w"], M["h"], M["dur"]
    vn = name(r, VNAMES, ext_of(fx))
    cur = M["size"] / 1e6
    target = r.choice([x for x in (1, 2, 3, 5, 8, 10, 16, 25) if x < max(cur * 0.8, 1.5)] or [max(1, round(cur / 3))])
    why = r.choice(["whatsapp", "email", "discord", "the upload form", "slack"])
    req = ask(r, "{v} has to be under {t} MB for {w}, keep the whole thing", "get {v} below {t}MB so I can send it on {w}",
              "compress {v} to less than {t} MB", v=vn, t=target, w=why)
    ab = 64 if target < 4 else 96
    total = target * 8000 * 0.9 / D
    vb = int(total - ab)
    sw = W if vb > 2500 or W <= 1280 else 1280
    if vb < 700 and sw > 854:
        sw = 854
    vf = f" -vf scale={even(sw)}:-2" if sw < W else ""
    if vb < 100:
        return None
    inter = ",yadif" if fx == "interlaced.mp4" else ""
    if inter:
        vf = f" -vf yadif,scale={even(sw)}:-2" if sw < W else " -vf yadif"
    cmd = (f"ffmpeg -i {q(vn)}{vf} -c:v libx264 -preset veryfast -b:v {vb}k -maxrate {vb}k -bufsize {2 * vb}k -pix_fmt yuv420p "
           f"-c:a aac -b:a {ab}k -movflags +faststart {o(stem(vn) + '_small.mp4')}")
    think = (f"{target} MB over {D:.1f}s is ~{int(total)} kb/s total with 10% headroom: {ab}k audio, {vb}k video, capped with maxrate"
             + (f"; scale to {even(sw)}px wide so that bitrate still looks clean" if vf else "") + ".")
    spec = {"match": "*.mp4", "max_mb": target, "dur": [D - 0.3, D + 0.3], "vcodec": "h264"}
    return dict(files={vn: fx}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("B14")
def split_chunks(r):
    fx = r.choice(["talk.mp4", "cuts.mp4", "land.mp4", "small.mp4", "webcam.mp4"])
    M = m(fx)
    D = M["dur"]
    vn = name(r, VNAMES, ext_of(fx))
    if r.random() < 0.6:
        n = r.choice([s for s in (5, 10, 15, 30) if s < D])
        k = -(-int(D * 1000) // (n * 1000))
        req = ask(r, "split {v} into {n} second clips", "cut {v} into {n}s pieces for my status/stories", "chop {v} into {n} second parts",
                  v=vn, n=n)
        cmd = (f"ffmpeg -i {q(vn)} {ENC} -force_key_frames 'expr:gte(t,n_forced*{n})' {AENC} -f segment -segment_time {n} "
               f"-segment_time_delta 0.05 -reset_timestamps 1 {o(stem(vn) + '_%02d.mp4')}")
        think = f"Re-encode with a keyframe every {n}s so the segment muxer can cut exactly at {n}s (segment_time_delta absorbs timestamp rounding); {D:.0f}s gives {k} parts."
        spec = {"match": "*.mp4", "count": k, "dur": [0.3, n + 0.4]}
        steps = [(think, cmd)]
    else:
        k = r.choice([2, 3, 4])
        L = round(D / k, 3)
        req = ask(r, "cut {v} into {k} equal parts", "split {v} in {k} same-length pieces", v=vn, k=k)
        steps = []
        for i in range(k):
            ss = f"-ss {round(i * L, 3)} " if i else ""
            tt = f"-t {L} " if i < k - 1 else ""
            steps.append((f"Part {i + 1} of {k}: {round(i * L, 2)}s, {L:.2f}s long." if i else f"{D:.1f}s / {k} = {L:.2f}s per part; re-encode each so the cuts are exact.",
                          f"ffmpeg {ss}-i {q(vn)} {tt}{ENC} {AENC} {o(f'{stem(vn)}_part{i + 1}.mp4')}"))
        spec = {"match": "*.mp4", "count": k, "dur": [L - 0.4, L + 0.4]}
    return dict(files={vn: fx}, ask=req, spec=[spec], steps=steps)


CAPTIONS = ["Full episode out now", "Link in bio", "Wait for it...", "New video!", "Watch till the end", "Episode 12", "Swipe up"]


@gen("B15")
def highlight(r):
    fx = r.choice(["talk.mp4", "land.mp4", "webcam.mp4", "cuts.mp4"])
    M = m(fx)
    W, H, D = M["w"], M["h"], M["dur"]
    vn = name(r, VNAMES, ext_of(fx))
    a = r.choice([0, 2, 4, 6, 8])
    b = min(D, a + r.choice([3, 4, 5, 6]))
    shape = r.choice(["9:16", "9:16", "1:1"])
    aw, ah = ASPECTS[shape]
    cw, ch = crop_to(W, H, aw, ah)
    tw, th = (1080, 1920) if shape == "9:16" else (1080, 1080)
    files = {vn: fx}
    if fx == "talk.mp4" and r.random() < 0.5:
        sn = stem(vn) + ".srt"
        files[sn] = "captions.srt"
        req = ask(r, "make a {s} highlight of {a}-{b}s from {v} with the subtitles from {sr} burned in", "{s} clip of {v} from {a} to {b} seconds, captions from {sr}",
                  s=shape, a=a, b=b, v=vn, sr=sn)
        vf = f"subtitles=in/{sn},crop={cw}:{ch},scale={tw}:{th}"
        cmd = f"ffmpeg -ss {a} -to {b} -copyts -i {q(vn)} -vf \"{vf}\" -output_ts_offset -{a} {ENC} {AENC} {o(stem(vn) + '_highlight.mp4')}"
        think = f"-copyts keeps the original timestamps so the subtitles line up after seeking to {a}s; burn them before cropping to {shape}, then reset the start."
    else:
        cap = r.choice(CAPTIONS)
        req = ask(r, "teaser from {v}: {a}s to {b}s, {s}, with the caption '{c}' at the bottom", "short {s} clip of {v} ({a}-{b}s) that says '{c}'",
                  s=shape, a=a, b=b, v=vn, c=cap)
        vf = f"crop={cw}:{ch},scale={tw}:{th},drawtext=text='{cap}':fontcolor=white:fontsize=56:borderw=3:x=(w-text_w)/2:y=h-160"
        cmd = f"ffmpeg -ss {a} -to {b} -i {q(vn)} -vf \"{vf}\" {ENC} {AENC} {o(stem(vn) + '_teaser.mp4')}"
        think = f"Seek {a}-{b}s, center-crop to {shape} ({cw}x{ch}), scale to {tw}x{th}, drawtext the caption near the bottom."
    return dict(files=files, ask=req, spec=[{"match": "*.mp4", "aspect": [aw, ah], "min_short": 600, "dur": [b - a - 0.35, b - a + 0.35]}],
                steps=[(think, cmd)])


@gen("B16")
def styled_subs(r):
    fx = "talk.mp4"
    vn = name(r, VNAMES, ".mp4")
    sn = r.choice(["subs.srt", "captions.srt", stem(vn) + ".srt", "lines.srt", "transcript.srt"])
    styles = [("big yellow text", "Fontsize=30,PrimaryColour=&H00FFFF&"), ("white with a thick black outline", "Outline=3,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&"),
              ("a black box behind the text", "BorderStyle=3,Outline=1,Shadow=0,BackColour=&H80000000&"), ("at the top of the screen", "Alignment=8,Fontsize=24"),
              ("small and subtle", "Fontsize=16"), ("bold, large", "Fontsize=32,Bold=1")]
    desc, st = r.choice(styles)
    req = ask(r, "burn {s} into {v}, {d}", "hardcode the subtitles from {s} on {v} ({d})", "put {s} on {v} as burned-in captions, {d}", s=sn, v=vn, d=desc)
    cmd = f"ffmpeg -i {q(vn)} -vf \"subtitles=in/{sn}:force_style='{st}'\" {ENC} -c:a copy {o(stem(vn) + '_subs.mp4')}"
    think = f"The subtitles filter renders the SRT onto the frames; force_style ({st}) gives {desc}. Audio is copied."
    sp = TRUTH["speech"]
    spec = {"match": "*.mp4", "wh": [1280, 720], "dur": [15.7, 16.4], "frame_unlike": [[r1((sp[1][0] + sp[1][1]) / 2), f"in/{vn}@{r1((sp[1][0] + sp[1][1]) / 2)}", 0.003]],
            "frame_like": [[r1((sp[0][1] + sp[1][0]) / 2), f"in/{vn}@{r1((sp[0][1] + sp[1][0]) / 2)}", 0.0025]]}
    return dict(files={vn: fx, sn: "captions.srt"}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("B17")
def thumbnail(r):
    fx = r.choice(["talk.mp4", "land.mp4", "cuts.mp4", "webcam.mp4", "version_a.mp4"])
    M = m(fx)
    vn = name(r, VNAMES, ext_of(fx))
    t = r.choice([1, 2, 3, 5])
    title = r.choice(TITLES + ["My Trip", "EP 12", "WATCH THIS", "Day in my life"])
    size = r.choice([("youtube thumbnail", 1280, 720), ("1080x1080 cover", 1080, 1080), ("full size", M["w"], M["h"])])
    ext = r.choice([".jpg", ".png"])
    req = ask(r, "{k} from {v} at {t}s with the title '{x}'", "grab a frame from {v} at {t} seconds and put '{x}' on it ({k}, {e})",
              k=size[0], v=vn, t=t, x=title, e=ext[1:])
    W, H = size[1], size[2]
    cw, ch = crop_to(M["w"], M["h"], W, H)
    vf = f"crop={cw}:{ch},scale={W}:{H},drawtext=text='{title}':fontcolor=white:fontsize={H // 7}:borderw=5:x=(w-text_w)/2:y=(h-text_h)/2"
    cmd = f"ffmpeg -ss {t} -i {q(vn)} -frames:v 1 -vf \"{vf}\"{' -q:v 2' if ext == '.jpg' else ''} {o(stem(vn) + '_thumb' + ext)}"
    think = f"Seek to {t}s, take one frame, crop to {W}:{H} shape and scale, drawtext the title with an outline so it reads on any background."
    return dict(files={vn: fx}, ask=req, spec=[{"match": "*" + ext, "wh": [W, H]}], steps=[(think, cmd)])


@gen("B18")
def audiogram(r):
    au = r.choice(["narration.m4a", "music_bed.mp3", "voice_m.m4a", "a.m4a", "music.mp3"])
    img = r.choice(["cover.jpg", "photo.jpg", "title.png"])
    an, im = name(r, ["episode", "clip", "snippet", "show", "podcast", "ep12"], ext_of(au)), name(r, ["art", "cover", "artwork", "bg"], ext_of(img))
    D = m(au)["dur"]
    shape = r.choice([("square 1080", 1080, 1080), ("9:16", 1080, 1920), ("16:9", 1920, 1080)])
    wave = r.random() < 0.7
    n = r.choice([None, 10, 15]) if D > 16 else None
    req = ask(r, "make a {s} video for {a} with {i} as the background" + (" and a moving waveform" if wave else "") + ("" if not n else ", first {n} seconds"),
              "audiogram: {a} over {i}, {s}" + (", with waveform" if wave else "") + ("" if not n else ", just the first {n}s"), s=shape[0], a=an, i=im, n=n)
    W, H = shape[1], shape[2]
    T = min(D, n) if n else D
    tt = f"-t {n} " if n else ""
    bg = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[bg]"
    if wave:
        fc = f"{bg};[1:a]showwaves=s={W}x{H // 5}:mode=cline:colors=white[w];[bg][w]overlay=0:{H - H // 4}:shortest=1,format=yuv420p[v]"
        think = f"Loop the image as a {W}x{H} background, draw showwaves from the audio and overlay it near the bottom; -shortest ends with the audio."
    else:
        fc = f"{bg};[bg]format=yuv420p[v]"
        think = f"Loop the image scaled/cropped to {W}x{H}; the video lasts as long as the audio (-shortest)."
    cmd = f"ffmpeg -loop 1 -framerate 25 -i {q(im)} {tt}-i {q(an)} -filter_complex \"{fc}\" -map \"[v]\" -map 1:a -shortest {ENC} {AENC} {o(stem(an) + '.mp4')}"
    return dict(files={an: au, im: img}, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "has_audio": True, "dur": [T - 0.4, T + 0.6]}],
                steps=[(think, cmd)])


# ═════════════════════════ C. audio & voice ═════════════════════════

@gen("C19")
def voice_fx(r):
    au = r.choice(["voice_m.m4a", "voice_f.m4a", "narration.m4a"])
    an = name(r, ANAMES + ["memo", "announcement", "call"], ext_of(au))
    D = m(au)["dur"]
    fx = r.choice(["higher", "chipmunk", "deeper", "trailer", "echo", "telephone"])
    ext = r.choice([".m4a", ".mp3", ".wav"])
    out = o(stem(an) + "_" + fx + ext)
    ac = {".m4a": "-c:a aac", ".mp3": "-c:a libmp3lame -b:a 192k", ".wav": ""}[ext]
    spec = {"match": "*" + ext, "has_audio": True}
    if fx in ("higher", "chipmunk", "deeper", "trailer"):
        p = {"higher": r.choice([1.25, 1.33]), "chipmunk": r.choice([1.7, 1.8, 2.0]), "deeper": r.choice([0.8, 0.75]), "trailer": r.choice([0.65, 0.7])}[fx]
        req = ask(r, {"higher": "make the voice in {a} higher pitched", "chipmunk": "make {a} sound like a chipmunk", "deeper": "make my voice in {a} deeper",
                      "trailer": "make {a} sound like a movie trailer voice, really deep"}[fx], a=an)
        cmd = f"ffmpeg -i {q(an)} -af rubberband=pitch={p} {ac} {out}"
        think = f"rubberband pitch={p} shifts the pitch without changing the length (asetrate would change the speed too)."
        spec.update(dur=[D - 0.3, D + 0.3], pitch_ratio=[f"in/{an}", p * 0.8, p * 1.2])
    elif fx == "echo":
        if au != "narration.m4a":
            return None
        req = ask(r, "add an echo to {a}, like a big hall", "give {a} a reverb/echo effect", a=an)
        cmd = f"ffmpeg -i {q(an)} -af 'aecho=0.8:0.85:300|600:0.4|0.25' {ac} {out}"
        think = "aecho with two taps (300 and 600 ms) and decays 0.4/0.25 gives a hall-like echo; the | needs quoting."
        e = TRUTH["speech"][0][1]
        spec.update(dur=[D - 0.3, D + 1.5], loud_at=[[e + 0.1, e + 0.5, -65]])
    else:
        req = ask(r, "make {a} sound like it's coming through a phone", "old telephone effect on {a}", a=an)
        cmd = f"ffmpeg -i {q(an)} -af highpass=f=300,lowpass=f=3400,lowpass=f=3400 {ac} {out}"
        think = "A telephone line passes roughly 300-3400 Hz: highpass + a doubled lowpass for a steeper cut."
        spec.update(dur=[D - 0.3, D + 0.3], band_max=[5000, -70])
    return dict(files={an: au}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("C20")
def voice_clean(r):
    an = name(r, ["memo", "podcast_raw", "note", "recording", "interview_audio", "voice"], ".wav")
    hz = TRUTH["hum_hz"]
    kind = r.choice(["hum", "hum_loud", "hum_hiss", "probe"])
    ext = r.choice([".wav", ".mp3", ".m4a"])
    ac = {".m4a": "-c:a aac", ".mp3": "-c:a libmp3lame -b:a 160k", ".wav": ""}[ext]
    out = o(stem(an) + "_clean" + ext)
    base = f"bandreject=f={hz}:width_type=q:w=1,highpass=f={int(hz * 2.4)},highpass=f={int(hz * 2.4)}"
    spec = {"match": "*" + ext, "has_audio": True, "band_max": [hz, -60], "loud_at": [[0.5, 2.0, -40]]}
    steps = []
    if kind == "probe":
        req = ask(r, "there's a buzz in {a}, can you find what it is and remove it?", "something hums in the background of {a}, fix it", a=an)
        steps.append(("Measure first: a mains hum shows up as a steady low tone; astats/volume in a narrow band around 50 and 60 Hz will tell which.",
                      f"ffmpeg -i {q(an)} -af bandpass=f=50:width_type=q:w=30,volumedetect -f null -"))
        steps.append((f"The 50 Hz band is loud, so it's 50 Hz mains hum: notch it and high-pass at {int(hz * 2.4)} Hz (speech has little below that).",
                      f"ffmpeg -i {q(an)} -af {base} {ac} {out}"))
    else:
        req = ask(r, {"hum": "remove the hum from {a}", "hum_loud": "clean up {a}: get rid of the hum and bring it to podcast loudness",
                      "hum_hiss": "{a} has a low buzz and some hiss, clean it up"}[kind], a=an)
        chain = base + {"hum": "", "hum_loud": ",afftdn,loudnorm=I=-16:TP=-1.5", "hum_hiss": ",afftdn=nr=12"}[kind]
        ar = " -ar 48000" if kind == "hum_loud" else ""
        steps.append((f"The hum is {hz} Hz mains (fundamental + harmonics): bandreject at {hz} Hz plus a steep highpass at {int(hz * 2.4)} Hz"
                      + {"hum": ".", "hum_loud": ", then afftdn and loudnorm to -16 LUFS (-ar 48000 after loudnorm).", "hum_hiss": ", afftdn for the hiss."}[kind],
                      f"ffmpeg -i {q(an)} -af {chain}{ar} {ac} {out}"))
        if kind == "hum_loud":
            spec["lufs"] = [-19, -13]
    return dict(files={an: "hum.wav"}, ask=req, spec=[spec], steps=steps)


@gen("C21")
def ducking(r):
    base = r.choice(["talk.mp4", "webcam.mp4", "narration.m4a"])
    mu = "music_bed.mp3"  # its tone frequency is known (TRUTH["music_hz"]), so ducking can be measured
    is_v = base.endswith(".mp4")
    bn = name(r, VNAMES, ".mp4") if is_v else name(r, ANAMES, ".m4a")
    mn = name(r, MNAMES, ext_of(mu))
    lvl = r.choice([0.3, 0.4, 0.5])
    req = ask(r, "add {m} under {b} and make it duck whenever someone talks", "background music {m} for {b}, quieter under the voice",
              "mix {m} into {b} with ducking", b=bn, m=mn)
    fc = (f"[0:a]{AN},asplit=2[sc][vo];[1:a]{AN},volume={lvl}[m];[m][sc]sidechaincompress=threshold=0.02:ratio=12:attack=20:release=300[duck];"
          f"[vo][duck]amix=inputs=2:duration=first:normalize=0[a]")
    if is_v:
        cmd = f"ffmpeg -i {q(bn)} -i {q(mn)} -filter_complex \"{fc}\" -map 0:v -map \"[a]\" -c:v copy {AENC} {o(stem(bn) + '_music.mp4')}"
        ext = ".mp4"
    else:
        cmd = f"ffmpeg -i {q(bn)} -i {q(mn)} -filter_complex \"{fc}\" -map \"[a]\" -c:a libmp3lame -b:a 192k {o(stem(bn) + '_mix.mp3')}"
        ext = ".mp3"
    think = (f"sidechaincompress lowers the music ({lvl}x) whenever the voice (the sidechain, split from the original) is above "
             f"the threshold; amix with duration=first keeps the original length.")
    sp, pa = TRUTH["speech"], TRUTH["pauses"]
    spec = {"match": "*" + ext, "dur": [15.7, 16.5], "band_drop": [TRUTH["music_hz"], [sp[1][0] + 0.3, sp[1][1] - 0.3], [pa[1][0] + 0.5, pa[1][1] - 0.4], 4]}
    return dict(files={bn: base, mn: mu}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("C22")
def remove_pauses(r):
    base = r.choice(["talk.mp4", "narration.m4a", "webcam.mp4"])
    is_v = base.endswith(".mp4")
    bn = name(r, VNAMES, ".mp4") if is_v else name(r, ANAMES + ["take", "answer"], ".m4a")
    keep = r.choice([None, 0.3, 0.5])
    req = ask(r, "cut out the long pauses in {b}", "remove the silences from {b} so it's tighter", "my {b} has gaps where I stopped talking, " +
              ("shorten each to {k}s" if keep else "cut them"), b=bn, k=keep)
    probe = (f"Find the pauses first: silencedetect at -35 dB, at least 0.5 s.", f"ffmpeg -i {q(bn)} -af silencedetect=n=-35dB:d=0.5 -f null -")
    sp = TRUTH["speech"]
    pad = keep or 0.15
    kr = [(max(0, round(a - pad / 2, 2)), round(b + pad / 2, 2)) for a, b in sp]
    sel = ranges_expr(kr)
    total = sum(b - a for a, b in kr)
    think = (f"Keep the speech stretches " + ", ".join(f"{a}-{b}s" for a, b in kr) + f" (each padded {pad}s) and drop the rest; select/aselect + "
             "setpts/asetpts keep picture and sound together.")
    if is_v:
        cmd = (f"ffmpeg -i {q(bn)} -vf \"select='{sel}',setpts=N/FRAME_RATE/TB\" -af \"aselect='{sel}',asetpts=N/SR/TB\" {ENC} {AENC} "
               f"{o(stem(bn) + '_tight.mp4')}")
        ext = ".mp4"
    else:
        cmd = f"ffmpeg -i {q(bn)} -af \"aselect='{sel}',asetpts=N/SR/TB\" {AENC} {o(stem(bn) + '_tight.m4a')}"
        ext = ".m4a"
    spec = {"match": "*" + ext, "dur": [total - 0.5, total + 0.5], "max_silence": 1.0}
    return dict(files={bn: base}, ask=req, spec=[spec], steps=[probe, (think, cmd)])


@gen("C23")
def fix_sync(r):
    if r.random() < 0.55:
        vn = name(r, VNAMES + ["clap", "sync_test"], ".mp4")
        ms = TRUTH["sync_ms"]
        told = r.random() < 0.6
        req = (ask(r, "the audio in {v} is about {n} ms late, fix it", "sound lags the picture by roughly {n}ms in {v}", v=vn, n=ms) if told else
               ask(r, "lip sync is off in {v}, the sound comes late. the flash and the beep should line up", v=vn))
        steps = []
        if not told:
            steps.append(("Measure it: the first flash shows as a jump in brightness (signalstats YAVG), the first beep as the end of the first silence.",
                          f"ffmpeg -i {q(vn)} -af silencedetect=n=-30dB:d=0.05 -f null -"))
        steps.append((f"Audio is {ms} ms late: read the same file twice and take the audio {ms / 1000}s earlier (-ss on the second input), video untouched.",
                      f"ffmpeg -i {q(vn)} -ss {ms / 1000} -i {q(vn)} -map 0:v -map 1:a -c:v copy {AENC} -shortest {o(stem(vn) + '_synced.mp4')}"))
        spec = {"match": "*.mp4", "av_sync_ms": [-50, 50]}
        return dict(files={vn: "sync.mp4"}, ask=req, spec=[spec], steps=steps)
    fx = r.choice(VID_SPEECH)
    vn = name(r, VNAMES, ".mp4")
    ms = r.choice([100, 200, 250, 400, 500])
    req = ask(r, "delay the audio of {v} by {n} ms", "the sound in {v} is {n}ms early, push it back", v=vn, n=ms)
    cmd = f"ffmpeg -i {q(vn)} -af adelay={ms}:all=1 -c:v copy {AENC} {o(stem(vn) + '_delayed.mp4')}"
    spec = {"match": "*.mp4", "quiet_at": [[0.0, ms / 1000 - 0.03, -60]], "loud_at": [[ms / 1000 + 0.2, TRUTH["speech"][0][1] + ms / 1000 - 0.2, -45]]}
    return dict(files={vn: fx}, ask=req, spec=[spec], steps=[(f"adelay={ms}:all=1 shifts every channel {ms} ms later; video is copied.", cmd)])


@gen("C24")
def tracks(r):
    vn = name(r, ["movie", "film", "show", "dub", "episode", "lecture"], ".mkv")
    kind = r.choice(["keep_eng", "keep_hin", "default_eng", "extract_hin", "mp4_eng", "list"])
    probe = ("Check which audio track is which language first.", f"ffprobe -v error -show_entries stream=index,codec_type:stream_tags=language -of compact {q(vn)}")
    if kind == "keep_eng":
        req = ask(r, "{v} has two audio tracks, keep only the english one", "remove the hindi audio from {v}", v=vn)
        st = ("Track a:1 is eng: map the video and only that track, copy everything.", f"ffmpeg -i {q(vn)} -map 0:v -map 0:a:m:language:eng -c copy {o(stem(vn) + '_eng.mkv')}")
        spec = {"match": "*.mkv|*.mp4", "has_video": True, "audio_streams": 1, "audio_lang": "eng"}
    elif kind == "keep_hin":
        req = ask(r, "keep just the hindi audio in {v}", "drop the english track from {v}", v=vn)
        st = ("Map the video and the hin track only.", f"ffmpeg -i {q(vn)} -map 0:v -map 0:a:m:language:hin -c copy {o(stem(vn) + '_hin.mkv')}")
        spec = {"match": "*.mkv|*.mp4", "has_video": True, "audio_streams": 1, "audio_lang": "hin"}
    elif kind == "default_eng":
        req = ask(r, "make english the default audio track of {v}, keep the other one", "{v} plays hindi first, switch it to english by default", v=vn)
        st = ("Reorder: eng (a:1) first and mark it default, hin second.", f"ffmpeg -i {q(vn)} -map 0:v -map 0:a:1 -map 0:a:0 -c copy -disposition:a:0 default -disposition:a:1 0 {o(stem(vn) + '.mkv')}")
        spec = {"match": "*.mkv", "audio_streams": 2, "audio_lang": "eng"}
    elif kind == "extract_hin":
        req = ask(r, "pull the hindi audio out of {v} as an mp3", "save the hindi track of {v} as mp3", v=vn)
        st = ("Map only the hin track and encode it to mp3.", f"ffmpeg -i {q(vn)} -map 0:a:m:language:hin -c:a libmp3lame -b:a 160k {o(stem(vn) + '_hindi.mp3')}")
        spec = {"match": "*.mp3", "dur": [2.0, 6.3]}
    elif kind == "mp4_eng":
        req = ask(r, "turn {v} into an mp4 that plays the english audio on any device", "mp4 of {v} with english sound", v=vn)
        st = ("Video copies as-is (h264); keep only the eng track as AAC so phones pick the right one.",
              f"ffmpeg -i {q(vn)} -map 0:v -map 0:a:m:language:eng -c:v copy {AENC} {o(stem(vn) + '.mp4')}")
        spec = {"match": "*.mp4", "audio_streams": 1, "audio_lang": "eng", "acodec": "aac"}
    else:
        req = ask(r, "what audio tracks does {v} have?", "which languages are in {v}?", v=vn)
        return dict(files={vn: "multilang.mkv"}, ask=req, spec={"outputs": [], "answer": {"must": [r"hin", r"eng"], "no_files": True}},
                    steps=[probe], final="Two audio tracks: track 1 is Hindi (hin, the default) and track 2 is English (eng), both AAC.")
    return dict(files={vn: "multilang.mkv"}, ask=req, spec=[spec], steps=[probe, st])


@gen("C25")
def channels(r):
    kind = r.choice(["to_stereo", "to_mono", "whisper", "left_only"])
    if kind == "to_stereo":
        an = name(r, ["rec", "memo", "voice", "mic"], ".wav")
        req = ask(r, "{a} is mono, make it stereo", "convert {a} to stereo", a=an)
        return dict(files={an: "hum.wav"}, ask=req, spec=[{"match": "*.wav", "ach": 2}],
                    steps=[("-ac 2 duplicates the mono channel to left and right.", f"ffmpeg -i {q(an)} -ac 2 {o(stem(an) + '_stereo.wav')}")])
    src = r.choice(["narration.m4a", "talk.mp4", "music_bed.mp3", "webcam.mp4"])
    an = name(r, ANAMES + VNAMES[:6], ext_of(src))
    if kind == "to_mono":
        req = ask(r, "convert {a} to a mono mp3", "make {a} mono (mp3) for a phone line", a=an)
        return dict(files={an: src}, ask=req, spec=[{"match": "*.mp3", "ach": 1}],
                    steps=[("-ac 1 downmixes to mono; -vn drops any video.", f"ffmpeg -i {q(an)} -vn -ac 1 -c:a libmp3lame -b:a 96k {o(stem(an) + '_mono.mp3')}")])
    if kind == "whisper":
        req = ask(r, "give me the audio of {a} as a 16kHz mono wav for transcription", "prep {a} for whisper", a=an)
        return dict(files={an: src}, ask=req, spec=[{"match": "*.wav", "ach": 1, "sr": 16000}],
                    steps=[("Whisper wants 16 kHz mono PCM: -vn -ac 1 -ar 16000 -c:a pcm_s16le.", f"ffmpeg -i {q(an)} -vn -ac 1 -ar 16000 -c:a pcm_s16le {o(stem(an) + '_16k.wav')}")])
    req = ask(r, "keep only the left channel of {a}, as mono", "{a}: use just the left channel", a=an)
    return dict(files={an: src}, ask=req, spec=[{"match": "*", "ach": 1}],
                steps=[("pan=mono|c0=FL keeps the front-left channel only.", f"ffmpeg -i {q(an)} -vn -af 'pan=mono|c0=FL' {o(stem(an) + '_left.wav')}")])


# ═════════════════════════ D. fixing problem files ═════════════════════════

@gen("D26")
def compat(r):
    fx = r.choice(["odd.webm", "odd.webm", "big.mov", "phone.mov", "web.webm"])
    M = m(fx)
    vn = name(r, VNAMES + ["export"], ext_of(fx))
    dev = r.choice(["my iphone", "my smart tv", "powerpoint on windows", "whatsapp", "my car's screen", "an old android phone"])
    req = ask(r, "{v} won't play on {d}, convert it", "make {v} work on {d}", "{d} can't open {v}, fix it", v=vn, d=dev)
    odd = M["w"] % 2 or M["h"] % 2
    vf = " -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2'" if odd else ""
    why = {"odd.webm": f"VP9/Opus in WebM; and {M['w']}x{M['h']} is odd, which H.264 yuv420p can't encode, so round down to even",
           "big.mov": "ProRes + PCM is an editing format", "phone.mov": "HEVC isn't supported everywhere", "web.webm": "VP9/Opus in WebM"}[fx]
    cmd = f"ffmpeg -i {q(vn)}{vf} -c:v libx264 -crf 22 -pix_fmt yuv420p -c:a aac -b:a 160k -movflags +faststart {o(stem(vn) + '.mp4')}"
    think = f"{why}: H.264 yuv420p + AAC in MP4 with faststart plays on practically everything."
    return dict(files={vn: fx}, ask=req, spec=[{"match": "*.mp4", "vcodec": "h264", "pix_fmt": "yuv420p", "acodec": "aac", "dur": [M["dur"] - 0.3, M["dur"] + 0.3]}],
                steps=[(think, cmd)])


@gen("D27")
def vfr(r):
    vn = name(r, ["screen_rec", "capture", "obs", "recording", "zoom_call"], ".mp4")
    F = r.choice([24, 25, 30, 30, 60])
    req = ask(r, "{v} is variable frame rate and drifts in my editor, make it constant {f}fps", "convert {v} to constant frame rate ({f} fps)",
              "premiere says {v} is VFR, fix it at {f}fps", v=vn, f=F)
    steps = []
    if r.random() < 0.4:
        steps.append(("Confirm it's VFR: look at the frame timestamps.", f"ffprobe -v error -select_streams v -show_entries frame=pts_time -read_intervals %+2 -of csv=p=0 {q(vn)}"))
    steps.append((f"fps={F} resamples to a fixed {F} fps grid (duplicating/dropping frames), re-encoded; audio re-encoded alongside.",
                  f"ffmpeg -i {q(vn)} -vf fps={F} {ENC} {AENC} {o(stem(vn) + '_cfr.mp4')}"))
    return dict(files={vn: "vfr.mp4"}, ask=req, spec=[{"match": "*.mp4", "cfr": True, "fps": F, "dur": [11.6, 12.4], "has_audio": True}], steps=steps)


@gen("D28")
def deinterlace(r):
    vn = name(r, ["tape", "camcorder", "broadcast", "wedding", "old_footage", "vhs_rip"], ".mp4")
    double = r.random() < 0.35
    req = ask(r, "{v} has comb lines when things move, fix it" + (", keep the smooth motion" if double else ""), "deinterlace {v}" + (" at double frame rate" if double else ""),
              v=vn)
    f = "yadif=mode=send_field" if double else r.choice(["yadif", "bwdif"])
    think = f"Those are interlacing artifacts: {f.split('=')[0]} deinterlaces" + (" and outputs one frame per field (2x frame rate)." if double else ".")
    spec = {"match": "*.mp4", "progressive": True, "dur": [9.7, 10.3]}
    if double:
        spec["fps"] = 60
    return dict(files={vn: "interlaced.mp4"}, ask=req, spec=[spec], steps=[(think, f"ffmpeg -i {q(vn)} -vf {f} {ENC} {AENC} {o(stem(vn) + '_progressive.mp4')}")])


@gen("D29")
def aspect_fix(r):
    if r.random() < 0.5:
        vn = name(r, ["squished", "clip", "thin", "export"], ".mp4")
        req = ask(r, "everything in {v} looks too thin, it should be 16:9 1280x720", "{v} is 960x720 but should be widescreen 16:9, fix it", v=vn)
        return dict(files={vn: "squashed.mp4"}, ask=req, spec=[{"match": "*.mp4", "wh": [1280, 720], "sar": [1, 1]}],
                    steps=[("The picture itself is squeezed: scale it to 1280x720 and set square pixels.", f"ffmpeg -i {q(vn)} -vf scale=1280:720,setsar=1 {ENC} {AENC} {o(stem(vn) + '_169.mp4')}")])
    vn = name(r, ["stretched", "wide", "clip", "export"], ".mp4")
    req = ask(r, "{v} plays stretched too wide, the pixels are fine", "fix the aspect ratio of {v}, it looks stretched", v=vn)
    return dict(files={vn: "wrongsar.mp4"}, ask=req, spec=[{"match": "*.mp4", "wh": [1280, 720], "sar": [1, 1]}],
                steps=[("Check the aspect flags first.", f"ffprobe -v error -select_streams v -show_entries stream=width,height,sample_aspect_ratio,display_aspect_ratio -of compact {q(vn)}"),
                       ("SAR is 4:3, so players stretch it; the pixels are 1280x720 already: setsar=1 fixes the display.", f"ffmpeg -i {q(vn)} -vf setsar=1 {ENC} -c:a copy {o(stem(vn) + '_fixed.mp4')}")])


@gen("D30")
def brighten(r):
    vn = name(r, ["night", "gym", "underexposed", "party", "basement", "evening"], ".mp4")
    req = ask(r, "{v} is way too dark, brighten it", "fix the exposure in {v}, it looks dull and dark", "brighten up {v}", v=vn)
    b, c = r.choice([(0.3, 1.15), (0.28, 1.2), (0.32, 1.1)])
    steps = []
    if r.random() < 0.4:
        steps.append(("Measure the brightness first (average luma).", f"ffmpeg -i {q(vn)} -vf \"fps=1,signalstats,metadata=print:key=lavfi.signalstats.YAVG\" -f null -"))
    steps.append((f"Average luma is ~52/255, so lift it: eq brightness={b} with a little contrast ({c}) to avoid a washed-out look.",
                  f"ffmpeg -i {q(vn)} -vf eq=brightness={b}:contrast={c} {ENC} -c:a copy {o(stem(vn) + '_bright.mp4')}"))
    return dict(files={vn: "dark.mp4"}, ask=req, spec=[{"match": "*.mp4", "yavg": [85, 190], "dur": [9.7, 10.3]}], steps=steps)


@gen("D31")
def salvage(r):
    vn = name(r, ["recording", "stream", "partial", "obs_crash", "capture"], ".mkv")
    to = r.choice([".mp4", ".mkv"])
    req = ask(r, "{v} got cut off when my pc crashed, save what you can" + (" as mp4" if to == ".mp4" else ""), "{v} is half-written and won't open properly, recover it", v=vn)
    cmd = f"ffmpeg -err_detect ignore_err -i {q(vn)} -c copy {o(stem(vn) + '_fixed' + to)}"
    think = "The file just ends mid-write; the intact part demuxes fine, so remux it with -c copy into a clean container with a proper index."
    return dict(files={vn: "broken.mkv"}, ask=req, spec=[{"match": "*" + to, "has_video": True, "dur": [12, 31]}], steps=[(think, cmd)])


# ═════════════════════════ E. images ═════════════════════════

def photos(r, k=None):
    ids = sorted(r.sample(range(1, 7), k)) if k else range(1, 7)
    folder = r.choice(["photos", "trip", "pics", "camera", "shoot"])
    return {f"{folder}/img_{i:02d}.jpg": f"photos/img_{i:02d}.jpg" for i in ids}, folder


@gen("E32")
def batch_img(r):
    files, d = photos(r)
    kind = r.choice(["webp", "thumbs", "size", "png"])
    if kind == "webp":
        n = r.choice([1200, 1600, 1920])
        req = ask(r, "convert all the photos in {d}/ to webp for my website, max {n}px", "webp versions of everything in {d}, longest side {n}", d=d, n=n)
        cmd, spec = f"magick {{in}} -resize '{n}x{n}>' -quality 82 out/{{stem}}.webp", {"match": "*.webp", "count": 6, "max_side": n}
    elif kind == "thumbs":
        n = r.choice([200, 300, 400])
        req = ask(r, "make {n}px thumbnails of every photo in {d}/", "thumbnail all of {d} at {n} pixels", d=d, n=n)
        cmd, spec = f"magick {{in}} -resize '{n}x{n}>' out/{{stem}}_thumb.jpg", {"match": "*.jpg", "count": 6, "max_side": n}
    elif kind == "size":
        kb = r.choice([200, 300, 500])
        req = ask(r, "shrink every photo in {d} to under {k} KB", "all photos below {k}KB please", d=d, k=kb)
        cmd, spec = f"magick {{in}} -resize '1200x1200>' -quality 75 out/{{stem}}.jpg", {"match": "*.jpg", "count": 6, "max_mb": kb / 1000}
    else:
        req = ask(r, "convert the photos in {d}/ to png", "png copies of all the {d} pictures", d=d)
        cmd, spec = "magick {in} out/{stem}.png", {"match": "*.png", "count": 6}
    return dict(files=files, ask=req, spec=[spec], steps=[("One command per file with foreach; '>' only shrinks larger images (quoted so the shell doesn't read it).", cmd, f"in/{d}/*.jpg")])


@gen("E33")
def product(r):
    pn = name(r, ["product", "item", "shoe", "mug", "bag", "watch"], ".png")
    bg = r.choice([("white", "white", [255, 255, 255]), ("light gray (#f3f4f6)", "'#f3f4f6'", [243, 244, 246]), ("pure white", "white", [255, 255, 255]),
                   ("light beige (#f5f0e6)", "'#f5f0e6'", [245, 240, 230])])
    n = r.choice([1000, 1080, 1200, 1600, 2000])
    ext = r.choice([".jpg", ".png"])
    req = ask(r, "{p} as a square {n}x{n} {e} on a {b} background with some padding", "marketplace image: {p} on {b}, {n}px square",
              p=pn, n=n, b=bg[0], e=ext[1:])
    inner = int(n * 0.85)
    cmd = f"magick in/{pn} -resize {inner}x{inner} -background {bg[1]} -gravity center -extent {n}x{n} -flatten {o(stem(pn) + '_square' + ext)}"
    think = f"Fit the product in {inner}px (about 7% padding each side), centre it on a {n}x{n} {bg[0]} canvas, flatten the transparency."
    return dict(files={pn: "product.png"}, ask=req, spec=[{"match": "*" + ext, "wh": [n, n], "corner_color": bg[2]}], steps=[(think, cmd)])


QUOTES = ["Stay curious", "Less, but better.", "Ship it", "Make it simple", "Keep going", "Done is better than perfect", "Good morning"]


@gen("E34")
def text_on_image(r):
    src = r.choice(["cover.jpg", "photo.jpg", "photos/img_02.jpg", "photos/img_04.jpg"])
    M = m(src) if src in META else {"w": 1300, "h": 850}
    if src.startswith("photos/"):
        from checks import magick_fx
        w, h = map(int, magick_fx(os.path.join(FIX, src), "%w %h").split())
        M = {"w": w, "h": h}
    im = name(r, ["bg", "photo", "pic", "texture", "image"], ".jpg")
    kind = r.choice(["center", "meme", "post"])
    if kind == "meme":
        a, b = r.choice([("WHEN THE CODE WORKS", "ON THE FIRST TRY"), ("ME", "ALSO ME"), ("MONDAY", "AGAIN")])
        req = ask(r, "meme: '{a}' on top and '{b}' at the bottom of {i}", "make a meme from {i}: top '{a}', bottom '{b}'", a=a, b=b, i=im)
        cmd = (f"magick in/{im} -fill white -stroke black -strokewidth 4 -pointsize {M['h'] // 10} -gravity north -annotate +0+20 '{a}' "
               f"-gravity south -annotate +0+20 '{b}' {o(stem(im) + '_meme.jpg')}")
        spec = {"match": "*.jpg", "wh": [M["w"], M["h"]], "frame_unlike": [[0, f"in/{im}", 0.004]]}
        think = "Classic meme text: white with a black stroke, top and bottom via gravity north/south."
    elif kind == "post":
        qt = r.choice(QUOTES)
        req = ask(r, "1080x1080 instagram quote post from {i} with '{q}' in the middle", "quote card: '{q}' on {i}, square 1080", q=qt, i=im)
        cmd = f"magick in/{im} -resize 1080x1080^ -gravity center -extent 1080x1080 -fill white -pointsize 90 -annotate +0+0 '{qt}' {o('quote_post.jpg')}"
        spec = {"match": "*.jpg", "wh": [1080, 1080]}
        think = "Fill-crop to 1080x1080 (the ^ makes the smaller side fit), then centre the quote."
    else:
        qt = r.choice(QUOTES)
        req = ask(r, "put '{q}' in big white letters across the middle of {i}", "write '{q}' on {i}, centred", q=qt, i=im)
        cmd = f"magick in/{im} -gravity center -fill white -stroke black -strokewidth 3 -pointsize {M['h'] // 9} -annotate +0+0 '{qt}' {o(stem(im) + '_quote.jpg')}"
        spec = {"match": "*.jpg", "wh": [M["w"], M["h"]], "frame_unlike": [[0, f"in/{im}", 0.004]]}
        think = "annotate with gravity center; a thin black stroke keeps white text readable on light areas."
    return dict(files={im: src}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("E35")
def slideshow(r):
    k = r.choice([3, 4, 5, 6])
    files, d = photos(r, k)
    per = r.choice([2, 3, 4])
    W, H = r.choice([(1920, 1080), (1280, 720), (1080, 1920), (1080, 1080)])
    music = r.random() < 0.6
    fade = r.random() < 0.35
    fl = " with crossfades" if fade else ""
    req = ask(r, "slideshow of the photos in {d}, {p}s each{f}" + (", with {m} playing" if music else "") + ", {w}x{h}",
              "turn the {d} pictures into a video, {p} seconds per photo{f}, {w}x{h}" + (", music: {m}" if music else ""),
              d=d, p=per, f=fl, m="song.mp3", w=W, h=H)
    names = sorted(files)
    ins = " ".join(f"-loop 1 -t {per} -i {q(n)}" for n in names)
    norm = ";".join(f"[{i}:v]{fit(W, H, 25)}[v{i}]" for i in range(k))
    if fade:
        chain, prev = [], "v0"
        for i in range(1, k):
            chain.append(f"[{prev}][v{i}]xfade=duration=0.5:offset={round(i * per - 0.5 * i, 2)}[x{i}]")
            prev = f"x{i}"
        fc, T = norm + ";" + ";".join(chain), k * per - 0.5 * (k - 1)
        last = prev
    else:
        fc, T, last = norm + ";" + "".join(f"[v{i}]" for i in range(k)) + f"concat=n={k}:v=1:a=0[v]", k * per, "v"
    if music:
        files["song.mp3"] = r.choice(MUSIC)
        cmd = f"ffmpeg {ins} -i in/song.mp3 -filter_complex \"{fc}\" -map \"[{last}]\" -map {k}:a {ENC} {AENC} -shortest {o('slideshow.mp4')}"
    else:
        cmd = f"ffmpeg {ins} -filter_complex \"{fc}\" -map \"[{last}]\" {ENC} {o('slideshow.mp4')}"
    think = (f"Each photo is a looped {per}s input, fitted to {W}x{H} (photos differ in size, so scale+pad each)" +
             (", chained with 0.5s xfades" if fade else ", concatenated") + (f"; the music stops with the pictures (-shortest)." if music else "."))
    return dict(files=files, ask=req, spec=[{"match": "*.mp4", "wh": [W, H], "dur": [T - 0.4, T + 0.5], **({"has_audio": True} if music else {})}],
                steps=[(think, cmd)])


@gen("E36")
def screenshot_polish(r):
    sn = name(r, ["screenshot", "app", "ui", "dashboard", "screen"], ".png")
    kind = r.choice(["round", "shadow", "border"])
    if kind == "round":
        rad = r.choice([16, 24, 30, 40])
        req = ask(r, "rounded corners on {s} ({n}px), transparent outside", "give {s} rounded corners", s=sn, n=rad)
        cmd = (f"magick in/{sn} \\( +clone -alpha extract -draw 'fill black polygon 0,0 0,{rad} {rad},0 fill white circle {rad},{rad} {rad},0' "
               f"\\( +clone -flip \\) -compose Multiply -composite \\( +clone -flop \\) -compose Multiply -composite \\) -alpha off "
               f"-compose CopyOpacity -composite {o(stem(sn) + '_rounded.png')}")
        spec, think = {"match": "*.png", "corner_alpha": True}, f"Build a mask with {rad}px quarter-circles in each corner (draw one, flip/flop for the rest) and use it as the alpha channel."
    elif kind == "shadow":
        pad = r.choice([40, 60, 80])
        req = ask(r, "put {s} on a white background with {p}px padding and a soft shadow", "blog-ready {s}: shadow, white margin {p}px", s=sn, p=pad)
        cmd = (f"magick in/{sn} \\( +clone -background black -shadow 50x20+0+15 \\) +swap -background white -layers merge +repage "
               f"-bordercolor white -border {pad} {o(stem(sn) + '_shadow.png')}")
        spec, think = {"match": "*.png", "corner_color": [255, 255, 255]}, "-shadow makes a blurred copy; merge it under the image on white, then add the margin."
    else:
        w = r.choice([800, 1000, 1200])
        req = ask(r, "add a thin gray border around {s} and make it {w}px wide", "{s}: 2px border, {w} wide", s=sn, w=w)
        cmd = f"magick in/{sn} -resize {w - 4}x -bordercolor '#9ca3af' -border 2 {o(stem(sn) + '_border.png')}"
        spec, think = {"match": "*.png", "max_side": w}, f"Resize to {w - 4}px so the 2px border on each side lands exactly at {w}."
    return dict(files={sn: "screenshot.png"}, ask=req, spec=[spec], steps=[(think, cmd)])


@gen("E37")
def scans(r):
    nm = r.choice([("scan1.jpg", "scan2.jpg", "scan3.jpg"), ("p1.jpg", "p2.jpg", "p3.jpg"), ("page_a.jpg", "page_b.jpg", "page_c.jpg")])
    files = {n: f"page{i + 1}.jpg" for i, n in enumerate(nm)}
    kind = r.choice(["pdf", "gray_pdf", "gray_jpg"])
    lst = " ".join(f"in/{n}" for n in nm)
    if kind == "gray_jpg":
        req = ask(r, "make {l} grayscale so they're easier to read", "black and white versions of the scans {l}", l=", ".join(nm))
        return dict(files=files, ask=req, spec=[{"match": "*.jpg", "count": 3, "gray": True}],
                    steps=[("-colorspace Gray plus -normalize stretches the contrast of a scanned page.", "magick {in} -colorspace Gray -normalize out/{stem}_bw.jpg", "in/*.jpg")])
    req = ask(r, "combine {l} into one pdf" + (" in grayscale" if kind == "gray_pdf" else ""), "one pdf from these scans: {l}", l=", ".join(nm))
    g = " -colorspace Gray" if kind == "gray_pdf" else ""
    return dict(files=files, ask=req, spec=[{"match": "*.pdf", "pdf_pages": 3}],
                steps=[("magick writes each input as one PDF page, in the order given.", f"magick {lst}{g} {o('scans.pdf')}")])


# ═════════════════════════ F. metadata ═════════════════════════

def shift(dt, hours):
    import datetime
    t = datetime.datetime.strptime(dt, "%Y:%m:%d %H:%M:%S") + datetime.timedelta(hours=hours)
    return t.strftime("%Y:%m:%d %H:%M:%S")


@gen("F38")
def dates(r):
    files, d = photos(r)
    kind = r.choice(["rename", "shift", "set"])
    if kind == "rename":
        req = ask(r, "rename the photos in {d} by the date they were taken", "name every photo in {d} after its capture date, like 2024-05-03_091200.jpg", d=d)
        cmd = f"exiftool -o out/ -d '%Y-%m-%d_%H%M%S%%-c.%%e' '-FileName<DateTimeOriginal' in/{d}"
        return dict(files=files, ask=req, spec=[{"match": "2024-05-0*.jpg", "count": 6}],
                    steps=[("exiftool renames from DateTimeOriginal with a date format; %-c adds a counter if two match, -o out/ writes copies.", cmd)])
    if kind == "shift":
        h = r.choice([-5, -3, -2, 1, 2, 3, 8])
        ahead = h < 0
        req = ask(r, "the camera clock was {n} hours {w}, fix the date taken on all the photos in {d}", "shift the capture time of every photo in {d} by {s}{n} hours",
                  n=abs(h), w="ahead" if ahead else "behind", d=d, s="-" if ahead else "+")
        op = "-=" if h < 0 else "+="
        cmd = f"exiftool -o out/ '-AllDates{op}{abs(h)}' in/{d}"
        k = r.choice(range(1, 7))
        return dict(files=files, ask=req, spec=[{"match": f"img_{k:02d}.jpg", "exif_eq": {"DateTimeOriginal": shift(TRUTH["photo_dates"][k - 1], h)}}],
                    steps=[(f"Clock {'ahead' if ahead else 'behind'} by {abs(h)}h, so {'subtract' if ahead else 'add'} {abs(h)} hours: AllDates{op}{abs(h)} shifts DateTimeOriginal, CreateDate and ModifyDate together.", cmd)])
    k = r.choice(range(1, 7))
    fn = f"img_{k:02d}.jpg"
    y, mo, dd, hh = r.choice([2023, 2024, 2025]), r.randint(1, 12), r.randint(1, 28), r.randint(6, 21)
    want = f"{y}:{mo:02d}:{dd:02d} {hh:02d}:00:00"
    req = ask(r, "set the date taken of {f} to {dd} {m} {y}, {h}:00", "change when {f} was taken to {y}-{mo:02d}-{dd:02d} {h}:00", f=fn, dd=dd,
              m=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][mo - 1], y=y, h=hh, mo=mo)
    return dict(files={fn: f"photos/{fn}"}, ask=req, spec=[{"match": "*.jpg", "exif_eq": {"DateTimeOriginal": want}}],
                steps=[("Write DateTimeOriginal (exiftool date format YYYY:MM:DD HH:MM:SS) to a copy in out/.", f"exiftool -o out/ '-DateTimeOriginal={want}' in/{fn}")])


@gen("F39")
def privacy(r):
    files, d = photos(r)
    kind = r.choice(["share", "all", "name"])
    if kind == "share":
        req = ask(r, "remove location and personal info from the photos in {d} before I share them, keep the date", "strip GPS and my name from {d}, but keep when they were taken", d=d)
        return dict(files=files, ask=req, spec=[{"match": "*.jpg", "count": 6, "no_gps": True, "has_date": True, "exif_absent": ["Artist"]}],
                    steps=[("Delete the GPS group and the Artist tag only, so DateTimeOriginal stays.", f"exiftool -o out/ -gps:all= -Artist= in/{d}")])
    if kind == "all":
        k = r.choice(range(1, 7))
        fn = f"img_{k:02d}.jpg"
        req = ask(r, "strip ALL metadata from {f}", "remove every bit of exif from {f}", f=fn)
        return dict(files={fn: f"photos/{fn}"}, ask=req, spec=[{"match": "*.jpg", "no_gps": True, "exif_absent": ["Artist", "Make", "DateTimeOriginal"]}],
                    steps=[("-all= removes every writable tag.", f"exiftool -all= -o out/ in/{fn}")])
    req = ask(r, "remove the photographer name from all photos in {d}", "take my name (Artist) out of the {d} photos", d=d)
    return dict(files=files, ask=req, spec=[{"match": "*.jpg", "count": 6, "exif_absent": ["Artist"]}],
                steps=[("Only the Artist tag goes; everything else stays.", f"exiftool -o out/ -Artist= in/{d}")])


@gen("F40")
def report(r):
    if r.random() < 0.5:
        files, d = photos(r)
        req = ask(r, "list the date taken of every photo in {d} as csv", "spreadsheet of when each photo in {d} was taken", d=d)
        rows = "\n".join(f"{d}/img_{i + 1:02d}.jpg,{dt}" for i, dt in enumerate(TRUTH["photo_dates"]))
        return dict(files=files, ask=req, spec={"outputs": [], "answer": {"must": ["img_01", "img_06", TRUTH["photo_dates"][0], TRUTH["photo_dates"][5]]}},
                    steps=[("exiftool -csv prints one row per file.", f"exiftool -csv -DateTimeOriginal in/{d}")],
                    final=f"SourceFile,DateTimeOriginal\n{rows}")
    vids = r.sample(["talk.mp4", "cuts.mp4", "webcam.mp4", "lowres.mp4", "version_a.mp4"], 3)
    names = [name(r, VNAMES, ".mp4", avoid=set()) for _ in vids]
    if len(set(names)) < 3:
        names = ["a.mp4", "b.mp4", "c.mp4"]
    files = dict(zip(names, vids))
    req = ask(r, "make a csv of every video here with its duration and resolution", "table of these videos: length and size in pixels")
    steps = [(f"Probe {n}: width, height, duration." if i else "One ffprobe per file (csv output), then assemble the table.",
              f"ffprobe -v error -select_streams v:0 -show_entries stream=width,height:format=duration -of csv=p=0 {q(n)}") for i, n in enumerate(names)]
    rows = "\n".join(f"{n},{m(v)['dur']:.1f},{m(v)['w']},{m(v)['h']}" for n, v in files.items())
    return dict(files=files, ask=req, spec={"outputs": [], "answer": {"must": [re_esc(n) for n in names] + [str(m(vids[0])["w"])]}},
                steps=steps, final=f"file,duration_s,width,height\n{rows}")


def re_esc(s):
    import re
    return re.escape(s)


# ═════════════════════════ G. questions ═════════════════════════

@gen("G41")
def diagnose(r):
    kind = r.choice(["big", "broken", "vfr", "odd", "interlaced", "sar"])
    P = {
        "big": ("big.mov", ["master", "export", "raw"], ".mov", "why is {v} so huge? it's only 5 seconds",
                "ffprobe -v error -show_entries stream=codec_name,bit_rate,profile:format=bit_rate -of compact {f}",
                [r"prores", r"(pcm|uncompress)"], "It's ProRes 422 HQ video with uncompressed 24-bit PCM audio: an editing/mastering format at ~50 Mb/s. Convert to H.264 + AAC to make it small."),
        "broken": ("broken.mkv", ["recording", "stream", "capture"], ".mkv", "what's wrong with {v}? it says 30 seconds but stops early",
                   "ffmpeg -v error -i {f} -f null -", [r"(truncat|incomplete|cut off|ends early|unfinished)"],
                   "The file is truncated: the recording stopped mid-write, so the header claims 30 s but only the first part of the data exists. The intact part can be remuxed."),
        "vfr": ("vfr.mp4", ["screen", "obs", "capture"], ".mp4", "is {v} variable frame rate?",
                "ffprobe -v error -select_streams v -show_entries frame=pts_time -of csv=p=0 {f}", [r"\b(yes|variable|vfr)\b"],
                "Yes: the frame interval changes partway through (about 30, then 12, then 24 fps), so it's variable frame rate. Converting with fps=30 makes it constant."),
        "odd": ("odd.webm", ["export", "clip"], ".webm", "why does {v} fail when I convert it to h264?",
                "ffprobe -v error -show_entries stream=width,height,codec_name -of compact {f}", [r"(odd|divisible|even|853|481)"],
                "It's 853x481: H.264 with yuv420p needs an even width and height. Scale with scale=trunc(iw/2)*2:trunc(ih/2)*2 (852x480) first."),
        "interlaced": ("interlaced.mp4", ["tape", "camcorder", "broadcast"], ".mp4", "why does {v} have comb lines when things move?",
                       "ffmpeg -i {f} -vf idet -frames:v 200 -f null -", [r"interlac"], "It's interlaced (top field first): each frame holds two fields from different moments. Deinterlace with yadif or bwdif."),
        "sar": ("wrongsar.mp4", ["stretched", "clip"], ".mp4", "why does {v} look stretched in VLC?",
                "ffprobe -v error -select_streams v -show_entries stream=width,height,sample_aspect_ratio,display_aspect_ratio -of compact {f}", [r"(4:3|aspect|sar)"],
                "The pixels are 1280x720 but the sample aspect ratio is flagged 4:3, so players stretch it to a wider display aspect. setsar=1 fixes it."),
    }[kind]
    fx, nms, ext, qq, cmd, must, ans = P
    vn = r.choice(nms) + ext
    return dict(files={vn: fx}, ask=ask(r, qq, v=vn), spec={"outputs": [], "answer": {"must": must, "no_files": True}},
                steps=[("Look at the file before answering.", cmd.format(f=q(vn)))], final=ans)


@gen("G42")
def timestamps(r):
    kind = r.choice(["scenes", "silence", "flash"])
    if kind == "scenes":
        vn = name(r, VNAMES + ["montage"], ".mp4")
        c = TRUTH["cuts"]
        return dict(files={vn: "cuts.mp4"}, ask=ask(r, "at what times does the scene change in {v}?", "find the cuts in {v}", v=vn),
                    spec={"outputs": [], "answer": {"must": [r"\b4(\.0+)?\b", r"9\.5", r"\b15(\.0+)?\b"], "no_files": True}},
                    steps=[("Scene detection: select frames whose scene score is above 0.3 and print their times with showinfo.",
                            f"ffmpeg -i {q(vn)} -vf \"select='gt(scene,0.3)',showinfo\" -f null -")],
                    final=f"The scene changes at {c[0]} s, {c[1]} s and {c[2]} s.")
    if kind == "silence":
        an = name(r, ANAMES + ["interview"], ".m4a")
        pa = TRUTH["pauses"]
        return dict(files={an: "narration.m4a"}, ask=ask(r, "where are the silent parts in {a}?", "list the pauses in {a}", a=an),
                    spec={"outputs": [], "answer": {"must": [r"1\.6", r"6\.0", r"8\.4", r"12\.0"], "no_files": True}},
                    steps=[("silencedetect prints the start and end of every quiet stretch (-35 dB, at least 0.5 s).", f"ffmpeg -i {q(an)} -af silencedetect=n=-35dB:d=0.5 -f null -")],
                    final=f"Silences at about {pa[0][0]:.1f}-{pa[0][1]:.1f} s and {pa[1][0]:.1f}-{pa[1][1]:.1f} s (plus a quiet tail after {TRUTH['speech'][-1][1]:.1f} s).")
    vn = name(r, ["sync", "clap", "test"], ".mp4")
    return dict(files={vn: "sync.mp4"}, ask=ask(r, "when does the white flash happen in {v}?", "timestamps of the flashes in {v}", v=vn),
                spec={"outputs": [], "answer": {"must": [r"\b2(\.0+)?\s*s", r"\b7(\.0+)?\s*s"], "no_files": True}},
                steps=[("Brightness per frame: signalstats YAVG jumps on a white frame.", f"ffmpeg -i {q(vn)} -vf \"signalstats,metadata=print:key=lavfi.signalstats.YAVG\" -f null -")],
                final="The flashes are at 2.0 s and 7.0 s.")


@gen("G43")
def compare(r):
    a, b = r.choice([("take_1.mp4", "take_2.mp4"), ("a.mp4", "b.mp4"), ("old.mp4", "new.mp4"), ("v1.mp4", "v2.mp4")])
    good_first = r.random() < 0.5
    files = {a: "version_a.mp4" if good_first else "version_b.mp4", b: "version_b.mp4" if good_first else "version_a.mp4"}
    good = a if good_first else b
    bad = b if good_first else a
    req = ask(r, "which of {a} and {b} has better picture quality? they're the same size", "compare {a} and {b}: which one should I upload?", a=a, b=b)
    steps = [(f"Same resolution on paper, so compare bitrates.", f"ffprobe -v error -show_entries stream=width,height,bit_rate -of compact {q(a)}"),
             ("Now the other one.", f"ffprobe -v error -show_entries stream=width,height,bit_rate -of compact {q(b)}")]
    return dict(files=files, ask=req, spec={"outputs": [], "answer": {"must": [re_esc(good)], "no_files": True}}, steps=steps,
                final=f"{good} is better: both are 1280x720, but {good} has a far higher bitrate, while {bad} is starved (~150 kb/s) and upscaled from a tiny source, so it's blocky.")


# ═════════════════════════ H. behaviour ═════════════════════════

@gen("H44", weight=1.5)
def recover(r):
    """A first attempt that runs but is wrong (or fails); the model checks, notices, and fixes it. The wrong attempt has weight 0."""
    kind = r.choice(["odd", "gif", "size", "speed", "overlay_end"])
    if kind == "odd":
        vn = name(r, VNAMES, ".webm")
        return dict(files={vn: "odd.webm"}, ask=ask(r, "convert {v} to mp4 (h264)", "mp4 version of {v} please", v=vn),
                    spec=[{"match": "*.mp4", "vcodec": "h264", "dur": [7.7, 8.4]}],
                    broken=f"ffmpeg -i {q(vn)} -c:v libx264 -pix_fmt yuv420p -c:a aac {o(stem(vn) + '.mp4')}",
                    steps=[("libx264 rejected the size: 853x481 is odd and yuv420p needs even dimensions. Round both down to even.",
                            f"ffmpeg -y -i {q(vn)} -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p -c:a aac {o(stem(vn) + '.mp4')}")])
    if kind == "gif":
        fx = r.choice(["talk.mp4", "land.mp4", "version_a.mp4"])
        vn = name(r, VNAMES, ".mp4")
        mb = r.choice([1, 1.5, 2])
        n = 4
        return dict(files={vn: fx}, ask=ask(r, "gif of the first {n} seconds of {v}, under {m} MB", "{n}s gif from {v}, max {m}MB", v=vn, n=n, m=mb),
                    spec=[{"match": "*.gif", "max_mb": mb, "dur": [n - 0.5, n + 0.5]}],
                    broken_ok=f"ffmpeg -t {n} -i {q(vn)} -vf fps=25 {o(stem(vn) + '.gif')}",
                    steps=[(f"Check the size against the {mb} MB limit.", f"ffprobe -v error -show_entries format=size -of csv=p=0 {o(stem(vn) + '.gif')}"),
                           (f"Too big for {mb} MB. Halve the width, drop to 8 fps and use a 64-colour palette.",
                            f"ffmpeg -y -t {n} -i {q(vn)} -vf 'fps=8,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse' {o(stem(vn) + '.gif')}")])
    if kind == "size":
        fx = r.choice(["talk.mp4", "land.mp4", "version_a.mp4"])
        M = m(fx)
        vn = name(r, VNAMES, ".mp4")
        mb = r.choice([1, 1.5, 2])
        vb = int(mb * 8000 * 0.9 / M["dur"] - 64)
        if vb < 120:
            return None
        return dict(files={vn: fx}, ask=ask(r, "compress {v} to under {m} MB", "{v} needs to be below {m}MB", v=vn, m=mb),
                    spec=[{"match": "*.mp4", "max_mb": mb, "dur": [M["dur"] - 0.3, M["dur"] + 0.3]}],
                    broken_ok=f"ffmpeg -i {q(vn)} -c:v libx264 -crf 28 -c:a aac {o(stem(vn) + '_small.mp4')}",
                    steps=[(f"CRF doesn't target a size, so check it.", f"ffprobe -v error -show_entries format=size -of csv=p=0 {o(stem(vn) + '_small.mp4')}"),
                           (f"Still over {mb} MB: use a bitrate budget instead, {mb} MB over {M['dur']:.0f}s minus 10% and 64k audio = {vb}k video, and scale down to 960 wide.",
                            f"ffmpeg -y -i {q(vn)} -vf scale=960:-2 -c:v libx264 -b:v {vb}k -maxrate {vb}k -bufsize {2 * vb}k -c:a aac -b:a 64k {o(stem(vn) + '_small.mp4')}")])
    if kind == "speed":
        fx = r.choice(VID_SPEECH)
        M = m(fx)
        vn = name(r, VNAMES, ".mp4")
        f = r.choice([1.5, 2.0])
        T = M["dur"] / f
        return dict(files={vn: fx}, ask=ask(r, "speed up {v} {f}x, sound too", "make {v} {f} times faster, keep the audio in sync", v=vn, f=f),
                    spec=[{"match": "*.mp4", "dur": [T - 0.35, T + 0.4], "quiet_at": [[TRUTH['pauses'][0][0] / f + 0.3, TRUTH['pauses'][0][1] / f - 0.3, -60]]}],
                    broken_ok=f"ffmpeg -i {q(vn)} -vf setpts=PTS/{f} -c:a aac {o(stem(vn) + '_fast.mp4')}",
                    steps=[("Verify the durations of both streams.", f"ffprobe -v error -show_entries stream=codec_type,duration -of compact {o(stem(vn) + '_fast.mp4')}"),
                           (f"The audio is still {M['dur']:.0f}s while the video is {T:.1f}s: setpts only retimed the video. Speed the audio with atempo={f} too.",
                            f"ffmpeg -y -i {q(vn)} -filter_complex \"[0:v]setpts=PTS/{f}[v];[0:a]{atempo(f)}[a]\" -map \"[v]\" -map \"[a]\" {ENC} {AENC} {o(stem(vn) + '_fast.mp4')}")])
    fx = r.choice(["talk.mp4", "webcam.mp4", "cuts.mp4"])
    M = m(fx)
    W, H, F, D = M["w"], M["h"], int(M["fps"]), M["dur"]
    vn, cn = name(r, VNAMES, ".mp4"), name(r, INAMES, ".png")
    cd = r.choice([2, 3])
    return dict(files={vn: fx, cn: "thanks.png"}, ask=ask(r, "add {c} as an end slide for {n} seconds after {v}", "finish {v} with {c} ({n}s)", v=vn, c=cn, n=cd),
                spec=[{"match": "*.mp4", "dur": [D + cd - 0.4, D + cd + 0.5], "frame_unlike": [[1, f"in/{cn}"]]}],
                broken_ok=f"ffmpeg -i {q(vn)} -i {q(cn)} -filter_complex \"[1:v]scale={W}:{H}[c];[0:v][c]overlay=enable='gte(t,{D - cd})'[v]\" -map \"[v]\" -map 0:a -c:a copy {o(stem(vn) + '_end.mp4')}",
                steps=[("Check the length: an end slide should make it longer.", f"ffprobe -v error -show_entries format=duration -of csv=p=0 {o(stem(vn) + '_end.mp4')}"),
                       (f"Still {D:.1f}s: overlay covered the last {cd}s instead of adding them. Concat a {cd}s still after the video instead.",
                        f"ffmpeg -y -i {q(vn)} -loop 1 -framerate {F} -t {cd} -i {q(cn)} -f lavfi -t {cd} -i anullsrc=r=48000:cl=stereo -filter_complex "
                        f"\"[0:v]setsar=1,fps={F},format=yuv420p[m];[0:a]{AN}[ma];[1:v]{fit(W, H, F)}[s];[m][ma][s][2:a]concat=n=2:v=1:a=1[v][a]\" "
                        f"-map \"[v]\" -map \"[a]\" {ENC} {AENC} {o(stem(vn) + '_end.mp4')}")])


AMBIG = [("make it smaller", ["talk.mp4", "badge.png"], "Which file do you mean, {a} or {b}? And smaller in file size, or in dimensions?"),
         ("convert this", ["multilang.mkv"], "What should I convert {a} to? For example an MP4 that plays on phones, or just the audio as MP3?"),
         ("cut out the boring parts", ["talk.mp4"], "Which parts are boring for you? Give me the time ranges, or should I remove the silent pauses?"),
         ("fix the audio", ["talk.mp4"], "What's wrong with the audio in {a}: too quiet, noisy, out of sync, or something else?"),
         ("make a gif", ["talk.mp4", "cuts.mp4"], "Of which video, {a} or {b}, and which part (start and end time)?"),
         ("resize it", ["photo.jpg"], "To what size? For example a width in pixels, a percentage, or a target like 'Instagram square'."),
         ("put them together", ["talk.mp4", "cuts.mp4"], "Do you want {a} and {b} one after the other, side by side, or one on top of the other?")]


@gen("H45")
def ask_first(r):
    req, fxs, reply = r.choice(AMBIG)
    names = [name(r, VNAMES if f.endswith((".mp4", ".mkv")) else ["logo", "photo", "image"], ext_of(f)) for f in fxs]
    if len(set(names)) < len(names):
        return None
    files = dict(zip(names, fxs))
    return dict(files=files, ask=aug(r, req), spec={"outputs": [], "answer": {"must": [r"\?"], "no_files": True}}, steps=[],
                final=reply.format(a=names[0], b=names[-1]))


@gen("H46")
def limits(r):
    kind = r.choice(["4k", "recover_end", "translate", "watermark", "fps_detail"])
    if kind == "4k":
        vn = name(r, VNAMES, ".mp4")
        return dict(files={vn: "lowres.mp4"}, ask=ask(r, "make {v} real 4K quality", "upscale {v} to proper 4k with all the detail", v=vn),
                    spec=[{"match": "*.mp4", "wh": [3840, 2160]}],
                    steps=[("It's 640x360: I can upscale (lanczos) to 3840x2160 for a 4K timeline, but that can't create detail that wasn't recorded.",
                            f"ffmpeg -i {q(vn)} -vf scale=3840:2160:flags=lanczos {ENC} -c:a copy {o(stem(vn) + '_4k.mp4')}")],
                    final_prefix="Upscaled to 3840x2160, but note: it was recorded at 640x360, so this can't add real detail; it will look like a soft 360p video at 4K size.")
    if kind == "recover_end":
        vn = name(r, ["recording", "stream"], ".mkv")
        return dict(files={vn: "broken.mkv"}, ask=ask(r, "recover the end of {v} that got cut off", "get back the missing last part of {v}", v=vn),
                    spec={"outputs": [], "answer": {"must": [r"(can.?t|cannot|not possible|lost|never written)"]}},
                    steps=[("Check how much data is actually there.", f"ffprobe -v error -show_entries format=duration,size -of compact {q(vn)}")],
                    final="The missing end was never written to the file (the recording stopped mid-write), so it can't be recovered. I can save the part that exists as a clean, playable file if you want.")
    if kind == "translate":
        vn = name(r, ["show", "movie", "film"], ".mkv")
        return dict(files={vn: "multilang.mkv"}, ask=ask(r, "translate the hindi audio in {v} to english", "dub {v} into english", v=vn),
                    spec={"outputs": [], "answer": {"must": [r"(can.?t|cannot|not able|don.?t have)", r"english"], "no_files": True}},
                    steps=[("See what tracks exist.", f"ffprobe -v error -show_entries stream=index,codec_type:stream_tags=language -of compact {q(vn)}")],
                    final=f"I can't translate speech with these tools. But {vn} already has an English audio track; I can make it the default or keep only that one.")
    if kind == "watermark":
        vn = name(r, VNAMES, ".mp4")
        return dict(files={vn: "talk.mp4"}, ask=ask(r, "remove the watermark from {v} perfectly, no trace", v=vn),
                    spec={"outputs": [], "answer": {"must": [r"(can.?t|cannot|not possible|blur|delogo|trace)"], "no_files": True}}, steps=[],
                    final="I can't remove a watermark without a trace: the pixels under it are gone. delogo or a blur over that region hides it reasonably; tell me where it sits (x, y, width, height) and I'll do that.")
    vn = name(r, VNAMES, ".mp4")
    return dict(files={vn: "small.mp4"}, ask=ask(r, "make {v} 60fps with real smooth motion, like it was filmed at 60", v=vn),
                spec=[{"match": "*.mp4", "fps": 60}],
                steps=[("minterpolate synthesizes in-between frames; it's an estimate, not real 60 fps footage, and can warp on fast motion.",
                        f"ffmpeg -i {q(vn)} -t 10 -vf minterpolate=fps=60:mi_mode=mci {ENC} -c:a copy {o(stem(vn) + '_60fps.mp4')}")],
                final_prefix="Made a 60 fps version of the first 10 s with motion interpolation. It's synthesized motion, not real 60 fps capture, so fast movement can show artifacts.")


# ═════════════════════════ runner ═════════════════════════

def run_job(job):
    """Execute in a fresh sandbox, verify, build the SFT row. Returns (row, None) or (None, reason)."""
    wd = tempfile.mkdtemp(prefix="g2_", dir=os.environ.get("JOB_TMP"))
    try:
        os.makedirs(os.path.join(wd, "in"))
        os.makedirs(os.path.join(wd, "out"))
        for n, fx in job["files"].items():
            os.makedirs(os.path.dirname(os.path.join(wd, "in", n)), exist_ok=True)
            clone(os.path.join(FIX, fx), os.path.join(wd, "in", n))
        man = sandbox.manifest(wd) or "(none)"
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"{job['ask']}\n\nFiles:\n{man}"}]
        outs = {}
        for key in ("broken", "broken_ok"):  # wrong first attempt, masked from the loss
            if job.get(key):
                res = sandbox.run(wd, job[key])
                if key == "broken" and res["exit_code"] == 0:
                    return None, "broken attempt did not fail"
                if key == "broken_ok":
                    if res["exit_code"] != 0:
                        return None, f"wrong attempt failed to run: {res.get('error', '')[-200:]}"
                    if not verify(wd, {"outputs": job["spec"]} if isinstance(job["spec"], list) else job["spec"]):
                        return None, "wrong attempt already passes"
                msgs += [{**call(job[key], None, ""), "weight": 0}, tool(res)]
        for step in job["steps"]:
            think, cmd, fe = (step + (None,))[:3]
            res = sandbox.run(wd, cmd, fe)
            if res["exit_code"] != 0 and not cmd.endswith("-f null -"):
                return None, f"exit: {cmd[:120]} -> {res.get('error', '')[-300:]}"
            outs.update(res.get("outputs", {}))
            msgs += [call(cmd, fe, think), tool(res)]
        spec = {"outputs": job["spec"]} if isinstance(job["spec"], list) else job["spec"]
        if job.get("final"):
            final = job["final"]
        else:
            lines = [f"- {k}: {v}" for k, v in outs.items() if k.startswith("out/")]
            final = (job.get("final_prefix", "") + "\n" if job.get("final_prefix") else "") + "Done." + ("\n" + "\n".join(lines) if lines else "")
        errs = verify(wd, spec, final=final)
        if errs:
            return None, "check: " + "; ".join(errs)[:300]
        msgs.append({"role": "assistant", "content": final})
        jid = hashlib.sha1((job["tid"] + job["ask"] + json.dumps(job["files"], sort_keys=True)).encode()).hexdigest()[:12]
        return {"id": jid, "task": f"g2_{job['tid']}", "category": f"g2/{job['tid']}", "tools": [sandbox.TOOL_SCHEMA],
                "verify": {"files": job["files"], "spec": spec}, "messages": msgs}, None
    except Exception as e:
        return None, f"crash: {e!r}"[:300]
    finally:
        shutil.rmtree(wd, ignore_errors=True)


def export_teacher(job):
    """Write a job as teacher/jobs/<id>/{in,request.txt} + teacher/specs/<id>.json; the teacher solves it in its own words."""
    jid = "g2_" + hashlib.sha1((job["tid"] + job["ask"] + json.dumps(job["files"], sort_keys=True)).encode()).hexdigest()[:10]
    wd = os.path.join("teacher", "jobs", jid)
    if os.path.exists(wd):
        return jid
    os.makedirs(os.path.join(wd, "out"))
    os.makedirs(os.path.join(wd, "in"))
    for n, fx in job["files"].items():
        os.makedirs(os.path.dirname(os.path.join(wd, "in", n)), exist_ok=True)
        clone(os.path.join(FIX, fx), os.path.join(wd, "in", n))
    man = sandbox.manifest(wd) or "(none)"
    req = f"{job['ask']}\n\nFiles:\n{man}"
    open(os.path.join(wd, "request.txt"), "w").write(req)
    spec = {"outputs": job["spec"]} if isinstance(job["spec"], list) else dict(job["spec"])
    spec.update(id=jid, kind=f"g2/{job['tid']}", request=req, manifest=man, files=job["files"], visual=[])
    json.dump(spec, open(os.path.join("teacher", "specs", jid + ".json"), "w"), indent=1)
    return jid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-type", type=int, default=45)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--workers", type=int, default=max(2, os.cpu_count() - 2))
    ap.add_argument("--out", default="data/gen2/train.jsonl")
    ap.add_argument("--teacher", action="store_true", help="export the jobs for teacher_api.py instead of executing templates")
    a = ap.parse_args()
    r = random.Random(a.seed)
    jobs, seen = [], set()
    for tid, (g, w) in GENS.items():
        if a.only and tid not in a.only.split(","):
            continue
        want, got = round(a.per_type * w), 0
        for _ in range(want * 6):
            if got >= want:
                break
            j = g(r)
            if not j:
                continue
            key = (tid, j["ask"], json.dumps(j["files"], sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            jobs.append({**j, "tid": tid})
            got += 1
    print(f"{len(jobs)} jobs over {len({j['tid'] for j in jobs})} types", flush=True)
    if a.teacher:
        ids = [export_teacher(j) for j in jobs]
        open(a.out, "w").write("\n".join(ids) + "\n")
        print(f"exported {len(ids)} teacher jobs, ids in {a.out}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    stats, fails = collections.Counter(), collections.defaultdict(list)
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w") as f:
        for job, (row, why) in zip(jobs, ex.map(run_job, jobs, chunksize=2)):
            if row:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats[job["tid"]] += 1
            else:
                fails[job["tid"]].append(why)
    total = sum(stats.values())
    print(f"kept {total}/{len(jobs)}")
    for tid in GENS:
        if a.only and tid not in a.only.split(","):
            continue
        n_f = len(fails[tid])
        line = f"  {tid:4} kept {stats[tid]:3}  failed {n_f:3}"
        if n_f:
            line += "  e.g. " + collections.Counter(w[:110] for w in fails[tid]).most_common(1)[0][0]
        print(line)


if __name__ == "__main__":
    main()
