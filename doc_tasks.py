"""Batch 4: features from the ffmpeg / ImageMagick / SoX / ExifTool docs that the 46 task types don't cover.

Each feature has a request, a reference command (proves its check is passable) and a check that rejects an unchanged copy.
They register into gen2's generator table, so gen2's runner and teacher export are reused:

    python3 doc_tasks.py --per-type 2 --out /tmp/x.jsonl            # validate references (template rows)
    python3 doc_tasks.py --teacher --per-type 3 --out ids.txt       # export for teacher_api.py
"""
import os

import gen2
from gen2 import META, TRUTH, FIX, ask, name, q, o, stem

V = ["talk.mp4", "short.mp4", "webcam.mp4", "version_a.mp4", "lowres.mp4"]
A = ["narration.m4a", "music.mp3", "a.wav", "voice_m.m4a"]
I = ["photo.jpg", "cover.jpg", "screenshot.png", "thanks.png"]
POOL = {"v": V, "a": A, "i": I}
NAMES = {"v": gen2.VNAMES, "a": gen2.ANAMES + gen2.MNAMES, "i": ["photo", "pic", "image", "shot", "img", "picture", "banner"]}
_lufs = {}


def lufs_of(fx):
    if fx not in _lufs:
        from teacher import lufs
        _lufs[fx] = lufs(os.path.join(FIX, fx))
    return _lufs[fx]


def feature(fid, kind, asks, cmd, chk=None, fx=None, ext=None, suffix="", thr=0.01, final=None, slots=None):
    """kind: v/a/i (effect on one input of that type, default checks) or q (question; chk gives the regexes)."""
    slots = slots or ([kind] if kind in "vai" else [])

    def g(r):
        files, c, used = {}, {}, set()
        for k, sl in enumerate(slots):
            src = (fx or {}).get(sl + ("2" if k and sl in slots[:k] else "")) or r.choice(POOL[sl[0]])
            if callable(src):
                src = src(r)
            key = sl + ("2" if sl in slots[:k] else "")
            nm = name(r, NAMES[sl[0]], os.path.splitext(src)[1], avoid=used)
            used.add(nm)
            files[nm] = src
            c[key], c[key.upper()], c[key + "_fx"] = nm, q(nm), src
        first = next(iter(files), None)
        M = META.get(files[first], {}) if first else {}
        W, H, D = M.get("w"), M.get("h"), M.get("dur")
        e = ext or (os.path.splitext(first)[1] if first and kind in "ai" else ".mp4")
        out = (stem(first) if first else "out") + (suffix or "_" + fid.lower()) + e
        c.update(W=W, H=H, D=D, O=o(out), out=out, M=M, fade_out=round(D - 1, 2) if D else 0, fade_out3=round(D - 3, 2) if D else 0)
        req = ask(r, *asks, **c)
        if kind == "q":
            spec = {"outputs": [], "answer": {"must": chk(c), "no_files": True}}
        else:
            dur = [D - 0.4, D + 0.4] if D else None
            if kind == "v":
                base = {"match": "*" + e, "wh": [W, H], "dur": dur, "frame_unlike": [[2, f"in/{c.get('v')}@2", thr]]}
            elif kind == "a":
                base = {"match": "*" + e, "dur": dur, "audio_unlike": [f"in/{c.get('a')}", 1.0]}
            elif kind == "i":
                base = {"match": "*" + e, "wh": [W, H], "frame_unlike": [[0, f"in/{c.get('i')}", thr]]}
            else:
                base = {"match": "*" + e}
            base.update(chk(c) if chk else {})
            spec = [{k: v for k, v in base.items() if v is not None}]
        job = dict(files=files, ask=req, spec=spec, steps=[("", cmd.format(**c))])
        if final:
            job["final"] = final(c)
        return job
    gen2.GENS[fid] = (g, 1.0)


def same(c):
    return {}


# ───────────── video effects (same size and length, picture changed) ─────────────
VF = [
    ("VF01", ["mirror {v} horizontally", "flip {v} left to right"], "hflip", 0.01),
    ("VF02", ["turn {v} upside down", "flip {v} vertically"], "vflip", 0.01),
    ("VF03", ["rotate {v} 180 degrees", "{v} was filmed upside down, rotate it 180"], "hflip,vflip", 0.01),
    ("VF04", ["make {v} black and white", "grayscale version of {v}"], "hue=s=0", -1),
    ("VF05", ["give {v} a sepia, old film look", "sepia tone for {v}"], "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131", 0.01),
    ("VF06", ["add a dark vignette around the edges of {v}", "vignette effect on {v}"], "vignette=PI/4", 0.003),
    ("VF07", ["blur the whole of {v} heavily", "make {v} really blurry"], "gblur=sigma=12", 0.003),
    ("VF08", ["pixelate {v}, big blocks", "mosaic / pixel effect over all of {v}"], "pixelize=w=24:h=24", 0.003),
    ("VF09", ["invert the colors of {v}", "negative effect on {v}"], "negate", 0.05),
    ("VF10", ["edge detection effect on {v}, like a sketch", "make {v} look like line art"], "edgedetect", 0.05),
    ("VF11", ["make the colors in {v} pop, double the saturation", "boost saturation of {v}"], "eq=saturation=2", -1),
    ("VF12", ["more contrast in {v}", "increase the contrast of {v} a lot"], "eq=contrast=1.6", 0.01),
    ("VF13", ["make {v} darker", "{v} is too bright, bring it down"], "eq=brightness=-0.25", 0.02),
    ("VF14", ["shift the hues in {v} by 90 degrees", "trippy color shift on {v}"], "hue=h=90", -1),
    ("VF15", ["warm up the colors of {v}, like golden hour", "make {v} look warmer"], "colortemperature=temperature=3000", 0.002),
    ("VF16", ["vintage color grade on {v}", "give {v} a retro look"], "curves=preset=vintage", 0.005),
    ("VF17", ["add film grain to {v}", "noisy, grainy look for {v}"], "noise=alls=40:allf=t", 0.002),
    ("VF18", ["draw a solid red box at x=100 y=80, 300x200, on {v}", "put a red rectangle over part of {v} (100,80, 300 by 200)"],
     "drawbox=x=100:y=80:w=300:h=200:color=red:t=fill", 0.003),
    ("VF19", ["burn a running timer into the top left corner of {v}", "show the playback time on {v}"],
     "drawtext=text='%{{pts\\:hms}}':x=20:y=20:fontsize=40:fontcolor=white:box=1:boxcolor=black@0.6", 0.0006),
    ("VF20", ["add '(c) Studio North' in the bottom right of {v}", "text watermark 'Studio North' bottom-right on {v}"],
     "drawtext=text='(c) Studio North':x=w-tw-20:y=h-th-20:fontsize=36:fontcolor=white@0.8", 0.0006),
    ("VF21", ["ghosting / motion trail effect on {v}", "make movement in {v} leave trails"], "lagfun=decay=0.95", 0.0),
    ("VF22", ["RGB split glitch effect on {v}", "chromatic aberration look for {v}"], "chromashift=cbh=12:crh=-12", -1),
    ("VF23", ["brighten {v} by one stop", "+1 exposure on {v}"], "exposure=exposure=1", 0.008),
    ("VF24", ["mirror the left half of {v} onto the right", "symmetric mirror effect on {v}"],
     "crop=iw/2:ih:0:0,split[l][r];[r]hflip[f];[l][f]hstack", 0.002),
]
for fid, asks, vf, thr in VF:
    feature(fid, "v", asks, f"ffmpeg -i {{V}} -vf \"{vf}\" -c:v libx264 -crf 22 -pix_fmt yuv420p -c:a copy {{O}}",
            chk=(lambda c: {"frame_unlike": None}) if thr == 0.0 else (lambda c: {"frame_unlike": None, "chroma_unlike": [f"in/{c['v']}", 1.5]}) if thr < 0 else None,
            thr=thr, fx={"v": lambda r: r.choice(["talk.mp4", "short.mp4", "version_a.mp4"])} if thr < 0 else None)

# ───────────── video: time, layout, generation ─────────────
feature("VT01", "v", ["play {v} backwards, sound too", "reverse {v}"], "ffmpeg -i {V} -vf reverse -af areverse -c:v libx264 -pix_fmt yuv420p {O}",
        chk=lambda c: {"frame_unlike": None, "frame_like": [[1, f"in/{c['v']}@{round(c['D'] - 1, 2)}", 0.05]]}, fx={"v": lambda r: r.choice(["version_a.mp4", "short.mp4"])})
feature("VT02", "v", ["boomerang of {v}: forwards then backwards", "make {v} play forward and then in reverse"],
        "ffmpeg -i {V} -filter_complex \"[0:v]split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[v]\" -map \"[v]\" -an -c:v libx264 -pix_fmt yuv420p {O}",
        chk=lambda c: {"dur": [2 * c["D"] - 0.4, 2 * c["D"] + 0.4], "frame_unlike": None}, fx={"v": "version_a.mp4"})
feature("VT03", "v", ["loop {v} three times", "repeat {v} 3x back to back"], "ffmpeg -stream_loop 2 -i {V} -c copy {O}",
        chk=lambda c: {"dur": [3 * c["D"] - 0.5, 3 * c["D"] + 0.5], "frame_unlike": None})
feature("VT04", "v", ["timelapse of {v}, 10x faster, no sound", "speed {v} up 10 times, drop the audio"],
        "ffmpeg -i {V} -vf setpts=PTS/10 -an -c:v libx264 -pix_fmt yuv420p {O}",
        chk=lambda c: {"dur": [c["D"] / 10 - 0.4, c["D"] / 10 + 0.4], "frame_unlike": None}, fx={"v": lambda r: r.choice(["small.mp4", "land.mp4"])})
feature("VT05", "v", ["half speed slow motion of {v}, audio slowed too", "slow {v} down to 0.5x"],
        "ffmpeg -i {V} -filter_complex \"[0:v]setpts=2*PTS[v];[0:a]atempo=0.5[a]\" -map \"[v]\" -map \"[a]\" -c:v libx264 -pix_fmt yuv420p {O}",
        chk=lambda c: {"dur": [2 * c["D"] - 0.4, 2 * c["D"] + 0.4], "frame_unlike": None})
feature("VT06", "v", ["convert {v} to 24 fps", "change the frame rate of {v} to 24"], "ffmpeg -i {V} -vf fps=24 -c:v libx264 -pix_fmt yuv420p -c:a copy {O}",
        chk=lambda c: {"fps": 24, "frame_unlike": None})
feature("VT07", "v", ["stabilize the shaky footage in {v}", "{v} is shaky, steady it"], "ffmpeg -i {V} -vf deshake -c:v libx264 -pix_fmt yuv420p -c:a copy {O}",
        chk=lambda c: {"frame_unlike": None})
feature("VT08", "v", ["crop {v} to a centered square", "square crop of {v} from the middle"], "ffmpeg -i {V} -vf crop=ih:ih -c:v libx264 -pix_fmt yuv420p -c:a copy {O}",
        chk=lambda c: {"wh": [c["H"], c["H"]], "frame_unlike": None})
feature("VT09", "v", ["keep only the top-left quarter of {v}", "crop {v} to its top left quarter"],
        "ffmpeg -i {V} -vf crop=iw/2:ih/2:0:0 -c:v libx264 -pix_fmt yuv420p -c:a copy {O}", chk=lambda c: {"wh": [c["W"] // 2, c["H"] // 2], "frame_unlike": None})
feature("VT10", "v", ["make {v} square by adding black bars, don't crop anything", "letterbox {v} into a 1:1 frame"],
        "ffmpeg -i {V} -vf \"pad=iw:iw:0:(ow-ih)/2\" -c:v libx264 -pix_fmt yuv420p -c:a copy {O}", chk=lambda c: {"wh": [c["W"], c["W"]], "frame_unlike": None})
feature("VT11", "v", ["vertical 1080x1920 version of {v} with a blurred copy filling the background", "9:16 of {v}, no crop, blurred background bars"],
        "ffmpeg -i {V} -filter_complex \"[0:v]split[a][b];[a]scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,gblur=sigma=30[bg];"
        "[b]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]\" -map \"[v]\" -map 0:a -c:v libx264 -pix_fmt yuv420p -c:a copy {O}",
        chk=lambda c: {"wh": [1080, 1920], "frame_unlike": None})
feature("VT12", "v", ["2x2 grid of {v} repeated four times", "show {v} four times in a grid"],
        "ffmpeg -i {V} -filter_complex \"[0:v]scale=iw/2:ih/2,split=4[a][b][c][d];[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]\" -map \"[v]\" -map 0:a -c:v libx264 -pix_fmt yuv420p -c:a copy {O}",
        chk=lambda c: {"frame_unlike": [[2, f"in/{c['v']}@2", 0.003]]}, fx={"v": lambda r: r.choice(["talk.mp4", "short.mp4", "version_a.mp4"])})
feature("VT13", "v", ["add a 1 second fade in from black and fade out at the end of {v}, audio too", "fade {v} in and out"],
        "ffmpeg -i {V} -vf \"fade=t=in:d=1,fade=t=out:st={fade_out}:d=1\" -af \"afade=t=in:d=1,afade=t=out:st={fade_out}:d=1\" -c:v libx264 -pix_fmt yuv420p {O}",
        chk=lambda c: {"frame_unlike": [[0.05, f"in/{c['v']}@0.05", 0.03], [-0.05, f"in/{c['v']}@{round(c['D'] - 0.05, 2)}", 0.03]]})
feature("VT14", "v", ["put {i} in the top right corner of {v} at half opacity", "semi-transparent logo {i} top-right on {v}"],
        "ffmpeg -i {V} -i {I} -filter_complex \"[1:v]scale=160:-1,format=rgba,colorchannelmixer=aa=0.5[l];[0:v][l]overlay=W-w-20:20[v]\" -map \"[v]\" -map 0:a -c:v libx264 -pix_fmt yuv420p -c:a copy {O}",
        chk=lambda c: {"frame_unlike": [[2, f"in/{c['v']}@2", 0.0005]]}, slots=["v", "i"], fx={"i": "badge.png"})
feature("VT15", "v", ["contact sheet of {v}: a 4x2 grid of frames, one every 2 seconds", "thumbnail grid image for {v}, 8 frames"],
        "ffmpeg -i {V} -vf \"fps=1/2,scale=320:-2,tile=4x2\" -frames:v 1 {O}", ext=".jpg",
        chk=lambda c: {"wh": [1280, 2 * (320 * c["H"] // c["W"] // 2 * 2)], "dur": None, "frame_unlike": None})
feature("VT16", "v", ["pick the most representative frame of {v} as a jpg", "best thumbnail frame from {v}"],
        "ffmpeg -i {V} -vf thumbnail -frames:v 1 {O}", ext=".jpg", chk=lambda c: {"dur": None, "frame_unlike": None})
feature("VT17", "v", ["animated zoom into {i}: 5 second 1280x720 video", "slow ken burns zoom on {i}, 5s"],
        "ffmpeg -loop 1 -i {I} -vf \"scale=2560:-2,zoompan=z='min(zoom+0.0015,1.3)':d=125:s=1280x720:fps=25\" -t 5 -c:v libx264 -pix_fmt yuv420p {O}",
        slots=["i"], chk=lambda c: {"wh": [1280, 720], "dur": [4.6, 5.4], "frame_unlike": None})
feature("VT18", "x", ["make a 10 second 1080p color bars test video with a 1 kHz tone", "smpte bars + 1kHz test tone, 10s, full hd"],
        "ffmpeg -f lavfi -i smptehdbars=s=1920x1080:d=10 -f lavfi -i sine=f=1000:d=10 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest {O}",
        chk=lambda c: {"wh": [1920, 1080], "dur": [9.6, 10.4], "has_audio": True})
feature("VT19", "x", ["5 second countdown video, white numbers on black, 720p", "make a 5-4-3-2-1 countdown clip"],
        "ffmpeg -f lavfi -i color=black:s=1280x720:d=5 -vf \"drawtext=text='%{{eif\\:5-t\\:d}}':fontsize=300:fontcolor=white:x=(w-tw)/2:y=(h-th)/2\" -c:v libx264 -pix_fmt yuv420p {O}",
        chk=lambda c: {"wh": [1280, 720], "dur": [4.6, 5.4]})
feature("VT20", "x", ["10 second credits video: 'Directed by Sam Lee' scrolling up on black, 720p", "rolling credit 'Directed by Sam Lee', 10s"],
        "ffmpeg -f lavfi -i color=black:s=1280x720:d=10 -vf \"drawtext=text='Directed by Sam Lee':fontsize=60:fontcolor=white:x=(w-tw)/2:y=h-t*80\" -c:v libx264 -pix_fmt yuv420p {O}",
        chk=lambda c: {"wh": [1280, 720], "dur": [9.6, 10.4]})
feature("VT21", "a", ["spectrum visualizer video for {a}, 1280x720", "make an animated spectrum video from {a}"],
        "ffmpeg -i {A} -filter_complex \"[0:a]showspectrum=s=1280x720:slide=scroll,format=yuv420p[v]\" -map \"[v]\" -map 0:a -c:v libx264 -c:a aac {O}",
        ext=".mp4", chk=lambda c: {"wh": [1280, 720], "audio_unlike": None, "has_audio": True}, fx={"a": lambda r: r.choice(["narration.m4a", "voice_m.m4a"])})
feature("VT22", "a", ["waveform picture of {a}, 1600x400 png", "draw the waveform of {a} as an image"],
        "ffmpeg -i {A} -filter_complex showwavespic=s=1600x400 -frames:v 1 {O}", ext=".png", chk=lambda c: {"wh": [1600, 400], "dur": None, "audio_unlike": None})
feature("VT23", "a", ["spectrogram image of {a}", "make a spectrogram png from {a}"],
        "ffmpeg -i {A} -lavfi showspectrumpic=s=1024x512 {O}", ext=".png", chk=lambda c: {"dur": None, "audio_unlike": None})

# ───────────── containers / codecs ─────────────
for fid, asks, args, e, codec in [
    ("VC01", ["convert {v} to webm (vp9)", "vp9 webm version of {v} for the web"], "-c:v libvpx-vp9 -b:v 0 -crf 35 -c:a libopus", ".webm", "vp9"),
    ("VC02", ["re-encode {v} as hevc / h.265 to save space", "h265 version of {v}"], "-c:v libx265 -crf 28 -tag:v hvc1 -c:a copy", ".mp4", "hevc"),
    ("VC03", ["encode {v} as AV1", "av1 version of {v}"], "-c:v libsvtav1 -crf 35 -c:a copy", ".mp4", "av1"),
    ("VC04", ["convert {v} to ProRes for editing in Final Cut", "prores mov of {v}"], "-c:v prores_ks -profile:v 2 -c:a pcm_s16le", ".mov", "prores"),
    ("VC05", ["remux {v} into mkv without re-encoding", "put {v} in an mkv container, no quality loss"], "-c copy", ".mkv", "h264"),
]:
    feature(fid, "v", asks, f"ffmpeg -i {{V}} {args} {{O}}", ext=e, chk=(lambda cd: lambda c: {"vcodec": cd, "frame_unlike": None})(codec))
feature("VC06", "v", ["cut 2s to 7s out of {v} without re-encoding", "lossless trim of {v}: 2 to 7 seconds"], "ffmpeg -ss 2 -to 7 -i {V} -c copy {O}",
        chk=lambda c: {"dur": [4, 6.2], "frame_unlike": None})
feature("VC07", "v", ["set the title metadata of {v} to 'Summer Trip'", "{v}: title tag 'Summer Trip'"], "ffmpeg -i {V} -c copy -metadata title='Summer Trip' {O}",
        chk=lambda c: {"exif_eq": {"Title": "Summer Trip"}, "frame_unlike": None})
feature("VC08", "v", ["add {i} as the cover art / thumbnail of {v}", "embed {i} as the poster image in {v}"],
        "ffmpeg -i {V} -i {I} -map 0 -map 1 -c copy -disposition:v:1 attached_pic {O}", slots=["v", "i"], fx={"i": "cover.jpg"},
        chk=lambda c: {"frame_unlike": None})
feature("VC09", "v", ["add {s} to {v} as soft subtitles that can be turned off", "attach {s} as a subtitle track to {v}"],
        "ffmpeg -i {V} -i {S} -map 0 -map 1 -c copy -c:s mov_text -metadata:s:s:0 language=eng {O}", slots=["v", "s"],
        fx={"v": "talk.mp4", "s": "captions.srt"}, chk=lambda c: {"frame_unlike": None})
POOL["s"], NAMES["s"] = ["captions.srt"], ["subs", "captions", "english"]

# ───────────── audio effects ─────────────
AF = [
    ("AF01", ["add reverb to {a}", "make {a} sound like it's in a big room"], "ffmpeg -i {A} -af aecho=0.8:0.88:60:0.4 {O}"),
    ("AF02", ["chorus effect on {a}", "add a chorus to {a}"], "ffmpeg -i {A} -af chorus=0.5:0.9:50:0.4:0.25:2 {O}"),
    ("AF03", ["flanger effect on {a}", "jet / flanger sound on {a}"], "ffmpeg -i {A} -af flanger=depth=8 {O}"),
    ("AF04", ["phaser on {a}", "add a phaser effect to {a}"], "ffmpeg -i {A} -af aphaser=type=t:speed=2 {O}"),
    ("AF05", ["tremolo effect on {a}", "make the volume of {a} wobble"], "ffmpeg -i {A} -af tremolo=f=6:d=0.9 {O}"),
    ("AF07", ["add distortion to {a}", "overdrive {a}, make it crunchy"], "sox {A} {O} overdrive 25"),
    ("AF08", ["boost the bass in {a}", "more low end on {a}"], "ffmpeg -i {A} -af bass=g=12 {O}"),
    ("AF09", ["make {a} sound muffled, like through a wall", "cut the highs on {a}"], "ffmpeg -i {A} -af lowpass=f=600,lowpass=f=600 {O}"),
    ("AF10", ["boost 1 kHz by 10 dB in {a}", "eq: +10dB at 1000 Hz on {a}"], "ffmpeg -i {A} -af equalizer=f=1000:t=q:w=1:g=10 {O}"),
    ("AF11", ["compress the dynamics of {a} so quiet and loud parts are closer", "add compression to {a}"],
     "ffmpeg -i {A} -af acompressor=threshold=-24dB:ratio=6:makeup=8 {O}"),
    ("AF12", ["make {a} 6 dB louder", "turn up {a} by 6dB"], "ffmpeg -i {A} -af volume=6dB {O}"),
    ("AF13", ["make {a} quieter, half the volume", "turn {a} down to 50%"], "ffmpeg -i {A} -af volume=0.5 {O}"),
    ("AF14", ["mute {a} between 6.5 and 7.5 seconds", "silence {a} from 6.5s to 7.5s"], "ffmpeg -i {A} -af \"volume=enable='between(t,6.5,7.5)':volume=0\" {O}"),
    ("AF15", ["make {a} play backwards", "reverse the audio {a}"], "ffmpeg -i {A} -af areverse {O}"),
    ("AF16", ["remove noise from {a} using its own noise profile", "sox noise reduction on {a}"],
     "sox {A} -n noiseprof tmp_noise.prof && sox {A} {O} noisered tmp_noise.prof 0.3"),
    ("AF17", ["raise the pitch of {a} by 3 semitones, same length", "{a} 3 semitones higher"], "ffmpeg -i {A} -af rubberband=pitch=1.189 {O}"),
]
for fid, asks, cmd in AF:
    ck = None
    if fid == "AF14":
        ck = lambda c: {"quiet_at": [[6.6, 7.4, -60]], "audio_unlike": None}
    elif fid == "AF15":
        ck = lambda c: {"audio_unlike": None, "loud_at": [[16 - 14.2, 16 - 12.3, -45]], "quiet_at": [[16 - 11.8, 16 - 8.7, -60]]}
    elif fid == "AF17":
        ck = lambda c: {"audio_unlike": None, "pitch_ratio": [f"in/{c['a']}", 1.08, 1.32]}
    voice = fid in ("AF07", "AF10", "AF14", "AF15", "AF16", "AF17")
    pool = {"AF07": lambda r: r.choice(["a.wav", "hum.wav"]), "AF10": lambda r: r.choice(["narration.m4a", "voice_m.m4a"]), "AF14": "narration.m4a", "AF15": "narration.m4a", "AF16": "hum.wav"}.get(fid)
    feature(fid, "a", asks, cmd, chk=ck, fx={"a": pool or (lambda r: r.choice(["narration.m4a", "voice_m.m4a", "voice_f.m4a"]))} if voice else None,
            ext=".wav" if cmd.startswith("sox") else None)

# ───────────── audio: time, format, generation ─────────────
feature("AT01", "a", ["speed {a} up 1.5x without changing the pitch", "play {a} 1.5 times faster, normal voice"], "ffmpeg -i {A} -af atempo=1.5 {O}",
        chk=lambda c: {"dur": [c["D"] / 1.5 - 0.4, c["D"] / 1.5 + 0.4], "audio_unlike": None})
feature("AT02", "a", ["speed {a} up 1.5x the old-school way, pitch goes up too", "tape-speed 1.5x on {a}"],
        "ffmpeg -i {A} -af asetrate=48000*1.5,aresample=48000 {O}", fx={"a": lambda r: r.choice(["narration.m4a", "voice_m.m4a"])},
        chk=lambda c: {"dur": [c["D"] / 1.5 - 0.4, c["D"] / 1.5 + 0.4], "audio_unlike": None, "pitch_ratio": [f"in/{c['a']}", 1.3, 1.7]})
feature("AT03", "a", ["slow {a} to 80% speed with sox", "tempo 0.8 for {a}, keep pitch"], "sox {A} {O} tempo 0.8", ext=".wav",
        chk=lambda c: {"dur": [c["D"] / 0.8 - 0.5, c["D"] / 0.8 + 0.5], "audio_unlike": None}, fx={"a": "a.wav"})
feature("AT04", "a", ["fade {a} in over 3 seconds and out over 3 seconds", "3s fade in and fade out on {a}"],
        "ffmpeg -i {A} -af \"afade=t=in:d=3,afade=t=out:st={fade_out3}:d=3\" {O}", fx={"a": lambda r: r.choice(["music_bed.mp3", "music.mp3"])},
        chk=lambda c: {"quiet_at": [[0, 0.15, -40]], "audio_unlike": None})
feature("AT05", "a", ["trim the silence off the end of {a}", "remove the trailing silence in {a}"],
        "ffmpeg -i {A} -af \"areverse,silenceremove=start_periods=1:start_threshold=-50dB,areverse\" {O}", fx={"a": "narration.m4a"},
        chk=lambda c: {"dur": [14.1, 15.2], "audio_unlike": None})
feature("AT06", "a", ["add 2 seconds of silence at the end of {a}", "pad {a} with 2s of silence"], "ffmpeg -i {A} -af apad=pad_dur=2 {O}",
        chk=lambda c: {"dur": [c["D"] + 1.6, c["D"] + 2.4], "audio_unlike": None})
feature("AT07", "a", ["join {a} and then {a2} into one file", "append {a2} after {a}"],
        "ffmpeg -i {A} -i {A2} -filter_complex \"[0:a]aresample=48000,aformat=channel_layouts=stereo[x];[1:a]aresample=48000,aformat=channel_layouts=stereo[y];[x][y]concat=n=2:v=0:a=1\" {O}",
        slots=["a", "a"], fx={"a": "voice_m.m4a", "a2": "voice_f.m4a"}, ext=".m4a",
        chk=lambda c: {"dur": [3.046 + 2.483 - 0.4, 3.046 + 2.483 + 0.4], "audio_unlike": None})
feature("AT08", "a", ["loop {a} until it's 2 minutes long", "repeat {a} to fill 120 seconds"], "ffmpeg -stream_loop -1 -i {A} -t 120 {O}",
        chk=lambda c: {"dur": [119.5, 120.5], "audio_unlike": None}, fx={"a": lambda r: r.choice(["music.mp3", "a.wav", "music_bed.mp3"])})
feature("AT09", "a", ["just the first 10 seconds of {a}", "cut {a} down to its first 10s"], "ffmpeg -i {A} -t 10 {O}",
        chk=lambda c: {"dur": [9.7, 10.3], "audio_unlike": None}, fx={"a": lambda r: r.choice(["music.mp3", "a.wav", "narration.m4a", "music_bed.mp3"])})
feature("AT10", "a", ["convert {a} to 24-bit wav", "24 bit wav of {a}"], "ffmpeg -i {A} -c:a pcm_s24le {O}", ext=".wav",
        chk=lambda c: {"acodec": "pcm_s24le", "audio_unlike": None})
feature("AT11", "a", ["opus file of {a} at 32 kbps for voice", "small opus version of {a}"], "ffmpeg -i {A} -c:a libopus -b:a 32k {O}", ext=".opus",
        chk=lambda c: {"acodec": "opus", "audio_unlike": None})
feature("AT12", "a", ["ogg vorbis version of {a}", "convert {a} to .ogg"], "ffmpeg -i {A} -c:a libvorbis -q:a 5 {O}", ext=".ogg",
        chk=lambda c: {"acodec": "vorbis", "audio_unlike": None})
feature("AT13", "a", ["lossless flac of {a}", "flac version of {a}"], "ffmpeg -i {A} -c:a flac {O}", ext=".flac",
        chk=lambda c: {"acodec": "flac", "audio_unlike": None}, fx={"a": lambda r: r.choice(["a.wav", "hum.wav"])})
feature("AT14", "a", ["make an iphone ringtone from the first 30 seconds of {a}", "m4r ringtone, 30s, from {a}"],
        "ffmpeg -i {A} -t 30 -c:a aac -b:a 192k -f ipod {O}", ext=".m4r", fx={"a": lambda r: r.choice(["music.mp3", "music_bed.mp3"])},
        chk=lambda c: {"dur": [29.6, 30.4], "audio_unlike": None})
feature("AT15", "a", ["resample {a} to 22.05 kHz", "lower the sample rate of {a} to 22050"], "ffmpeg -i {A} -ar 22050 {O}",
        chk=lambda c: {"sr": 22050, "audio_unlike": None})
feature("AT16", "a", ["split {a} into separate left and right mono files", "one wav per channel from {a}"],
        "ffmpeg -i {A} -filter_complex \"channelsplit=channel_layout=stereo[l][r]\" -map \"[l]\" {OL} -map \"[r]\" {OR}".replace("{OL}", "out/left.wav").replace("{OR}", "out/right.wav"),
        chk=lambda c: {"match": "*.wav", "count": 2, "ach": 1, "audio_unlike": None, "dur": None})
feature("AT17", "a", ["merge {a} and {a2} into one stereo file, first on the left, second on the right", "{a} left channel, {a2} right channel"],
        "sox -M {A} {A2} {O}", slots=["a", "a"], fx={"a": "hum.wav", "a2": "hum.wav"}, ext=".wav", chk=lambda c: {"ach": 2, "audio_unlike": None, "dur": None})
feature("AT18", "x", ["generate a 5 second 440 Hz tone as wav", "make a 440hz beep, 5s"], "ffmpeg -f lavfi -i sine=f=440:d=5 {O}", ext=".wav",
        chk=lambda c: {"dur": [4.7, 5.3], "loud_at": [[1, 4, -40]]})
feature("AT19", "x", ["10 seconds of white noise as mp3", "make a white noise track, 10s"], "ffmpeg -f lavfi -i anoisesrc=d=10:c=white:a=0.5 -c:a libmp3lame {O}",
        ext=".mp3", chk=lambda c: {"dur": [9.6, 10.4], "loud_at": [[1, 9, -30]]})
feature("AT20", "x", ["5 second sine sweep from 100 Hz to 10 kHz", "make a frequency sweep wav, 100-10000Hz, 5s"], "sox -n {O} synth 5 sine 100-10000",
        ext=".wav", chk=lambda c: {"dur": [4.7, 5.3]})
feature("AT21", "a", ["normalize {a} to -23 LUFS for broadcast", "EBU R128 loudness on {a}"], "ffmpeg -i {A} -af loudnorm=I=-23:TP=-2 -ar 48000 {O}",
        chk=lambda c: {"lufs": [-25, -21], "audio_unlike": None}, fx={"a": lambda r: r.choice(["narration.m4a", "music.mp3"])})

# ───────────── image effects (same size, picture changed) ─────────────
IF = [
    ("IF01", ["sepia version of {i}", "make {i} look old, sepia"], "-sepia-tone 80%", 0.01),
    ("IF02", ["black and white {i}", "grayscale {i}"], "-colorspace Gray", 0.003),
    ("IF03", ["invert the colors of {i}", "negative of {i}"], "-negate", 0.05),
    ("IF04", ["charcoal drawing effect on {i}", "make {i} look like a charcoal sketch"], "-charcoal 2", 0.03),
    ("IF05", ["oil painting effect on {i}", "make {i} look painted"], "-paint 4", 0.0003),
    ("IF06", ["vignette on {i}", "darken the corners of {i}"], "-background black -vignette 0x60", 0.003),
    ("IF07", ["blur {i} a lot", "heavy blur on {i}"], "-blur 0x12", 0.001),
    ("IF08", ["swirl effect in the middle of {i}", "twist {i} with a swirl"], "-swirl 270", 0.003),
    ("IF09", ["give {i} a blue tint", "colorize {i} blue, 30%"], "-fill blue -colorize 30%", 0.01),
    ("IF10", ["make {i} more saturated and a bit brighter", "boost colors of {i}"], "-modulate 115,170", 0.003),
    ("IF11", ["posterize {i} to 4 levels", "poster effect, few colors, on {i}"], "-posterize 4", 0.003),
    ("IF12", ["emboss effect on {i}", "make {i} look embossed"], "-emboss 2", 0.02),
    ("IF13", ["pixelate {i}", "pixel-art blocks on {i}"], "-scale 5% -scale 2000%", 0.001),
    ("IF14", ["mirror {i}", "flip {i} left to right"], "-flop", 0.003),
    ("IF15", ["old photo look for {i}: sepia plus vignette", "vintage {i}"], "-sepia-tone 80% -background black -vignette 0x80", 0.01),
]
for fid, asks, op, thr in IF:
    ck = (lambda c: {"wh": None}) if fid == "IF13" else None
    feature(fid, "i", asks, f"magick in/{{i}} {op} {{O}}", thr=thr, chk=ck, fx={"i": "cover.jpg" if fid == "IF14" else lambda r: r.choice(["photo.jpg", "cover.jpg"])})

# ───────────── image: geometry, format, composition ─────────────
feature("IG01", "i", ["rotate {i} 90 degrees clockwise", "turn {i} sideways (90° right)"], "magick in/{i} -rotate 90 {O}",
        chk=lambda c: {"wh": [c["H"], c["W"]], "frame_unlike": None})
feature("IG02", "i", ["crop an 800x800 square from the center of {i}", "center crop {i} to 800x800"], "magick in/{i} -gravity center -crop 800x800+0+0 +repage {O}", fx={"i": lambda r: r.choice(["photo.jpg", "cover.jpg", "screenshot.png"])},
        chk=lambda c: {"wh": [800, 800], "frame_unlike": None})
feature("IG03", "i", ["make {i} half the size", "scale {i} down to 50%"], "magick in/{i} -resize 50% {O}",
        chk=lambda c: {"wh": [round(c["W"] / 2), round(c["H"] / 2)], "frame_unlike": None})
feature("IG04", "i", ["resize {i} to exactly 500x500, stretching is fine", "force {i} to 500 by 500"], "magick in/{i} -resize 500x500! {O}",
        chk=lambda c: {"wh": [500, 500], "frame_unlike": None})
feature("IG05", "i", ["300x300 thumbnail of {i}, fill and crop", "square 300px thumb of {i}, no bars"], "magick in/{i} -resize 300x300^ -gravity center -extent 300x300 {O}",
        chk=lambda c: {"wh": [300, 300], "frame_unlike": None})
feature("IG06", "i", ["add a 20px white border to {i}", "white frame, 20 pixels, around {i}"], "magick in/{i} -bordercolor white -border 20 {O}",
        chk=lambda c: {"wh": [c["W"] + 40, c["H"] + 40], "frame_unlike": None})
feature("IG07", "i", ["trim the empty transparent space around {i}", "auto-crop {i} to its content"], "magick in/{i} -trim +repage {O}",
        fx={"i": "product.png"}, chk=lambda c: {"wh": None, "max_side": 899, "frame_unlike": None})
feature("IG08", "i", ["polaroid style version of {i}", "make {i} look like a polaroid"], "magick in/{i} -resize 800x800 -bordercolor snow -background black +polaroid {O}",
        ext=".png", chk=lambda c: {"wh": None, "frame_unlike": None})
feature("IG09", "i", ["caption 'Day 1' under {i} on a white strip", "add a white bar with 'Day 1' below {i}"],
        "magick in/{i} -gravity south -background white -splice 0x100 -pointsize 60 -annotate +0+20 'Day 1' {O}",
        chk=lambda c: {"wh": [c["W"], c["H"] + 100], "frame_unlike": None})
feature("IG10", "i", ["put {i2} in the bottom right corner of {i} as a watermark", "stamp logo {i2} bottom right on {i}"],
        "magick in/{i} in/{i2} -gravity southeast -geometry +30+30 -composite {O}", slots=["i", "i"], fx={"i2": "badge.png"},
        chk=lambda c: {"frame_unlike": [[0, f"in/{c['i']}", 0.0005]]})
feature("IG11", "i", ["put {i} and {i2} side by side in one image", "combine {i} and {i2} horizontally"], "magick in/{i} in/{i2} +append {O}",
        slots=["i", "i"], fx={"i": "thanks.png", "i2": "title.png"}, chk=lambda c: {"wh": [2560, 720], "frame_unlike": None})
feature("IG12", "i", ["stack {i} on top of {i2}", "combine {i} and {i2} vertically"], "magick in/{i} in/{i2} -append {O}",
        slots=["i", "i"], fx={"i": "thanks.png", "i2": "title.png"}, chk=lambda c: {"wh": [1280, 1440], "frame_unlike": None})
feature("IG13", "i", ["favicon.ico from {i} with 16, 32 and 48 px sizes", "make a multi-size icon from {i}"],
        "magick in/{i} -define icon:auto-resize=16,32,48 out/favicon.ico", fx={"i": "badge.png"}, ext=".ico", chk=lambda c: {"wh": None, "frame_unlike": None})
feature("IG14", "i", ["progressive jpeg of {i} for faster web loading", "save {i} as a progressive jpg"], "magick in/{i} -interlace Plane -quality 85 {O}",
        ext=".jpg", fx={"i": lambda r: r.choice(["photo.jpg", "cover.jpg"])}, chk=lambda c: {"frame_unlike": None, "exif_eq": {"EncodingProcess": "Progressive DCT, Huffman coding"}})
feature("IG15", "i", ["get {i} under 100 KB as a jpg", "compress {i} to below 100KB"], "magick in/{i} -resize '1600x1600>' -define jpeg:extent=100kb {O}",
        ext=".jpg", chk=lambda c: {"wh": None, "max_mb": 0.1, "frame_unlike": None}, fx={"i": lambda r: r.choice(["photo.jpg", "cover.jpg"])})
feature("IG16", "i", ["avif version of {i}", "convert {i} to avif"], "magick in/{i} -quality 60 {O}", ext=".avif", chk=lambda c: {"frame_unlike": None})
feature("IG17", "i", ["multi-page tiff from {i} and {i2}", "combine {i} and {i2} into one tif"], "magick in/{i} in/{i2} {O}", slots=["i", "i"],
        fx={"i": "thanks.png", "i2": "title.png"}, ext=".tif", chk=lambda c: {"frame_unlike": None})
feature("IG18", "i", ["animated gif cycling through {i} and {i2}, one second each, looping", "make a looping gif from {i} and {i2}"],
        "magick -delay 100 in/{i} in/{i2} -loop 0 {O}", slots=["i", "i"], fx={"i": "thanks.png", "i2": "title.png"}, ext=".gif",
        chk=lambda c: {"frame_unlike": None})
feature("IG19", "i", ["remove all metadata from {i}", "strip the exif from {i}"], "magick in/{i} -strip {O}", fx={"i": "photos/img_01.jpg"}, ext=".jpg",
        chk=lambda c: {"wh": None, "frame_unlike": None, "exif_absent": ["Artist", "Make", "GPSLatitude"]})

# ───────────── exiftool writes ─────────────
for fid, asks, args, eq in [
    ("EX01", ["set the copyright of {i} to '(c) 2024 Jamie Rivera'", "add a copyright notice '(c) 2024 Jamie Rivera' to {i}"],
     "'-Copyright=(c) 2024 Jamie Rivera'", {"Copyright": "(c) 2024 Jamie Rivera"}),
    ("EX02", ["add the comment 'shot on the beach' to {i}", "write a comment into {i}: shot on the beach"], "'-Comment=shot on the beach'", {"Comment": "shot on the beach"}),
    ("EX03", ["set the location of {i} to the Eiffel Tower (48.8584 N, 2.2945 E)", "geotag {i}: 48.8584, 2.2945"],
     "-GPSLatitude=48.8584 -GPSLatitudeRef=N -GPSLongitude=2.2945 -GPSLongitudeRef=E", {"GPSLatitude": "48 deg 51' 30.24\" N", "GPSLongitude": "2 deg 17' 40.20\" E"}),
    ("EX04", ["change the photographer name in {i} to Alex Kim", "artist tag of {i} should be Alex Kim"], "'-Artist=Alex Kim'", {"Artist": "Alex Kim"}),
    ("EX05", ["set the title of {i} to 'Sunset'", "give {i} the title Sunset"], "-XMP:Title=Sunset", {"Title": "Sunset"}),
]:
    feature(fid, "i", asks, f"exiftool -o out/ {args} in/{{i}}", fx={"i": lambda r: f"photos/img_0{r.randint(1, 6)}.jpg"}, ext=".jpg",
            chk=(lambda e: lambda c: {"wh": None, "frame_unlike": None, "exif_eq": e})(eq))
feature("EX06", "i", ["copy the date and camera info from {i2} into {i}", "{i} should get the exif of {i2}"],
        "exiftool -o out/ -TagsFromFile in/{i2} -DateTimeOriginal -Make -Model in/{i}", slots=["i", "i"], fx={"i": "cover.jpg", "i2": "photos/img_03.jpg"},
        ext=".jpg", chk=lambda c: {"wh": None, "frame_unlike": None, "exif_eq": {"DateTimeOriginal": TRUTH["photo_dates"][2], "Make": "Canon"}})
feature("EX07", "v", ["set the recording date of {v} to 1 June 2024 10:00", "change the creation date of {v} to 2024-06-01 10:00"],
        "exiftool -o out/ '-CreateDate=2024:06:01 10:00:00' {V}", ext=".mp4", chk=lambda c: {"frame_unlike": None, "exif_eq": {"CreateDate": "2024:06:01 10:00:00"}})

# ───────────── questions (answer in words) ─────────────
QS = [
    ("Q01", "v", ["how many frames are in {v}?", "frame count of {v}?"], "ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 {V}",
     lambda c: [rf"\b{int(round(c['M']['fps'] * c['D']))}\b"], lambda c: f"{c['v']} has {int(round(c['M']['fps'] * c['D']))} frames."),
    ("Q02", "v", ["what codec is {v} encoded with?", "which video codec does {v} use?"], "ffprobe -v error -show_entries stream=codec_name,codec_type -of compact {V}",
     lambda c: [r"(?i)(h\.?264|avc)"], lambda c: "The video is H.264 (AVC) and the audio is AAC."),
    ("Q03", "a", ["how loud is {a} in LUFS?", "what's the integrated loudness of {a}?"], "ffmpeg -i {A} -af ebur128=framelog=quiet -f null -",
     lambda c: [rf"-{abs(round(lufs_of(c['a_fx'])))}(\.\d)?|-{abs(round(lufs_of(c['a_fx']))) - 1}\.[5-9]|-{abs(round(lufs_of(c['a_fx'])))}"],
     lambda c: f"Integrated loudness is about {lufs_of(c['a_fx']):.1f} LUFS."),
    ("Q04", "a", ["what sample rate is {a}?", "sample rate and channels of {a}?"], "ffprobe -v error -show_entries stream=sample_rate,channels -of compact {A}",
     lambda c: [rf"{c['M']['asr']}|{c['M']['asr'] / 1000:g}\s*k"], lambda c: f"{c['a']} is {c['M']['asr']} Hz, {c['M']['ach']} channel(s)."),
    ("Q05", "v", ["does {v} have any audio?", "is there sound in {v}?"], "ffprobe -v error -show_entries stream=codec_type -of csv=p=0 {V}",
     lambda c: [r"(?i)\bno\b|no audio|without (any )?audio|silent"], lambda c: f"No, {c['v']} has only a video stream, no audio.",),
    ("Q06", "i", ["what camera took {i}?", "which camera model shot {i}?"], "exiftool -Make -Model in/{i}",
     lambda c: [r"(?i)canon", r"(?i)R6"], lambda c: "It was taken with a Canon EOS R6."),
    ("Q07", "i", ["where was {i} taken?", "what are the GPS coordinates of {i}?"], "exiftool -n -GPSLatitude -GPSLongitude in/{i}",
     lambda c: [r"12\.97", r"77\.59"], lambda c: "At 12.97 N, 77.59 E (Bengaluru, India)."),
    ("Q08", "i", ["what are the dimensions of {i}?", "how big is {i} in pixels?"], "magick identify in/{i}",
     lambda c: [rf"{c['W']}\s*(x|×|by)\s*{c['H']}"], lambda c: f"{c['i']} is {c['W']}x{c['H']} pixels."),
    ("Q09", "v", ["what's the bitrate of {v}?", "how many kbps is {v}?"], "ffprobe -v error -show_entries format=bit_rate -of csv=p=0 {V}",
     lambda c: [r"(?i)\d+(\.\d+)?\s*(kb|kbit|mb|mbit)"], lambda c: f"About {c['M']['size'] * 8 / c['D'] / 1000:.0f} kb/s overall."),
]
for fid, k, asks, cmd, must, fin in QS:
    feature(fid, "q", asks, cmd, chk=must, final=fin, slots=[k],
            fx={"v": lambda r: r.choice(["screenrec.mp4"]) } if fid == "Q05" else {"i": lambda r: f"photos/img_0{r.choice([1, 3, 5])}.jpg"} if fid in ("Q06", "Q07") else None)


if __name__ == "__main__":
    import sys
    base = {k for k in list(gen2.GENS) if not k[:2] in ("VF", "VT", "VC", "AF", "AT", "IF", "IG", "EX") and not (k[0] == "Q" and k[1:].isdigit())}
    for k in base:
        gen2.GENS.pop(k)  # batch 4 only
    print(len(gen2.GENS), "doc features", file=sys.stderr)
    gen2.main()
