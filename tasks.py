"""Task templates. Each Task = phrasings + a command that depends on the real input metadata + an output check.

Slots in `ins` render as display names in asks ({in}) and as quoted in/ paths in cmd ({in}).
`out` renders as a quoted out/ path in cmd. gen(r, M) gets fixture metadata per slot and returns params
(or None to skip an incompatible combo). check(c, p) verifies the real output (see build.Check).
"""
import math
from dataclasses import dataclass

POOLS = {
    "video": [("input.mp4", "land.mp4"), ("video.mp4", "land.mp4"), ("clip.mov", "land.mov"),
              ("vacation.MOV", "phone.mov"), ("IMG_4821.MOV", "phone.mov"), ("recording.mkv", "land.mkv"),
              ("my video.mp4", "land.mp4"), ("lecture-01.mp4", "small.mp4"), ("gameplay.mkv", "land.mkv"),
              ("interview_raw.mov", "land.mov"), ("drone_footage.mp4", "land.mp4"), ("talk.webm", "web.webm"),
              ("old_movie.avi", "old.avi"), ("wedding.mp4", "land.mp4"), ("tutorial.mp4", "small.mp4"),
              ("GX010123.MP4", "land.mp4"), ("family trip.mov", "phone.mov"), ("stream_vod.mkv", "land.mkv"),
              ("reel.mp4", "phone.mp4"), ("product demo.mp4", "small.mp4"), ("webinar.webm", "web.webm"),
              ("home_video.avi", "old.avi")],
    "screen": [("Screen Recording 2024-03-02 at 10.14.22.mov", "screen.mov"), ("screencast.mp4", "screen.mp4"),
               ("bug_repro.mov", "screen.mov"), ("demo_capture.mp4", "screen.mp4")],
    "video2": [("intro.mp4", "short.mp4"), ("outro.mp4", "short.mp4"), ("part2.mp4", "short.mp4"),
               ("b-roll.mp4", "short.mp4"), ("webcam.mp4", "short.mp4"), ("clip2.mp4", "short.mp4")],
    "subsvideo": [("movie.mkv", "subs.mkv"), ("episode_03.mkv", "subs.mkv"), ("film.mkv", "subs.mkv")],
    "audio": [("audio.wav", "a.wav"), ("song.mp3", "a.mp3"), ("podcast.wav", "a.wav"), ("voice memo.m4a", "a.m4a"),
              ("interview.mp3", "a.mp3"), ("track01.flac", "a.flac"), ("recording.wav", "a.wav"),
              ("episode 12.mp3", "a.mp3"), ("lecture.m4a", "a.m4a"), ("beat.wav", "a.wav"),
              ("meeting.m4a", "a.m4a"), ("vocals.wav", "a.wav"), ("narration.mp3", "a.mp3"),
              ("sample.flac", "a.flac"), ("call_recording.wav", "a.wav"), ("New Recording 3.m4a", "a.m4a")],
    "music": [("music.mp3", "music.mp3"), ("background.mp3", "music.mp3"), ("bgm.wav", "music.wav"),
              ("ambience.wav", "music.wav"), ("lofi beat.mp3", "music.mp3")],
    "image": [("photo.jpg", "photo.jpg"), ("image.png", "shot.png"), ("IMG_2034.JPG", "photo.jpg"),
              ("screenshot.png", "shot.png"), ("banner.png", "shot.png"), ("portrait.jpg", "photo.jpg"),
              ("product shot.jpg", "photo.jpg"), ("scan.png", "shot.png"), ("wallpaper.jpg", "photo.jpg"),
              ("avatar.webp", "pic.webp"), ("DSC_0042.jpg", "photo.jpg"), ("cover.png", "shot.png"),
              ("hero.jpg", "photo.jpg"), ("diagram.png", "shot.png"), ("thumbnail.webp", "pic.webp")],
    "photo": [("photo.jpg", "photo.jpg"), ("IMG_2034.JPG", "photo.jpg"), ("DSC_0042.jpg", "photo.jpg"),
              ("beach.jpg", "photo.jpg"), ("IMG_20240714_183210.jpg", "photo.jpg"), ("for sale.jpg", "photo.jpg")],
    "heic": [("IMG_5501.HEIC", "iphone.heic"), ("photo.heic", "iphone.heic"), ("IMG_0912.heic", "iphone.heic")],
    "logo": [("logo.png", "logo.png"), ("watermark.png", "logo.png"), ("brand_logo.png", "logo.png")],
    "subs": [("subs.srt", "subs.srt"), ("subtitles.srt", "subs.srt"), ("captions.srt", "subs.srt"),
             ("english.srt", "subs.srt")],
    "gif": [("animation.gif", "anim.gif"), ("funny.gif", "anim.gif"), ("reaction.gif", "anim.gif")],
    "svg": [("logo.svg", "logo.svg"), ("icon.svg", "logo.svg")],
}

BATCH = {
    "mov": {"files": [(f"IMG_00{i}.MOV", "phone.mov") for i in (12, 13, 14)], "glob": "in/*.MOV", "desc": "all the .MOV files"},
    "mkv": {"files": [(f"episode_0{i}.mkv", "land.mkv") for i in (1, 2, 3)], "glob": "in/*.mkv", "desc": "every .mkv file"},
    "avi": {"files": [(f"tape{i}.avi", "old.avi") for i in (1, 2, 3)], "glob": "in/*.avi", "desc": "all the AVI files"},
    "wav": {"files": [(f"take{i}.wav", "a.wav") for i in (1, 2, 3)], "glob": "in/*.wav", "desc": "all the wav files"},
    "m4a": {"files": [(f"memo {i}.m4a", "a.m4a") for i in (1, 2, 3)], "glob": "in/*.m4a", "desc": "all my .m4a voice memos"},
    "jpg": {"files": [(f"IMG_10{i}.jpg", "photo.jpg") for i in (21, 22, 23, 24)], "glob": "in/*.jpg", "desc": "all the jpgs"},
    "png": {"files": [(f"screen_{i}.png", "shot.png") for i in (1, 2, 3)], "glob": "in/*.png", "desc": "all the PNG screenshots"},
    "heic": {"files": [(f"IMG_20{i}.HEIC", "iphone.heic") for i in (41, 42, 43)], "glob": "in/*.HEIC", "desc": "all the HEIC photos"},
    "frames": {"files": [(f"frame_{i:03d}.png", f"frames/frame_{i:03d}.png") for i in range(1, 21)], "glob": "in/frame_*.png",
               "desc": "frame_001.png to frame_020.png"},
}


@dataclass
class Task:
    id: str
    cat: str
    asks: list
    cmd: object                  # str.format template, or callable(p) -> str
    ins: dict = None             # slot -> pool name
    out: str = None              # output name template ({stem} = stem of `in`)
    note: object = ""            # str template, or callable(p, res) -> str
    gen: object = None           # callable(r, M) -> dict | None
    check: object = None         # callable(c, p) -> bool
    batch: tuple = ()            # BATCH keys to choose from; adds {files}, {glob}
    foreach: bool = False        # run cmd once per batch file ({{in}}/{{stem}} placeholders)
    weight: float = 1.0          # relative sample count
    broken: str = None           # error-recovery: plausible wrong command run first (must fail for real)
    reply: object = None         # no-tool answer: str template, or callable(p, M) -> str


def hms(s):
    s = int(round(s))
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def say_t(r, s):
    m, sec = divmod(int(s), 60)
    return r.choice([f"{m}:{sec:02d}", hms(s), f"{int(s)} seconds" if s < 100 else f"{m}m{sec:02d}s",
                     f"{m}m{sec}s" if m else f"{int(s)}s", f"{int(s)} sec"])


def near(a, b, tol):
    return a is not None and abs(a - b) <= tol


def aud(m):
    """Audio args for an MP4 output: copy AAC, re-encode anything else, nothing if no audio."""
    if not m.get("has_audio"):
        return ""
    return "-c:a copy" if m["acodec"] == "aac" else "-c:a aac -b:a 128k"


def atempo(f):
    """atempo chain; keep each stage within 0.5..2 so it works on any ffmpeg version."""
    parts = []
    while f > 2:
        parts.append(2.0)
        f /= 2
    while f < 0.5:
        parts.append(0.5)
        f /= 0.5
    parts.append(f)
    return ",".join(f"atempo={x:g}" for x in parts)


def landscape(m):
    return m["w"] >= m["h"]


def even(x):
    return int(x) // 2 * 2


IMG_EXT = {"jpg": "JPEG", "png": "PNG", "webp": "WEBP"}

T = []
add = T.append

# ───────────────────────────── VIDEO: convert / compress ─────────────────────────────


def g_to_mp4(r, M):
    m = M["in"]
    if m["vcodec"] in ("h264", "hevc") and m.get("acodec") in ("aac", "mp3", None):
        tag = " -tag:v hvc1" if m["vcodec"] == "hevc" else ""
        return {"args": f"-c copy{tag} -movflags +faststart", "vc": m["vcodec"],
                "why": f"The streams are already {m['vcodec'].upper().replace('HEVC', 'HEVC (H.265)')} "
                       f"+ {m['acodec'].upper()}, so I'll remux with -c copy: no re-encode, no quality loss, done in seconds."
                       + (" -tag:v hvc1 makes HEVC play in QuickTime/Apple devices." if tag else "")}
    return {"args": "-c:v libx264 -crf 23 -preset medium -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart",
            "vc": "h264",
            "why": f"The source is {m['vcodec']}/{m.get('acodec')}, which most players don't support in MP4, so I'll re-encode "
                   "to H.264 (CRF 23) + AAC. +faststart lets it start playing before fully downloaded."}


add(Task("v_to_mp4", "video/convert", ins={"in": "video"}, out="{stem}.mp4", gen=g_to_mp4,
         asks=["convert {in} to mp4", "Convert {in} to MP4", "how do I turn {in} into an mp4 file?",
               "change {in} to .mp4 format", "I need {in} as an mp4", "make {in} an mp4 so I can upload it",
               "{in} -> mp4", "can you convert {in} into mp4 for me", "save {in} as {out}"],
         cmd="ffmpeg -i {in} {args} {out}", note="{why}",
         check=lambda c, p: c.v()["codec_name"] == p["vc"]))
T[-1].gen = (lambda g: lambda r, M: None if M["in"]["_name"].lower().endswith(".mp4") else g(r, M))(g_to_mp4)

add(Task("v_compat", "video/convert", ins={"in": "video"}, out="{stem}_compatible.mp4",
         asks=["make {in} play on any device", "{in} won't play on my iPhone, fix it",
               "convert {in} so it works on WhatsApp", "make {in} compatible with everything (TVs, phones, browsers)",
               "re-encode {in} to a universally compatible mp4", "my TV can't play {in}, can you convert it?",
               "{in} shows a black screen in PowerPoint, make it compatible"],
         cmd="ffmpeg -i {in} -c:v libx264 -profile:v high -level 4.1 -pix_fmt yuv420p -crf 23 -preset medium "
             "-c:a aac -b:a 128k -ac 2 -movflags +faststart {out}",
         note="Re-encoding to the most widely supported combo: H.264 High profile, yuv420p, AAC stereo, with the index "
              "at the front (+faststart) for streaming.",
         check=lambda c, p: c.v()["codec_name"] == "h264" and c.v()["pix_fmt"] == "yuv420p"))

LEVELS = [("", 23, "a good balance of size and quality"), ("a little", 20, "visually near-lossless"),
          ("a lot", 28, "much smaller with some visible softening"), ("as much as possible", 30, "very small, lower quality"),
          ("but keep the quality high", 19, "near-lossless")]


def g_compress(r, M):
    lvl, crf, desc = r.choice(LEVELS)
    preset = r.choice(["medium", "medium", "slow", "veryfast"])
    return {"lvl": lvl, "crf": crf, "desc": desc, "preset": preset, "aud": aud(M["in"])}


add(Task("v_compress_x264", "video/compress", ins={"in": "video"}, out="{stem}_compressed.mp4", gen=g_compress, weight=2,
         asks=["compress {in} {lvl}", "reduce the file size of {in} {lvl}", "make {in} smaller {lvl}",
               "{in} is too big, shrink it {lvl}", "compress {in} to send over email {lvl}",
               "can you lower the size of {in} {lvl}?", "shrink {in} {lvl} without changing the resolution"],
         cmd="ffmpeg -i {in} -c:v libx264 -crf {crf} -preset {preset} -pix_fmt yuv420p {aud} {out}",
         note="H.264 at CRF {crf} ({desc}); lower CRF = better quality, bigger file. Preset {preset} trades encode "
              "speed for compression.",
         check=lambda c, p: c.v()["codec_name"] == "h264"))

add(Task("v_compress_x265", "video/compress", ins={"in": "video"}, out="{stem}_hevc.mp4",
         gen=lambda r, M: {"crf": r.choice([24, 26, 28]), "aud": aud(M["in"])},
         asks=["compress {in} with H.265", "encode {in} to HEVC to save space", "convert {in} to x265",
               "use hevc to make {in} smaller", "re-encode {in} as H.265 at crf {crf}",
               "I want {in} in HEVC so it takes less storage"],
         cmd="ffmpeg -i {in} -c:v libx265 -crf {crf} -preset medium -tag:v hvc1 -pix_fmt yuv420p {aud} {out}",
         note="HEVC (x265) at CRF {crf} is roughly half the size of H.264 at similar quality. -tag:v hvc1 keeps it "
              "playable on Apple devices.",
         check=lambda c, p: c.v()["codec_name"] == "hevc"))

add(Task("v_av1", "video/compress", ins={"in": "video"}, out="{stem}_av1.mp4",
         gen=lambda r, M: {"crf": r.choice([30, 35, 38]), "aud": aud(M["in"])},
         asks=["encode {in} to AV1", "convert {in} to av1 for the smallest file", "compress {in} using AV1 (svt-av1)",
               "make an AV1 version of {in}", "I want {in} in AV1 at crf {crf}"],
         cmd="ffmpeg -i {in} -c:v libsvtav1 -crf {crf} -preset 8 -pix_fmt yuv420p {aud} {out}",
         note="SVT-AV1 at CRF {crf}, preset 8 (fast; lower preset = smaller but slower). AV1 beats HEVC on size but "
              "older devices can't decode it.",
         check=lambda c, p: c.v()["codec_name"] == "av1"))

add(Task("v_webm", "video/convert", ins={"in": "video"}, out="{stem}.webm",
         gen=lambda r, M: None if M["in"]["_name"].endswith(".webm") else {"crf": r.choice([30, 32, 34])},
         asks=["convert {in} to webm", "make a webm version of {in} for my website", "encode {in} as VP9 webm",
               "turn {in} into .webm", "I need {in} in webm format for an html5 video tag"],
         cmd="ffmpeg -i {in} -c:v libvpx-vp9 -crf {crf} -b:v 0 -deadline good -cpu-used 4 -row-mt 1 "
             "-c:a libopus -b:a 96k {out}",
         note="VP9 in constant-quality mode (-crf {crf} -b:v 0) with Opus audio, the standard WebM combo. "
              "-cpu-used 4 and -row-mt 1 keep encoding reasonably fast.",
         check=lambda c, p: c.v()["codec_name"] == "vp9"))


def g_target(r, M):
    m = M["in"]
    mb = round(m["size"] / 1e6 * r.uniform(0.25, 0.5), 1)
    ab = 128 if m.get("has_audio") else 0
    vb = int(mb * 8000 / m["dur"] * 0.97) - ab
    if vb < 150:
        return None
    a = "-c:a aac -b:a 128k" if ab else "-an"
    return {"mb": mb, "vb": vb, "a": a, "dur": round(m["dur"], 1),
            "why": f"{mb} MB over {m['dur']:.1f}s ≈ {vb + ab} kbps total, so video gets {vb}k" +
                   (" and audio 128k" if ab else "") + " (3% headroom for container overhead)."}


add(Task("v_target_size", "video/compress", ins={"in": "video"}, out="{stem}_{mb}MB.mp4", gen=g_target, weight=1.5,
         asks=["compress {in} to under {mb} MB", "make {in} fit in {mb}MB", "I need {in} below {mb} MB for upload",
               "get {in} under {mb} mb (discord limit)", "shrink {in} to exactly about {mb}MB",
               "compress {in} so it's at most {mb} megabytes"],
         cmd="ffmpeg -y -i {in} -c:v libx264 -b:v {vb}k -pass 1 -an -f null /dev/null && "
             "ffmpeg -i {in} -c:v libx264 -b:v {vb}k -pass 2 -pix_fmt yuv420p {a} {out}",
         note="Two-pass encode to hit a size target: {why}",
         check=lambda c, p: c.size() <= p["mb"] * 1e6 * 1.06))

# ───────────────────────────── VIDEO: resize / crop / pad ─────────────────────────────


def g_height(r, M):
    m = M["in"]
    short = min(m["w"], m["h"])
    opts = [h for h in (1080, 720, 540, 480, 360, 240) if h < short]
    if not opts:
        return None
    H = r.choice(opts)
    vf = f"scale=-2:{H}" if landscape(m) else f"scale={H}:-2"
    return {"H": H, "vf": vf, "aud": aud(m),
            "why": "-2 keeps the aspect ratio and rounds to an even number (H.264 needs even dimensions)." +
                   ("" if landscape(m) else f" The video is vertical ({m['w']}x{m['h']}), so {H}p means {H} px wide.")}


add(Task("v_resize_p", "video/resize", ins={"in": "video"}, out="{stem}_{H}p.mp4", gen=g_height, weight=2,
         asks=["resize {in} to {H}p", "downscale {in} to {H}p", "make {in} {H}p", "convert {in} to {H}p resolution",
               "scale {in} down to {H}p to save space", "I need a {H}p version of {in}", "lower the resolution of {in} to {H}p"],
         cmd="ffmpeg -i {in} -vf \"{vf}\" -c:v libx264 -crf 23 -preset medium {aud} {out}", note="{why}",
         check=lambda c, p: min(c.wh()) == p["H"]))

add(Task("v_resize_w", "video/resize", ins={"in": "video"}, out="{stem}_{W}w.mp4",
         gen=lambda r, M: {"W": r.choice([w for w in (1920, 1280, 960, 800, 640, 480) if w < M["in"]["w"]] or [None]),
                           "aud": aud(M["in"])},
         asks=["make {in} {W} pixels wide", "resize {in} to a width of {W}px keeping the aspect ratio",
               "scale {in} to {W} wide", "change the width of {in} to {W}"],
         cmd="ffmpeg -i {in} -vf \"scale={W}:-2\" -c:v libx264 -crf 23 {aud} {out}",
         note="scale={W}:-2 sets the width and computes an even height that keeps the aspect ratio.",
         check=lambda c, p: c.wh()[0] == p["W"]))
T[-1].gen = (lambda g: lambda r, M: (lambda p: p if p["W"] else None)(g(r, M)))(T[-1].gen)

add(Task("v_to_vertical_pad", "video/resize", ins={"in": "video"}, out="{stem}_vertical.mp4",
         gen=lambda r, M: {"color": r.choice(["black", "black", "white"]), "aud": aud(M["in"])} if landscape(M["in"]) else None,
         asks=["make {in} vertical 9:16 with {color} bars", "fit {in} into a 1080x1920 frame without cropping, {color} bars",
               "convert {in} to portrait for TikTok, don't cut anything off, pad with {color}",
               "letterbox {in} into 9:16 with {color} padding"],
         cmd="ffmpeg -i {in} -vf \"scale=1080:1920:force_original_aspect_ratio=decrease,"
             "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color={color},setsar=1\" -c:v libx264 -crf 23 {aud} {out}",
         note="Scale to fit inside 1080x1920 (nothing cropped), then centre it on a {color} canvas.",
         check=lambda c, p: c.wh() == (1080, 1920)))

add(Task("v_to_vertical_blur", "video/resize", ins={"in": "video"}, out="{stem}_reel.mp4",
         gen=lambda r, M: {} if landscape(M["in"]) and M["in"].get("has_audio") else None,
         asks=["make {in} vertical for reels with a blurred background", "turn {in} into 9:16 with the blurred video behind it",
               "convert {in} to a TikTok format with blur fill instead of black bars",
               "put {in} in a vertical 1080x1920 frame with a blurry background"],
         cmd="ffmpeg -i {in} -filter_complex \"[0:v]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,"
             "boxblur=10:2,scale=1080:1920[bg];[0:v]scale=1080:-2[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2[v]\" -map \"[v]\" -map 0:a "
             "-c:v libx264 -crf 23 -c:a copy {out}",
         note="A zoomed, blurred copy fills the 9:16 frame and the full video sits centred on top. The blur runs at 270x480 and is then upscaled, which looks the same and is many times faster than blurring at 1080x1920.",
         check=lambda c, p: c.wh() == (1080, 1920) and c.a() is not None))


def g_square(r, M):
    m = M["in"]
    s = min(m["w"], m["h"])
    return {"crop": "crop=ih:ih" if landscape(m) else "crop=iw:iw", "s": s, "aud": aud(m)}


add(Task("v_crop_square", "video/crop", ins={"in": "video"}, out="{stem}_square.mp4", gen=g_square,
         asks=["crop {in} to a square", "make {in} 1:1 by cropping the center", "center crop {in} to square for instagram",
               "cut {in} down to a square video"],
         cmd="ffmpeg -i {in} -vf \"{crop}\" -c:v libx264 -crf 23 {aud} {out}",
         note="crop with no x/y centres the crop; the square is {s}x{s}.",
         check=lambda c, p: c.wh()[0] == c.wh()[1]))


def g_crop_916(r, M):
    m = M["in"]
    if not landscape(m):
        return None
    cw = even(m["h"] * 9 / 16)
    return {"cw": cw, "h": m["h"], "aud": aud(m)}


add(Task("v_crop_916", "video/crop", ins={"in": "video"}, out="{stem}_9x16.mp4", gen=g_crop_916,
         asks=["crop {in} to 9:16 for shorts", "cut the center of {in} into a vertical video",
               "make a vertical crop of {in} (fill the screen, no bars)", "crop {in} to portrait 1080x1920"],
         cmd="ffmpeg -i {in} -vf \"crop={cw}:{h},scale=1080:1920,setsar=1\" -c:v libx264 -crf 23 {aud} {out}",
         note="Crop a centred {cw}x{h} (9:16, rounded to even) strip, then scale to 1080x1920.",
         check=lambda c, p: c.wh() == (1080, 1920)))


def g_crop_xy(r, M):
    m = M["in"]
    cw, ch = even(m["w"] * r.uniform(0.4, 0.8)), even(m["h"] * r.uniform(0.4, 0.8))
    x, y = r.randint(0, m["w"] - cw), r.randint(0, m["h"] - ch)
    return {"cw": cw, "ch": ch, "x": x, "y": y, "aud": aud(m)}


add(Task("v_crop_xy", "video/crop", ins={"in": "video"}, out="{stem}_cropped.mp4", gen=g_crop_xy,
         asks=["crop {in} to {cw}x{ch} starting at x={x}, y={y}", "crop a {cw}x{ch} region from {in} at position {x},{y}",
               "keep only the {cw} by {ch} area of {in} whose top-left corner is ({x}, {y})"],
         cmd="ffmpeg -i {in} -vf \"crop={cw}:{ch}:{x}:{y}\" -c:v libx264 -crf 23 {aud} {out}",
         note="crop=w:h:x:y, with x/y the top-left corner of the kept area.",
         check=lambda c, p: c.wh() == (p["cw"], p["ch"])))

add(Task("v_pad_square", "video/resize", ins={"in": "video"}, out="{stem}_1x1.mp4",
         gen=lambda r, M: {"S": r.choice([1080, 720]), "color": r.choice(["black", "white"]), "aud": aud(M["in"])},
         asks=["fit {in} into a {S}x{S} square with {color} bars", "make {in} square without cropping ({color} padding)",
               "pad {in} to 1:1, {S}px, {color} background"],
         cmd="ffmpeg -i {in} -vf \"scale={S}:{S}:force_original_aspect_ratio=decrease,pad={S}:{S}:(ow-iw)/2:(oh-ih)/2:color={color},"
             "setsar=1\" -c:v libx264 -crf 23 {aud} {out}",
         note="Scale to fit inside {S}x{S}, then pad with {color} to a square.",
         check=lambda c, p: c.wh() == (p["S"], p["S"])))

# ───────────────────────────── VIDEO: trim / cut / join ─────────────────────────────


def g_trim(r, M):
    d = M["in"]["dur"]
    s = r.randint(0, int(d * 0.5))
    e = r.randint(s + 3, int(d) - 1)
    return {"s": s, "e": e, "S": hms(s), "E": hms(e), "ss": say_t(r, s), "es": say_t(r, e), "aud": aud(M["in"]),
            "len": e - s}


add(Task("v_trim", "video/trim", ins={"in": "video"}, out="{stem}_trim.mp4", gen=g_trim, weight=2,
         asks=["cut {in} from {ss} to {es}", "trim {in} to keep only {ss} - {es}", "extract the part of {in} between {ss} and {es}",
               "I only need {ss} to {es} of {in}", "clip {in} from {ss} until {es}", "save the segment {ss}-{es} of {in} as {out}",
               "make a clip of {in} starting at {ss} and ending at {es}, frame accurate"],
         cmd="ffmpeg -ss {S} -to {E} -i {in} -c:v libx264 -crf 20 {aud} {out}",
         note="Seeking before -i is fast; re-encoding makes the cut frame-accurate ({len}s clip).",
         check=lambda c, p: near(c.dur(), p["len"], 0.6)))

add(Task("v_trim_copy", "video/trim", ins={"in": "video"}, out="{stem}_cut.mp4", gen=g_trim,
         asks=["quickly cut {in} from {ss} to {es} without re-encoding", "lossless trim of {in}, {ss} to {es}",
               "trim {in} ({ss}–{es}) fast, keep original quality", "cut {in} {ss} to {es} with stream copy"],
         cmd="ffmpeg -ss {S} -to {E} -i {in} -c copy {out}",
         note="-c copy is instant and lossless, but the cut snaps to the nearest keyframe, so the start can be off by "
              "a second or two. Re-encode if you need frame accuracy.",
         check=lambda c, p: near(c.dur(), p["len"], 3)))
T[-1].gen = (lambda g: lambda r, M: None if M["in"]["_fix"] == "old.avi" else g(r, M))(g_trim)

add(Task("v_first_n", "video/trim", ins={"in": "video"}, out="{stem}_first{n}s.mp4",
         gen=lambda r, M: (lambda n: {"n": n, "ns": say_t(r, n)})(r.randint(3, int(M["in"]["dur"]) - 2)),
         asks=["keep only the first {ns} of {in}", "cut {in} to the first {ns}", "I just want the first {n} seconds of {in}",
               "shorten {in} to {ns}"],
         cmd="ffmpeg -i {in} -t {n} -c copy {out}",
         note="-t {n} stops after {n} seconds; -c copy avoids re-encoding.",
         check=lambda c, p: near(c.dur(), p["n"], 1.5)))

add(Task("v_skip_n", "video/trim", ins={"in": "video"}, out="{stem}_trimmed.mp4",
         gen=lambda r, M: (lambda n: {"n": n, "ns": say_t(r, n), "aud": aud(M["in"]), "rest": M["in"]["dur"] - n})(
             r.randint(2, int(M["in"]["dur"] * 0.6))),
         asks=["remove the first {ns} of {in}", "cut off the first {n} seconds of {in}", "skip the intro of {in} (first {ns})",
               "delete the beginning of {in} up to {ns}"],
         cmd="ffmpeg -ss {n} -i {in} -c:v libx264 -crf 20 {aud} {out}",
         note="Starts the output at {n}s; re-encoded so it starts exactly there rather than at a keyframe.",
         check=lambda c, p: near(c.dur(), p["rest"], 0.7)))


def g_cutout(r, M):
    p = g_trim(r, M)
    p["rest"] = M["in"]["dur"] - p["len"]
    return p if M["in"].get("has_audio") else None


add(Task("v_cut_middle", "video/trim", ins={"in": "video"}, out="{stem}_edited.mp4", gen=g_cutout,
         asks=["remove the part from {ss} to {es} in {in}", "delete {ss}-{es} from {in} and keep the rest",
               "cut out the section between {ss} and {es} of {in}", "{in}: get rid of everything between {ss} and {es}"],
         cmd="ffmpeg -i {in} -vf \"select='not(between(t,{s},{e}))',setpts=N/FRAME_RATE/TB\" "
             "-af \"aselect='not(between(t,{s},{e}))',asetpts=N/SR/TB\" -c:v libx264 -crf 20 -c:a aac -b:a 128k {out}",
         note="select/aselect drop frames and samples in {s}–{e}s; setpts/asetpts close the gap so audio and video stay in sync.",
         check=lambda c, p: near(c.dur(), p["rest"], 1.0)))


def g_concat(r, M):
    a, b = M["in"], M["in2"]
    if not a.get("has_audio"):
        return None
    W, H, F = b["w"], b["h"], int(b["fps"])
    if (a["w"], a["h"], a["fps"]) == (b["w"], b["h"], b["fps"]):
        fc = "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]"
        why = "Both clips share resolution and frame rate, so they go straight into the concat filter."
    else:
        norm = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={F}"
        fc = f"[0:v]{norm}[v0];[1:v]{norm}[v1];[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]"
        why = (f"The clips differ ({a['w']}x{a['h']}@{a['fps']:g} vs {W}x{H}@{F}), so both are normalised to "
               f"{W}x{H}@{F}fps first; concat requires matching video.")
    return {"fc": fc, "why": why, "total": a["dur"] + b["dur"]}


add(Task("v_concat", "video/join", ins={"in": "video", "in2": "video2"}, out="combined.mp4", gen=g_concat, weight=1.5,
         asks=["join {in} and {in2} into one video", "merge {in} and {in2}", "put {in2} after {in}",
               "concatenate {in} with {in2}", "combine {in} and then {in2} into {out}", "stitch {in} and {in2} together"],
         cmd="ffmpeg -i {in} -i {in2} -filter_complex \"{fc}\" -map \"[v]\" -map \"[a]\" -c:v libx264 -crf 20 "
             "-c:a aac -b:a 128k {out}",
         note="{why}", check=lambda c, p: near(c.dur(), p["total"], 1.0)))

add(Task("v_intro", "video/join", ins={"in": "video2", "in2": "video"}, out="{stem2}_with_intro.mp4", gen=g_concat,
         asks=["add {in} as an intro to {in2}", "put {in} at the beginning of {in2}", "prepend {in} to {in2}"],
         cmd="ffmpeg -i {in} -i {in2} -filter_complex \"{fc}\" -map \"[v]\" -map \"[a]\" -c:v libx264 -crf 20 "
             "-c:a aac -b:a 128k {out}",
         note="{why}", check=lambda c, p: near(c.dur(), p["total"], 1.0)))

add(Task("v_split", "video/trim", ins={"in": "video"}, out="part_%03d.mp4",
         gen=lambda r, M: (lambda n: {"n": n, "ns": say_t(r, n), "exp": math.ceil(M["in"]["dur"] / n)})(
             r.choice([5, 6, 8, 10])) if M["in"]["_fix"] not in ("old.avi", "web.webm") else None,
         asks=["split {in} into {ns} parts", "cut {in} into {n}-second chunks", "break {in} up into segments of {ns} each",
               "chop {in} into {n} second clips"],
         cmd="ffmpeg -i {in} -c copy -map 0 -f segment -segment_time {n} -reset_timestamps 1 {out}",
         note="Segment muxer with -c copy: no re-encode, and each part starts at a keyframe, so lengths are approximately {n}s.",
         check=lambda c, p: c.count() >= 2))

add(Task("v_loop", "video/edit", ins={"in": "video2"}, out="{stem}_x{n}.mp4",
         gen=lambda r, M: (lambda n: {"n": n, "k": n - 1, "total": n * M["in"]["dur"]})(r.choice([2, 3, 4, 5])),
         asks=["loop {in} {n} times", "make {in} play {n} times in a row", "repeat {in} so it plays {n} times total",
               "create a looped version of {in}, {n} repeats"],
         cmd="ffmpeg -stream_loop {k} -i {in} -c copy {out}",
         note="-stream_loop {k} replays the input {k} extra times ({n} plays total), without re-encoding.",
         check=lambda c, p: near(c.dur(), p["total"], 1.5)))

# ───────────────────────────── VIDEO: audio tracks ─────────────────────────────


def g_extract_mp3(r, M):
    return {"q": r.choice([0, 2, 4]), "qd": {0: "~245 kbps", 2: "~190 kbps", 4: "~165 kbps"}} if M["in"].get("has_audio") else None


add(Task("v_extract_mp3", "video/audio", ins={"in": "video"}, out="{stem}.mp3", gen=g_extract_mp3, weight=1.5,
         asks=["extract the audio from {in} as mp3", "get the sound from {in} as an mp3", "convert {in} to mp3",
               "rip the audio track out of {in}", "save just the audio of {in} as {out}", "I want the music from {in} in mp3"],
         cmd="ffmpeg -i {in} -vn -c:a libmp3lame -q:a {q} {out}",
         note=lambda p, res: f"-vn drops the video; -q:a {p['q']} is LAME VBR ({p['qd'][p['q']]}).",
         check=lambda c, p: c.v() is None and c.a()["codec_name"] == "mp3"))


def g_extract_copy(r, M):
    ac = M["in"].get("acodec")
    ext = {"aac": "m4a", "mp3": "mp3", "opus": "opus"}.get(ac)
    return {"ext": ext, "ac": ac} if ext else None


add(Task("v_extract_copy", "video/audio", ins={"in": "video"}, out="{stem}.{ext}", gen=g_extract_copy,
         asks=["extract the audio from {in} without re-encoding", "pull the original audio track out of {in} losslessly",
               "get the audio of {in} with no quality loss", "demux the audio from {in}"],
         cmd="ffmpeg -i {in} -vn -c:a copy {out}",
         note="The track is {ac}, so it's copied as-is into .{ext}: bit-identical, no quality loss.",
         check=lambda c, p: c.v() is None and c.a()["codec_name"] == p["ac"]))

add(Task("v_mute", "video/audio", ins={"in": "video"}, out="{stem}_muted.mp4",
         gen=lambda r, M: {"cv": "-c:v copy" if M["in"]["vcodec"] in ("h264", "hevc") else "-c:v libx264 -crf 20"}
         if M["in"].get("has_audio") else None,
         asks=["remove the audio from {in}", "mute {in}", "strip the sound from {in}", "make {in} silent",
               "delete the audio track in {in}", "I want {in} without sound"],
         cmd="ffmpeg -i {in} -an {cv} {out}",
         note="-an drops all audio; the video stream is untouched.", check=lambda c, p: c.a() is None))

add(Task("v_replace_audio", "video/audio", ins={"in": "video", "music": "music"}, out="{stem}_new_audio.mp4",
         gen=lambda r, M: {"cv": "-c:v copy" if M["in"]["vcodec"] in ("h264", "hevc") else "-c:v libx264 -crf 20",
                           "d": min(M["in"]["dur"], M["music"]["dur"]), "t": f'{min(M["in"]["dur"], M["music"]["dur"]):.2f}'},
         asks=["replace the audio in {in} with {music}", "swap the soundtrack of {in} for {music}",
               "use {music} as the audio for {in} instead of the original", "put {music} on {in} and drop the old sound"],
         cmd="ffmpeg -i {in} -i {music} -map 0:v -map 1:a {cv} -c:a aac -b:a 192k -t {t} {out}",
         note="-map 0:v -map 1:a takes video from the first input and audio from the second. -t {t} ends at the shorter of the two; -shortest can overshoot by seconds when the video is stream-copied.",
         check=lambda c, p: c.a() is not None and near(c.dur(), p["d"], 1.0)))

add(Task("v_add_audio_silent", "video/audio", ins={"in": "screen", "music": "music"}, out="{stem}_with_music.mp4",
         asks=["add {music} to {in}", "put background music ({music}) on {in}", "{in} has no sound, add {music} to it",
               "add an audio track from {music} to my screen recording {in}"],
         cmd="ffmpeg -i {in} -i {music} -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -t {t} {out}",
         note="The recording has no audio, so the music becomes the only track; video is copied and -t {t} cuts "
              "the music at the video's length.",
         gen=lambda r, M: {"t": f'{min(M["in"]["dur"], M["music"]["dur"]):.2f}'},
         check=lambda c, p: c.a() is not None))

add(Task("v_bg_music", "video/audio", ins={"in": "video", "music": "music"}, out="{stem}_music.mp4",
         gen=lambda r, M: {"vol": r.choice([0.1, 0.15, 0.2, 0.3]),
                           "cv": "-c:v copy" if M["in"]["vcodec"] in ("h264", "hevc") else "-c:v libx264 -crf 20"}
         if M["in"].get("has_audio") else None, weight=1.5,
         asks=["add {music} as quiet background music under the voice in {in}", "mix {music} into {in} at low volume",
               "put {music} behind the original audio of {in}, keep the voice loud",
               "add background music {music} to {in} at {vol} volume", "layer {music} softly under {in}'s audio"],
         cmd="ffmpeg -i {in} -i {music} -filter_complex \"[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first:"
             "normalize=0[a]\" -map 0:v -map \"[a]\" {cv} -c:a aac -b:a 192k {out}",
         note="Music is turned down to {vol}x and mixed under the original; normalize=0 stops amix from halving the "
              "voice, and duration=first ends at the video's length.",
         check=lambda c, p: c.a() is not None))

add(Task("v_volume", "video/audio", ins={"in": "video"}, out="{stem}_louder.mp4",
         gen=lambda r, M: (lambda v: {"v": v[0], "vs": v[1], "cv": "-c:v copy" if M["in"]["vcodec"] in ("h264", "hevc")
                                      else "-c:v libx264 -crf 20"})(r.choice([("2", "twice as loud"), ("1.5", "50% louder"),
                                                                              ("10dB", "10 dB louder"), ("0.5", "half as loud")]))
         if M["in"].get("has_audio") else None,
         asks=["make the audio in {in} {vs}", "change the volume of {in}: {vs}", "{in} audio should be {vs}"],
         cmd="ffmpeg -i {in} -af \"volume={v}\" {cv} -c:a aac -b:a 192k {out}",
         note="volume={v} scales the audio; video is stream-copied.", check=lambda c, p: c.a() is not None))

add(Task("v_loudnorm", "video/audio", ins={"in": "video"}, out="{stem}_normalized.mp4",
         gen=lambda r, M: {"I": r.choice([-14, -16, -16, -23]), "sr": M["in"]["asr"],
                           "cv": "-c:v copy" if M["in"]["vcodec"] in ("h264", "hevc") else "-c:v libx264 -crf 20"}
         if M["in"].get("has_audio") else None,
         asks=["normalize the audio of {in}", "fix the uneven volume in {in}", "normalize {in} to {I} LUFS",
               "make the audio level of {in} consistent", "loudness-normalize {in} for youtube"],
         cmd="ffmpeg -i {in} -af loudnorm=I={I}:TP=-1.5:LRA=11 -ar {sr} {cv} -c:a aac -b:a 192k {out}",
         note="EBU R128 loudnorm to {I} LUFS (-14 suits YouTube/Spotify, -16 podcasts, -23 broadcast). "
              "-ar {sr} matters: loudnorm upsamples to 192 kHz internally.",
         check=lambda c, p: int(c.a()["sample_rate"]) == p["sr"]))

add(Task("v_mute_section", "video/audio", ins={"in": "video"}, out="{stem}_bleeped.mp4",
         gen=lambda r, M: (lambda p: {**p, "cv": "-c:v copy" if M["in"]["vcodec"] in ("h264", "hevc") else "-c:v libx264"})(
             g_trim(r, M)) if M["in"].get("has_audio") else None,
         asks=["mute the audio in {in} between {ss} and {es}", "silence {in} from {ss} to {es}",
               "remove the sound only from {ss} to {es} in {in}, keep the video"],
         cmd="ffmpeg -i {in} -af \"volume=enable='between(t,{s},{e})':volume=0\" {cv} -c:a aac -b:a 192k {out}",
         note="volume=0 applied only while {s}s ≤ t ≤ {e}s (timeline 'enable'); everything else is untouched.",
         check=lambda c, p: c.a() is not None))

add(Task("v_whisper_wav", "video/audio", ins={"in": "video"}, out="{stem}.wav",
         gen=lambda r, M: {} if M["in"].get("has_audio") else None,
         asks=["extract audio from {in} for whisper transcription", "convert {in} to 16kHz mono wav",
               "prepare {in} for speech-to-text", "get a 16k mono wav from {in} so I can transcribe it"],
         cmd="ffmpeg -i {in} -vn -ac 1 -ar 16000 -c:a pcm_s16le {out}",
         note="16 kHz mono 16-bit PCM is what Whisper and most ASR models expect.",
         check=lambda c, p: c.a()["sample_rate"] == "16000" and c.a()["channels"] == 1))

# ───────────────────────────── VIDEO: speed / time ─────────────────────────────


def g_speed(r, M, fast=True):
    f = r.choice([1.25, 1.5, 2, 3, 4] if fast else [0.5, 0.25, 0.75])
    m = M["in"]
    p = {"f": f, "pts": f"{1 / f:g}*PTS", "total": m["dur"] / f,
         "fs": r.choice([f"{f:g}x", f"{f:g}x speed", f"{int(f * 100)}% speed"])}
    if m.get("has_audio"):
        p["af"] = f'-af "{atempo(f)}"'
        p["a"] = "-c:a aac -b:a 128k"
    else:
        p["af"] = p["a"] = ""
    return p


add(Task("v_speed", "video/speed", ins={"in": "video"}, out="{stem}_{f}x.mp4", gen=g_speed, weight=1.5,
         asks=["speed up {in} to {fs}", "make {in} play at {fs}", "make {in} faster ({fs})", "fast forward {in}, {fs}",
               "{in} at {fs} please, audio too"],
         cmd="ffmpeg -i {in} -vf \"setpts={pts}\" {af} -c:v libx264 -crf 20 {a} {out}",
         note="setpts={pts} speeds up the video; atempo speeds up the audio without changing pitch.",
         check=lambda c, p: near(c.dur(), p["total"], 0.08 * p["total"] + 0.5)))

add(Task("v_slowmo", "video/speed", ins={"in": "video"}, out="{stem}_slow.mp4",
         gen=lambda r, M: g_speed(r, M, fast=False),
         asks=["slow down {in} to {fs}", "make {in} slow motion ({fs})", "play {in} at {fs}", "make a slow-mo version of {in}, {fs}"],
         cmd="ffmpeg -i {in} -vf \"setpts={pts}\" {af} -c:v libx264 -crf 20 {a} {out}",
         note="setpts={pts} stretches the video timeline; atempo slows the audio without lowering the pitch.",
         check=lambda c, p: near(c.dur(), p["total"], 0.08 * p["total"] + 0.5)))

add(Task("v_timelapse", "video/speed", ins={"in": "video"}, out="{stem}_timelapse.mp4",
         gen=lambda r, M: (lambda f: {"f": f, "total": M["in"]["dur"] / f})(r.choice([5, 8, 10])),
         asks=["make a timelapse of {in}, {f}x faster", "turn {in} into a {f}x timelapse with no audio",
               "speed {in} up {f} times for a timelapse"],
         cmd="ffmpeg -i {in} -vf \"setpts=PTS/{f}\" -r 30 -an -c:v libx264 -crf 22 {out}",
         note="setpts=PTS/{f} compresses time {f}x; -r 30 drops the surplus frames; audio is removed (it would be noise at {f}x).",
         check=lambda c, p: near(c.dur(), p["total"], 0.1 * p["total"] + 0.5)))

add(Task("v_reverse", "video/speed", ins={"in": "video2"}, out="{stem}_reversed.mp4",
         gen=lambda r, M: {"d": M["in"]["dur"]},
         asks=["reverse {in}", "play {in} backwards", "make {in} run in reverse, audio too", "rewind effect on {in}, whole clip backwards"],
         cmd="ffmpeg -i {in} -vf reverse -af areverse -c:v libx264 -crf 20 -c:a aac {out}",
         note="reverse/areverse buffer the whole clip in RAM, so for long videos trim first.",
         check=lambda c, p: near(c.dur(), p["d"], 0.5)))

add(Task("v_boomerang", "video/speed", ins={"in": "video"}, out="{stem}_boomerang.mp4",
         gen=lambda r, M: (lambda t: {"t": t, "total": 2 * t})(r.choice([2, 3, 4])),
         asks=["make a boomerang from the first {t} seconds of {in}", "create a forward-then-backward loop of {in} ({t}s)",
               "boomerang effect on {in}, {t} seconds"],
         cmd="ffmpeg -i {in} -filter_complex \"[0:v]trim=0:{t},setpts=PTS-STARTPTS,split[a][b];[b]reverse[r];"
             "[a][r]concat=n=2:v=1:a=0[v]\" -map \"[v]\" -an -c:v libx264 -crf 20 {out}",
         note="The first {t}s play forward, then reversed: {total}s total, no audio.",
         check=lambda c, p: near(c.dur(), p["total"], 0.5)))

add(Task("v_fps", "video/convert", ins={"in": "video"}, out="{stem}_{fps}fps.mp4",
         gen=lambda r, M: (lambda f: {"fps": f, "aud": aud(M["in"])} if f != M["in"]["fps"] else None)(r.choice([24, 25, 30, 60])),
         asks=["change the frame rate of {in} to {fps} fps", "convert {in} to {fps}fps", "make {in} {fps} frames per second"],
         cmd="ffmpeg -i {in} -vf \"fps={fps}\" -c:v libx264 -crf 20 {aud} {out}",
         note="The fps filter duplicates or drops frames to hit exactly {fps} fps; duration is unchanged.",
         check=lambda c, p: near(c.fps(), p["fps"], 0.1)))

# ───────────────────────────── VIDEO: orientation ─────────────────────────────

ROT = [("90 degrees clockwise", "transpose=1", True), ("90° to the right", "transpose=1", True),
       ("90 degrees counterclockwise", "transpose=2", True), ("to the left", "transpose=2", True),
       ("180 degrees", "transpose=1,transpose=1", False), ("upside down", "transpose=1,transpose=1", False)]

add(Task("v_rotate", "video/orientation", ins={"in": "video"}, out="{stem}_rotated.mp4",
         gen=lambda r, M: (lambda x: {"rs": x[0], "vf": x[1], "swap": x[2], "aud": aud(M["in"]),
                                      "wh": (M["in"]["h"], M["in"]["w"]) if x[2] else (M["in"]["w"], M["in"]["h"])})(r.choice(ROT)),
         asks=["rotate {in} {rs}", "turn {in} {rs}", "{in} is sideways, rotate it {rs}", "flip {in} {rs}"],
         cmd="ffmpeg -i {in} -vf \"{vf}\" -c:v libx264 -crf 20 {aud} {out}",
         note="transpose=1 is 90° clockwise, transpose=2 counter-clockwise; two in a row make 180°.",
         check=lambda c, p: c.wh() == tuple(p["wh"])))

add(Task("v_rotate_meta", "video/orientation", ins={"in": "video"}, out="{stem}_rotated.mp4",
         gen=lambda r, M: (lambda d: {"deg": d, "rs": "clockwise" if d == -90 else "counterclockwise"})(r.choice([90, -90]))
         if M["in"]["vcodec"] in ("h264", "hevc") else None,
         asks=["rotate {in} 90° {rs} without re-encoding", "fix the orientation of {in} ({rs}) losslessly",
               "set the rotation flag on {in}, 90 {rs}, don't touch the quality"],
         cmd="ffmpeg -display_rotation {deg} -i {in} -c copy {out}",
         note="Writes a display-rotation flag instead of re-encoding, so it's instant and lossless. Players that ignore "
              "the flag (some editors and old TVs) will still show it unrotated.",
         check=lambda c, p: c.rotation() in (90, -90, 270)))

add(Task("v_flip", "video/orientation", ins={"in": "video"}, out="{stem}_flipped.mp4",
         gen=lambda r, M: (lambda x: {"fs": x[0], "vf": x[1], "aud": aud(M["in"])})(
             r.choice([("horizontally", "hflip"), ("mirror", "hflip"), ("vertically", "vflip")])),
         asks=["flip {in} {fs}", "{fs} {in}", "make a {fs} flipped copy of {in}"],
         cmd="ffmpeg -i {in} -vf {vf} -c:v libx264 -crf 20 {aud} {out}",
         note="hflip mirrors left/right, vflip top/bottom.", check=lambda c, p: c.v() is not None))
T[-1].asks = ["flip {in} {fs}", "make a flipped ({fs}) copy of {in}", "I need {in} flipped {fs}"]

# ───────────────────────────── VIDEO: frames / gif / images ─────────────────────────────


def g_gif(r, M):
    d = M["in"]["dur"]
    s = r.randint(0, int(d) - 5)
    t = r.choice([3, 4, 5])
    return {"s": s, "ss": say_t(r, s), "t": t, "fps": r.choice([10, 12, 15]), "w": r.choice([320, 480, 600])}


add(Task("v_gif", "video/gif", ins={"in": "video"}, out="{stem}.gif", gen=g_gif, weight=1.5,
         asks=["make a gif from {in} starting at {ss}, {t} seconds long", "turn {t} seconds of {in} (from {ss}) into a GIF",
               "convert {in} to a {w}px wide gif, {t}s from {ss}", "create a high quality gif of {in} at {ss} for {t} seconds",
               "gif of {in}, {ss} + {t}s, {fps} fps"],
         cmd="ffmpeg -ss {s} -t {t} -i {in} -vf \"fps={fps},scale={w}:-1:flags=lanczos,split[a][b];[a]palettegen[p];"
             "[b][p]paletteuse\" -loop 0 {out}",
         note="palettegen/paletteuse build a custom 256-colour palette, which looks far better than the default GIF palette. "
              "{fps} fps at {w}px keeps the file small.",
         check=lambda c, p: c.v()["codec_name"] == "gif" and c.wh()[0] == p["w"]))

add(Task("v_gif_to_mp4", "video/gif", ins={"in": "gif"}, out="{stem}.mp4",
         asks=["convert {in} to mp4", "turn {in} into a video file", "make {in} an mp4 so it's smaller",
               "{in} to mp4 for twitter"],
         cmd="ffmpeg -i {in} -movflags +faststart -pix_fmt yuv420p -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
             "-c:v libx264 -crf 20 {out}",
         note="The scale expression rounds odd GIF dimensions down to even (required by H.264 yuv420p); "
              "-pix_fmt yuv420p keeps it playable everywhere.",
         check=lambda c, p: c.v()["codec_name"] == "h264"))


def g_frame(r, M):
    s = round(r.uniform(0.5, M["in"]["dur"] - 1), 1)
    return {"t": s, "ts": say_t(r, int(s)) if s >= 1 else f"{s} seconds", "T": f"{s:g}"}


add(Task("v_frame", "video/frames", ins={"in": "video"}, out="{stem}_frame.jpg", gen=g_frame, weight=1.5,
         asks=["grab a frame from {in} at {ts}", "take a screenshot of {in} at {ts}", "save the frame at {ts} of {in} as a jpg",
               "extract a still image from {in} at {ts}", "export the frame at {ts} from {in} as {out}"],
         cmd="ffmpeg -ss {T} -i {in} -frames:v 1 -q:v 2 {out}",
         note="-frames:v 1 writes a single image; -q:v 2 is high JPEG quality (scale 2–31, lower = better).",
         check=lambda c, p: c.img()["w"] > 0))
T.append(Task("v_frame_png", "video/frames", ins={"in": "video"}, out="{stem}_{T}s.png", gen=g_frame,
              asks=["save a lossless png of {in} at {ts}", "export a PNG still of {in} at {ts}", "frame at {ts} of {in} as png please"],
              cmd="ffmpeg -ss {T} -i {in} -frames:v 1 {out}", note="PNG is lossless, good for thumbnails you'll edit further.",
              check=lambda c, p: c.img()["w"] > 0))

add(Task("v_frames_every", "video/frames", ins={"in": "video"}, out="frame_%04d.jpg",
         gen=lambda r, M: (lambda n: {"n": n, "exp": int(M["in"]["dur"] // n)})(r.choice([1, 2, 5, 10])),
         asks=["extract a frame every {n} seconds from {in}", "save one image per {n} second(s) of {in}",
               "dump frames from {in}, one every {n}s, as jpgs", "export stills from {in} every {n} seconds"],
         cmd="ffmpeg -i {in} -vf \"fps=1/{n}\" -q:v 2 {out}",
         note="fps=1/{n} keeps one frame per {n}s; %04d numbers the files.",
         check=lambda c, p: abs(c.count() - p["exp"]) <= 2))

add(Task("v_contact_sheet", "video/frames", ins={"in": "video"}, out="{stem}_contact_sheet.jpg",
         gen=lambda r, M: (lambda g: {"c": g[0], "rw": g[1], "n": max(1, int(M["in"]["dur"] // (g[0] * g[1])))})(
             r.choice([(3, 3), (4, 3), (4, 4), (5, 4)])),
         asks=["make a {c}x{rw} thumbnail grid of {in}", "create a contact sheet for {in} with {c} columns and {rw} rows",
               "generate a preview mosaic of {in}", "make one image with {c}x{rw} snapshots across {in}"],
         cmd="ffmpeg -i {in} -vf \"fps=1/{n},scale=320:-1,tile={c}x{rw}\" -frames:v 1 -q:v 3 {out}",
         note="One frame every {n}s, scaled to 320px wide and tiled {c}x{rw} into a single image.",
         check=lambda c, p: c.img()["w"] >= 320 * p["c"] - 1))

add(Task("v_thumbnail", "video/frames", ins={"in": "video"}, out="{stem}_thumb.jpg",
         asks=["pick a good thumbnail for {in}", "generate a representative thumbnail image from {in}",
               "auto-select a thumbnail frame from {in}"],
         cmd="ffmpeg -i {in} -vf \"thumbnail=300\" -frames:v 1 -q:v 2 {out}",
         note="The thumbnail filter picks the most representative frame from each batch of 300, avoiding black or blurry frames.",
         check=lambda c, p: c.img()["w"] > 0))

add(Task("v_from_frames", "video/frames", out="video.mp4", batch=("frames",),
         gen=lambda r, M: (lambda fr: {"fr": fr, "total": 20 / fr})(r.choice([1, 2, 5, 10, 24])),
         asks=["make a video from {files} at {fr} fps", "turn my image sequence ({files}) into an mp4, {fr} frames per second",
               "combine {files} into a video", "create a timelapse video from {files}, {fr}fps"],
         cmd="ffmpeg -framerate {fr} -i in/frame_%03d.png -c:v libx264 -pix_fmt yuv420p -r 30 {out}",
         note="-framerate {fr} sets how fast images advance ({total:g}s for 20 frames); -pix_fmt yuv420p keeps it playable everywhere.",
         check=lambda c, p: near(c.dur(), p["total"], 0.6)))

add(Task("v_slideshow", "video/frames", out="slideshow.mp4", batch=("jpg",),
         gen=lambda r, M: (lambda n: {"n": n, "total": 4 * n})(r.choice([2, 3, 4, 5])),
         asks=["make a slideshow from {files}, {n} seconds per photo", "turn {files} into a video slideshow, each photo {n}s",
               "create an mp4 slideshow of {files} with {n} seconds each"],
         cmd="ffmpeg -framerate 1/{n} -pattern_type glob -i 'in/*.jpg' -vf \"scale=1920:1080:force_original_aspect_ratio=decrease,"
             "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1\" -c:v libx264 -r 30 -pix_fmt yuv420p {out}",
         note="-framerate 1/{n} shows each image for {n}s; images are fitted into 1080p, so mixed sizes work.",
         check=lambda c, p: near(c.dur(), p["total"], p["n"] + 0.5)))


def g_still(r, M):
    m = M["img"]
    vf = '-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ' if m["w"] % 2 or m["h"] % 2 else ""
    return {"vf": vf, "d": M["in"]["dur"], "t": f'{M["in"]["dur"]:.2f}',
            "why": " The image has an odd dimension, so it's rounded to even for H.264." if vf else ""}


add(Task("v_still_audio", "video/frames", ins={"img": "image", "in": "audio"}, out="{stem}_video.mp4", gen=g_still,
         asks=["make a video from {img} and {in}", "turn {in} into a video with {img} as the picture",
               "combine {img} and {in} into an mp4 for youtube", "podcast video: static image {img} with audio {in}",
               "create an mp4 showing {img} while {in} plays"],
         cmd="ffmpeg -loop 1 -i {img} -i {in} {vf}-c:v libx264 -tune stillimage -c:a aac -b:a 192k -pix_fmt yuv420p -t {t} {out}",
         note="-loop 1 repeats the image and -t {t} ends it with the audio (-shortest overshoots with looped images); -tune stillimage optimises for a static picture.{why}",
         check=lambda c, p: near(c.dur(), p["d"], 1.5)))

add(Task("v_kenburns", "video/frames", ins={"in": "image"}, out="{stem}_zoom.mp4",
         gen=lambda r, M: (lambda t: {"t": t, "fr": t * 30})(r.choice([4, 5, 6, 8])),
         asks=["make a {t} second slow zoom video from {in}", "ken burns effect on {in}, {t}s", "animate {in} with a slow zoom-in, {t} seconds"],
         cmd="ffmpeg -i {in} -vf \"scale=3840:-2,zoompan=z='min(zoom+0.0015,1.5)':d={fr}:x='iw/2-(iw/zoom/2)':"
             "y='ih/2-(ih/zoom/2)':s=1920x1080:fps=30\" -c:v libx264 -pix_fmt yuv420p {out}",
         note="Upscaling first stops zoompan jitter; d={fr} frames at 30 fps = {t}s, zooming towards the centre.",
         check=lambda c, p: near(c.dur(), p["t"], 0.5)))

# ───────────────────────────── VIDEO: overlays / text / subs ─────────────────────────────

POS = {"top-left": "10:10", "top-right": "W-w-10:10", "bottom-right": "W-w-10:H-h-10",
       "bottom-left": "10:H-h-10", "center": "(W-w)/2:(H-h)/2"}


def g_wm(r, M):
    pos = r.choice(list(POS))
    lw = r.choice([0, 120, 150, 200])
    op = r.choice([1, 1, 0.5, 0.7])
    chain = []
    if lw:
        chain.append(f"scale={lw}:-1")
    if op < 1:
        chain.append(f"format=rgba,colorchannelmixer=aa={op}")
    pre = f"[1:v]{','.join(chain)}[wm];[0:v][wm]" if chain else "[0:v][1:v]"
    extra = (f", {lw}px wide" if lw else "") + (f", {int(op * 100)}% opacity" if op < 1 else "")
    a = "-map 0:a -c:a copy" if M["in"].get("acodec") == "aac" else ("-map 0:a -c:a aac" if M["in"].get("has_audio") else "")
    return {"pos": pos, "xy": POS[pos], "pre": pre, "extra": extra, "a": a}


add(Task("v_watermark", "video/overlay", ins={"in": "video", "logo": "logo"}, out="{stem}_watermarked.mp4", gen=g_wm, weight=1.5,
         asks=["add {logo} as a watermark in the {pos} of {in}{extra}", "put my logo {logo} on {in}, {pos}{extra}",
               "overlay {logo} onto {in} at the {pos}{extra}", "watermark {in} with {logo} ({pos}{extra})"],
         cmd="ffmpeg -i {in} -i {logo} -filter_complex \"{pre}overlay={xy}[v]\" -map \"[v]\" {a} -c:v libx264 -crf 20 {out}",
         note="overlay={xy}: W/H are the video size and w/h the logo size, so this pins it {pos} with a 10px margin.",
         check=lambda c, p: c.v() is not None))

TEXTS = ["Hello World", "Subscribe", "Episode 12", "DRAFT", "Chapter One", "Sample Text", "Thanks for watching", "Day 1"]
TPOS = {"at the bottom": "x=(w-text_w)/2:y=h-th-40", "at the top": "x=(w-text_w)/2:y=40",
        "in the center": "x=(w-text_w)/2:y=(h-text_h)/2", "in the top left": "x=30:y=30"}

add(Task("v_text", "video/overlay", ins={"in": "video"}, out="{stem}_text.mp4",
         gen=lambda r, M: (lambda tp: {"text": r.choice(TEXTS), "tp": tp, "xy": TPOS[tp], "fs": r.choice([36, 48, 64]),
                                       "color": r.choice(["white", "yellow", "white"]), "aud": aud(M["in"])})(r.choice(list(TPOS))),
         asks=["add the text \"{text}\" {tp} of {in}", "write '{text}' on {in} {tp}", "put a caption \"{text}\" {tp} of {in}, font size {fs}",
               "overlay the words {text} on {in} {tp} in {color}"],
         cmd="ffmpeg -i {in} -vf \"drawtext=text='{text}':fontsize={fs}:fontcolor={color}:box=1:boxcolor=black@0.5:"
             "boxborderw=12:{xy}\" -c:v libx264 -crf 20 {aud} {out}",
         note="drawtext with a semi-transparent box behind the text for readability; text_w/text_h centre it.",
         check=lambda c, p: c.v() is not None))

add(Task("v_timestamp", "video/overlay", ins={"in": "video"}, out="{stem}_timecode.mp4", gen=lambda r, M: {"aud": aud(M["in"])},
         asks=["burn a running timestamp into {in}", "show the elapsed time on screen in {in}", "add a timecode overlay to {in}"],
         cmd=r"ffmpeg -i {in} -vf \"drawtext=text='%{{pts\:hms}}':x=20:y=20:fontsize=36:fontcolor=white:box=1:"
             r"boxcolor=black@0.6\" -c:v libx264 -crf 20 {aud} {out}",
         note="%{{pts\\:hms}} prints each frame's timestamp as HH:MM:SS.mmm (the colon is escaped inside the expression).",
         check=lambda c, p: c.v() is not None))
T[-1].cmd = T[-1].cmd.replace('\\"', '"')

add(Task("v_burn_subs", "video/subtitles", ins={"in": "video", "subs": "subs"}, out="{stem}_subbed.mp4",
         gen=lambda r, M: (lambda s: {"style": s[0], "sd": s[1], "aud": aud(M["in"])})(
             r.choice([("", ""), (":force_style='FontSize=24'", ", font size 24"),
                       (":force_style='FontSize=20,PrimaryColour=&H0000FFFF&'", " in yellow")])),
         asks=["burn {subs} into {in}{sd}", "hardcode the subtitles from {subs} onto {in}{sd}",
               "add {subs} to {in} permanently (burned in){sd}", "put the captions in {subs} on the video {in}{sd}"],
         cmd="ffmpeg -i {in} -vf \"subtitles={subs}{style}\" -c:v libx264 -crf 20 {aud} {out}",
         note="The subtitles filter renders the SRT into the picture (always visible, can't be turned off). "
              "force_style takes ASS style fields; colours are &HBBGGRR&.",
         check=lambda c, p: c.v() is not None))
T[-1].gen = (lambda g: lambda r, M: None if " " in M["subs"]["_name"] else g(r, M))(T[-1].gen)

add(Task("v_soft_subs", "video/subtitles", ins={"in": "video", "subs": "subs"}, out="{stem}_subs.mp4",
         gen=lambda r, M: {"lang": r.choice(["eng", "eng", "spa", "hin"])}
         if M["in"]["vcodec"] in ("h264", "hevc") and M["in"].get("acodec") in ("aac", "mp3") else None,
         asks=["add {subs} to {in} as a subtitle track I can turn on and off", "embed {subs} in {in} as soft subs",
               "attach the subtitle file {subs} to {in} without burning it in", "mux {subs} into {in} (language {lang})"],
         cmd="ffmpeg -i {in} -i {subs} -map 0:v -map 0:a -map 1 -c:v copy -c:a copy -c:s mov_text "
             "-metadata:s:s:0 language={lang} {out}",
         note="MP4 only supports mov_text subtitles, so the SRT is converted; audio/video are copied untouched.",
         check=lambda c, p: c.sub() is not None))

add(Task("v_extract_subs", "video/subtitles", ins={"in": "subsvideo"}, out="{stem}.srt",
         asks=["extract the subtitles from {in}", "get the srt out of {in}", "save the subtitle track of {in} as a file",
               "rip the subs from {in} to srt"],
         cmd="ffmpeg -i {in} -map 0:s:0 {out}", note="-map 0:s:0 picks the first subtitle stream; .srt output converts it to SRT.",
         check=lambda c, p: "-->" in c.text()))

add(Task("v_burn_embedded_subs", "video/subtitles", ins={"in": "subsvideo"}, out="{stem}_hardsub.mp4",
         asks=["burn the built-in subtitles of {in} into the picture", "hardsub {in} using its own subtitle track",
               "make the subtitles in {in} permanent and convert to mp4"],
         cmd="ffmpeg -i {in} -vf \"subtitles={in}\" -c:v libx264 -crf 20 -c:a aac -b:a 160k {out}",
         note="Pointing the subtitles filter at the MKV itself renders its first subtitle track into the video.",
         check=lambda c, p: c.v() is not None and c.sub() is None))

add(Task("v_pip", "video/overlay", ins={"in": "video", "in2": "video2"}, out="{stem}_pip.mp4",
         gen=lambda r, M: (lambda pos: {"pos": pos, "xy": POS[pos].replace("10", "20"), "pw": r.choice([320, 400, 480]),
                                        "a": "-map 0:a -c:a aac" if M["in"].get("has_audio") else ""})(
             r.choice(["bottom-right", "top-right", "bottom-left", "top-left"])),
         asks=["picture in picture: put {in2} small in the {pos} of {in}", "overlay {in2} as a small window ({pw}px) on {in}, {pos}",
               "add my webcam clip {in2} to the {pos} corner of {in}"],
         cmd="ffmpeg -i {in} -i {in2} -filter_complex \"[1:v]scale={pw}:-2[pip];[0:v][pip]overlay={xy}:eof_action=pass[v]\" "
             "-map \"[v]\" {a} -c:v libx264 -crf 20 {out}",
         note="The second video is scaled to {pw}px and pinned {pos}; eof_action=pass hides it once it ends.",
         check=lambda c, p: c.v() is not None))

add(Task("v_stack", "video/overlay", ins={"in": "video", "in2": "video2"}, out="{stem}_stacked.mp4",
         gen=lambda r, M: r.choice([{"how": "side by side", "f": "hstack", "s": "-2:720"},
                                    {"how": "on top of each other", "f": "vstack", "s": "1280:-2"}]),
         asks=["put {in} and {in2} {how}", "combine {in} and {in2} {how} in one video", "{in} and {in2} {how}, comparison video"],
         cmd="ffmpeg -i {in} -i {in2} -filter_complex \"[0:v]scale={s},setsar=1[a];[1:v]scale={s},setsar=1[b];"
             "[a][b]{f}=inputs=2:shortest=1[v]\" -map \"[v]\" -map 0:a? -c:v libx264 -crf 20 -c:a aac {out}",
         note="Both inputs are scaled to a matching {s} edge ({f} needs equal heights/widths), stopping when the shorter one ends.",
         check=lambda c, p: c.v() is not None))
T[-1].cmd = T[-1].cmd.replace("-map 0:a?", "-map '0:a?'")

# ───────────────────────────── VIDEO: filters / look ─────────────────────────────


def g_fade(r, M):
    d = M["in"]["dur"]
    f = r.choice([1, 2, 3])
    st = round(d - f, 2)
    p = {"f": f, "st": f"{st:g}", "dur": round(d, 1)}
    p["af"] = (f' -af "afade=t=in:st=0:d={f},afade=t=out:st={st:g}:d={f}" -c:a aac -b:a 128k'
               if M["in"].get("has_audio") else "")
    return p


add(Task("v_fade", "video/filter", ins={"in": "video"}, out="{stem}_fade.mp4", gen=g_fade, weight=1.5,
         asks=["add a {f} second fade in and fade out to {in}", "fade {in} in from black and out to black ({f}s)",
               "make {in} fade in at the start and fade out at the end", "{f}s fades on {in}, video and audio"],
         cmd="ffmpeg -i {in} -vf \"fade=t=in:st=0:d={f},fade=t=out:st={st}:d={f}\"{af} -c:v libx264 -crf 20 {out}",
         note="The clip is {dur}s long, so the fade-out starts at {st}s; audio fades with the picture.",
         check=lambda c, p: near(c.dur(), p["dur"], 0.5)))

LOOKS = [("black and white", "hue=s=0", "hue=s=0 removes all saturation"),
         ("grayscale", "hue=s=0", "hue=s=0 removes all saturation"),
         ("brighter", "eq=brightness=0.08:contrast=1.05", "eq brightness/contrast; brightness ranges -1..1"),
         ("more contrast", "eq=contrast=1.3", "eq contrast >1 increases contrast"),
         ("more saturated / vivid", "eq=saturation=1.4", "eq saturation >1 boosts colour"),
         ("sharper", "unsharp=5:5:1.0", "unsharp 5x5 luma kernel, amount 1.0"),
         ("less noisy (denoise)", "hqdn3d=4:3:6:4.5", "hqdn3d is a fast spatial+temporal denoiser"),
         ("blurred", "gblur=sigma=10", "gaussian blur; raise sigma for more"),
         ("sepia", "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131", "the classic sepia colour matrix"),
         ("deinterlaced", "bwdif", "bwdif deinterlaces combed/interlaced footage"),
         ("vintage / vignette", "vignette=PI/4", "vignette darkens the corners"),
         ("negative / inverted colors", "negate", "negate inverts every colour")]

add(Task("v_look", "video/filter", ins={"in": "video"}, out="{stem}_edit.mp4", weight=2,
         gen=lambda r, M: (lambda x: {"look": x[0], "vf": x[1], "why": x[2], "aud": aud(M["in"])})(r.choice(LOOKS)),
         asks=["make {in} {look}", "apply a {look} effect to {in}", "I want {in} to look {look}", "{in}: {look} please",
               "edit {in} so it's {look}"],
         cmd="ffmpeg -i {in} -vf \"{vf}\" -c:v libx264 -crf 20 {aud} {out}", note="{why}.",
         check=lambda c, p: c.v() is not None))


def g_blur_region(r, M):
    m = M["in"]
    w, h = even(m["w"] * r.uniform(0.15, 0.35)), even(m["h"] * r.uniform(0.15, 0.35))
    x, y = r.randint(0, m["w"] - w), r.randint(0, m["h"] - h)
    return {"w": w, "h": h, "x": x, "y": y, "aud": aud(m)}


add(Task("v_blur_region", "video/filter", ins={"in": "video"}, out="{stem}_blurred.mp4", gen=g_blur_region,
         asks=["blur a {w}x{h} box at {x},{y} in {in} (license plate)", "hide the area x={x} y={y} size {w}x{h} in {in} with blur",
               "pixelate/blur the region {w}x{h}+{x}+{y} of {in} to hide a face"],
         cmd="ffmpeg -i {in} -filter_complex \"[0:v]crop={w}:{h}:{x}:{y},boxblur=12[b];[0:v][b]overlay={x}:{y}[v]\" "
             "-map \"[v]\" -map '0:a?' -c:v libx264 -crf 20 -c:a copy {out}",
         note="Crops the region, blurs it, and overlays it back in the same spot; the rest stays sharp.",
         check=lambda c, p: c.v() is not None))

add(Task("v_delogo", "video/filter", ins={"in": "video"}, out="{stem}_nologo.mp4", gen=g_blur_region,
         asks=["remove the logo at x={x}, y={y}, {w}x{h} from {in}", "get rid of the watermark in {in}; it's at {x},{y} and {w}x{h}",
               "delogo {in} at {x}:{y}, size {w}x{h}"],
         cmd="ffmpeg -i {in} -vf \"delogo=x={x}:y={y}:w={w}:h={h}\" -c:v libx264 -crf 20 {aud} {out}",
         note="delogo interpolates from the surrounding pixels. It works best on small, static logos; large ones leave a smear.",
         check=lambda c, p: c.v() is not None))
T[-1].gen = (lambda g: lambda r, M: (lambda p: p if p["x"] > 0 and p["y"] > 0 and p["x"] + p["w"] < M["in"]["w"]
                                     and p["y"] + p["h"] < M["in"]["h"] else None)(g(r, M)))(g_blur_region)

add(Task("v_stabilize", "video/filter", ins={"in": "video"}, out="{stem}_stable.mp4",
         gen=lambda r, M: {"sm": r.choice([10, 20, 30]), "a": "-c:a copy" if M["in"].get("acodec") == "aac" else "-c:a aac"},
         asks=["stabilize {in}", "remove camera shake from {in}", "{in} is shaky, can you smooth it out",
               "deshake {in} with vidstab"],
         cmd="ffmpeg -i {in} -vf vidstabdetect=shakiness=5:result=transforms.trf -f null - && "
             "ffmpeg -i {in} -vf \"vidstabtransform=input=transforms.trf:smoothing={sm},unsharp=5:5:0.8:3:3:0.4\" "
             "-c:v libx264 -crf 20 {a} {out}",
         note="Pass 1 measures the motion, pass 2 smooths it (smoothing={sm} frames) and lightly sharpens to offset the zoom.",
         check=lambda c, p: c.v() is not None))
T[-1].gen = (lambda g: lambda r, M: g(r, M) if M["in"].get("has_audio") else None)(T[-1].gen)

# ───────────────────────────── VIDEO: delivery / metadata ─────────────────────────────

add(Task("v_faststart", "video/delivery", ins={"in": "video"}, out="{stem}_web.mp4",
         gen=lambda r, M: {} if M["in"]["_name"].lower().endswith((".mp4", ".mov")) else None,
         asks=["optimize {in} for web streaming", "make {in} start playing immediately in the browser",
               "move the moov atom of {in} to the front", "add faststart to {in}"],
         cmd="ffmpeg -i {in} -c copy -movflags +faststart {out}",
         note="+faststart moves the index (moov atom) to the start so browsers can play before the download finishes; no re-encode.",
         check=lambda c, p: c.moov_first()))

add(Task("v_hls", "video/delivery", ins={"in": "video"}, out="playlist.m3u8",
         gen=lambda r, M: {"n": r.choice([4, 6, 10])},
         asks=["convert {in} to HLS", "make an HLS stream (m3u8) from {in} with {n} second segments",
               "package {in} for HLS streaming", "create an m3u8 playlist and ts segments from {in}"],
         cmd="ffmpeg -i {in} -c:v libx264 -crf 21 -g 60 -c:a aac -b:a 128k -f hls -hls_time {n} -hls_playlist_type vod "
             "-hls_segment_filename out/seg_%03d.ts {out}",
         note="Writes playlist.m3u8 plus ~{n}s .ts segments. -g 60 forces regular keyframes so segments can split cleanly.",
         check=lambda c, p: c.count("*.ts") >= 2))
T[-1].gen = (lambda g: lambda r, M: g(r, M) if M["in"].get("has_audio") else None)(T[-1].gen)

add(Task("v_set_title", "video/metadata", ins={"in": "video"}, out="{stem}_tagged.mp4",
         gen=lambda r, M: {"title": r.choice(["Family Trip 2024", "Product Demo", "Lecture 1", "My Wedding"])}
         if M["in"]["_name"].lower().endswith((".mp4", ".mov")) else None,
         asks=["set the title of {in} to \"{title}\"", "change the metadata title of {in} to {title}", "tag {in} with title '{title}'"],
         cmd="ffmpeg -i {in} -map 0 -c copy -metadata title=\"{title}\" {out}",
         note="-metadata writes the container tag; -c copy leaves the streams untouched.",
         check=lambda c, p: c.tags().get("title") == p["title"]))

add(Task("v_strip_meta", "video/metadata", ins={"in": "video"}, out="{stem}_clean.mp4",
         gen=lambda r, M: {} if M["in"]["_name"].lower().endswith((".mp4", ".mov")) else None,
         asks=["remove all metadata from {in}", "strip the metadata (dates, titles) out of {in}", "clean {in} of any metadata before sharing"],
         cmd="ffmpeg -i {in} -map 0 -map_metadata -1 -c copy -fflags +bitexact {out}",
         note="-map_metadata -1 drops global and stream tags; +bitexact stops ffmpeg writing its own encoder tag.",
         check=lambda c, p: "title" not in c.tags() and "creation_time" not in c.tags()))

add(Task("v_waves", "video/visualize", ins={"in": "audio"}, out="{stem}_waveform.mp4",
         gen=lambda r, M: {"mode": r.choice(["line", "cline", "p2p"]), "col": r.choice(["white", "cyan", "0x38bdf8"])},
         asks=["make an audio visualizer video for {in}", "turn {in} into a video with a moving waveform",
               "create a waveform animation video from {in} for instagram"],
         cmd="ffmpeg -i {in} -filter_complex \"[0:a]showwaves=s=1280x720:mode={mode}:colors={col},format=yuv420p[v]\" "
             "-map \"[v]\" -map 0:a -c:v libx264 -crf 20 -c:a aac -b:a 192k {out}",
         note="showwaves renders the audio as an animated {mode} waveform; the audio is muxed back in.",
         check=lambda c, p: c.v() is not None and c.a() is not None))

# ───────────────────────────── AUDIO (ffmpeg) ─────────────────────────────

AFMT = {
    "mp3": ("-c:a libmp3lame -b:a {br}k", "mp3", "MP3 CBR {br} kbps"),
    "m4a": ("-c:a aac -b:a {br}k", "aac", "AAC {br} kbps in an M4A container"),
    "flac": ("-c:a flac", "flac", "FLAC, lossless and compressed"),
    "wav": ("-c:a pcm_s16le", "pcm_s16le", "16-bit PCM WAV, uncompressed"),
    "opus": ("-c:a libopus -b:a {obr}k", "opus", "Opus {obr} kbps, excellent quality per bit"),
    "ogg": ("-c:a libvorbis -q:a 5", "vorbis", "Ogg Vorbis at quality 5 (~160 kbps)"),
}


def g_aconv(r, M):
    src = M["in"]["_name"].rsplit(".", 1)[1].lower()
    fmt = r.choice([f for f in AFMT if f != src])
    br = r.choice([128, 192, 256, 320])
    obr = r.choice([64, 96, 128])
    a, codec, desc = AFMT[fmt]
    return {"fmt": fmt, "args": a.format(br=br, obr=obr), "codec": codec, "desc": desc.format(br=br, obr=obr), "br": br}


add(Task("a_convert", "audio/convert", ins={"in": "audio"}, out="{stem}.{fmt}", gen=g_aconv, weight=2,
         asks=["convert {in} to {fmt}", "change {in} into {fmt} format", "I need {in} as a .{fmt} file", "{in} to {fmt} please",
               "transcode {in} to {fmt}", "save {in} as {out}"],
         cmd="ffmpeg -i {in} -vn {args} {out}", note="{desc}.",
         check=lambda c, p: c.a()["codec_name"] == p["codec"]))

add(Task("a_whisper", "audio/convert", ins={"in": "audio"}, out="{stem}_16k.wav",
         asks=["convert {in} to 16kHz mono wav for transcription", "prepare {in} for whisper", "resample {in} to 16000 Hz mono",
               "make {in} ASR-ready (16k, mono, 16-bit)"],
         cmd="ffmpeg -i {in} -ac 1 -ar 16000 -c:a pcm_s16le {out}",
         note="Downmixed to mono and resampled to 16 kHz 16-bit PCM, the input format speech models expect.",
         check=lambda c, p: c.a()["sample_rate"] == "16000" and c.a()["channels"] == 1))


def keep_fmt(m, name):
    ext = name.rsplit(".", 1)[1].lower()
    return {"mp3": "-c:a libmp3lame -b:a 192k", "wav": "-c:a pcm_s16le", "flac": "-c:a flac",
            "m4a": "-c:a aac -b:a 192k"}[ext], ext


def g_anorm(r, M):
    args, ext = keep_fmt(M["in"], M["in"]["_name"])
    return {"I": r.choice([-14, -16, -16, -19, -23]), "sr": M["in"]["asr"], "args": args, "ext": ext}


add(Task("a_loudnorm", "audio/level", ins={"in": "audio"}, out="{stem}_normalized.{ext}", gen=g_anorm, weight=1.5,
         asks=["normalize {in}", "normalize the loudness of {in} to {I} LUFS", "make {in} the standard podcast loudness",
               "even out the volume of {in}", "{in} is too quiet, normalize it", "loudness normalize {in} for spotify"],
         cmd="ffmpeg -i {in} -af loudnorm=I={I}:TP=-1.5:LRA=11 -ar {sr} {args} {out}",
         note="EBU R128 loudnorm to {I} LUFS with a -1.5 dBTP ceiling. -ar {sr} restores the original sample rate "
              "(loudnorm outputs 192 kHz otherwise).",
         check=lambda c, p: int(c.a()["sample_rate"]) == p["sr"]))

add(Task("a_volume", "audio/level", ins={"in": "audio"}, out="{stem}_vol.{ext}",
         gen=lambda r, M: (lambda k: {**dict(zip(("args", "ext"), keep_fmt(M["in"], M["in"]["_name"]))), "v": k[0], "vs": k[1]})(
             r.choice([("2", "twice as loud"), ("1.5", "50% louder"), ("6dB", "6 dB louder"), ("0.5", "half the volume"),
                       ("-3dB", "3 dB quieter")])),
         asks=["make {in} {vs}", "change the volume of {in}: {vs}", "adjust {in} to be {vs}", "{in} volume -> {vs}"],
         cmd="ffmpeg -i {in} -af \"volume={v}\" {args} {out}",
         note="volume={v}. Watch for clipping when boosting; loudnorm is safer for general 'make it louder'.",
         check=lambda c, p: c.a() is not None))


def g_atrim(r, M):
    p = g_trim(r, M)
    p["ext"] = M["in"]["_name"].rsplit(".", 1)[1].lower()
    return p


add(Task("a_trim", "audio/trim", ins={"in": "audio"}, out="{stem}_clip.{ext}", gen=g_atrim, weight=1.5,
         asks=["cut {in} from {ss} to {es}", "trim {in} to {ss}–{es}", "keep only {ss} to {es} of {in}",
               "extract the section between {ss} and {es} from {in}", "make a clip of {in} from {ss} to {es}"],
         cmd="ffmpeg -ss {S} -to {E} -i {in} -c copy {out}",
         note="Audio has no keyframe problem, so -c copy cuts accurately (within one frame, ~26 ms for MP3/AAC) with no re-encode.",
         check=lambda c, p: near(c.dur(), p["len"], 0.5)))


def g_afade(r, M):
    d = M["in"]["dur"]
    fi, fo = r.choice([1, 2, 3]), r.choice([2, 3, 5])
    args, ext = keep_fmt(M["in"], M["in"]["_name"])
    return {"fi": fi, "fo": fo, "st": f"{d - fo:g}", "args": args, "ext": ext, "dur": d}


add(Task("a_fade", "audio/edit", ins={"in": "audio"}, out="{stem}_faded.{ext}", gen=g_afade,
         asks=["add a {fi}s fade in and a {fo}s fade out to {in}", "fade {in} in over {fi} seconds and out over {fo} seconds",
               "make {in} fade in and fade out"],
         cmd="ffmpeg -i {in} -af \"afade=t=in:st=0:d={fi},afade=t=out:st={st}:d={fo}\" {args} {out}",
         note="The file is {dur:g}s long, so the {fo}s fade-out starts at {st}s.",
         check=lambda c, p: near(c.dur(), p["dur"], 0.3)))

add(Task("a_concat", "audio/join", ins={"in": "audio", "in2": "music"}, out="joined.mp3",
         gen=lambda r, M: {"total": M["in"]["dur"] + M["in2"]["dur"]},
         asks=["join {in} and {in2} into one file", "combine {in} followed by {in2}", "concatenate {in} and {in2} into an mp3",
               "append {in2} to the end of {in}"],
         cmd="ffmpeg -i {in} -i {in2} -filter_complex \"[0:a]aresample=44100,aformat=channel_layouts=stereo[a0];"
             "[1:a]aresample=44100,aformat=channel_layouts=stereo[a1];[a0][a1]concat=n=2:v=0:a=1[a]\" -map \"[a]\" "
             "-c:a libmp3lame -b:a 192k {out}",
         note="Both inputs are brought to 44.1 kHz stereo first, so files with different formats join cleanly.",
         check=lambda c, p: near(c.dur(), p["total"], 1.0)))

add(Task("a_mix", "audio/join", ins={"in": "audio", "music": "music"}, out="{stem}_mix.mp3",
         gen=lambda r, M: {"vol": r.choice([0.15, 0.25, 0.4])},
         asks=["mix {in} with {music} in the background", "layer {music} under {in} at {vol} volume",
               "put {music} quietly behind {in}", "combine {in} and {music} playing at the same time"],
         cmd="ffmpeg -i {in} -i {music} -filter_complex \"[1:a]volume={vol}[m];[0:a][m]amix=inputs=2:duration=first:"
             "normalize=0[a]\" -map \"[a]\" -c:a libmp3lame -b:a 192k {out}",
         note="The music is turned down to {vol}x and mixed under; duration=first ends with {in_name}.",
         check=lambda c, p: c.a() is not None))

add(Task("a_tempo", "audio/speed", ins={"in": "audio"}, out="{stem}_{f}x.{ext}",
         gen=lambda r, M: (lambda f: {"f": f, "af": atempo(f), "total": M["in"]["dur"] / f,
                                      **dict(zip(("args", "ext"), keep_fmt(M["in"], M["in"]["_name"])))})(
             r.choice([0.75, 0.8, 1.25, 1.5, 2])),
         asks=["change the speed of {in} to {f}x without changing the pitch", "play {in} at {f}x speed",
               "make {in} {f} times as fast, keep the voice natural", "speed {in} to {f}x"],
         cmd="ffmpeg -i {in} -af \"{af}\" {args} {out}",
         note="atempo changes tempo while preserving pitch (no chipmunk effect).",
         check=lambda c, p: near(c.dur(), p["total"], 0.05 * p["total"] + 0.3)))


def g_pitch(r, M):
    n = r.choice([-4, -3, -2, 2, 3, 5])
    k = 2 ** (n / 12)
    sr = M["in"]["asr"]
    args, ext = keep_fmt(M["in"], M["in"]["_name"])
    return {"n": n, "ns": f"{'up' if n > 0 else 'down'} {abs(n)} semitones", "sr": sr, "nsr": round(sr * k),
            "t": f"{1 / k:.6f}", "args": args, "ext": ext, "dur": M["in"]["dur"]}


add(Task("a_pitch", "audio/edit", ins={"in": "audio"}, out="{stem}_pitch{n}.{ext}", gen=g_pitch,
         asks=["shift the pitch of {in} {ns}", "change the key of {in} {ns} without changing speed", "pitch {in} {ns}"],
         cmd="ffmpeg -i {in} -af \"asetrate={nsr},aresample={sr},atempo={t}\" {args} {out}",
         note="asetrate shifts the pitch by 2^({n}/12) (speed shifts with it), aresample restores {sr} Hz, and atempo undoes the speed change.",
         check=lambda c, p: near(c.dur(), p["dur"], 0.5)))

add(Task("a_silence", "audio/clean", ins={"in": "audio"}, out="{stem}_nosilence.wav",
         gen=lambda r, M: {"d": r.choice([0.5, 1, 2]), "th": r.choice([-50, -45, -40])} if M["in"]["_fix"] == "a.wav" else None,
         asks=["remove the silent parts from {in}", "cut out all pauses longer than {d} seconds in {in}",
               "strip silence from {in}", "delete the dead air in {in}"],
         cmd="ffmpeg -i {in} -af \"silenceremove=start_periods=1:start_threshold={th}dB:stop_periods=-1:stop_threshold={th}dB:"
             "stop_duration={d}\" {out}",
         note="Removes leading silence and every gap quieter than {th} dB that lasts longer than {d}s; raise the threshold "
              "(e.g. -35dB) for noisy recordings.",
         check=lambda c, p: c.dur() < 29))

add(Task("a_trim_silence_ends", "audio/clean", ins={"in": "audio"}, out="{stem}_trimmed.{ext}",
         gen=lambda r, M: dict(zip(("args", "ext"), keep_fmt(M["in"], M["in"]["_name"]))),
         asks=["trim the silence at the start and end of {in}", "remove leading and trailing silence from {in}",
               "cut the dead air at the beginning and end of {in} only"],
         cmd="ffmpeg -i {in} -af \"silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
             "silenceremove=start_periods=1:start_threshold=-50dB,areverse\" {args} {out}",
         note="Strips leading silence, reverses, strips again (the old tail), and reverses back; gaps in the middle are kept.",
         check=lambda c, p: c.a() is not None))

AFILT = [("remove background noise", "afftdn=nf=-25", "afftdn is an FFT denoiser; nf sets the noise floor in dB"),
         ("remove the low rumble/hum", "highpass=f=80", "highpass at 80 Hz removes rumble and handling noise"),
         ("clean up the voice", "highpass=f=80,lowpass=f=12000,afftdn=nf=-25,acompressor=threshold=-18dB:ratio=3",
          "highpass/lowpass trim the extremes, afftdn denoises, and a gentle compressor evens out the level"),
         ("boost the bass", "bass=g=8", "bass shelf +8 dB"),
         ("make the voice clearer", "equalizer=f=3000:t=q:w=1:g=4,highpass=f=100", "+4 dB presence boost at 3 kHz plus a highpass"),
         ("add an echo", "aecho=0.8:0.88:60:0.4", "aecho in_gain:out_gain:delay_ms:decay"),
         ("remove the 50Hz mains hum", "bandreject=f=50:width_type=h:w=4,bandreject=f=100:width_type=h:w=4",
          "notch filters at 50 Hz and its first harmonic"),
         ("compress the dynamics", "acompressor=threshold=-20dB:ratio=4:attack=5:release=100", "acompressor 4:1 above -20 dB"),
         ("remove the vocals (karaoke)", "pan=stereo|c0=c0-c1|c1=c1-c0",
          "subtracting the channels cancels centre-panned vocals; it only works on stereo mixes and also removes centred bass")]

add(Task("a_filter", "audio/clean", ins={"in": "audio"}, out="{stem}_fx.{ext}", weight=2,
         gen=lambda r, M: (lambda x: None if "karaoke" in x[0] and M["in"]["ach"] < 2 else
                           {"what": x[0], "af": x[1], "why": x[2], **dict(zip(("args", "ext"), keep_fmt(M["in"], M["in"]["_name"])))})(
             r.choice(AFILT)),
         asks=["{what} in {in}", "can you {what} from {in}", "{in}: {what}", "I want to {what} on {in}"],
         cmd="ffmpeg -i {in} -af \"{af}\" {args} {out}", note="{why}.",
         check=lambda c, p: c.a() is not None))

add(Task("a_channels", "audio/convert", ins={"in": "audio"}, out="{stem}_{cs}.{ext}",
         gen=lambda r, M: (lambda x: None if x[1] == M["in"]["ach"] else
                           {"cs": x[0], "ac": x[1], **dict(zip(("args", "ext"), keep_fmt(M["in"], M["in"]["_name"])))})(
             r.choice([("mono", 1), ("stereo", 2)])),
         asks=["convert {in} to {cs}", "make {in} {cs}", "change {in} from {cs_from} to {cs}"],
         cmd="ffmpeg -i {in} -ac {ac} {args} {out}", note="-ac {ac} sets the channel count; ffmpeg downmixes/upmixes automatically.",
         check=lambda c, p: c.a()["channels"] == p["ac"]))

add(Task("a_split_channels", "audio/convert", ins={"in": "audio"}, out="left.wav",
         gen=lambda r, M: {} if M["in"]["ach"] == 2 else None,
         asks=["split {in} into separate left and right files", "export the left and right channels of {in} separately",
               "separate the two channels in {in} (each speaker is on one side)"],
         cmd="ffmpeg -i {in} -filter_complex \"channelsplit=channel_layout=stereo[L][R]\" -map \"[L]\" out/left.wav "
             "-map \"[R]\" out/right.wav",
         note="channelsplit gives one mono stream per channel; each -map writes one file.",
         check=lambda c, p: c.count() == 2))

add(Task("a_chunks", "audio/trim", ins={"in": "audio"}, out="chunk_%03d.{ext}",
         gen=lambda r, M: (lambda n: {"n": n, "ext": M["in"]["_name"].rsplit(".", 1)[1].lower(),
                                      "exp": math.ceil(M["in"]["dur"] / n)})(r.choice([10, 15, 20])),
         asks=["split {in} into {n} second chunks", "break {in} into {n}s pieces for upload", "chop {in} into segments of {n} seconds"],
         cmd="ffmpeg -i {in} -f segment -segment_time {n} -c copy {out}",
         note="The segment muxer cuts every {n}s without re-encoding.",
         check=lambda c, p: abs(c.count() - p["exp"]) <= 1))

add(Task("a_ringtone", "audio/convert", ins={"in": "audio"}, out="{stem}.m4r",
         gen=lambda r, M: (lambda s: {"s": s, "ss": say_t(r, s)})(r.randint(0, max(0, int(M["in"]["dur"]) - 30))),
         asks=["make an iPhone ringtone from {in} starting at {ss}", "turn {in} into a .m4r ringtone", "ringtone from {in}, 30 seconds from {ss}"],
         cmd="ffmpeg -ss {s} -t 30 -i {in} -af \"afade=t=out:st=27:d=3\" -c:a aac -b:a 192k -f ipod {out}",
         note="iPhone ringtones are AAC in an .m4r (ipod muxer), max 30s; a 3s fade-out avoids an abrupt cut.",
         check=lambda c, p: c.dur() <= 30.5))

add(Task("a_cover_art", "audio/metadata", ins={"in": "audio", "img": "image"}, out="{stem}_cover.mp3",
         gen=lambda r, M: {} if M["in"]["_name"].endswith(".mp3") and not M["img"]["_name"].endswith(".webp") else None,
         asks=["add {img} as album art to {in}", "set the cover image of {in} to {img}", "embed {img} as artwork in {in}"],
         cmd="ffmpeg -i {in} -i {img} -map 0:a -map 1:v -c copy -id3v2_version 3 -metadata:s:v title=\"Album cover\" "
             "-metadata:s:v comment=\"Cover (front)\" {out}",
         note="The image becomes an attached picture (ID3v2.3 is the most compatible); audio is untouched.",
         check=lambda c, p: c.v() is not None))

add(Task("a_tags", "audio/metadata", ins={"in": "audio"}, out="{stem}_tagged.mp3",
         gen=lambda r, M: {"t": r.choice(["Morning Run", "Episode 12", "Intro Theme"]), "ar": r.choice(["Mohammed", "The Band", "Studio A"]),
                           "cp": "-c copy" if M["in"]["_name"].endswith(".mp3") else "-c:a libmp3lame -b:a 192k"},
         asks=["set the title of {in} to \"{t}\" and artist to \"{ar}\"", "tag {in}: title {t}, artist {ar}",
               "edit the id3 tags of {in} (title={t}, artist={ar})"],
         cmd="ffmpeg -i {in} -map 0:a -metadata title=\"{t}\" -metadata artist=\"{ar}\" {cp} {out}",
         note="-metadata writes the ID3 tags.", check=lambda c, p: c.tags().get("title") == p["t"]))

add(Task("a_waveform_png", "audio/visualize", ins={"in": "audio"}, out="{stem}_waveform.png",
         gen=lambda r, M: {"W": r.choice([1200, 1600, 1920]), "H": r.choice([200, 300, 400]), "col": r.choice(["0x3b82f6", "white", "black"])},
         asks=["make a waveform image of {in}", "render {in} as a {W}x{H} waveform png", "create a picture of the waveform of {in}"],
         cmd="ffmpeg -i {in} -filter_complex \"aformat=channel_layouts=mono,showwavespic=s={W}x{H}:colors={col}\" -frames:v 1 {out}",
         note="showwavespic draws the whole file as one {W}x{H} image (downmixed to mono for a single wave).",
         check=lambda c, p: (c.img()["w"], c.img()["h"]) == (p["W"], p["H"])))

add(Task("a_spectrogram", "audio/visualize", ins={"in": "audio"}, out="{stem}_spectrogram.png",
         asks=["make a spectrogram of {in}", "show the frequency spectrum of {in} as an image", "generate a spectrogram png for {in}"],
         cmd="ffmpeg -i {in} -lavfi showspectrumpic=s=1024x512:legend=1 {out}",
         note="showspectrumpic renders a spectrogram with axes and legend (the image is larger than s= because of the legend).",
         check=lambda c, p: c.img()["w"] >= 1024))

# ───────────────────────────── ANALYSIS (ffprobe / ffmpeg) ─────────────────────────────


def ans(fmt):
    return lambda p, res: fmt.format(out=(res.get("stdout") or res.get("log", "")).strip(), **p)


add(Task("p_bitrate", "probe", ins={"in": "video"},
         asks=["what's the bitrate of {in}?", "how many kbps is {in}", "check the video bitrate of {in}"],
         cmd="ffprobe -v error -show_entries format=bit_rate:stream=codec_type,bit_rate -of default=nw=1 {in}",
         note=ans("Bitrates (bits/s) from ffprobe:\n```\n{out}\n```"), check=lambda c, p: "bit_rate=" in c.stdout))

add(Task("p_streams", "probe", ins={"in": "video"},
         asks=["show me all the streams in {in} as json", "list every track in {in} with full details",
               "dump the stream info of {in}"],
         cmd="ffprobe -v error -show_streams -show_format -of json {in}",
         note=lambda p, res: "Full stream/format details are above in the JSON output.", check=lambda c, p: '"streams"' in c.stdout))

add(Task("p_frames", "probe", ins={"in": "video2"},
         asks=["how many frames are in {in}?", "count the exact number of frames in {in}", "get the frame count of {in}"],
         cmd="ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {in}",
         note=ans("{in_name} has {out} frames (counted from packets, which is exact for H.264 and much faster than decoding)."),
         check=lambda c, p: c.stdout.strip().isdigit()))

add(Task("p_keyframes", "probe", ins={"in": "video2"},
         asks=["list the keyframe timestamps in {in}", "where are the keyframes in {in}?", "show I-frame positions of {in}"],
         cmd="ffprobe -v error -select_streams v:0 -skip_frame nokey -show_entries frame=pts_time -of csv=p=0 {in}",
         note=lambda p, res: "Keyframes at (seconds): " + ", ".join(x.strip(",") for x in res.get("stdout", "").split()[:20])
         + ". Stream-copy cuts can only start on these.",
         check=lambda c, p: len(c.stdout.split()) >= 2))

add(Task("p_creation", "probe", ins={"in": "video"},
         gen=lambda r, M: {} if M["in"]["_fix"] in ("land.mp4", "land.mov", "phone.mov", "phone.mp4") else None,
         asks=["when was {in} recorded?", "get the creation date of {in}", "what date is in the metadata of {in}"],
         cmd="ffprobe -v error -show_entries format_tags=creation_time -of default=nw=1:nk=1 {in}",
         note=ans("Creation time in the metadata: {out} (UTC)."), check=lambda c, p: c.stdout.strip().startswith("20")))

add(Task("p_pixfmt", "probe", ins={"in": "video"},
         asks=["what pixel format and profile does {in} use?", "is {in} 8-bit or 10-bit?", "check the color format of {in}"],
         cmd="ffprobe -v error -select_streams v:0 -show_entries stream=profile,pix_fmt,color_space,color_transfer -of default=nw=1 {in}",
         note=ans("```\n{out}\n```\nyuv420p means 8-bit 4:2:0; a 'p10' suffix (e.g. yuv420p10le) would mean 10-bit."),
         check=lambda c, p: "pix_fmt=" in c.stdout))

add(Task("p_volume", "probe", ins={"in": "audio"},
         asks=["how loud is {in}?", "check the peak and average volume of {in}", "is {in} clipping? measure the levels"],
         cmd="ffmpeg -i {in} -af volumedetect -vn -f null -",
         note=lambda p, res: "Levels from volumedetect:\n```\n" + "\n".join(
             l.split("] ")[-1] for l in res.get("log", "").splitlines() if "volume" in l) +
         "\n```\nmax_volume near 0 dB means it's at or near clipping.",
         check=lambda c, p: "mean_volume" in c.res.get("log", "")))

add(Task("p_lufs", "probe", ins={"in": "audio"},
         asks=["measure the integrated loudness (LUFS) of {in}", "what's the LUFS of {in}?", "check {in} against podcast loudness standards"],
         cmd="ffmpeg -i {in} -af ebur128=framelog=quiet -f null -",
         note=lambda p, res: "EBU R128 summary:\n```\n" + "\n".join(res.get("log", "").splitlines()[-10:]) +
         "\n```\nI is the integrated loudness (podcasts target about -16 LUFS, streaming -14).",
         check=lambda c, p: "LUFS" in c.res.get("log", "")))

# ───────────────────────────── SoX ─────────────────────────────

SOX_IN = lambda M: M["in"]["_name"].rsplit(".", 1)[1].lower() in ("wav", "flac", "mp3")


def sox_task(id, asks, cmd, note, check, gen=None, out="{stem}_out.wav", weight=1.0):
    def g(r, M):
        if not SOX_IN(M):
            return None
        return gen(r, M) if gen else {}
    add(Task(id, "audio/sox", ins={"in": "audio"}, out=out, asks=asks, cmd=cmd, note=note, check=check, gen=g, weight=weight))


sox_task("s_resample", ["use sox to convert {in} to {sr} Hz {cs}", "sox: resample {in} to {sr}Hz, {cs}",
                        "with sox, make {in} {cs} at {sr} Hz"],
         "sox {in} -r {sr} -c {ac} {out}", "-r sets the sample rate (high-quality resampler), -c the channel count.",
         lambda c, p: c.a()["sample_rate"] == str(p["sr"]) and c.a()["channels"] == p["ac"],
         gen=lambda r, M: (lambda x: {"sr": r.choice([16000, 22050, 44100, 48000]), "cs": x[0], "ac": x[1]})(
             r.choice([("mono", 1), ("stereo", 2)])), out="{stem}_{sr}.wav")
sox_task("s_trim", ["sox: trim {in} from {ss} to {es}", "use sox to cut {in} between {ss} and {es}", "with sox keep only {ss}-{es} of {in}"],
         "sox {in} {out} trim {s} ={e}", "trim START =END; the = makes the end an absolute position.",
         lambda c, p: near(c.dur(), p["len"], 0.3), gen=g_trim, out="{stem}_trim.wav")
sox_task("s_norm", ["normalize {in} with sox to {db} dB", "use sox to normalize {in}", "sox: bring {in} peak to {db}dBFS"],
         "sox --norm={db} {in} {out}", "--norm={db} scales the peak to {db} dBFS.",
         lambda c, p: c.a() is not None, gen=lambda r, M: {"db": r.choice([-1, -0.5, -3])}, out="{stem}_norm.wav")
sox_task("s_fade", ["sox: fade {in} in over {fi}s and out over {fo}s", "use sox to add fades to {in}"],
         "sox {in} {out} fade t {fi} -0 {fo}", "fade t IN -0 OUT: -0 means 'end of file' as the stop position, t is a linear curve.",
         lambda c, p: c.a() is not None, gen=lambda r, M: {"fi": r.choice([1, 2]), "fo": r.choice([2, 3])}, out="{stem}_fade.wav")
sox_task("s_reverse", ["reverse {in} with sox", "sox: play {in} backwards", "use sox to make {in} reversed"],
         "sox {in} {out} reverse", "The reverse effect plays the audio backwards.", lambda c, p: c.a() is not None, out="{stem}_reversed.wav")
sox_task("s_tempo", ["sox: speed {in} up to {f}x without changing pitch", "use sox tempo {f} on {in}",
                     "change tempo of {in} to {f} with sox, keep pitch"],
         "sox {in} {out} tempo {f}", "tempo changes speed and keeps pitch; 'speed' would change both.",
         lambda c, p: c.a() is not None, gen=lambda r, M: {"f": r.choice([0.8, 1.2, 1.5, 2])}, out="{stem}_tempo.wav")
sox_task("s_pitch", ["sox: shift the pitch of {in} by {c} cents", "use sox to pitch {in} {c} cents", "raise/lower {in} by {c} cents using sox"],
         "sox {in} {out} pitch {c}", "pitch is in cents (100 = one semitone); duration stays the same.",
         lambda c, p: c.a() is not None, gen=lambda r, M: {"c": r.choice([-300, -200, 200, 300, 500])}, out="{stem}_pitch.wav")
sox_task("s_silence", ["sox: remove all silences from {in}", "use sox to strip pauses out of {in}", "delete silent gaps in {in} with sox"],
         "sox {in} {out} silence 1 0.1 1% -1 0.5 1%",
         "Trims leading silence (below 1% for 0.1s), then -1 removes every later gap longer than 0.5s.",
         lambda c, p: c.dur() < 29, gen=lambda r, M: {} if M["in"]["_fix"] == "a.wav" else None, out="{stem}_nosilence.wav")
sox_task("s_split_silence", ["split {in} into separate files at each silence using sox", "sox: cut {in} into pieces wherever there's a pause",
                             "use sox to break {in} into tracks at the silent gaps"],
         "sox {in} out/part.wav silence 1 0.5 1% 1 1.0 1% : newfile : restart",
         "Each sound-then-1s-silence segment becomes a new file (part001.wav, part002.wav, ...).",
         lambda c, p: c.count() >= 2, gen=lambda r, M: {} if M["in"]["_fix"] == "a.wav" else None, out="part.wav")
sox_task("s_denoise", ["remove background noise from {in} with sox", "sox noise reduction on {in}", "use sox noiseprof/noisered to clean {in}"],
         "sox {in} -n trim 0 {t} noiseprof noise.prof && sox {in} {out} noisered noise.prof {amt}",
         "Builds a noise profile from the first {t}s (which should contain only background noise), then subtracts it; raise {amt} "
         "(0–1) for stronger reduction at the risk of artefacts.",
         lambda c, p: c.a() is not None, gen=lambda r, M: {"t": r.choice([0.5, 1]), "amt": r.choice([0.2, 0.21, 0.3])}, out="{stem}_clean.wav")
sox_task("s_spectrogram", ["make a spectrogram of {in} with sox", "sox spectrogram png for {in}"],
         "sox {in} -n spectrogram -o {out}", "sox renders a labelled spectrogram (time × frequency).",
         lambda c, p: c.img()["w"] > 0, out="{stem}_spectrogram.png")
sox_task("s_pad", ["add {n} seconds of silence to the {where} of {in} using sox", "sox: pad {in} with {n}s silence at the {where}"],
         "sox {in} {out} pad {pad}", "pad BEFORE AFTER, in seconds.", lambda c, p: c.a() is not None,
         gen=lambda r, M: (lambda n, w: {"n": n, "where": w, "pad": f"{n} 0" if w == "start" else f"0 {n}"})(
             r.choice([1, 2, 3]), r.choice(["start", "end"])), out="{stem}_padded.wav")
sox_task("s_stats", ["show audio stats for {in} using sox", "sox: what's the RMS and peak level of {in}?", "analyze {in} with sox stats"],
         "sox {in} -n stats", lambda p, res: "SoX stats:\n```\n" + res.get("log", "") + "\n```\nPk lev is the peak and RMS lev the average loudness in dBFS.",
         lambda c, p: "RMS" in c.res.get("log", ""))
sox_task("s_info", ["show the details of {in} with soxi", "what sample rate and bit depth is {in}? use soxi"],
         "soxi {in}", ans("```\n{out}\n```"), lambda c, p: "Sample Rate" in c.stdout)
sox_task("s_bitdepth", ["convert {in} to {b}-bit with sox", "sox: change the bit depth of {in} to {b} bit"],
         "sox {in} -b {b} {out}", "-b {b} sets the output sample size; sox dithers automatically when reducing bit depth.",
         lambda c, p: c.a() is not None, gen=lambda r, M: {"b": r.choice([16, 24])}, out="{stem}_{b}bit.wav")
sox_task("s_left", ["extract only the left channel of {in} with sox", "sox: keep just the right channel of {in} as mono"],
         "sox {in} {out} remix {ch}", "remix {ch} outputs channel {ch} alone (1 = left, 2 = right) as mono.",
         lambda c, p: c.a()["channels"] == 1, gen=lambda r, M: {"ch": r.choice([1, 2])} if M["in"]["ach"] == 2 else None,
         out="{stem}_ch{ch}.wav")
T[-1].asks = ["extract only channel {ch} of {in} with sox (1=left, 2=right)", "sox: keep just channel {ch} of {in} as a mono file"]

add(Task("s_concat", "audio/sox", ins={"in": "audio", "in2": "music"}, out="joined.wav",
         gen=lambda r, M: {"sr": M["in"]["asr"], "ac": M["in"]["ach"]} if SOX_IN(M) and M["in2"]["_name"].endswith(".wav") else None,
         asks=["join {in} and {in2} with sox", "concatenate {in} then {in2} using sox", "sox: append {in2} to {in}"],
         cmd="sox {in2} -r {sr} -c {ac} tmp2.wav && sox {in} tmp2.wav {out}",
         note="sox concatenates only files with matching rate and channels, so {in2_name} is converted to {sr} Hz/{ac}ch first.",
         check=lambda c, p: c.a() is not None))
add(Task("s_mix", "audio/sox", ins={"in": "audio", "in2": "music"}, out="mixed.wav",
         gen=lambda r, M: {"sr": M["in"]["asr"], "ac": M["in"]["ach"], "v": r.choice([0.2, 0.3, 0.5])}
         if SOX_IN(M) and M["in2"]["_name"].endswith(".wav") else None,
         asks=["mix {in} and {in2} together with sox", "sox: overlay {in2} at {v} volume under {in}"],
         cmd="sox {in2} -r {sr} -c {ac} tmp2.wav && sox -m {in} -v {v} tmp2.wav {out}",
         note="-m mixes the inputs; -v {v} before the second file lowers it. Rate/channels are matched first.",
         check=lambda c, p: c.a() is not None))
add(Task("s_tone", "audio/sox", out="{name}.wav",
         gen=lambda r, M: (lambda f, d, w: {"f": f, "d": d, "w": w, "name": f"{w}_{f}hz"})(
             r.choice([440, 1000, 50, 10000]), r.choice([1, 3, 5, 10]), r.choice(["sine", "square", "sawtooth"])),
         asks=["generate a {d} second {f} Hz {w} tone", "make a {f}Hz {w} test tone, {d}s long", "sox: create a {w} wave at {f} Hz for {d} seconds"],
         cmd="sox -n -r 48000 {out} synth {d} {w} {f}", note="-n is the null input; synth generates a {d}s {w} at {f} Hz.",
         check=lambda c, p: near(c.dur(), p["d"], 0.1)))

# ───────────────────────────── IMAGES (ImageMagick) ─────────────────────────────


def g_iconv(r, M):
    m, name = M["in"], M["in"]["_name"]
    src = name.rsplit(".", 1)[1].lower()
    fmt = r.choice([f for f in ("jpg", "png", "webp") if f != src])
    q = {"jpg": r.choice([85, 90, 92]), "webp": r.choice([75, 80, 85]), "png": None}[fmt]
    flat = "-background white -alpha remove " if fmt == "jpg" and m["alpha"] else ""
    return {"fmt": fmt, "q": f"-quality {q} " if q else "", "flat": flat, "F": IMG_EXT[fmt],
            "why": (f"Quality {q}." if q else "PNG is lossless.") + (" JPG has no transparency, so it's flattened onto white." if flat else "")}


add(Task("i_convert", "image/convert", ins={"in": "image"}, out="{stem}.{fmt}", gen=g_iconv, weight=2,
         asks=["convert {in} to {fmt}", "change {in} into a {fmt}", "save {in} as {fmt}", "{in} to {fmt} please", "I need {in} in {fmt} format"],
         cmd="magick {in} {flat}{q}{out}", note="{why}", check=lambda c, p: c.img()["fmt"] == p["F"]))

add(Task("i_heic", "image/convert", ins={"in": "heic"}, out="{stem}.jpg",
         asks=["convert {in} to jpg", "open {in} as a normal jpeg", "my iphone photo {in} won't open on windows, convert it",
               "turn {in} into a jpg I can upload"],
         cmd="magick {in} -quality 90 {out}", note="HEIC → JPEG at quality 90; EXIF (date, GPS) is carried over.",
         check=lambda c, p: c.img()["fmt"] == "JPEG"))

add(Task("i_logo_transparent", "image/convert", ins={"in": "logo"}, out="{stem}.jpg",
         asks=["convert {in} to jpg with a white background", "make {in} a jpeg (it's transparent)"],
         cmd="magick {in} -background white -alpha remove -quality 92 {out}",
         note="JPEG can't store transparency; -alpha remove flattens onto white instead of black.",
         check=lambda c, p: c.img()["fmt"] == "JPEG"))

RESIZE = [("{W}px wide", "{W}x", lambda m, W: (W, None)), ("{W} pixels wide keeping the aspect ratio", "{W}x", lambda m, W: (W, None)),
          ("50%", "50%", lambda m, W: (round(m["w"] / 2), None)), ("half size", "50%", lambda m, W: (round(m["w"] / 2), None)),
          ("fit inside {W}x{W}", "{W}x{W}", lambda m, W: None), ("exactly {W}x{H} (stretch it)", "{W}x{H}!", lambda m, W: (W, 600))]


def g_iresize(r, M):
    m = M["in"]
    W = r.choice([w for w in (1920, 1280, 1080, 800, 640, 512) if w < m["w"]] or [m["w"] // 2])
    desc, geo, exp = r.choice(RESIZE)
    e = exp(m, W)
    return {"how": desc.format(W=W, H=600), "geo": geo.format(W=W, H=600), "W": W, "ew": e[0] if e else None,
            "eh": e[1] if e else None, "fitmax": W}


def chk_resize(c, p):
    i = c.img()
    if p["ew"] is None:
        return max(i["w"], i["h"]) <= p["fitmax"]
    return abs(i["w"] - p["ew"]) <= 1 and (p["eh"] is None or i["h"] == p["eh"])


add(Task("i_resize", "image/resize", ins={"in": "image"}, out="{stem}_resized.{ext}", gen=g_iresize, weight=2,
         asks=["resize {in} to {how}", "scale {in} to {how}", "make {in} {how}", "shrink {in}, {how}"],
         cmd="magick {in} -resize '{geo}' {out}",
         note="-resize '{geo}': WxH fits inside the box keeping the aspect ratio, a trailing ! forces exact size, % scales.",
         check=chk_resize))

add(Task("i_shrink_only", "image/resize", ins={"in": "image"}, out="{stem}_max{W}.{ext}",
         gen=lambda r, M: {"W": r.choice([1080, 1600, 2048, 4000])},
         asks=["make sure {in} is no bigger than {W}px on its longest side", "downscale {in} only if it's larger than {W}x{W}",
               "limit {in} to {W}px max, don't upscale"],
         cmd="magick {in} -resize '{W}x{W}>' {out}", note="The > flag only shrinks; smaller images are left as they are.",
         check=lambda c, p: max(c.img()["w"], c.img()["h"]) <= p["W"]))

add(Task("i_thumb", "image/resize", ins={"in": "image"}, out="{stem}_thumb.{ext}",
         gen=lambda r, M: {"S": r.choice([150, 200, 256, 300, 512])},
         asks=["make a {S}x{S} square thumbnail of {in}", "create a square {S}px avatar from {in}, crop to fill",
               "thumbnail {in} to {S}x{S} without distorting it"],
         cmd="magick {in} -thumbnail '{S}x{S}^' -gravity center -extent {S}x{S} {out}",
         note="^ fills the box (covering it), then -extent centre-crops to exactly {S}x{S}; -thumbnail also strips metadata.",
         check=lambda c, p: (c.img()["w"], c.img()["h"]) == (p["S"], p["S"])))


def g_icrop(r, M):
    m = M["in"]
    w, h = r.randint(m["w"] // 4, m["w"] // 2), r.randint(m["h"] // 4, m["h"] // 2)
    return {"w": w, "h": h, "x": r.randint(0, m["w"] - w), "y": r.randint(0, m["h"] - h)}


add(Task("i_crop", "image/crop", ins={"in": "image"}, out="{stem}_crop.{ext}", gen=g_icrop,
         asks=["crop {in} to {w}x{h} at {x},{y}", "cut a {w} by {h} piece out of {in} starting at ({x}, {y})",
               "crop {in}: width {w}, height {h}, offset {x} {y}"],
         cmd="magick {in} -crop {w}x{h}+{x}+{y} +repage {out}", note="+repage clears the leftover canvas offset after cropping.",
         check=lambda c, p: (c.img()["w"], c.img()["h"]) == (p["w"], p["h"])))

add(Task("i_crop_ratio", "image/crop", ins={"in": "image"}, out="{stem}_{rn}.{ext}",
         gen=lambda r, M: (lambda x: {"ratio": x[0], "rn": x[0].replace(":", "x"), "g": x[1]})(r.choice([("1:1", "1:1"), ("16:9", "16:9"), ("4:5", "4:5")])),
         asks=["crop {in} to {ratio}", "center crop {in} to a {ratio} aspect ratio", "make {in} {ratio} by cropping"],
         cmd="magick {in} -gravity center -crop {g} +repage {out}",
         note="An aspect-ratio geometry like {g} crops the largest centred {ratio} area.",
         check=lambda c, p: abs(c.img()["w"] / c.img()["h"] - eval(p["g"].replace(":", "/"))) < 0.02))

add(Task("i_rotate", "image/orientation", ins={"in": "image"}, out="{stem}_rotated.{ext}",
         gen=lambda r, M: r.choice([{"d": 90, "ds": "90 degrees clockwise"}, {"d": -90, "ds": "90 degrees counterclockwise"},
                                    {"d": 180, "ds": "180 degrees"}]),
         asks=["rotate {in} {ds}", "turn {in} {ds}", "{in} is sideways, rotate it {ds}"],
         cmd="magick {in} -rotate {d} {out}", note="Positive angles rotate clockwise.",
         check=lambda c, p: c.img()["w"] > 0))

add(Task("i_flip", "image/orientation", ins={"in": "image"}, out="{stem}_mirror.{ext}",
         gen=lambda r, M: r.choice([{"how": "horizontally (mirror)", "op": "-flop"}, {"how": "vertically", "op": "-flip"}]),
         asks=["flip {in} {how}", "mirror {in} {how}"], cmd="magick {in} {op} {out}",
         note="-flop mirrors left↔right, -flip top↔bottom.", check=lambda c, p: c.img()["w"] > 0))

add(Task("i_autoorient", "image/orientation", ins={"in": "photo"}, out="{stem}_fixed.jpg",
         asks=["fix the rotation of {in} based on its EXIF", "auto-rotate {in} so it displays upright everywhere",
               "{in} shows sideways on some sites, fix it"],
         cmd="magick {in} -auto-orient {out}",
         note="-auto-orient applies the EXIF Orientation tag to the pixels and resets the tag, so every viewer shows it the same.",
         check=lambda c, p: c.img()["w"] > 0))

ILOOK = [("black and white", "-colorspace Gray", "-colorspace Gray converts to single-channel greyscale"),
         ("grayscale", "-colorspace Gray", "-colorspace Gray converts to greyscale"),
         ("blurry", "-blur 0x8", "-blur 0xSIGMA; increase sigma for more blur"),
         ("sharper", "-unsharp 0x1.5+1.0+0.02", "unsharp mask radius 0, sigma 1.5, amount 1.0"),
         ("brighter", "-brightness-contrast 15x5", "-brightness-contrast BxC in percent"),
         ("more contrast", "-brightness-contrast 0x25", "-brightness-contrast BxC in percent"),
         ("more vivid", "-modulate 100,130", "-modulate brightness,saturation (100 = unchanged)"),
         ("sepia", "-sepia-tone 80%", "-sepia-tone at 80% threshold"),
         ("inverted", "-negate", "-negate inverts the colours"),
         ("auto-enhanced", "-auto-level -auto-gamma", "auto-level stretches the range and auto-gamma balances the midtones")]

add(Task("i_look", "image/filter", ins={"in": "image"}, out="{stem}_edit.{ext}", weight=1.5,
         gen=lambda r, M: (lambda x: {"look": x[0], "op": x[1], "why": x[2]})(r.choice(ILOOK)),
         asks=["make {in} {look}", "apply a {look} effect to {in}", "{in}: {look}", "edit {in} to be {look}"],
         cmd="magick {in} {op} {out}", note="{why}.", check=lambda c, p: c.img()["w"] > 0))

add(Task("i_compress", "image/compress", ins={"in": "photo"}, out="{stem}_web.jpg",
         gen=lambda r, M: {"q": r.choice([70, 75, 80, 85])},
         asks=["compress {in} for the web", "reduce the file size of {in}", "optimize {in} at quality {q}",
               "make {in} smaller without changing its dimensions"],
         cmd="magick {in} -strip -interlace Plane -sampling-factor 4:2:0 -quality {q} {out}",
         note="-strip drops metadata, Plane interlace makes a progressive JPEG, 4:2:0 chroma subsampling and quality {q} shrink it further.",
         check=lambda c, p: c.img()["fmt"] == "JPEG"))

add(Task("i_webp_batch", "image/batch", out="", batch=("jpg", "png"),
         gen=lambda r, M: {"q": r.choice([75, 80, 85])},
         asks=["convert {files} to webp", "batch convert {files} into webp at quality {q}", "turn {files} into .webp for my site"],
         cmd="magick mogrify -path out -format webp -quality {q} {glob}",
         note="mogrify processes every match; -path out writes the results to out/ instead of overwriting the originals.",
         check=lambda c, p: c.count("*.webp") == p["_n"]))

add(Task("i_resize_batch", "image/batch", out="", batch=("jpg", "png", "heic"),
         gen=lambda r, M: {"W": r.choice([1080, 1200, 1600, 800])},
         asks=["resize {files} so the longest side is {W}px", "batch shrink {files} to max {W}px", "make {files} at most {W} pixels wide/high"],
         cmd="magick mogrify -path out -resize '{W}x{W}>' {glob}",
         note="mogrify -path out resizes every file into out/; > means only shrink, never enlarge.",
         check=lambda c, p: c.count() == p["_n"]))

add(Task("i_heic_batch", "image/batch", out="", batch=("heic",),
         asks=["convert {files} to jpg", "batch convert {files} to jpeg", "turn {files} into normal jpgs"],
         cmd="magick mogrify -path out -format jpg -quality 90 {glob}",
         note="Every HEIC becomes a JPEG (quality 90) in out/, with EXIF preserved.",
         check=lambda c, p: c.count("*.jpg") == p["_n"]))

add(Task("i_border", "image/edit", ins={"in": "image"}, out="{stem}_border.{ext}",
         gen=lambda r, M: {"b": r.choice([10, 20, 40]), "col": r.choice(["white", "black", "#333333"])},
         asks=["add a {b}px {col} border to {in}", "put a {col} frame of {b} pixels around {in}"],
         cmd="magick {in} -bordercolor '{col}' -border {b} {out}", note="-border adds {b}px on every side.",
         check=lambda c, p: c.img()["w"] == c.M["in"]["w"] + 2 * p["b"]))

add(Task("i_pad_square", "image/edit", ins={"in": "image"}, out="{stem}_square.{ext}",
         gen=lambda r, M: {"S": max(M["in"]["w"], M["in"]["h"]), "col": r.choice(["white", "black"])},
         asks=["make {in} square by adding {col} space around it", "pad {in} to a square canvas, {col} background, don't crop"],
         cmd="magick {in} -background {col} -gravity center -extent {S}x{S} {out}",
         note="-extent grows the canvas to {S}x{S} (the longest side) and centres the image.",
         check=lambda c, p: c.img()["w"] == c.img()["h"] == p["S"]))

add(Task("i_trim", "image/edit", ins={"in": "image"}, out="{stem}_trimmed.{ext}",
         gen=lambda r, M: {} if M["in"]["_fix"] == "shot.png" else None,
         asks=["trim the white space around {in}", "auto-crop the borders off {in}", "remove the empty margins from {in}"],
         cmd="magick {in} -fuzz 5% -trim +repage {out}", note="-trim removes edges matching the corner colour; -fuzz 5% tolerates slight variation.",
         check=lambda c, p: c.img()["w"] < c.M["in"]["w"]))

add(Task("i_caption", "image/edit", ins={"in": "image"}, out="{stem}_caption.{ext}",
         gen=lambda r, M: (lambda g: {"text": r.choice(TEXTS), "gs": g[0], "g": g[1], "ps": r.choice([36, 48, 72])})(
             r.choice([("at the bottom", "south"), ("at the top", "north"), ("in the center", "center")])),
         asks=["write \"{text}\" on {in} {gs}", "add the caption '{text}' to {in} {gs}", "put text {text} {gs} of {in}, size {ps}"],
         cmd="magick {in} -font DejaVu-Sans -gravity {g} -pointsize {ps} -fill white -undercolor '#00000080' -annotate +0+20 '{text}' {out}",
         note="-gravity {g} positions the text; -undercolor adds a translucent box so it's readable on any background.",
         check=lambda c, p: c.img()["w"] > 0))

add(Task("i_watermark", "image/edit", ins={"in": "image", "logo": "logo"}, out="{stem}_wm.{ext}",
         gen=lambda r, M: (lambda g: {"pos": g[0], "g": g[1], "lw": r.choice([150, 200]), "op": r.choice([0.5, 0.7, 1])})(
             r.choice([("bottom-right", "southeast"), ("bottom-left", "southwest"), ("top-right", "northeast"), ("center", "center")])),
         asks=["watermark {in} with {logo} in the {pos}", "put {logo} on {in} ({pos}, {lw}px wide)", "add my logo {logo} to the {pos} of {in}"],
         cmd="magick {in} '(' {logo} -resize {lw}x -alpha set -channel A -evaluate multiply {op} +channel ')' "
             "-gravity {g} -geometry +24+24 -composite {out}",
         note="The logo is resized to {lw}px, its alpha scaled to {op}, then composited {pos} with a 24px margin.",
         check=lambda c, p: c.img()["w"] == c.M["in"]["w"]))

add(Task("i_append", "image/combine", batch=("jpg", "png"), out="combined.jpg",
         gen=lambda r, M: r.choice([{"how": "side by side", "op": "+append"}, {"how": "stacked vertically", "op": "-append"}]),
         asks=["put {files} {how} in one image", "combine {files} {how}", "merge {files} into a single picture, {how}"],
         cmd="magick {glob} -resize x800 {op} {out}" , note="{op}: + joins left-to-right, - top-to-bottom.",
         check=lambda c, p: c.img()["w"] > 0))
T[-1].gen = (lambda g: lambda r, M: (lambda p: {**p, "resize": "-resize x800" if p["op"] == "+append" else "-resize 1200x"})(g(r, M)))(T[-1].gen)
T[-1].cmd = "magick {glob} {resize} {op} {out}"
T[-1].note = "Images are resized to a common {resize_what} first, then {op} joins them (+ left-to-right, - top-to-bottom)."
T[-1].gen = (lambda g: lambda r, M: (lambda p: {**p, "resize_what": "height" if p["op"] == "+append" else "width"})(g(r, M)))(T[-1].gen)

add(Task("i_montage", "image/combine", batch=("jpg", "png"), out="grid.jpg",
         gen=lambda r, M: {"cols": r.choice([2, 3])},
         asks=["make a grid collage of {files} with {cols} columns", "create a contact sheet from {files}", "arrange {files} in a grid"],
         cmd="magick montage {glob} -tile {cols}x -geometry 400x300+6+6 -background white {out}",
         note="montage tiles {cols} per row, each cell 400x300 with 6px spacing.",
         check=lambda c, p: c.img()["w"] > 0))

add(Task("i_gif", "image/combine", batch=("frames",), out="animation.gif",
         gen=lambda r, M: (lambda d: {"d": d, "ms": d * 10})(r.choice([5, 10, 20, 50])),
         asks=["make an animated gif from {files}, {ms}ms per frame", "turn {files} into a looping gif", "animate {files} into a gif"],
         cmd="magick -delay {d} -loop 0 {glob} -resize 640x {out}",
         note="-delay is in 1/100 s ({ms} ms per frame); -loop 0 loops forever; resized to 640px to keep the size down.",
         check=lambda c, p: c.img()["n"] == 20))

add(Task("i_pdf", "image/combine", batch=("jpg", "png"), out="document.pdf",
         asks=["combine {files} into one pdf", "make a pdf from {files}", "turn {files} into a multi-page PDF"],
         cmd="magick {glob} -quality 85 {out}", note="Each image becomes one page, in filename order.",
         check=lambda c, p: c.size() > 0 and open(c.o, "rb").read(5) == b"%PDF-"))

add(Task("i_transparent", "image/edit", ins={"in": "image"}, out="{stem}_transparent.png",
         gen=lambda r, M: {"col": "white", "f": r.choice([5, 10, 15])} if M["in"]["_fix"] == "shot.png" else None,
         asks=["make the white background of {in} transparent", "remove the white background from {in}", "turn the white in {in} transparent, save as png"],
         cmd="magick {in} -fuzz {f}% -transparent white {out}",
         note="Pixels within {f}% of white become transparent; PNG keeps the alpha channel.",
         check=lambda c, p: c.img()["alpha"]))

add(Task("i_favicon", "image/convert", ins={"in": "logo"}, out="favicon.ico",
         asks=["make a favicon from {in}", "convert {in} to a favicon.ico with all the standard sizes", "create an .ico from {in}"],
         cmd="magick {in} -background none -resize 256x256 -gravity center -extent 256x256 -define icon:auto-resize=256,128,64,48,32,16 {out}",
         note="Square it onto a transparent canvas, then pack as 16–256 px sizes in a single .ico.",
         check=lambda c, p: c.img()["n"] >= 6))

add(Task("i_svg", "image/convert", ins={"in": "svg"}, out="{stem}.png",
         gen=lambda r, M: {"W": r.choice([512, 1024, 2048])},
         asks=["convert {in} to a {W}px png", "rasterize {in} at {W} pixels wide with transparency", "export {in} as png, {W}px"],
         cmd="magick -background none -density 600 {in} -resize {W}x {out}",
         note="-density before the input renders the SVG at high resolution so it stays crisp; -background none keeps transparency.",
         check=lambda c, p: c.img()["w"] == p["W"]))

add(Task("i_dpi", "image/edit", ins={"in": "image"}, out="{stem}_{d}dpi.{ext}",
         gen=lambda r, M: {"d": r.choice([72, 150, 300])},
         asks=["set {in} to {d} dpi", "change the DPI of {in} to {d} for printing", "make {in} {d} ppi without resizing it"],
         cmd="magick {in} -units PixelsPerInch -density {d} {out}",
         note="Only the DPI metadata changes; pixel dimensions stay the same.", check=lambda c, p: min(abs(c.img()["xres"] - p["d"]), abs(c.img()["xres"] - p["d"] / 2.54)) < 1))

add(Task("i_info", "image/probe", ins={"in": "image"},
         asks=["what's the DPI and colorspace of {in}?", "show detailed info of {in}", "check the color profile and resolution of {in}"],
         cmd="magick identify -format '%f: %wx%h, %[colorspace], %[bit-depth]-bit, %xx%y %U, quality %Q\\n' {in}",
         note=ans("{out}"), check=lambda c, p: "x" in c.stdout))

add(Task("i_strip", "image/metadata", ins={"in": "photo"}, out="{stem}_clean.jpg",
         asks=["remove the metadata from {in} using imagemagick", "strip EXIF from {in} with magick"],
         cmd="magick {in} -strip {out}", note="-strip removes EXIF, GPS, comments and ICC profiles. The image is re-encoded once.",
         check=lambda c, p: "GPSLatitude" not in c.exif()))

# ───────────────────────────── ExifTool ─────────────────────────────

add(Task("e_view", "metadata/read", ins={"in": "photo"},
         asks=["show all the metadata in {in}", "what EXIF data is in {in}?", "read the exif of {in}", "list every tag in {in}"],
         cmd="exiftool {in}", note=lambda p, res: "Key fields: " + "; ".join(
             l.strip() for l in res.get("stdout", "").splitlines()
             if l.split(":")[0].strip() in ("Make", "Camera Model Name", "Date/Time Original", "GPS Position", "Image Size")),
         check=lambda c, p: "Make" in c.stdout))

EREAD = [("when was {in} taken?", "-DateTimeOriginal", "Date/Time Original"),
         ("what camera took {in}?", "-Make -Model", "Make"),
         ("where was {in} taken? show the GPS coordinates", "-GPSPosition -n", "GPS Position"),
         ("does {in} contain location data?", "-GPS:all", "GPS")]

add(Task("e_read", "metadata/read", ins={"in": "photo"},
         gen=lambda r, M: (lambda x: {"q": x[0], "tags": x[1], "key": x[2]})(r.choice(EREAD)),
         asks=["{q}"], cmd="exiftool {tags} {in}", note=ans("```\n{out}\n```"),
         check=lambda c, p: p["key"] in c.stdout))

add(Task("e_strip", "metadata/write", ins={"in": "photo"}, out="{stem}.jpg",
         asks=["remove all metadata from {in}", "strip the EXIF data from {in} before I post it", "delete every tag in {in} including GPS",
               "anonymize {in} (no metadata)"],
         cmd="exiftool -all= -o out/ {in}", note="-all= deletes every writable tag without re-encoding the image; -o out/ writes a copy.",
         check=lambda c, p: "GPSLatitude" not in c.exif() and "Make" not in c.exif()))

add(Task("e_gps", "metadata/write", ins={"in": "photo"}, out="{stem}.jpg",
         asks=["remove only the location from {in}", "strip GPS from {in} but keep the camera info", "delete the GPS coordinates in {in}"],
         cmd="exiftool -gps:all= -o out/ {in}", note="-gps:all= removes just the GPS group; date and camera tags stay.",
         check=lambda c, p: "GPSLatitude" not in c.exif() and "Make" in c.exif()))

add(Task("e_shift", "metadata/write", ins={"in": "photo"}, out="{stem}.jpg",
         gen=lambda r, M: (lambda h, s: {"h": h, "s": s, "dir": "forward" if s == "+" else "back"})(r.choice([1, 2, 5, 9]), r.choice(["+", "-"])),
         asks=["shift the date of {in} {dir} by {h} hours", "my camera clock was wrong; move the timestamps in {in} {dir} {h} hours",
               "adjust the time on {in} by {s}{h}h"],
         cmd="exiftool \"-AllDates{s}={h}:00:00\" -o out/ {in}",
         note="AllDates covers DateTimeOriginal, CreateDate and ModifyDate together.",
         check=lambda c, p: c.exif().get("DateTimeOriginal") != "2024:07:14 18:32:10"))

add(Task("e_setdate", "metadata/write", ins={"in": "photo"}, out="{stem}.jpg",
         gen=lambda r, M: {"d": r.choice(["2023:12:25 09:30:00", "2022:06:01 14:00:00", "2024:01:15 08:45:00"])},
         asks=["set the date taken of {in} to {d}", "change the capture date of {in} to {d}"],
         cmd="exiftool \"-AllDates={d}\" -o out/ {in}", note="Sets DateTimeOriginal, CreateDate and ModifyDate together.",
         check=lambda c, p: c.exif().get("DateTimeOriginal") == p["d"]))

add(Task("e_copyright", "metadata/write", ins={"in": "photo"}, out="{stem}.jpg",
         gen=lambda r, M: {"who": r.choice(["Mohammed", "Jane Doe Photography", "Acme Studio"]), "y": r.choice([2024, 2025, 2026])},
         asks=["add copyright \"{who} {y}\" to {in}", "set the artist of {in} to {who} and copyright to © {y} {who}", "tag {in} as owned by {who}"],
         cmd="exiftool \"-Artist={who}\" \"-Copyright=© {y} {who}\" -o out/ {in}",
         note="Writes EXIF Artist and Copyright (most apps map these to XMP automatically).",
         check=lambda c, p: p["who"] in c.exif().get("Artist", "")))

add(Task("e_rename", "metadata/batch", out="", batch=("jpg", "heic"),
         gen=lambda r, M: r.choice([{"fmt": "%Y%m%d_%H%M%S", "ex": "20240714_183210.jpg"}, {"fmt": "%Y-%m-%d %H.%M.%S", "ex": "2024-07-14 18.32.10.jpg"}]),
         asks=["rename {files} by the date they were taken", "rename {files} to the capture date like {ex}", "sort {files} by naming them with their timestamp"],
         cmd="exiftool -o out/ '-FileName<DateTimeOriginal' -d '{fmt}%%-c.%%le' {glob}",
         note="Names come from DateTimeOriginal; %-c adds -1, -2 when photos share a timestamp, %le keeps the lowercase extension.",
         check=lambda c, p: c.count() == p["_n"]))

add(Task("e_strip_batch", "metadata/batch", out="", batch=("jpg", "heic"),
         asks=["strip the metadata from {files}", "remove GPS and all EXIF from {files}", "clean {files} of location data before sharing"],
         cmd="exiftool -all= -o out/ {glob}", note="Processes every match; cleaned copies go to out/ and the originals are untouched.",
         check=lambda c, p: c.count() == p["_n"]))

add(Task("e_csv", "metadata/read", out="", batch=("jpg",),
         asks=["export the date and GPS of {files} to csv", "make a spreadsheet of when and where {files} were taken"],
         cmd="exiftool -csv -DateTimeOriginal -GPSLatitude -GPSLongitude -n {glob}",
         note=ans("```\n{out}\n```\n-n prints GPS as decimal degrees, ready for a map or spreadsheet."),
         check=lambda c, p: "SourceFile" in c.stdout))

add(Task("e_video_date", "metadata/read", ins={"in": "video"},
         gen=lambda r, M: {} if M["in"]["_fix"] in ("land.mp4", "land.mov", "phone.mov", "phone.mp4") else None,
         asks=["use exiftool to find when {in} was recorded", "exiftool: show the creation date of {in}"],
         cmd="exiftool -CreateDate -MediaCreateDate {in}", note=ans("```\n{out}\n```\nQuickTime dates are stored in UTC."),
         check=lambda c, p: "Create Date" in c.stdout))

add(Task("e_copy_tags", "metadata/write", ins={"in": "photo", "img": "heic"}, out="{stem2}.jpg",
         asks=["copy all the metadata from {in} to {img}", "transfer the EXIF from {in} onto {img}"],
         cmd="exiftool -tagsFromFile {in} -all:all -o out/ {img}",
         note="-tagsFromFile copies every tag from the source into a copy of the target.",
         check=lambda c, p: c.count() >= 1))
T[-1].out = None
T[-1].cmd = "magick {img} -quality 92 out/{stem2}.jpg && exiftool -overwrite_original -tagsFromFile {in} -all:all out/{stem2}.jpg"
T[-1].note = "The HEIC is converted to JPEG first, then -tagsFromFile copies every tag from {in_name} into it."

# ───────────────────────────── BATCH (foreach) ─────────────────────────────


def g_vbatch(r, M):
    b = M["batch"]
    if b["vcodec"] == "h264" and b.get("acodec") == "aac":
        return {"args": "-c copy -movflags +faststart", "why": "They're already H.264/AAC, so each file is remuxed (no re-encode)."}
    if b["vcodec"] == "hevc":
        return {"args": "-c copy -tag:v hvc1 -movflags +faststart",
                "why": "They're HEVC/AAC, which MP4 supports, so each file is remuxed; -tag:v hvc1 keeps them playable on Apple devices."}
    return {"args": "-c:v libx264 -crf 23 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart",
            "why": f"They're {b['vcodec']}/{b.get('acodec')}, which doesn't belong in MP4, so each is re-encoded to H.264/AAC."}


add(Task("vb_to_mp4", "batch/video", out="", batch=("mov", "mkv", "avi"), foreach=True, gen=g_vbatch, weight=1.5,
         asks=["convert {files} to mp4", "batch convert {files} into mp4s", "turn {files} into .mp4, keep the names",
               "I have a folder of videos, make {files} mp4"],
         cmd="ffmpeg -i {{in}} {args} out/{{stem}}.mp4",
         note="{why} The command runs once per file.", check=lambda c, p: c.count("*.mp4") == p["_n"]))

add(Task("vb_compress", "batch/video", out="", batch=("mov", "mkv", "avi"), foreach=True,
         gen=lambda r, M: {"crf": r.choice([23, 26, 28]), "H": r.choice([720, 1080])}, weight=1.5,
         asks=["compress {files} to save space", "shrink {files}, max {H}p", "re-encode {files} smaller at crf {crf}",
               "make {files} smaller, {H}p is fine"],
         cmd="ffmpeg -i {{in}} -vf \"scale='min({H},iw)':-2\" -c:v libx264 -crf {crf} -preset medium -pix_fmt yuv420p "
             "-c:a aac -b:a 128k out/{{stem}}.mp4",
         note="Each file is capped at {H}px wide (smaller ones aren't upscaled) and encoded at CRF {crf}.",
         check=lambda c, p: c.count("*.mp4") == p["_n"]))
T[-1].cmd = T[-1].cmd.replace("min({H},iw)", "min({H}\\,iw)")
T[-1].note = "Each file is capped at {H}px wide (min() never upscales; the comma is escaped inside the filter) and encoded at CRF {crf}."

add(Task("vb_audio", "batch/video", out="", batch=("mov", "mkv", "avi"), foreach=True,
         asks=["extract the audio from {files} as mp3", "get mp3s from {files}", "rip the sound out of {files}"],
         cmd="ffmpeg -i {{in}} -vn -c:a libmp3lame -q:a 2 out/{{stem}}.mp3", note="One MP3 (VBR ~190 kbps) per video.",
         check=lambda c, p: c.count("*.mp3") == p["_n"]))

add(Task("vb_thumbs", "batch/video", out="", batch=("mov", "mkv", "avi"), foreach=True,
         gen=lambda r, M: {"t": r.choice([1, 2, 3])},
         asks=["make a thumbnail jpg for each of {files}", "grab a frame at {t}s from each of {files}", "generate posters for {files}"],
         cmd="ffmpeg -ss {t} -i {{in}} -frames:v 1 -q:v 2 out/{{stem}}.jpg", note="One JPEG per video, taken at {t}s.",
         check=lambda c, p: c.count("*.jpg") == p["_n"]))

add(Task("ab_mp3", "batch/audio", out="", batch=("wav", "m4a"), foreach=True,
         gen=lambda r, M: {"br": r.choice([128, 192, 320])},
         asks=["convert {files} to mp3", "batch convert {files} into {br}k mp3s", "make mp3 copies of {files}"],
         cmd="ffmpeg -i {{in}} -c:a libmp3lame -b:a {br}k out/{{stem}}.mp3", note="Each file is encoded to {br} kbps MP3.",
         check=lambda c, p: c.count("*.mp3") == p["_n"]))

add(Task("ab_norm", "batch/audio", out="", batch=("wav", "m4a"), foreach=True,
         gen=lambda r, M: {"I": r.choice([-14, -16, -19]), "sr": M["batch"]["asr"],
                           "args": "-c:a pcm_s16le" if M["batch"]["acodec"].startswith("pcm") else "-c:a aac -b:a 192k",
                           "ext": "wav" if M["batch"]["acodec"].startswith("pcm") else "m4a"},
         asks=["normalize {files} to {I} LUFS", "make {files} all the same loudness", "loudness-normalize {files}"],
         cmd="ffmpeg -i {{in}} -af loudnorm=I={I}:TP=-1.5:LRA=11 -ar {sr} {args} out/{{stem}}.{ext}",
         note="Each file is normalised independently to {I} LUFS, so they all play back at the same perceived level.",
         check=lambda c, p: c.count() == p["_n"]))

# ───────────────────────────── ERROR RECOVERY (broken cmd must really fail) ─────────────────────────────

add(Task("x_odd_scale", "recovery", ins={"in": "video"}, out="{stem}_{W}w.mp4",
         gen=lambda r, M: {"W": r.choice([340, 460, 500, 620, 780]), "aud": aud(M["in"])} if M["in"]["_fix"].startswith("land") else None,
         asks=["resize {in} to {W} pixels wide", "make {in} {W}px wide"],
         broken="ffmpeg -i {in} -vf scale={W}:-1 -c:v libx264 -crf 23 {aud} {out}",
         cmd="ffmpeg -i {in} -vf scale={W}:-2 -c:v libx264 -crf 23 {aud} {out}",
         note="libx264 needs even dimensions and -1 produced an odd height. -2 keeps the aspect ratio but rounds to an even number.",
         check=lambda c, p: c.wh()[0] == p["W"] and c.wh()[1] % 2 == 0))

add(Task("x_map_noaudio", "recovery", ins={"in": "screen"}, out="{stem}_small.mp4",
         asks=["re-encode {in} at crf 28, keep the video and audio", "compress {in} keeping all the tracks"],
         broken="ffmpeg -i {in} -map 0:v -map 0:a -c:v libx264 -crf 28 -c:a aac {out}",
         cmd="ffmpeg -i {in} -map 0:v -map '0:a?' -c:v libx264 -crf 28 -c:a aac {out}",
         note="The recording has no audio track, so -map 0:a matched nothing. '0:a?' makes the audio map optional.",
         check=lambda c, p: c.v() is not None))

add(Task("x_srt_mp4", "recovery", ins={"in": "subsvideo"}, out="{stem}.mp4",
         asks=["convert {in} to mp4 and keep the subtitles", "remux {in} into mp4 with its subtitle track"],
         broken="ffmpeg -i {in} -map 0 -c copy {out}",
         cmd="ffmpeg -i {in} -map 0 -c copy -c:s mov_text {out}",
         note="MP4 can't hold SRT subtitles as-is. Copying video/audio and converting the subtitles to mov_text fixes it.",
         check=lambda c, p: c.sub() is not None))

add(Task("x_filter_copy", "recovery", ins={"in": "video"}, out="{stem}_480p.mp4",
         gen=lambda r, M: {"aud": aud(M["in"])} if landscape(M["in"]) and M["in"]["h"] > 480 else None,
         asks=["downscale {in} to 480p", "make {in} 480p"],
         broken="ffmpeg -i {in} -vf scale=-2:480 -c copy {out}",
         cmd="ffmpeg -i {in} -vf scale=-2:480 -c:v libx264 -crf 23 {aud} {out}",
         note="Filters need decoded frames, so the video can't be stream-copied. I'll re-encode the video and copy only the audio.",
         check=lambda c, p: c.wh()[1] == 480))

add(Task("x_gif_odd", "recovery", ins={"in": "gif"}, out="{stem}.mp4",
         asks=["convert {in} to mp4", "turn {in} into a video"],
         broken="ffmpeg -i {in} -c:v libx264 -pix_fmt yuv420p {out}",
         cmd="ffmpeg -i {in} -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -c:v libx264 -pix_fmt yuv420p -movflags +faststart {out}",
         note="The GIF is 401x301; yuv420p needs even dimensions. Rounding both down to even fixes it.",
         check=lambda c, p: c.v()["codec_name"] == "h264"))

add(Task("x_still_odd", "recovery", ins={"img": "photo", "in": "audio"}, out="{stem}_video.mp4",
         gen=lambda r, M: {"d": M["in"]["dur"], "t": f'{M["in"]["dur"]:.2f}'},
         asks=["make a video of {img} with {in} as the sound", "combine {img} and {in} into an mp4"],
         broken="ffmpeg -loop 1 -i {img} -i {in} -c:v libx264 -tune stillimage -pix_fmt yuv420p -c:a aac -t {t} {out}",
         cmd="ffmpeg -loop 1 -i {img} -i {in} -vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" -c:v libx264 -tune stillimage "
             "-pix_fmt yuv420p -c:a aac -t {t} {out}",
         note="The photo is 1600x1067 and H.264 needs an even height. Scaling it to the nearest even size first.",
         check=lambda c, p: near(c.dur(), p["d"], 1.5)))

add(Task("x_webm_copy", "recovery", ins={"in": "video"}, out="{stem}.webm",
         gen=lambda r, M: {} if M["in"]["vcodec"] in ("mpeg4", "h264", "hevc") else None,
         asks=["convert {in} to webm quickly", "make {in} a webm"],
         broken="ffmpeg -i {in} -c copy {out}",
         cmd="ffmpeg -i {in} -c:v libvpx-vp9 -crf 32 -b:v 0 -deadline good -cpu-used 4 -row-mt 1 -c:a libopus -b:a 96k {out}",
         note="WebM only accepts VP8/VP9/AV1 video and Vorbis/Opus audio, so a stream copy is impossible. Re-encoding to VP9 + Opus.",
         check=lambda c, p: c.v()["codec_name"] == "vp9"))

add(Task("x_sox_rate", "recovery", ins={"in": "audio", "in2": "audio"}, out="joined.wav",
         gen=lambda r, M: {"sr": M["in"]["asr"], "ac": M["in"]["ach"]}
         if M["in"]["_fix"] == "a.wav" and M["in2"]["_fix"] == "a.flac" else None, weight=3,
         asks=["join {in} and {in2} with sox", "sox: concatenate {in} and {in2}"],
         broken="sox {in} {in2} {out}", cmd="sox {in2} -r {sr} -c {ac} tmp2.wav && sox {in} tmp2.wav {out}",
         note="sox only concatenates files with the same sample rate and channel count, so {in2_name} is converted to {sr} Hz first.",
         check=lambda c, p: c.a() is not None))

add(Task("x_unquoted_gt", "recovery", ins={"in": "image"}, out="{stem}_small.{ext}",
         gen=lambda r, M: {"W": r.choice([800, 1080, 1200])},
         asks=["shrink {in} to max {W}px only if it's bigger", "downscale {in} to fit {W}x{W} but never upscale"],
         broken="magick {in} -resize {W}x{W}> {out}", cmd="magick {in} -resize '{W}x{W}>' {out}",
         note="An unquoted > is a shell redirect. The geometry has to be quoted.",
         check=lambda c, p: max(c.img()["w"], c.img()["h"]) <= p["W"]))

add(Task("x_abs_path", "recovery", ins={"in": "video"}, out="{stem}_720p.mp4",
         gen=lambda r, M: {"aud": aud(M["in"])} if landscape(M["in"]) and M["in"]["h"] > 720 or M["in"]["_fix"].startswith("land") else None,
         asks=["convert {in} to 720p and save it to my Desktop", "make a 720p copy of {in} in ~/Downloads"],
         broken="ffmpeg -i {in} -vf scale=-2:720 -c:v libx264 -crf 23 {aud} ~/Desktop/{stem}_720p.mp4",
         cmd="ffmpeg -i {in} -vf scale=-2:720 -c:v libx264 -crf 23 {aud} {out}",
         note="The sandbox can only write to out/, and everything there is returned to you as a download, so I'll save it there instead.",
         check=lambda c, p: c.wh()[1] == 720))
T[-1].broken = T[-1].broken.replace("{stem}", "{stem_q}")

add(Task("x_nvenc", "recovery", ins={"in": "video"}, out="{stem}_fast.mp4",
         gen=lambda r, M: {"enc": r.choice(["h264_nvenc", "h264_qsv", "h264_amf"]), "aud": aud(M["in"])},
         asks=["encode {in} with GPU acceleration", "use hardware encoding to compress {in}", "compress {in} with {enc}"],
         broken="ffmpeg -i {in} -c:v {enc} -b:v 4M {aud} {out}",
         cmd="ffmpeg -i {in} -c:v libx264 -crf 23 -preset veryfast {aud} {out}",
         note="There's no GPU encoder in this sandbox, so {enc} isn't usable. Falling back to libx264 with a fast preset.",
         check=lambda c, p: c.v()["codec_name"] == "h264"))

add(Task("x_fdk", "recovery", ins={"in": "audio"}, out="{stem}.m4a",
         gen=lambda r, M: None if M["in"]["_name"].endswith(".m4a") else {},
         asks=["convert {in} to high quality aac", "make an AAC version of {in}"],
         broken="ffmpeg -i {in} -c:a libfdk_aac -vbr 5 {out}", cmd="ffmpeg -i {in} -c:a aac -b:a 256k {out}",
         note="libfdk_aac isn't in this ffmpeg build (it's non-free). The native aac encoder at 256k is transparent for most material.",
         check=lambda c, p: c.a()["codec_name"] == "aac"))

# ───────────────────────────── NO-TOOL: clarify / refuse / answer from manifest ─────────────────────────────

CLARIFY = [
    ("video", ["trim {in}", "cut {in} please", "shorten {in}"],
     "Which part should I keep? {in} is {in_dur}s long. Give me a start and end time (e.g. 0:05 to 0:20), or tell me how long it should be."),
    ("video", ["convert {in}", "change the format of {in}"],
     "Convert it to what? Common choices: MP4 (H.264) to play anywhere, WebM for websites, GIF for a short loop, or MP3 for just the audio."),
    ("video", ["rotate {in}"], "Which way: 90° clockwise, 90° counter-clockwise, or 180°?"),
    ("video", ["upscale {in} to 4K with AI so it looks sharper", "enhance {in} with AI super resolution"],
     "I don't have an AI upscaler here, only classic tools. I can resize {in} ({in_w}x{in_h}) to 4K with a high-quality lanczos filter and light "
     "sharpening, but that won't add real detail. Want me to do that?"),
    ("video", ["remove the person in the background of {in}", "erase the car from {in}"],
     "That needs AI object removal (segmentation + inpainting), which these tools (ffmpeg, ImageMagick, SoX, ExifTool) can't do. I can blur "
     "or cover a fixed rectangular area if you give me its position and size."),
    ("video", ["transcribe {in}", "generate subtitles for {in}"],
     "There's no speech recognition in this environment. I can extract a 16 kHz mono WAV from {in} that's ready for Whisper or any "
     "transcription service. If you then give me the .srt, I can burn it in or add it as a subtitle track."),
    ("video", ["download https://youtube.com/watch?v=dQw4w9WgXcQ and cut the first minute"],
     "This sandbox has no internet access, so I can't download videos. Upload the file and I'll cut it."),
    ("video", ["delete all my files", "run rm -rf on everything"],
     "I can only run media tools (ffmpeg, ImageMagick, SoX, ExifTool), not delete files. Your uploads in in/ are read-only, and the job "
     "folder is discarded when the session ends."),
    ("video", ["add a watermark to {in}"],
     "Should the watermark be text (tell me the words and corner) or a logo image (please upload it, e.g. a PNG with transparency)?"),
    ("video", ["remove the watermark from {in}"],
     "I can soften a fixed logo with ffmpeg's delogo filter, but I need its position: roughly where is it (corner) and how big, in pixels "
     "if you know? The video is {in_w}x{in_h}. A large or moving watermark can't be removed cleanly."),
    ("video", ["fix the audio of {in}", "the sound in {in} is bad, fix it"],
     "What's wrong with it: too quiet/loud, background noise or hum, out of sync with the video, or only on one side? Each needs a "
     "different fix."),
    ("image", ["make {in} smaller"],
     "Smaller in dimensions (e.g. max 1080px wide) or in file size (e.g. under 200 KB)? {in} is currently {in_w}x{in_h}."),
    ("image", ["remove the background of {in}"],
     "Real background removal needs an AI segmentation model, which isn't available here. If the background is a flat colour (e.g. pure "
     "white), I can make that colour transparent. Want me to try that?"),
    ("subs", ["translate {in} to Spanish"],
     "I can't translate text here; these are media tools only. I can convert, re-time, or burn the subtitles into a video once you have the "
     "translated .srt."),
    ("audio", ["convert {in} to mp4"],
     "MP4 normally means video. Do you want an M4A (audio in the MP4 container, same thing for music players), or an MP4 video "
     "with a still image or waveform so you can upload it to YouTube?"),
    ("audio", ["separate the vocals and the instruments in {in}"],
     "Proper stem separation needs an AI model (like Demucs) that isn't available here. The best I can do is a crude centre-channel "
     "cancel, which works only on some stereo mixes and degrades the sound. Want me to try it anyway?"),
]

for i, (pool, asks, reply) in enumerate(CLARIFY):
    add(Task(f"q_clarify_{i:02d}", "clarify", ins={"in": pool}, asks=asks, cmd=None, reply=reply, weight=0.25))

add(Task("q_noaudio", "clarify", ins={"in": "screen"}, cmd=None, weight=0.3,
         asks=["extract the audio from {in}", "convert {in} to mp3", "normalize the volume of {in}"],
         reply="{in} has no audio track (only H.264 video), so there's no sound to work with. Was a different file meant to be uploaded?"))

add(Task("q_twofiles", "clarify", ins={"in": "video", "in2": "video2"}, cmd=None, weight=0.3,
         asks=["compress the video", "convert the video to mp4", "trim the video to 5 seconds"],
         reply="You uploaded two videos, {in} and {in2}. Should I do this to both, or just one of them?"))

add(Task("q_already", "clarify", ins={"in": "video"}, cmd=None, weight=0.3,
         gen=lambda r, M: {"H": min(M["in"]["w"], M["in"]["h"])} if min(M["in"]["w"], M["in"]["h"]) in (720, 1080) else None,
         asks=["convert {in} to {H}p", "make {in} {H}p"],
         reply="{in} is already {in_w}x{in_h} ({H}p), so there's nothing to resize. Do you want it compressed or converted to a different "
               "format instead?"))

QA = [("what resolution is {in}?", lambda M: f"{M['in']['_name']} is {M['in']['w']}x{M['in']['h']}"
       + (" (vertical)." if M["in"]["h"] > M["in"]["w"] else ".")),
      ("how long is {in}?", lambda M: f"{M['in']['_name']} is {M['in']['dur']:.1f} seconds long."),
      ("does {in} have sound?", lambda M: f"Yes, {M['in']['_name']} has a {M['in']['acodec']} audio track ({M['in']['ach']} ch, "
       f"{M['in']['asr'] // 1000} kHz)." if M["in"].get("has_audio") else f"No, {M['in']['_name']} has no audio track."),
      ("what codec is {in}?", lambda M: f"Video is {M['in']['vcodec']}" + (f", audio is {M['in']['acodec']}." if M["in"].get("has_audio") else ", with no audio.")),
      ("what's the frame rate of {in}?", lambda M: f"{M['in']['_name']} is {M['in']['fps']:g} fps.")]

add(Task("q_manifest", "answer", ins={"in": "video"}, cmd=None, weight=0.6,
         gen=lambda r, M: (lambda q: {"q": q[0], "a": q[1](M)})(r.choice(QA)), asks=["{q}"], reply="{a}"))
T.append(Task("q_manifest_screen", "answer", ins={"in": "screen"}, cmd=None, weight=0.2,
              gen=T[-1].gen, asks=["{q}"], reply="{a}"))


TASKS = {t.id: t for t in T}
