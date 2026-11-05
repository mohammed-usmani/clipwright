"""Builds the fixture media library once (fixtures/) and records per-file metadata (fixtures/meta.json)."""
import json
import os
import shutil
import subprocess
import sys

from sandbox import probe

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixtures")


def clone(src, dst):
    """Copy a file. On macOS it's an APFS clone (shares storage until either file changes); elsewhere a normal copy."""
    if sys.platform == "darwin" and subprocess.run(["cp", "-c", src, dst], capture_output=True).returncode == 0:
        return dst
    return shutil.copyfile(src, dst)

SRT = """1
00:00:01,000 --> 00:00:04,000
Welcome back to the channel.

2
00:00:05,000 --> 00:00:09,000
Today we're testing a few edits.

3
00:00:11,000 --> 00:00:15,000
Let's get started.

4
00:00:18,000 --> 00:00:24,000
Thanks for watching!
"""

V = "-c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p -g 60"
A = "-c:a aac -b:a 128k"
SRC = "-f lavfi -i testsrc2=s={s}:r={r}:d={d} -f lavfi -i sine=f={f}:d={d}:sample_rate=48000"

RECIPES = {
    # video
    "land.mp4": f"ffmpeg {SRC.format(s='1280x720', r=30, d=40, f=440)} -ac 2 {V} {A} -metadata title='Summer 2024' -metadata creation_time=2024-07-14T18:32:10Z -shortest {{o}}",
    "land.mov": "ffmpeg -i land.mp4 -c copy -map_metadata 0 {o}",
    "land.mkv": "ffmpeg -i land.mp4 -c copy {o}",
    "phone.mov": f"ffmpeg {SRC.format(s='1080x1920', r=30, d=15, f=520)} -ac 2 -c:v libx265 -preset ultrafast -crf 26 -tag:v hvc1 -pix_fmt yuv420p {A} -metadata creation_time=2024-03-02T09:15:00Z -shortest {{o}}",
    "phone.mp4": "ffmpeg -i phone.mov -c copy {o}",
    "small.mp4": f"ffmpeg {SRC.format(s='640x360', r=24, d=75, f=300)} -ac 2 {V} {A} -shortest {{o}}",
    "screen.mov": f"ffmpeg -f lavfi -i testsrc2=s=1280x720:r=30:d=20 {V} {{o}}",
    "screen.mp4": "ffmpeg -i screen.mov -c copy {o}",
    "web.webm": f"ffmpeg {SRC.format(s='854x480', r=30, d=25, f=350)} -ac 2 -c:v libvpx-vp9 -deadline realtime -cpu-used 8 -b:v 800k -c:a libopus -b:a 96k -shortest {{o}}",
    "old.avi": f"ffmpeg {SRC.format(s='640x480', r=25, d=25, f=400)} -ac 2 -c:v mpeg4 -q:v 5 -c:a libmp3lame -b:a 128k -shortest {{o}}",
    "subs.mkv": f"ffmpeg {SRC.format(s='1280x720', r=24, d=30, f=380)} -i subs.srt -map 0 -map 1 -map 2 -ac 2 {V} {A} -c:s srt -metadata:s:s:0 language=eng -shortest {{o}}",
    "short.mp4": f"ffmpeg {SRC.format(s='1280x720', r=30, d=10, f=600)} -ac 2 {V} {A} -shortest {{o}}",
    "anim.gif": "magick -delay 10 -size 401x301 xc:tomato xc:gold xc:teal xc:navy -loop 0 {o}",
    # audio: tone with 3 s silent gaps every 8 s (so silence tools have work to do)
    "a.wav": "ffmpeg -f lavfi -i \"aevalsrc=0.4*sin(2*PI*330*t)*gt(mod(t\\,8)\\,3)|0.4*sin(2*PI*335*t)*gt(mod(t\\,8)\\,3):s=44100:d=30\" -c:a pcm_s16le {o}",
    "a.mp3": "ffmpeg -f lavfi -i sine=f=262:d=45:sample_rate=44100 -f lavfi -i anoisesrc=d=45:a=0.02:r=44100 -filter_complex amix=inputs=2,aformat=channel_layouts=stereo -c:a libmp3lame -b:a 192k {o}",
    "a.m4a": "ffmpeg -f lavfi -i sine=f=200:d=60:sample_rate=48000 -f lavfi -i anoisesrc=d=60:a=0.05:r=48000 -filter_complex amix=inputs=2 -ac 1 -c:a aac -b:a 64k {o}",
    "a.flac": "ffmpeg -f lavfi -i sine=f=440:d=30:sample_rate=48000 -ac 2 -c:a flac {o}",
    "music.mp3": "ffmpeg -f lavfi -i \"sine=f=220:d=60:sample_rate=44100\" -f lavfi -i \"sine=f=330:d=60:sample_rate=44100\" -filter_complex amix=inputs=2,aformat=channel_layouts=stereo -c:a libmp3lame -b:a 192k {o}",
    "music.wav": "ffmpeg -i music.mp3 -c:a pcm_s16le {o}",
    # images
    "photo.jpg": "magick -size 1600x1067 gradient:skyblue-navy -fill white -draw 'circle 800,533 800,300' -quality 92 {o}",
    "shot.png": "magick -size 1440x900 xc:white -fill '#1e293b' -draw 'rectangle 120,90 1320,810' -fill '#38bdf8' -draw 'rectangle 200,160 900,400' {o}",
    "pic.webp": "magick -size 800x800 plasma:fractal -quality 85 {o}",
    "iphone.heic": "magick -size 1200x1600 gradient:orange-purple {o}",
    "logo.png": "magick -size 300x120 xc:none -fill 'rgba(255,255,255,0.9)' -draw 'roundrectangle 0,0 299,119 20,20' -fill '#0f172a' -draw 'circle 60,60 60,25' {o}",
    "logo.svg": None,
}

LOGO_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" width="120" height="48" viewBox="0 0 120 48">'
            '<rect width="120" height="48" rx="8" fill="#0f172a"/><circle cx="24" cy="24" r="14" fill="#38bdf8"/></svg>')

EXIF = ["-Make=Apple", "-Model=iPhone 15 Pro", "-DateTimeOriginal=2024:07:14 18:32:10",
        "-CreateDate=2024:07:14 18:32:10", "-GPSLatitude=17.385", "-GPSLatitudeRef=N",
        "-GPSLongitude=78.4867", "-GPSLongitudeRef=E", "-Artist=Mohammed", "-Software=Camera 17.5"]


def _sh(cmd):
    subprocess.run(cmd, shell=True, cwd=FIX, check=True, capture_output=True)


def build():
    os.makedirs(FIX, exist_ok=True)
    with open(os.path.join(FIX, "subs.srt"), "w") as f:
        f.write(SRT)
    with open(os.path.join(FIX, "logo.svg"), "w") as f:
        f.write(LOGO_SVG)
    for name, recipe in RECIPES.items():
        if recipe and not os.path.exists(os.path.join(FIX, name)):
            print("fixture", name)
            _sh(recipe.format(o=name))
    os.makedirs(os.path.join(FIX, "frames"), exist_ok=True)
    if not os.path.exists(os.path.join(FIX, "frames", "frame_020.png")):
        _sh("ffmpeg -f lavfi -i testsrc2=s=1280x720:r=1:d=20 frames/frame_%03d.png")
    for img in ("photo.jpg", "iphone.heic"):
        subprocess.run(["exiftool", "-overwrite_original", *EXIF, img], cwd=FIX, check=True, capture_output=True)
    meta = {}
    for name in sorted(os.listdir(FIX)):
        p = os.path.join(FIX, name)
        if not os.path.isfile(p) or name == "meta.json":
            continue
        meta[name] = describe(p)
    with open(os.path.join(FIX, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def describe(p):
    m = {"size": os.path.getsize(p)}
    ext = os.path.splitext(p)[1].lower()
    if ext in (".jpg", ".png", ".webp", ".heic", ".svg"):
        r = subprocess.run(["magick", "identify", "-format", "%w %h %A", p + "[0]"], capture_output=True, text=True)
        w, h, a = r.stdout.split()
        return {**m, "w": int(w), "h": int(h), "alpha": a.lower() not in ("undefined", "false")}
    info = probe(p)
    if not info:
        return m
    m["dur"] = float(info["format"].get("duration", 0))
    for s in info["streams"]:
        if s["codec_type"] == "video" and "vcodec" not in m:
            num, den = s["avg_frame_rate"].split("/")
            m.update(vcodec=s["codec_name"], w=s["width"], h=s["height"], fps=round(float(num) / float(den), 3))
        elif s["codec_type"] == "audio" and "acodec" not in m:
            m.update(acodec=s["codec_name"], ach=s["channels"], asr=int(s["sample_rate"]))
        elif s["codec_type"] == "subtitle":
            m["subs"] = True
    m["has_audio"] = "acodec" in m
    return m


if __name__ == "__main__":
    for k, v in build().items():
        print(k, v)
