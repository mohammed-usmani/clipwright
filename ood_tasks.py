"""Hand-written held-out test set: real-world phrasing, NOT produced by the task templates or the teacher.

No model was trained on these requests. Each task carries a reference solution, and `--validate` proves the checker is
fair: the reference must pass verify(), and an empty out/ must fail it.

    python3 ood_tasks.py --validate     # run every reference solution through the sandbox + verify()
    python3 ood_tasks.py --export       # write data/ood/test_ood.jsonl (same row format as test_multi.jsonl)
"""
import argparse
import json
import os
import shutil
import tempfile

import sandbox
from build import SYSTEM
from fixtures import FIX, clone
from teacher import verify

# (id, category, request, {in-name: fixture}, [output specs], [reference commands])
T = [
    # ── video ──
    ("o_v01", "video", "IMG_2231.MOV won't play on my windows laptop, can you make a version that works there?",
     {"IMG_2231.MOV": "phone.mov"}, [{"match": "*.mp4", "vcodec": "h264", "has_audio": True, "dur": [14, 16]}],
     ["ffmpeg -i in/IMG_2231.MOV -c:v libx264 -crf 23 -c:a aac out/IMG_2231.mp4"]),
    ("o_v02", "video", "I need to email trip.mp4 but the attachment limit is 5 MB. Keep the whole video please.",
     {"trip.mp4": "land.mp4"}, [{"match": "*.mp4", "max_mb": 5, "has_video": True, "has_audio": True, "dur": [39, 41]}],
     ["ffmpeg -i in/trip.mp4 -c:v libx264 -b:v 750k -maxrate 750k -bufsize 1500k -c:a aac -b:a 64k out/trip_small.mp4"]),
    ("o_v03", "video", "turn demo.mp4 into a vertical 9:16 video for instagram reels, keep the middle of the shot",
     {"demo.mp4": "land.mp4"}, [{"match": "*.mp4", "aspect": [9, 16], "min_short": 400, "has_audio": True, "dur": [39, 41]}],
     ["ffmpeg -i in/demo.mp4 -vf crop=404:720 -c:v libx264 -c:a copy out/demo_vertical.mp4"]),
    ("o_v04", "video", "just give me the bit from 0:12 to 0:20 of interview.mp4",
     {"interview.mp4": "land.mp4"}, [{"match": "*.mp4", "has_video": True, "has_audio": True, "dur": [7.5, 8.6]}],
     ["ffmpeg -ss 12 -to 20 -i in/interview.mp4 -c:v libx264 -c:a aac out/interview_clip.mp4"]),
    ("o_v05", "video", "the sound in lecture.mp4 is way too quiet, bring it up to normal YouTube loudness",
     {"lecture.mp4": "land.mp4"}, [{"match": "*.mp4", "has_video": True, "lufs": [-16, -12], "dur": [39, 41]}],
     ["ffmpeg -i in/lecture.mp4 -c:v copy -af loudnorm=I=-14:TP=-1.5:LRA=11 -ar 48000 -c:a aac out/lecture.mp4"]),
    ("o_v06", "video", "strip the sound out of clip.mp4, I'm going to put my own music on it later",
     {"clip.mp4": "short.mp4"}, [{"match": "*.mp4", "has_video": True, "has_audio": False, "dur": [9.5, 10.5]}],
     ["ffmpeg -i in/clip.mp4 -c:v copy -an out/clip_silent.mp4"]),
    ("o_v07", "video", "sunset.mp4 is too slow, make it 4 times faster. don't care about the audio",
     {"sunset.mp4": "land.mp4"}, [{"match": "*.mp4", "has_video": True, "dur": [9, 11]}],
     ["ffmpeg -i in/sunset.mp4 -vf setpts=PTS/4 -an -c:v libx264 out/sunset_4x.mp4"]),
    ("o_v08", "video", "can I get a gif of the first 3 seconds of cat.mp4? has to be under 2MB for slack",
     {"cat.mp4": "short.mp4"}, [{"match": "*.gif", "max_mb": 2, "dur": [2.5, 3.6]}],
     ["ffmpeg -t 3 -i in/cat.mp4 -vf fps=10,scale=480:-1:flags=lanczos out/cat.gif"]),
    ("o_v09", "video", "grab a still from vlog.mp4 at the 5 second mark, as a jpg",
     {"vlog.mp4": "land.mp4"}, [{"match": "*.jpg", "count": 1}],
     ["ffmpeg -ss 5 -i in/vlog.mp4 -frames:v 1 -q:v 2 out/vlog_5s.jpg"]),
    ("o_v10", "video", "sideways.mp4 was filmed with the phone on its side, rotate it 90 degrees so it's portrait",
     {"sideways.mp4": "land.mp4"}, [{"match": "*.mp4", "wh": [720, 1280], "has_audio": True}],
     ["ffmpeg -i in/sideways.mp4 -vf transpose=1 -c:v libx264 -c:a copy out/sideways_rotated.mp4"]),
    ("o_v11", "video", "old_home_video.avi needs to play on my iphone",
     {"old_home_video.avi": "old.avi"}, [{"match": "*.mp4|*.mov", "vcodec": "h264", "acodec": "aac", "dur": [24, 26]}],
     ["ffmpeg -i in/old_home_video.avi -c:v libx264 -pix_fmt yuv420p -c:a aac out/old_home_video.mp4"]),
    ("o_v12", "video", "put logo.png in the top right corner of promo.mp4",
     {"promo.mp4": "land.mp4", "logo.png": "logo.png"}, [{"match": "*.mp4", "wh": [1280, 720], "has_audio": True, "dur": [39, 41]}],
     ["ffmpeg -i in/promo.mp4 -i in/logo.png -filter_complex overlay=W-w-10:10 -c:v libx264 -c:a copy out/promo_logo.mp4"]),
    ("o_v13", "video", "convert recording.webm to mp4 so I can put it in powerpoint",
     {"recording.webm": "web.webm"}, [{"match": "*.mp4", "vcodec": "h264", "acodec": "aac", "dur": [24, 26]}],
     ["ffmpeg -i in/recording.webm -c:v libx264 -pix_fmt yuv420p -c:a aac out/recording.mp4"]),
    ("o_v14", "video", "screen_rec.mp4 is 720p, too big. make it 480p",
     {"screen_rec.mp4": "screen.mp4"}, [{"match": "*.mp4", "minside": 480, "dur": [19, 21]}],
     ["ffmpeg -i in/screen_rec.mp4 -vf scale=-2:480 -c:v libx264 out/screen_rec_480p.mp4"]),
    ("o_v15", "video", "join part1.mp4 and part2.mp4 together into one video",
     {"part1.mp4": "short.mp4", "part2.mp4": "short.mp4"}, [{"match": "*.mp4", "has_audio": True, "dur": [19, 21]}],
     ["ffmpeg -i in/part1.mp4 -i in/part2.mp4 -filter_complex \"[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]\" "
      "-map \"[v]\" -map \"[a]\" -c:v libx264 -c:a aac out/joined.mp4"]),
    ("o_v16", "video", "I only want the audio from podcast_video.mp4, as an mp3",
     {"podcast_video.mp4": "land.mp4"}, [{"match": "*.mp3", "acodec": "mp3", "has_video": False, "dur": [39, 41]}],
     ["ffmpeg -i in/podcast_video.mp4 -vn -c:a libmp3lame -q:a 2 out/podcast.mp3"]),
    ("o_v17", "video", "wedding.mp4 should fade in from black at the start",
     {"wedding.mp4": "land.mp4"}, [{"match": "*.mp4", "first_dark": True, "has_audio": True, "dur": [39, 41]}],
     ["ffmpeg -i in/wedding.mp4 -vf fade=t=in:st=0:d=2 -c:v libx264 -c:a copy out/wedding_fade.mp4"]),
    ("o_v18", "video", "video.mp4 is 24fps, I need it at 30fps",
     {"video.mp4": "small.mp4"}, [{"match": "*.mp4", "fps": 30, "dur": [74, 76]}],
     ["ffmpeg -i in/video.mp4 -vf fps=30 -c:v libx264 -c:a copy out/video_30fps.mp4"]),
    ("o_v19", "video", "cut the first 10 seconds off intro.mp4, the rest is fine",
     {"intro.mp4": "land.mp4"}, [{"match": "*.mp4", "has_audio": True, "dur": [29.4, 30.6]}],
     ["ffmpeg -ss 10 -i in/intro.mp4 -c:v libx264 -c:a aac out/intro_cut.mp4"]),
    ("o_v20", "video", "make the audio in talk.mp4 mono, don't touch the video",
     {"talk.mp4": "land.mp4"}, [{"match": "*.mp4", "ach": 1, "vcodec": "h264", "dur": [39, 41]}],
     ["ffmpeg -i in/talk.mp4 -c:v copy -ac 1 -c:a aac out/talk_mono.mp4"]),
    # ── audio ──
    ("o_a01", "audio", "voice memo.m4a → mp3 please, my car stereo only plays mp3",
     {"voice memo.m4a": "a.m4a"}, [{"match": "*.mp3", "acodec": "mp3", "dur": [59, 61]}],
     ["ffmpeg -i 'in/voice memo.m4a' -c:a libmp3lame -q:a 2 'out/voice memo.mp3'"]),
    ("o_a02", "audio", "song.wav is too big for whatsapp, make it a small mp3",
     {"song.wav": "a.wav"}, [{"match": "*.mp3", "acodec": "mp3", "max_mb": 1.5, "dur": [29, 31]}],
     ["ffmpeg -i in/song.wav -c:a libmp3lame -b:a 128k out/song.mp3"]),
    ("o_a03", "audio", "I only need the first 20 seconds of beat.mp3, with a fade out at the end",
     {"beat.mp3": "music.mp3"}, [{"match": "*.mp3", "dur": [19.5, 20.6]}],
     ["ffmpeg -i in/beat.mp3 -t 20 -af afade=t=out:st=17:d=3 out/beat_20s.mp3"]),
    ("o_a04", "audio", "convert interview.flac to a 16kHz mono wav, it's for a transcription tool",
     {"interview.flac": "a.flac"}, [{"match": "*.wav", "sr": 16000, "ach": 1, "dur": [29, 31]}],
     ["ffmpeg -i in/interview.flac -ar 16000 -ac 1 out/interview_16k.wav"]),
    ("o_a05", "audio", "normalize episode.wav to -16 LUFS for spotify and give it to me as mp3",
     {"episode.wav": "a.wav"}, [{"match": "*.mp3", "acodec": "mp3", "lufs": [-17.5, -14.5]}],
     ["ffmpeg -i in/episode.wav -af loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-8.96:measured_TP=-7.96:measured_LRA=4.80:"
      "measured_thresh=-19.02:offset=-2.39:linear=true -ar 44100 -c:a libmp3lame -b:a 192k out/episode.mp3"]),  # 2nd loudnorm pass
    ("o_a06", "audio", "stick intro.mp3 and talk.mp3 together into one file, intro first",
     {"intro.mp3": "a.mp3", "talk.mp3": "music.mp3"}, [{"match": "*.mp3|*.m4a|*.wav", "dur": [104, 106]}],
     ["ffmpeg -i in/intro.mp3 -i in/talk.mp3 -filter_complex \"[0:a][1:a]concat=n=2:v=0:a=1[a]\" -map \"[a]\" out/combined.mp3"]),
    ("o_a07", "audio", "make me a ringtone from song.mp3: 30 seconds starting at 0:15, as m4a",
     {"song.mp3": "music.mp3"}, [{"match": "*.m4a", "acodec": "aac", "dur": [29.5, 30.6]}],
     ["ffmpeg -ss 15 -t 30 -i in/song.mp3 -c:a aac -b:a 160k out/ringtone.m4a"]),
    ("o_a08", "audio", "speed up lecture.mp3 to 1.5x but don't make it sound like a chipmunk",
     {"lecture.mp3": "a.mp3"}, [{"match": "*.mp3", "dur": [29, 31]}],
     ["ffmpeg -i in/lecture.mp3 -af atempo=1.5 out/lecture_1.5x.mp3"]),
    ("o_a09", "audio", "turn recording.wav into mono",
     {"recording.wav": "a.wav"}, [{"match": "*.wav|*.mp3|*.flac", "ach": 1, "dur": [29, 31]}],
     ["ffmpeg -i in/recording.wav -ac 1 out/recording_mono.wav"]),
    ("o_a10", "audio", "add 2 seconds of silence at the start of jingle.wav",
     {"jingle.wav": "a.wav"}, [{"match": "*.wav|*.mp3", "dur": [31.6, 32.4]}],
     ["ffmpeg -i in/jingle.wav -af adelay=2000:all=1 out/jingle_delayed.wav"]),
    # ── image ──
    ("o_i01", "image", "my windows pc can't open IMG_0042.HEIC, can you turn it into a jpg?",
     {"IMG_0042.HEIC": "iphone.heic"}, [{"match": "*.jpg|*.jpeg", "count": 1}],
     ["magick in/IMG_0042.HEIC out/IMG_0042.jpg"]),
    ("o_i02", "image", "resize banner.jpg to exactly 1200x630 for a linkedin post, crop it if you have to",
     {"banner.jpg": "photo.jpg"}, [{"match": "*.jpg|*.jpeg|*.png", "wh": [1200, 630]}],
     ["magick in/banner.jpg -resize 1200x630^ -gravity center -extent 1200x630 out/banner_linkedin.jpg"]),
    ("o_i03", "image", "I need profile.jpg as a 400x400 square for my avatar",
     {"profile.jpg": "photo.jpg"}, [{"match": "*.jpg|*.jpeg|*.png", "wh": [400, 400]}],
     ["magick in/profile.jpg -resize 400x400^ -gravity center -extent 400x400 out/profile_400.jpg"]),
    ("o_i04", "image", "screenshot.png is huge, make it a jpg under 200KB",
     {"screenshot.png": "shot.png"}, [{"match": "*.jpg|*.jpeg", "max_mb": 0.2}],
     ["magick in/screenshot.png -quality 80 out/screenshot.jpg"]),
    ("o_i05", "image", "remove the location info from vacation.jpg before I post it online",
     {"vacation.jpg": "photo.jpg"}, [{"match": "*.jpg|*.jpeg", "no_gps": True}],
     ["exiftool -gps:all= -o out/vacation.jpg in/vacation.jpg"]),
    ("o_i06", "image", "convert sticker.webp to a png",
     {"sticker.webp": "pic.webp"}, [{"match": "*.png", "count": 1, "wh": [800, 800]}],
     ["magick in/sticker.webp out/sticker.png"]),
    ("o_i07", "image", "make all the photos in scans/ into a single pdf",
     {f"scans/page_{i:02d}.png": f"frames/frame_{i:03d}.png" for i in range(1, 9)}, [{"match": "*.pdf", "count": 1}],
     ["magick in/scans/page_*.png out/scans.pdf"]),
    ("o_i08", "image", "shrink photo.jpg so the longest side is 1080px for instagram",
     {"photo.jpg": "photo.jpg"}, [{"match": "*.jpg|*.jpeg", "wh": [1080, 720]}],
     ["magick in/photo.jpg -resize 1080x1080 out/photo_1080.jpg"]),
    ("o_i09", "image", "rotate photo.jpg 90 degrees clockwise",
     {"photo.jpg": "photo.jpg"}, [{"match": "*.jpg|*.jpeg", "wh": [1067, 1600]}],
     ["magick in/photo.jpg -rotate 90 out/photo_rotated.jpg"]),
    ("o_i10", "image", "make logo.png 2x bigger",
     {"logo.png": "logo.png"}, [{"match": "*.png", "wh": [600, 240]}],
     ["magick in/logo.png -resize 200% out/logo_2x.png"]),
    # ── multi-step: several deliverables, named by the user ──
    ("o_m01", "multi", "from trip.mp4 I need three things: highlight.mp4 = 15 seconds starting at 0:05, story.mp4 = a vertical "
     "9:16 version of that highlight, and thumb.jpg = a frame from the highlight",
     {"trip.mp4": "land.mp4"},
     [{"match": "highlight.mp4", "dur": [14.5, 15.6], "has_audio": True}, {"match": "story.mp4", "aspect": [9, 16], "min_short": 400, "dur": [14.5, 15.6]},
      {"match": "thumb.jpg", "count": 1}],
     ["ffmpeg -ss 5 -t 15 -i in/trip.mp4 -c:v libx264 -c:a aac out/highlight.mp4",
      "ffmpeg -i out/highlight.mp4 -vf crop=404:720 -c:v libx264 -c:a copy out/story.mp4",
      "ffmpeg -ss 3 -i out/highlight.mp4 -frames:v 1 out/thumb.jpg"]),
    ("o_m02", "multi", "for my podcast: make show.wav -16 LUFS and save it as episode.mp3, plus a preview.mp3 that's just the "
     "first 20 seconds of the episode",
     {"show.wav": "music.wav"},
     [{"match": "episode.mp3", "acodec": "mp3", "lufs": [-17.5, -14.5], "dur": [59, 61]}, {"match": "preview.mp3", "dur": [19.5, 20.6]}],
     ["ffmpeg -i in/show.wav -af loudnorm=I=-16:TP=-1.5 -ar 44100 -c:a libmp3lame -b:a 192k out/episode.mp3",
      "ffmpeg -i out/episode.mp3 -t 20 -c copy out/preview.mp3"]),
    ("o_m03", "multi", "add music.mp3 as background music to screen.mp4 (it has no sound right now). the result should be "
     "as long as the video",
     {"screen.mp4": "screen.mp4", "music.mp3": "music.mp3"}, [{"match": "*.mp4", "has_video": True, "has_audio": True, "dur": [19.5, 20.6]}],
     ["ffmpeg -i in/screen.mp4 -i in/music.mp3 -map 0:v -map 1:a -c:v copy -c:a aac -shortest out/screen_music.mp4"]),
    ("o_m04", "multi", "from game.mp4 make clip.mp4 (0:10 to 0:20) and also clip.gif of the same part, the gif under 3MB",
     {"game.mp4": "land.mp4"}, [{"match": "clip.mp4", "dur": [9.5, 10.6]}, {"match": "clip.gif", "max_mb": 3}],
     ["ffmpeg -ss 10 -to 20 -i in/game.mp4 -c:v libx264 -c:a aac out/clip.mp4",
      "ffmpeg -i out/clip.mp4 -vf fps=10,scale=480:-1 out/clip.gif"]),
    ("o_m05", "multi", "raw.mov is for youtube: convert it to an h264 mp4 at -14 LUFS, and give me a thumbnail.jpg from it too",
     {"raw.mov": "phone.mov"},
     [{"match": "*.mp4", "vcodec": "h264", "lufs": [-16, -12], "dur": [14, 16]}, {"match": "thumbnail.jpg", "count": 1}],
     ["ffmpeg -i in/raw.mov -c:v libx264 -crf 20 -pix_fmt yuv420p -af loudnorm=I=-14:TP=-1.5 -ar 48000 -c:a aac out/raw.mp4",
      "ffmpeg -ss 2 -i out/raw.mp4 -frames:v 1 out/thumbnail.jpg"]),
]


def setup(files):
    wd = tempfile.mkdtemp(prefix="ood_", dir=os.environ.get("JOB_TMP"))
    for name, fx in files.items():
        os.makedirs(os.path.dirname(os.path.join(wd, "in", name)), exist_ok=True)
        clone(os.path.join(FIX, fx), os.path.join(wd, "in", name))
    os.makedirs(os.path.join(wd, "out"))
    return wd


def spec_of(outs, tid=None):
    """outs is a list of output specs, or a full spec dict (outputs + answer) for question tasks."""
    spec = dict(outs) if isinstance(outs, dict) else {"outputs": outs}
    spec.setdefault("outputs", [])
    if tid:
        spec["id"] = tid
    return spec


def validate(tasks=None, only=""):
    """Every reference must pass verify(); an empty out/ (and empty reply) must fail. REPLY: entries are the final reply."""
    tasks = [t for t in (tasks or T) if t[0].startswith(tuple(only.split(",")))]
    bad = 0
    for tid, cat, req, files, outs, ref in tasks:
        wd = setup(files)
        try:
            spec = spec_of(outs)
            empty = verify(wd, spec, final="")
            errs, final = [], ""
            for cmd in ref:
                if cmd.startswith("REPLY:"):
                    final = cmd[6:].strip()
                    continue
                fe = None
                if cmd.startswith("FOREACH "):  # "FOREACH in/x/*.jpg :: cmd with {in} {stem}"
                    fe, cmd = cmd[8:].split(" :: ", 1)
                res = sandbox.run(wd, cmd, fe)
                if res["exit_code"] != 0:
                    errs.append(f"ref failed: {cmd[:60]} -> {res.get('error') or res.get('log', '')[-200:]}")
                    break
            errs += verify(wd, spec, final=final)
            ok = not errs and empty
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {tid}  {'; '.join(errs)[:220] if errs else ''}{'' if empty else ' (empty out/ passed!)'}",
                  flush=True)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    print(f"{len(tasks) - bad}/{len(tasks)} reference solutions pass and empty outputs fail")
    return bad == 0


def export(path, tasks=None, prefix="ood"):
    tasks = tasks or T
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for tid, cat, req, files, outs, ref in tasks:
            wd = setup(files)
            try:
                man = sandbox.manifest(wd)
            finally:
                shutil.rmtree(wd, ignore_errors=True)
            row = {"id": tid, "category": f"{prefix}/{cat}", "tools": [sandbox.TOOL_SCHEMA],
                   "verify": {"files": files, "spec": spec_of(outs, tid)},
                   "messages": [{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": f"{req}\n\nFiles:\n{man}"}],
                   "reference": ref}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks to {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--export", action="store_true")
    a = ap.parse_args()
    if a.validate:
        validate()
    if a.export:
        export(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "ood", "test_ood.jsonl"))
