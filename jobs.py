"""Layer 2: compose multi-step jobs (request + hidden spec) for the teacher.

python3 jobs.py --n 10 --seed 1      -> teacher/jobs/<id>/{in,out,request.txt}, teacher/specs/<id>.json
Specs live outside the job dir so the teacher can't read them.
"""
import argparse
import json
import os
import random
import shutil

import sandbox
from fixtures import FIX, clone
from tasks import BATCH, POOLS, say_t

HERE = os.path.dirname(os.path.abspath(__file__))
TEACH = os.path.join(HERE, "teacher")
PLAT = ["Instagram Reels", "TikTok", "YouTube Shorts"]
TITLES = ["Behind the Scenes", "Episode 12", "Product Launch", "Our Trip", "Highlights"]


def video_job(r, meta):
    name, fx = r.choice([v for v in POOLS["video"] if v[1] in ("land.mp4", "land.mov", "land.mkv", "phone.mov", "small.mp4")])
    m = meta[fx]
    st = {"files": {name: fx}, "say": [], "visual": [], "dur": m["dur"], "used": set(),
          "main": {"match": "*.mp4|*.mov|*.mkv", "has_video": True}, "extra": []}
    land = m["w"] > m["h"]
    ops = ["trim", "frame", "captions", "watermark", "title", "loud", "music", "mute", "fade", "speed", "intro", "size",
           "format", "gif", "thumb", "mp3", "meta", "fps"]
    k = r.choice([2, 3, 3, 4, 4, 5])
    for op in r.sample(ops, len(ops)):
        if len(st["used"]) >= k:
            break
        if op == "trim" and m["dur"] > 20:
            s = r.randint(0, int(m["dur"] * 0.3))
            e = r.randint(s + 10, min(int(m["dur"]) - 1, s + 30))
            st["trim"] = (s, e)
            st["dur"] = e - s
            st["say"].append(r.choice([f"Only keep {say_t(r, s)} to {say_t(r, e)}.",
                                       f"Cut it down to the part from {say_t(r, s)} to {say_t(r, e)}.",
                                       f"Use just the section {say_t(r, s)}–{say_t(r, e)}."]))
        elif op == "frame":
            if land and r.random() < 0.7:
                style = r.choice(["blur", "bars", "crop"])
                st["main"].update(aspect=[9, 16], min_short=720)
                st["say"].append(f"Make it vertical 9:16 for {r.choice(PLAT)}" +
                                 {"blur": " with a blurred background behind it.", "bars": " with black bars, don't crop anything.",
                                  "crop": ", cropped to fill the screen."}[style])
                if style == "blur":
                    st["visual"].append("Background is a blurred, zoomed copy of the video (not black bars)")
            elif min(m["w"], m["h"]) > 720:
                st["main"]["minside"] = 720
                st["say"].append(r.choice(["Downscale it to 720p.", "720p is enough."]))
            else:
                st["main"]["wh"] = [1080, 1080]
                st["say"].append("Make it square (1080x1080) for an Instagram post, no cropping, pad with black.")
        elif op == "captions" and m["dur"] >= 24 and "speed" not in st["used"]:
            sub = r.choice(POOLS["subs"])
            st["files"][sub[0]] = sub[1]
            st["captions"] = True
            st["say"].append(r.choice([f"Burn in the subtitles from {sub[0]}.", f"Hardcode the captions in {sub[0]}."]))
            st["visual"].append(f"Captions from {sub[0]} are burned in, readable, and in sync with the trimmed timeline "
                                "(cue 'Welcome back to the channel.' is at 0:01–0:04 of the ORIGINAL video)")
        elif op == "watermark":
            lg = r.choice(POOLS["logo"])
            corner = r.choice(["top-right", "bottom-right", "top-left", "bottom-left"])
            st["files"][lg[0]] = lg[1]
            st["say"].append(f"Put my logo ({lg[0]}) in the {corner} corner, fairly small.")
            st["visual"].append(f"Logo visible in the {corner} corner, small, not stretched")
        elif op == "title":
            t = r.choice(TITLES)
            st["say"].append(f"Add the title \"{t}\" at the top for the first 3 seconds.")
            st["visual"].append(f"Title '{t}' visible at the top during the first 3 s only")
        elif op == "loud" and "mute" not in st["used"]:
            target = r.choice([-14, -16])
            st["main"]["lufs"] = [target - 2, target + 2]
            st["say"].append(r.choice([f"Normalize the audio to {target} LUFS.", "Make the volume consistent and loud enough for social media."
                                       if target == -14 else "Loudness should be podcast standard (-16 LUFS)."]))
        elif op == "music" and "mute" not in st["used"]:
            mu = r.choice(POOLS["music"])
            st["files"][mu[0]] = mu[1]
            st["main"]["has_audio"] = True
            st["say"].append(f"Add {mu[0]} as quiet background music under the original sound.")
        elif op == "mute" and not st["used"] & {"loud", "music", "mp3"}:
            st["main"]["has_audio"] = False
            st["say"].append("Remove the audio completely.")
        elif op == "fade":
            st["main"]["first_dark"] = True
            st["say"].append("Fade in from black at the start and fade out at the end (audio too).")
        elif op == "speed" and "intro" not in st["used"] and "captions" not in st["used"]:
            f = r.choice([1.5, 2])
            st["speed"] = f
            st["say"].append(f"Speed it up to {f:g}x but keep voices sounding normal.")
        elif op == "intro" and "speed" not in st["used"]:
            iv = r.choice(POOLS["video2"])
            st["files"][iv[0]] = iv[1]
            st["intro"] = meta[iv[1]]["dur"]
            st["say"].append(f"Put {iv[0]} at the very beginning as an intro.")
        elif op == "size":
            st["size"] = True
            st["say"].append("Keep the final file under {MB} MB.")
        elif op == "format" and "size" not in st["used"]:
            if r.random() < 0.5:
                st["main"].update(match="*.webm", vcodec="vp9")
                st["say"].append("Export it as WebM for my website.")
            else:
                st["main"].update(vcodec="h264", pix_fmt="yuv420p")
                st["say"].append("It must play on old phones and smart TVs, so use the most compatible format.")
        elif op == "gif":
            st["extra"].append({"match": "*.gif", "max_w": 480, "dur": [2, 4.5]})
            st["say"].append("Also make a GIF preview of the first 3 seconds of the final video, max 480px wide.")
        elif op == "thumb":
            st["extra"].append({"match": "*.jpg"})
            st["say"].append("Also give me a JPG thumbnail.")
        elif op == "mp3" and "mute" not in st["used"]:
            st["extra"].append({"match": "*.mp3", "has_video": False, "has_audio": True})
            st["say"].append("And export the final audio separately as an MP3.")
        elif op == "meta":
            st["main"]["no_meta"] = True
            st["say"].append("Strip all metadata before I share it.")
        elif op == "fps" and m["fps"] != 24:
            st["main"]["fps"] = 24
            st["say"].append("Convert it to 24 fps for a cinematic feel.")
        else:
            continue
        st["used"].add(op)
    if len(st["used"]) < 2:
        return None
    if "mute" in st["used"]:
        st["say"] = [x.replace(" (audio too)", "") for x in st["say"]]
    d = st["dur"] / st.get("speed", 1) + st.get("intro", 0)
    st["main"]["dur"] = [round(d - 1, 2), round(d + 1, 2)]
    if st.get("size"):
        mb = max(2, round(d * 0.25))
        st["main"]["max_mb"] = mb
        st["say"] = [s.replace("{MB}", str(mb)) for s in st["say"]]
    for e in st["extra"]:
        if e["match"] == "*.mp3":
            e["dur"] = st["main"]["dur"]
    lead = r.choice([f"I've got {name}. ", f"Here's {name}. ", f"Edit {name} for me: ", f"{name} needs some work. ", ""])
    if r.random() < 0.35:
        req = (lead or f"For {name}:") + "\n" + "\n".join(f"- {s}" for s in st["say"])
    else:
        req = lead + " ".join(st["say"]) if lead else f"For {name}: " + " ".join(st["say"])
    return st, req


def audio_job(r, meta):
    name, fx = r.choice([a for a in POOLS["audio"] if a[1] in ("a.wav", "a.m4a", "a.mp3")])
    m = meta[fx]
    st = {"files": {name: fx}, "say": [], "visual": [], "used": set(), "main": {"match": "*.mp3", "has_audio": True}, "extra": []}
    st["say"].append(r.choice(["This is a podcast episode.", "It's a voice recording for my show.", "Raw interview audio."]))
    dur = m["dur"]
    if fx == "a.wav" and r.random() < 0.6:
        st["say"].append("Cut out the long silent pauses.")
        st["used"].add("silence")
        dur = None
    if r.random() < 0.7:
        st["main"]["lufs"] = [-18, -14]
        st["say"].append("Normalize it to -16 LUFS.")
        st["used"].add("loud")
    if r.random() < 0.5:
        mu = r.choice(POOLS["music"])
        st["files"][mu[0]] = mu[1]
        st["say"].append(f"Add the first 5 seconds of {mu[0]} as an intro jingle, fading out.")
        st["used"].add("jingle")
        if dur:
            dur += 5
    if r.random() < 0.5:
        st["main"]["ach"] = 1
        st["say"].append("Mono is fine.")
    br = r.choice([96, 128])
    st["say"].append(f"Final file as a {br} kbps MP3.")
    st["main"]["acodec"] = "mp3"
    if r.random() < 0.4:
        st["extra"].append({"match": "*.wav", "sr": 16000, "ach": 1})
        st["say"].append("Also give me a 16 kHz mono WAV of the cleaned audio for transcription.")
    if dur:
        st["main"]["dur"] = [dur - 1.5, dur + 1.5]
    else:
        st["main"]["dur"] = [5, m["dur"] - 3 + (5 if "jingle" in st["used"] else 0)]
    return st, f"{name}: " + " ".join(st["say"])


def image_job(r, meta):
    key = r.choice(["jpg", "heic"])
    B = BATCH[key]
    n = len(B["files"])
    st = {"files": dict(B["files"]), "say": [], "visual": [], "used": set(), "main": {"match": "*", "count": n}, "extra": []}
    fmt = r.choice(["jpg", "webp"])
    st["main"]["match"] = f"*.{fmt}"
    S = r.choice([1080, 1600, 2048])
    st["main"]["max_side"] = S
    st["say"].append(f"Resize {B['desc']} so the longest side is at most {S}px and save them as {fmt}.")
    if r.random() < 0.6:
        st["main"]["no_gps"] = True
        st["say"].append("Remove the location data but keep the date taken.")
        st["main"]["has_date"] = True
    if r.random() < 0.5:
        lg = r.choice(POOLS["logo"])
        st["files"][lg[0]] = lg[1]
        st["say"].append(f"Watermark each one with {lg[0]} in the bottom-right corner.")
        st["visual"].append("Logo watermark in the bottom-right of each photo")
    if r.random() < 0.4:
        st["extra"].append({"match": "*.pdf"})
        st["say"].append("Then combine them into one PDF too.")
    if r.random() < 0.4:
        st["extra"].append({"match": "contact*", "count": 1})
        st["say"].append("And make one contact-sheet image (name it contact_sheet) showing them all in a grid.")
    return st, "I uploaded some photos. " + " ".join(st["say"])


def make(n, seed):
    meta = json.load(open(os.path.join(FIX, "meta.json")))
    r = random.Random(seed)
    os.makedirs(os.path.join(TEACH, "jobs"), exist_ok=True)
    os.makedirs(os.path.join(TEACH, "specs"), exist_ok=True)
    made = []
    while len(made) < n:
        kind = r.choices(["video", "audio", "image"], weights=[6, 2, 2])[0]
        res = {"video": video_job, "audio": audio_job, "image": image_job}[kind](r, meta)
        if not res:
            continue
        st, req = res
        jid = f"{kind[0]}{seed:03d}_{len(made):04d}"
        wd = os.path.join(TEACH, "jobs", jid)
        shutil.rmtree(wd, ignore_errors=True)
        os.makedirs(os.path.join(wd, "in"))
        os.makedirs(os.path.join(wd, "out"))
        for fname, fx in st["files"].items():
            clone(os.path.join(FIX, fx), os.path.join(wd, "in", fname))
        manifest = sandbox.manifest(wd)
        with open(os.path.join(wd, "request.txt"), "w") as f:
            f.write(f"{req}\n\nFiles:\n{manifest}\n")
        spec = {"id": jid, "kind": kind, "request": req, "manifest": manifest, "files": st["files"],
                "outputs": [st["main"]] + st["extra"], "visual": st["visual"]}
        json.dump(spec, open(os.path.join(TEACH, "specs", f"{jid}.json"), "w"), indent=1)
        made.append(jid)
    return made


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    for j in make(a.n, a.seed):
        print(j)
