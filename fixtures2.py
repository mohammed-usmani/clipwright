"""Second fixture library: synthetic media with KNOWN ground truth, for the new task types (slides + voiceover, pitch,
ducking, sync, interlacing, VFR, aspect, brightness, broken files, multi-language, scans, product shots, quality...).

Two copies with different parameters (voices, resolution, frame rate, timings, colours):
  fixtures/           train set (task templates + teacher jobs use these)
  fixtures/test/      held-out test set only; never used to generate training data

    python3 fixtures2.py          # build both (skips files that exist), print the ground truth
Ground truth for each set is written to <dir>/truth.json so specs can use it.
"""
import json
import os
import subprocess
import sys

from fixtures import FIX, describe

SETS = {
    "train": dict(dir=FIX, voice_m="Daniel", voice_f="Samantha", voice_m2="Rishi", voice_f2="Karen",
                  s="1280x720", r=30, cuts=[4.0, 9.5, 15.0], cut_colors=["teal", "orange", "purple", "gold"],
                  sync_ms=300, flash=[2.0, 7.0], music=(65, 98), hum=50, card_bg="#0f172a", card_fg="white",
                  lines=["Welcome to the quarterly update.", "Revenue grew in every region this year.",
                         "Thanks for listening, see you next time."],
                  short_m="Thanks for watching. Don't forget to subscribe.",
                  short_f="That's everything for today. See you soon.",
                  hindi="नमस्ते, आज हम नए फ़ीचर देखेंगे।", page_tint="#f4ecd8", product="#2563eb", odd="853x481",
                  photo_dates=["2024:05:03 09:12:00", "2024:05:03 14:40:10", "2024:05:04 08:05:33", "2024:05:05 19:22:00",
                               "2024:05:06 07:59:59", "2024:05:06 21:30:00"]),
    "test": dict(dir=os.path.join(FIX, "test"), voice_m="Aman", voice_f="Moira", voice_m2="Aman", voice_f2="Tessa",
                 s="1920x1080", r=25, cuts=[3.0, 7.5, 12.0], cut_colors=["navy", "crimson", "olive", "silver"],
                 sync_ms=-250, flash=[1.5, 6.0], music=(247, 370), hum=60, card_bg="#fafaf9", card_fg="#111827",
                 lines=["Hi everyone, quick demo of the new editor.", "First we import the clips from the camera.",
                        "That's it, thanks so much for watching."],
                 short_m="That was the demo. Thanks a lot for your time.",
                 short_f="Thank you all for joining today's session.",
                 hindi="धन्यवाद, यह वीडियो देखने के लिए शुक्रिया।", page_tint="#eef2f7", product="#16a34a", odd="1279x717",
                 photo_dates=["2023:12:24 18:00:00", "2023:12:25 10:15:00", "2023:12:25 12:45:30", "2023:12:26 16:20:00",
                              "2023:12:31 23:59:00", "2024:01:01 00:10:00"]),
}
SLOT, GAP = 4.0, 2.0  # narration: 3 speech slots of 4 s, separated by 2 s of silence


def sh(cmd, cwd):
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"FAILED in {cwd}:\n{cmd}\n{r.stderr[-1500:]}")


def say(text, voice, out, cwd, slot=None):
    """TTS to m4a (48 kHz stereo). With slot, padded/trimmed to exactly slot seconds."""
    sh(f"say -v {voice} -o .tmp.aiff {json.dumps(text)}", cwd)
    pad = f"apad=whole_dur={slot},atrim=0:{slot}," if slot else ""
    sh(f"ffmpeg -y -v error -i .tmp.aiff -af {pad}aresample=48000 -ac 2 -c:a aac -b:a 128k {out}", cwd)
    os.remove(os.path.join(cwd, ".tmp.aiff"))


def build(p):
    d = p["dir"]
    os.makedirs(d, exist_ok=True)
    w, h = map(int, p["s"].split("x"))
    r, V, A = p["r"], "-c:v libx264 -preset veryfast -crf 23 -pix_fmt yuv420p", "-c:a aac -b:a 128k"
    need = lambda f: not os.path.exists(os.path.join(d, f))
    truth = {}

    # voices
    for name, voice, text in (("voice_m.m4a", p["voice_m"], p["short_m"]), ("voice_f.m4a", p["voice_f"], p["short_f"])):
        if need(name):
            say(text, voice, name, d)
    # narration with known speech slots: [0,4] [6,10] [12,16]
    if need("narration.m4a"):
        for i, line in enumerate(p["lines"]):
            say(line, p["voice_m2"] if i % 2 else p["voice_f2"], f".n{i}.m4a", d, slot=SLOT)
        sh("ffmpeg -y -v error -i .n0.m4a -i .n1.m4a -i .n2.m4a -filter_complex "
           f"\"anullsrc=r=48000:cl=stereo,atrim=0:{GAP}[g1];anullsrc=r=48000:cl=stereo,atrim=0:{GAP}[g2];"
           f"[0][g1][1][g2][2]concat=n=5:v=0:a=1\" {A} narration.m4a", d)
        for i in range(3):
            os.remove(os.path.join(d, f".n{i}.m4a"))
    truth["speech"] = [[0, SLOT], [SLOT + GAP, 2 * SLOT + GAP], [2 * (SLOT + GAP), 3 * SLOT + 2 * GAP]]
    truth["pauses"] = [[SLOT, SLOT + GAP], [2 * SLOT + GAP, 2 * (SLOT + GAP)]]
    dur = 3 * SLOT + 2 * GAP
    if need("talk.mp4"):  # talking-head stand-in: moving test pattern + the narration
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d={dur} -i narration.m4a {V} {A} -shortest talk.mp4", d)
    if need("music_bed.mp3"):
        f1, f2 = p["music"]
        sh(f"ffmpeg -y -v error -f lavfi -i sine=f={f1}:d=40:sample_rate=44100 -f lavfi -i sine=f={f2}:d=40:sample_rate=44100 "
           "-filter_complex amix=inputs=2,volume=0.6,aformat=channel_layouts=stereo -c:a libmp3lame -b:a 192k music_bed.mp3", d)
    truth["music_hz"] = p["music"][0]
    # cards / slides
    if need("thanks.png"):
        sh(f"magick -size {p['s']} xc:'{p['card_bg']}' -fill '{p['card_fg']}' -pointsize {h // 9} -gravity center "
           f"-annotate +0-{h // 20} 'Thank you' -pointsize {h // 24} -annotate +0+{h // 8} 'clipwright.dev' thanks.png", d)
    if need("title.png"):
        sh(f"magick -size {p['s']} gradient:'{p['cut_colors'][0]}'-'{p['cut_colors'][2]}' -fill white -pointsize {h // 10} "
           "-gravity center -annotate +0+0 'Episode 12' title.png", d)
    # hard cuts at known times
    if need("cuts.mp4"):
        b = [0] + p["cuts"] + [p["cuts"][-1] + 5]
        segs = "".join(f"color={c}:s={p['s']}:r={r}:d={b[i + 1] - b[i]}[c{i}];" for i, c in enumerate(p["cut_colors"]))
        cat = "".join(f"[c{i}]" for i in range(4))
        sh(f"ffmpeg -y -v error -f lavfi -i \"{segs}{cat}concat=n=4:v=1:a=0[out0]\" -f lavfi -i sine=f=440:d={b[-1]} "
           f"-map 0:v -map 1:a -ac 2 {V} {A} -shortest cuts.mp4", d)
    truth["cuts"] = p["cuts"]
    truth["cut_colors"] = p["cut_colors"]
    # A/V sync: white flashes at known times, beeps late/early by sync_ms
    if need("sync.mp4"):
        f1, f2 = p["flash"]
        o = p["sync_ms"] / 1000
        en = f"between(t\\,{f1}\\,{f1 + 0.08})+between(t\\,{f2}\\,{f2 + 0.08})"
        ex = f"0.5*sin(2*PI*1000*t)*(between(t\\,{f1 + o}\\,{f1 + o + 0.1})+between(t\\,{f2 + o}\\,{f2 + o + 0.1}))"
        sh(f"ffmpeg -y -v error -f lavfi -i color=black:s={p['s']}:r={r}:d=10 -f lavfi -i \"aevalsrc={ex}:s=48000:d=10\" "
           f"-vf \"drawbox=t=fill:c=white:enable='{en}'\" -ac 2 {V} {A} sync.mp4", d)
    truth["sync_ms"] = p["sync_ms"]
    # problem files
    if need("interlaced.mp4"):
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={2 * r}:d=10 -f lavfi -i sine=f=330:d=10 "
           f"-vf tinterlace=interleave_top,fieldorder=tff -c:v libx264 -flags +ilme+ildct -pix_fmt yuv420p "
           f"-ac 2 {A} interlaced.mp4", d)
    if need("vfr.mp4"):  # screen-recorder style: frame rate changes 30 -> 12 -> 24 fps; audio stays continuous
        segs = ";".join(f"testsrc2=s={p['s']}:r={fr}:d=4,settb=1/90000[s{k}]" for k, fr in enumerate((r, 12, 24)))
        sh(f"ffmpeg -y -v error -f lavfi -i \"{segs};[s0][s1][s2]concat=n=3:v=1:a=0[out0]\" -f lavfi -i sine=f=300:d=12 "
           f"-fps_mode passthrough -ac 2 {V} {A} vfr.mp4", d)
    if need("squashed.mp4"):  # picture squeezed to 3/4 width, square pixels: looks squashed, needs widening
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d=10 -f lavfi -i sine=f=350:d=10 "
           f"-vf scale={w * 3 // 4 // 2 * 2}:{h},setsar=1 -ac 2 {V} {A} squashed.mp4", d)
    if need("wrongsar.mp4"):  # right pixels, wrong flag: players stretch it 4:3 wider
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d=10 -f lavfi -i sine=f=360:d=10 "
           f"-vf setsar=4/3 -ac 2 {V} {A} wrongsar.mp4", d)
    if need("dark.mp4"):
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d=10 -f lavfi -i sine=f=370:d=10 "
           f"-vf eq=brightness=-0.28:contrast=0.6 -ac 2 {V} {A} dark.mp4", d)
    if need("broken.mkv"):  # recording that stopped mid-write: cut at 60%
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d=30 -f lavfi -i sine=f=390:d=30 -ac 2 {V} {A} .full.mkv", d)
        data = open(os.path.join(d, ".full.mkv"), "rb").read()
        open(os.path.join(d, "broken.mkv"), "wb").write(data[: int(len(data) * 0.6)])
        os.remove(os.path.join(d, ".full.mkv"))
    if need("multilang.mkv"):
        say(p["short_m"], p["voice_m"], ".en.m4a", d)
        say(p["hindi"], "Lekha", ".hi.m4a", d)
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d=6 -i .hi.m4a -i .en.m4a -map 0 -map 1 -map 2 "
           f"-t 6 {V} -c:a aac -metadata:s:a:0 language=hin -metadata:s:a:1 language=eng -disposition:a:0 default "
           "-disposition:a:1 0 multilang.mkv", d)
        os.remove(os.path.join(d, ".en.m4a"))
        os.remove(os.path.join(d, ".hi.m4a"))
    if need("hum.wav"):  # voice with mains hum and hiss
        say(" ".join(p["lines"]), p["voice_f"], ".v.m4a", d)
        sh(f"ffmpeg -y -v error -i .v.m4a -f lavfi -i sine=f={p['hum']}:d=30:sample_rate=48000 -f lavfi -i anoisesrc=a=0.01:d=30:r=48000 "
           "-filter_complex \"[1]volume=0.25[h];[0][h][2]amix=inputs=3:duration=first:normalize=0\" -ac 1 -c:a pcm_s16le hum.wav", d)
        os.remove(os.path.join(d, ".v.m4a"))
    truth["hum_hz"] = p["hum"]
    if need("big.mov"):  # oversized: ProRes + uncompressed audio
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d=5 -f lavfi -i sine=f=400:d=5:sample_rate=48000 "
           "-c:v prores_ks -profile:v 3 -c:a pcm_s24le -ac 2 big.mov", d)
    if need("version_a.mp4"):  # good quality
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s={p['s']}:r={r}:d=6 -f lavfi -i sine=f=410:d=6 -c:v libx264 -crf 18 "
           f"-pix_fmt yuv420p -ac 2 {A} version_a.mp4", d)
    if need("version_b.mp4"):  # same size on paper, but upscaled from 320 px at a starved bitrate
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s=320x180:r={r}:d=6 -f lavfi -i sine=f=410:d=6 -vf scale={w}:{h} "
           f"-c:v libx264 -b:v 150k -pix_fmt yuv420p -ac 2 -c:a aac -b:a 48k version_b.mp4", d)
    truth["better"] = "version_a.mp4"
    if need("lowres.mp4"):
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s=640x360:r={r}:d=8 -f lavfi -i sine=f=420:d=8 -ac 2 {V} {A} lowres.mp4", d)
    if need("webcam.mp4"):
        sh(f"ffmpeg -y -v error -f lavfi -i \"testsrc=s=640x480:r={r}:d={dur},hue=s=0\" -i narration.m4a {V} {A} -shortest webcam.mp4", d)
    if need("screenrec.mp4"):
        sh(f"ffmpeg -y -v error -f lavfi -i \"smptehdbars=s={p['s']}:r={r}:d={dur}\" {V} screenrec.mp4", d)
    # images
    if need("product.png"):
        sh(f"magick -size 900x700 xc:none -fill '{p['product']}' -draw 'roundrectangle 250,150 650,550 40,40' "
           "-fill white -draw 'circle 450,350 450,260' product.png", d)
    for i in range(1, 4):
        if need(f"page{i}.jpg"):
            sh(f"magick -size 1240x1754 xc:'{p['page_tint']}' -fill '#333' -pointsize 60 -annotate +120+200 'Page {i}' "
               f"-fill '#888' -draw 'rectangle 120,300 1120,{300 + 60 * (8 - i)}' -background '{p['page_tint']}' -rotate {i - 2}.5 "
               f"-quality 88 page{i}.jpg", d)
    truth["pages"] = 3
    if need("screenshot.png"):
        sh(f"magick -size 1440x900 xc:'#f8fafc' -fill '{p['product']}' -draw 'rectangle 0,0 1440,60' -fill '#334155' "
           "-draw 'rectangle 80,120 700,820' -fill '#cbd5e1' -draw 'rectangle 760,120 1360,460' screenshot.png", d)
    if need("cover.jpg"):
        sh(f"magick -size 1400x1400 gradient:'{p['cut_colors'][1]}'-'{p['cut_colors'][0]}' -fill white -pointsize 110 "
           "-gravity center -annotate +0+0 'The Show' -quality 90 cover.jpg", d)
    if need("photos"):
        os.makedirs(os.path.join(d, "photos"))
        for i, c in enumerate(p["cut_colors"] + ["white", "black"]):
            sh(f"magick -size {1200 + 100 * i}x{900 - 50 * i} gradient:{c}-gray -quality 90 photos/img_{i + 1:02d}.jpg", d)
    import checks
    truth["speech"] = checks.speech_segments(os.path.join(d, "narration.m4a"))  # measured, not the nominal slots
    truth["pauses"] = [[a[1], b[0]] for a, b in zip(truth["speech"], truth["speech"][1:])]
    # photo batch with camera metadata: dates, some with GPS (for rename-by-date, date shift, privacy tasks)
    dates = p["photo_dates"]
    for i, dt in enumerate(dates):
        f = os.path.join(d, "photos", f"img_{i + 1:02d}.jpg")
        gps = ["-GPSLatitude=12.97", "-GPSLatitudeRef=N", "-GPSLongitude=77.59", "-GPSLongitudeRef=E"] if i % 2 == 0 else []
        subprocess.run(["exiftool", "-overwrite_original", f"-DateTimeOriginal={dt}", f"-CreateDate={dt}", "-Make=Canon",
                        "-Model=EOS R6", "-Artist=Jamie Rivera", *gps, f], check=True, capture_output=True)
    truth["photo_dates"] = dates
    if need("captions.srt"):
        sp = truth["speech"]
        ts = lambda t: f"00:00:{int(t):02d},{int(round((t % 1) * 1000)):03d}"
        open(os.path.join(d, "captions.srt"), "w").write("\n".join(
            f"{i + 1}\n{ts(a)} --> {ts(b)}\n{line}\n" for i, ((a, b), line) in enumerate(zip(sp, p["lines"]))))
    if need("badge.png"):
        sh(f"magick -size 240x240 xc:none -fill '{p['product']}' -draw 'circle 120,120 120,10' -fill white -pointsize 90 "
           "-gravity center -annotate +0+0 'CW' badge.png", d)
    if need("odd.webm"):  # odd dimensions: H.264 in yuv420p needs even sizes, so a plain re-encode to mp4 fails
        sh(f"ffmpeg -y -v error -f lavfi -i testsrc2=s=1280x720:r={r}:d=8 -f lavfi -i sine=f=430:d=8 -vf scale={p['odd'].replace('x', ':')} -c:v libvpx-vp9 "
           "-deadline realtime -cpu-used 8 -b:v 1M -ac 2 -c:a libopus odd.webm", d)
    json.dump(truth, open(os.path.join(d, "truth.json"), "w"), indent=1)
    meta_path = os.path.join(d, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    for name in sorted(os.listdir(d)):
        fp = os.path.join(d, name)
        if os.path.isfile(fp) and name not in meta and not name.endswith(".json") and not name.startswith("."):
            meta[name] = describe(fp)
    json.dump(meta, open(meta_path, "w"), indent=1)
    return truth


if __name__ == "__main__":
    for k, p in SETS.items():
        print(k, json.dumps(build(p)))
