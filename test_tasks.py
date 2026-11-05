"""Held-out test set v2: ~200 hand-written tasks over the 46 task types (A-H), on the TEST fixtures (fixtures/test/,
different voices, sizes, frame rates and timings from the training fixtures). Never used to generate training data.

Each task: (id, type, request, {in-name: fixture}, output specs | full spec with "answer", [reference commands / REPLY:]).
The reference proves the check is passable; an empty out/ must fail it.

    python3 test_tasks.py --validate [--only a,b]   # every reference passes, every empty output fails
    python3 test_tasks.py --export                  # data/test_v2/test.jsonl
Test-fixture ground truth: fixtures/test/truth.json (speech [[0,2.79],[6.04,8.23],[12.01,14.46]], cuts 3/7.5/12 in
navy/crimson/olive/silver, 25 fps 1920x1080, sync -250 ms, hum 60 Hz, music 247 Hz).
"""
import argparse
import os

import ood_tasks

t = lambda name: f"test/{name}"  # test fixture
MP4 = lambda **k: {"match": "*.mp4", **k}
FHD = [1920, 1080]
ENC = "-c:v libx264 -preset veryfast -crf 22 -pix_fmt yuv420p -c:a aac -b:a 160k"

T = [
    # ── A1. end card / thank-you slide with voiceover ──
    ("a01", "A1 end card + voiceover",
     "add endcard.png at the end of presentation.mp4, shown for as long as outro_vo.m4a plays, with that recording as the sound",
     {"presentation.mp4": t("talk.mp4"), "endcard.png": t("thanks.png"), "outro_vo.m4a": t("voice_m.m4a")},
     [MP4(wh=FHD, dur=[18.1, 19.4], frame_like=[[2, "in/presentation.mp4@2"], [-0.4, "in/endcard.png"]],
          frame_unlike=[[2, "in/endcard.png"]], loud_at=[[16.3, 18.0, -45]])],
     ["ffmpeg -i in/presentation.mp4 -loop 1 -framerate 25 -i in/endcard.png -i in/outro_vo.m4a -filter_complex "
      "\"[1:v]scale=1920:1080,format=yuv420p,setsar=1,trim=duration=2.6[s];[2:a]aresample=48000,aformat=channel_layouts=stereo,"
      f"apad=whole_dur=2.6[sa];[0:v][0:a][s][sa]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {ENC} out/presentation_final.mp4"]),
    ("a02", "A1 end card + voiceover",
     "after the video ends, hold title.png on screen for 3 more seconds. silence is fine for that part. file: recap.mp4",
     {"recap.mp4": t("talk.mp4"), "title.png": t("title.png")},
     [MP4(wh=FHD, has_audio=True, dur=[18.7, 19.4], frame_like=[[-0.5, "in/title.png"], [5, "in/recap.mp4@5"]],
          loud_at=[[0.3, 2.5, -45]])],
     ["ffmpeg -i in/recap.mp4 -loop 1 -framerate 25 -t 3 -i in/title.png -f lavfi -t 3 -i anullsrc=r=48000:cl=stereo "
      "-filter_complex \"[1:v]scale=1920:1080,format=yuv420p,setsar=1[s];[0:v][0:a][s][2:a]concat=n=2:v=1:a=1[v][a]\" "
      f"-map \"[v]\" -map \"[a]\" {ENC} out/recap_end.mp4"]),
    ("a03", "A1 end card + voiceover",
     "finish webinar.mp4 with the thank you slide (thank_you.png) while thanks.m4a plays. call it final.mp4",
     {"webinar.mp4": t("talk.mp4"), "thank_you.png": t("thanks.png"), "thanks.m4a": t("voice_f.m4a")},
     [{"match": "final.mp4", "wh": FHD, "dur": [17.9, 19.4], "frame_like": [[-0.3, "in/thank_you.png"]],
       "frame_unlike": [[1, "in/thank_you.png"]], "loud_at": [[16.2, 17.9, -45]]}],
     ["ffmpeg -i in/webinar.mp4 -loop 1 -framerate 25 -i in/thank_you.png -i in/thanks.m4a -filter_complex "
      "\"[1:v]scale=1920:1080,format=yuv420p,setsar=1,trim=duration=2.3[s];[2:a]aresample=48000,aformat=channel_layouts=stereo,"
      f"apad=whole_dur=2.3[sa];[0:v][0:a][s][sa]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {ENC} out/final.mp4"]),
    ("a04", "A1 end card + voiceover",
     "add thanks.png as a closing slide to interview.mp4 with signoff.m4a as the voice. keep the interview's size",
     {"interview.mp4": t("webcam.mp4"), "thanks.png": t("thanks.png"), "signoff.m4a": t("voice_f.m4a")},
     [MP4(wh=[640, 480], dur=[17.9, 19.0], frame_like=[[3, "in/interview.mp4@3"]],
          frame_unlike=[[-0.4, "in/interview.mp4@15.5"]], loud_at=[[16.2, 17.9, -45]])],
     ["ffmpeg -i in/interview.mp4 -loop 1 -framerate 25 -i in/thanks.png -i in/signoff.m4a -filter_complex "
      "\"[1:v]scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,format=yuv420p,setsar=1,"
      "trim=duration=2.3[s];[2:a]aresample=48000,aformat=channel_layouts=stereo,apad=whole_dur=2.3[sa];"
      f"[0:v][0:a][s][sa]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {ENC} out/interview_end.mp4"]),

    # ── A2. intro title card from text only ──
    ("a05", "A2 text intro card",
     "start demo.mp4 with a 3 second black title card that says 'Product Demo' in white, then the video",
     {"demo.mp4": t("talk.mp4")},
     [MP4(wh=FHD, dur=[18.7, 19.4], first_dark=True, frame_like=[[5, "in/demo.mp4@2"], [12, "in/demo.mp4@9"]], has_audio=True)],
     ["ffmpeg -f lavfi -i color=black:s=1920x1080:r=25:d=3 -f lavfi -t 3 -i anullsrc=r=48000:cl=stereo -i in/demo.mp4 "
      "-filter_complex \"[0:v]drawtext=text='Product Demo':fontcolor=white:fontsize=96:x=(w-text_w)/2:y=(h-text_h)/2,"
      f"format=yuv420p,setsar=1[t];[t][1:a][2:v][2:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {ENC} out/demo_intro.mp4"]),
    ("a06", "A2 text intro card",
     "put a 2s intro that says \"Episode 4: Editing\" on a dark blue background before screen_capture.mp4",
     {"screen_capture.mp4": t("screenrec.mp4")},
     [MP4(wh=FHD, dur=[17.7, 18.4], frame_like=[[4, "in/screen_capture.mp4@2"]], frame_unlike=[[1, "in/screen_capture.mp4@1"]])],
     ["ffmpeg -f lavfi -i color=0x1e3a8a:s=1920x1080:r=25:d=2 -i in/screen_capture.mp4 -filter_complex "
      "\"[0:v]drawtext=text='Episode 4\\: Editing':fontcolor=white:fontsize=90:x=(w-text_w)/2:y=(h-text_h)/2,format=yuv420p,"
      "setsar=1[t];[1:v]format=yuv420p,setsar=1[m];[t][m]concat=n=2:v=1:a=0[v]\" -map \"[v]\" -c:v libx264 -crf 22 out/episode4.mp4"]),
    ("a07", "A2 text intro card",
     "I want a title card 'Quarterly Update' for about 3 seconds at the start of update.mp4, fading into the video",
     {"update.mp4": t("talk.mp4")},
     [MP4(wh=FHD, dur=[17.4, 19.5], has_audio=True, frame_like=[[-1, "in/update.mp4@15"]], frame_unlike=[[0.8, "in/update.mp4@0.8"]])],
     ["ffmpeg -f lavfi -i color=black:s=1920x1080:r=25:d=3 -f lavfi -t 3 -i anullsrc=r=48000:cl=stereo -i in/update.mp4 "
      "-filter_complex \"[0:v]drawtext=text='Quarterly Update':fontcolor=white:fontsize=96:x=(w-text_w)/2:y=(h-text_h)/2,"
      "format=yuv420p,setsar=1,settb=AVTB,fps=25[t];[2:v]format=yuv420p,setsar=1,settb=AVTB,fps=25[m];"
      "[t][m]xfade=transition=fade:duration=1:offset=2[v];[1:a][2:a]acrossfade=d=1[a]\" -map \"[v]\" -map \"[a]\" "
      f"{ENC} out/update_titled.mp4"]),
    ("a08", "A2 text intro card",
     "add a 3-second title 'Interview with Sam' before sam.mp4 and make sure the sound still lines up with the picture",
     {"sam.mp4": t("webcam.mp4")},
     [MP4(wh=[640, 480], dur=[18.7, 19.4], frame_like=[[6, "in/sam.mp4@3"]], loud_at=[[3.3, 5.5, -45]], quiet_at=[[0.3, 2.7, -55]])],
     ["ffmpeg -f lavfi -i color=black:s=640x480:r=25:d=3 -f lavfi -t 3 -i anullsrc=r=48000:cl=stereo -i in/sam.mp4 "
      "-filter_complex \"[0:v]drawtext=text='Interview with Sam':fontcolor=white:fontsize=40:x=(w-text_w)/2:y=(h-text_h)/2,"
      f"format=yuv420p,setsar=1[t];[2:v]setsar=1[m];[t][1:a][m][2:a]concat=n=2:v=1:a=1[v][a]\" -map \"[v]\" -map \"[a]\" {ENC} out/sam_titled.mp4"]),

    # ── A3. insert a clip at a timestamp ──
    ("a09", "A3 insert at timestamp",
     "insert ad.mp4 into main.mp4 at 0:07.5 (right where the red part ends), then carry on with the rest of main",
     {"main.mp4": t("cuts.mp4"), "ad.mp4": t("version_a.mp4")},
     [MP4(wh=FHD, dur=[22.6, 23.4], frame_like=[[6, "in/main.mp4@6"], [8.5, "in/ad.mp4@1"], [15, "in/main.mp4@9"]])],
     ["ffmpeg -i in/main.mp4 -i in/ad.mp4 -filter_complex \"[0:v]trim=0:7.5,setpts=PTS-STARTPTS[a];[0:a]atrim=0:7.5,asetpts=PTS-STARTPTS[aa];"
      "[0:v]trim=7.5,setpts=PTS-STARTPTS[b];[0:a]atrim=7.5,asetpts=PTS-STARTPTS[ba];[1:v]setsar=1[m];[1:a]aresample=44100[ma];"
      f"[a][aa][m][ma][b][ba]concat=n=3:v=1:a=1[v][o]\" -map \"[v]\" -map \"[o]\" {ENC} out/main_with_ad.mp4"]),
    ("a10", "A3 insert at timestamp",
     "at the 8 second mark of lecture.mp4, show sponsor.png for 4 seconds (no sound), then continue the lecture",
     {"lecture.mp4": t("talk.mp4"), "sponsor.png": t("title.png")},
     [MP4(wh=FHD, has_audio=True, dur=[19.7, 20.4], frame_like=[[10, "in/sponsor.png"], [14, "in/lecture.mp4@10"], [4, "in/lecture.mp4@4"]])],
     ["ffmpeg -i in/lecture.mp4 -loop 1 -framerate 25 -t 4 -i in/sponsor.png -f lavfi -t 4 -i anullsrc=r=48000:cl=stereo "
      "-filter_complex \"[0:v]trim=0:8,setpts=PTS-STARTPTS[a];[0:a]atrim=0:8,asetpts=PTS-STARTPTS[aa];[0:v]trim=8,setpts=PTS-STARTPTS[b];"
      "[0:a]atrim=8,asetpts=PTS-STARTPTS[ba];[1:v]scale=1920:1080,format=yuv420p,setsar=1[s];"
      f"[a][aa][s][2:a][b][ba]concat=n=3:v=1:a=1[v][o]\" -map \"[v]\" -map \"[o]\" {ENC} out/lecture_sponsor.mp4"]),
    ("a11", "A3 insert at timestamp",
     "cut screen.mp4 into talk.mp4 at 6 seconds. screen.mp4 has no audio, keep everything in sync after it",
     {"talk.mp4": t("talk.mp4"), "screen.mp4": t("screenrec.mp4")},
     [MP4(wh=FHD, dur=[31.6, 32.4], frame_like=[[10, "in/screen.mp4@4"], [25, "in/talk.mp4@9"]], loud_at=[[22.3, 24.0, -45]])],
     ["ffmpeg -i in/talk.mp4 -i in/screen.mp4 -f lavfi -t 16 -i anullsrc=r=48000:cl=stereo -filter_complex "
      "\"[0:v]trim=0:6,setpts=PTS-STARTPTS[a];[0:a]atrim=0:6,asetpts=PTS-STARTPTS[aa];[0:v]trim=6,setpts=PTS-STARTPTS[b];"
      "[0:a]atrim=6,asetpts=PTS-STARTPTS[ba];[1:v]setsar=1[m];[a][aa][m][2:a][b][ba]concat=n=3:v=1:a=1[v][o]\" "
      f"-map \"[v]\" -map \"[o]\" {ENC} out/talk_with_screen.mp4"]),
    ("a12", "A3 insert at timestamp",
     "drop selfie.mp4 into the middle of montage.mp4 at 0:03. the selfie is a different size, make it fit",
     {"montage.mp4": t("cuts.mp4"), "selfie.mp4": t("webcam.mp4")},
     [MP4(wh=FHD, dur=[32.6, 33.4], frame_like=[[2, "in/montage.mp4@2"], [25, "in/montage.mp4@9"]])],
     ["ffmpeg -i in/montage.mp4 -i in/selfie.mp4 -filter_complex \"[0:v]trim=0:3,setpts=PTS-STARTPTS[a];[0:a]atrim=0:3,asetpts=PTS-STARTPTS[aa];"
      "[0:v]trim=3,setpts=PTS-STARTPTS[b];[0:a]atrim=3,asetpts=PTS-STARTPTS[ba];[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,"
      "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25[m];[1:a]aresample=44100[ma];"
      f"[a][aa][m][ma][b][ba]concat=n=3:v=1:a=1[v][o]\" -map \"[v]\" -map \"[o]\" {ENC} out/montage_selfie.mp4"]),

    # ── A4. cut out several sections ──
    ("a13", "A4 remove sections",
     "remove 0:03-0:05 and 0:09-0:11 from vlog.mp4 and join what's left",
     {"vlog.mp4": t("talk.mp4")},
     [MP4(wh=FHD, has_audio=True, dur=[11.7, 12.3], frame_like=[[2, "in/vlog.mp4@2"], [3.5, "in/vlog.mp4@5.5"], [8, "in/vlog.mp4@12"]])],
     ["ffmpeg -i in/vlog.mp4 -vf \"select='not(between(t,3,5)+between(t,9,11))',setpts=N/FRAME_RATE/TB\" "
      f"-af \"aselect='not(between(t,3,5)+between(t,9,11))',asetpts=N/SR/TB\" {ENC} out/vlog_cut.mp4"]),
    ("a14", "A4 remove sections",
     "cut out the red section of colors.mp4 entirely",
     {"colors.mp4": t("cuts.mp4")},
     [MP4(wh=FHD, dur=[12.2, 12.8], frame_like=[[1, "in/colors.mp4@1"], [4, "in/colors.mp4@8.5"], [8, "in/colors.mp4@12.5"]])],
     ["ffmpeg -i in/colors.mp4 -vf \"select='not(between(t,3,7.5))',setpts=N/FRAME_RATE/TB\" "
      f"-af \"aselect='not(between(t,3,7.5))',asetpts=N/SR/TB\" {ENC} out/colors_no_red.mp4"]),
    ("a15", "A4 remove sections",
     "keep only 0-4s and 10-14s of clip.mp4, joined together",
     {"clip.mp4": t("talk.mp4")},
     [MP4(wh=FHD, has_audio=True, dur=[7.7, 8.3], frame_like=[[1, "in/clip.mp4@1"], [6, "in/clip.mp4@12"]])],
     ["ffmpeg -i in/clip.mp4 -vf \"select='between(t,0,4)+between(t,10,14)',setpts=N/FRAME_RATE/TB\" "
      f"-af \"aselect='between(t,0,4)+between(t,10,14)',asetpts=N/SR/TB\" {ENC} out/clip_parts.mp4"]),
    ("a16", "A4 remove sections",
     "drop the first 2 seconds, the last 3 seconds, and the bit between 8 and 9 seconds of speech.mp4",
     {"speech.mp4": t("talk.mp4")},
     [MP4(wh=FHD, has_audio=True, dur=[9.7, 10.3], frame_like=[[1, "in/speech.mp4@3"], [7, "in/speech.mp4@10"]])],
     ["ffmpeg -i in/speech.mp4 -vf \"select='between(t,2,8)+between(t,9,13)',setpts=N/FRAME_RATE/TB\" "
      f"-af \"aselect='between(t,2,8)+between(t,9,13)',asetpts=N/SR/TB\" {ENC} out/speech_trimmed.mp4"]),

    # ── A5. crossfade transitions ──
    ("a17", "A5 crossfade join",
     "join part1.mp4 and part2.mp4 with a 1 second crossfade between them, sound too",
     {"part1.mp4": t("talk.mp4"), "part2.mp4": t("cuts.mp4")},
     [MP4(wh=FHD, has_audio=True, dur=[31.6, 32.4], frame_like=[[5, "in/part1.mp4@5"], [25, "in/part2.mp4@10"]],
          frame_unlike=[[15.5, "in/part1.mp4@15.5", 0.03], [15.5, "in/part2.mp4@0.5", 0.03]])],
     ["ffmpeg -i in/part1.mp4 -i in/part2.mp4 -filter_complex \"[0:v]settb=AVTB,fps=25,format=yuv420p[a];[1:v]settb=AVTB,fps=25,"
      "format=yuv420p[b];[a][b]xfade=transition=fade:duration=1:offset=15[v];[0:a]aresample=48000[x];[1:a]aresample=48000[y];"
      f"[x][y]acrossfade=d=1[o]\" -map \"[v]\" -map \"[o]\" {ENC} out/joined.mp4"]),
    ("a18", "A5 crossfade join",
     "dissolve from take1.mp4 into take2.mp4, half a second",
     {"take1.mp4": t("version_a.mp4"), "take2.mp4": t("cuts.mp4")},
     [MP4(wh=FHD, dur=[22.2, 22.8], frame_like=[[2, "in/take1.mp4@2"], [20, "in/take2.mp4@14.5"]])],
     ["ffmpeg -i in/take1.mp4 -i in/take2.mp4 -filter_complex \"[0:v]settb=AVTB,fps=25,format=yuv420p[a];[1:v]settb=AVTB,fps=25,"
      "format=yuv420p[b];[a][b]xfade=transition=dissolve:duration=0.5:offset=5.5[v];[0:a][1:a]acrossfade=d=0.5[o]\" "
      f"-map \"[v]\" -map \"[o]\" {ENC} out/dissolve.mp4"]),
    ("a19", "A5 crossfade join",
     "stitch a.mp4, b.mp4 and c.mp4 together in that order with a 1s fade between each",
     {"a.mp4": t("cuts.mp4"), "b.mp4": t("version_a.mp4"), "c.mp4": t("version_b.mp4")},
     [MP4(wh=FHD, dur=[26.6, 27.4], frame_like=[[2, "in/a.mp4@2"], [19, "in/b.mp4@3"]])],
     ["ffmpeg -i in/a.mp4 -i in/b.mp4 -i in/c.mp4 -filter_complex \"[0:v]settb=AVTB,fps=25,format=yuv420p[a];[1:v]settb=AVTB,fps=25,"
      "format=yuv420p[b];[2:v]settb=AVTB,fps=25,format=yuv420p[c];[a][b]xfade=duration=1:offset=16[ab];[ab][c]xfade=duration=1:offset=21[v];"
      f"[0:a][1:a]acrossfade=d=1[x];[x][2:a]acrossfade=d=1[o]\" -map \"[v]\" -map \"[o]\" {ENC} out/abc.mp4"]),
    ("a20", "A5 crossfade join",
     "join intro.m4a and bed.mp3 into one track with a 3 second crossfade, mp3 please",
     {"intro.m4a": t("narration.m4a"), "bed.mp3": t("music_bed.mp3")},
     [{"match": "*.mp3", "has_audio": True, "dur": [52.5, 53.5]}],
     ["ffmpeg -i in/intro.m4a -i in/bed.mp3 -filter_complex \"[0:a]aresample=44100[a];[a][1:a]acrossfade=d=3\" -c:a libmp3lame "
      "-b:a 192k out/joined.mp3"]),

    # ── A6. voiceover at a timestamp (original lowered) ──
    ("a21", "A6 voiceover at time",
     "add narration.m4a as a voiceover starting at 0:05 in tutorial.mp4 (the video is silent right now)",
     {"tutorial.mp4": t("screenrec.mp4"), "narration.m4a": t("voice_f.m4a")},
     [MP4(wh=FHD, has_audio=True, dur=[15.7, 16.4], loud_at=[[5.3, 7.0, -45]], quiet_at=[[0.5, 4.5, -60]])],
     ["ffmpeg -i in/tutorial.mp4 -i in/narration.m4a -filter_complex \"[1:a]adelay=5000:all=1,apad[a]\" -map 0:v -map \"[a]\" "
      "-c:v copy -c:a aac -shortest out/tutorial_vo.mp4"]),
    ("a22", "A6 voiceover at time",
     "put my_voice.m4a over clip.mp4 starting at 6 seconds, and turn the original sound way down while I'm talking",
     {"clip.mp4": t("cuts.mp4"), "my_voice.m4a": t("voice_m.m4a")},
     [MP4(wh=FHD, dur=[16.7, 17.4], loud_at=[[6.3, 8.2, -40]], band_drop=[440, [6.4, 8.2], [1, 4], 8])],
     ["ffmpeg -i in/clip.mp4 -i in/my_voice.m4a -filter_complex \"[0:a]volume='if(between(t,6,8.7),0.15,1)':eval=frame[bg];"
      "[1:a]adelay=6000:all=1[v];[bg][v]amix=inputs=2:duration=first:normalize=0[a]\" -map 0:v -map \"[a]\" -c:v copy -c:a aac out/clip_vo.mp4"]),
    ("a23", "A6 voiceover at time",
     "replace the audio between 0:04 and 0:07 of promo.mp4 with line.m4a, keep the rest of the original sound",
     {"promo.mp4": t("cuts.mp4"), "line.m4a": t("voice_f.m4a")},
     [MP4(wh=FHD, dur=[16.7, 17.4], loud_at=[[4.3, 6.0, -40], [9, 12, -40]], band_drop=[440, [4.3, 6.8], [1, 3.5], 20])],
     ["ffmpeg -i in/promo.mp4 -i in/line.m4a -filter_complex \"[0:a]volume='if(between(t,4,7),0,1)':eval=frame[bg];"
      "[1:a]adelay=4000:all=1[v];[bg][v]amix=inputs=2:duration=first:normalize=0[a]\" -map 0:v -map \"[a]\" -c:v copy -c:a aac out/promo_vo.mp4"]),
    ("a24", "A6 voiceover at time",
     "the last 3 seconds of teaser.mp4 need vo.m4a as narration on top of the existing sound",
     {"teaser.mp4": t("version_a.mp4"), "vo.m4a": t("voice_m.m4a")},
     [MP4(wh=FHD, dur=[5.8, 6.3], loud_at=[[3.3, 5.4, -40]])],
     ["ffmpeg -i in/teaser.mp4 -i in/vo.m4a -filter_complex \"[1:a]adelay=3000:all=1[v];[0:a][v]amix=inputs=2:duration=first:normalize=0[a]\" "
      "-map 0:v -map \"[a]\" -c:v copy -c:a aac out/teaser_vo.mp4"]),

    # ── A7. speed ramp (only part sped up / slowed) ──
    ("a25", "A7 partial speed",
     "speed up colors.mp4 2x, but only from 0:07.5 onwards. no audio needed",
     {"colors.mp4": t("cuts.mp4")},
     [MP4(wh=FHD, dur=[11.9, 12.6], frame_like=[[5, "in/colors.mp4@5"], [9, "in/colors.mp4@10.5"], [11, "in/colors.mp4@14.5"]])],
     ["ffmpeg -i in/colors.mp4 -filter_complex \"[0:v]trim=0:7.5,setpts=PTS-STARTPTS[a];[0:v]trim=7.5,setpts=(PTS-STARTPTS)/2[b];"
      "[a][b]concat=n=2:v=1:a=0[v]\" -map \"[v]\" -an -c:v libx264 -crf 22 out/colors_ramp.mp4"]),
    ("a26", "A7 partial speed",
     "keep the first 3 seconds of shot.mp4 at normal speed and play the rest in half-speed slow motion, video only",
     {"shot.mp4": t("cuts.mp4")},
     [MP4(wh=FHD, dur=[30.5, 31.5], frame_like=[[2, "in/shot.mp4@2"], [10, "in/shot.mp4@6.5"], [20, "in/shot.mp4@11.5"]])],
     ["ffmpeg -i in/shot.mp4 -filter_complex \"[0:v]trim=0:3,setpts=PTS-STARTPTS[a];[0:v]trim=3,setpts=(PTS-STARTPTS)*2[b];"
      "[a][b]concat=n=2:v=1:a=0[v]\" -map \"[v]\" -an -c:v libx264 -crf 22 out/shot_slowmo.mp4"]),
    ("a27", "A7 partial speed",
     "fast forward the boring middle of meeting.mp4 (4s to 12s) at 4x speed, sound included, rest normal",
     {"meeting.mp4": t("talk.mp4")},
     [MP4(wh=FHD, has_audio=True, dur=[9.6, 10.4], frame_like=[[2, "in/meeting.mp4@2"], [5, "in/meeting.mp4@8"], [8, "in/meeting.mp4@14"]])],
     ["ffmpeg -i in/meeting.mp4 -filter_complex \"[0:v]trim=0:4,setpts=PTS-STARTPTS[a];[0:a]atrim=0:4,asetpts=PTS-STARTPTS[aa];"
      "[0:v]trim=4:12,setpts=(PTS-STARTPTS)/4[b];[0:a]atrim=4:12,asetpts=PTS-STARTPTS,atempo=2,atempo=2[ba];"
      "[0:v]trim=12,setpts=PTS-STARTPTS[c];[0:a]atrim=12,asetpts=PTS-STARTPTS[ca];[a][aa][b][ba][c][ca]concat=n=3:v=1:a=1[v][o]\" "
      f"-map \"[v]\" -map \"[o]\" {ENC} out/meeting_ff.mp4"]),
    ("a28", "A7 partial speed",
     "make the part from 0:03 to 0:07.5 of reel.mp4 three times faster, leave everything else as is (video only is ok)",
     {"reel.mp4": t("cuts.mp4")},
     [MP4(wh=FHD, dur=[13.7, 14.3], frame_like=[[2, "in/reel.mp4@2"], [6, "in/reel.mp4@9"], [12, "in/reel.mp4@15"]])],
     ["ffmpeg -i in/reel.mp4 -filter_complex \"[0:v]trim=0:3,setpts=PTS-STARTPTS[a];[0:v]trim=3:7.5,setpts=(PTS-STARTPTS)/3[b];"
      "[0:v]trim=7.5,setpts=PTS-STARTPTS[c];[a][b][c]concat=n=3:v=1:a=0[v]\" -map \"[v]\" -an -c:v libx264 -crf 22 out/reel_ramp.mp4"]),

    # ── A8. combined edits (the live failure: speed + pitch) ──
    ("a29", "A8 combined edits",
     "speed up vlog.mp4 to 1.5x and make the voice higher pitched, like a chipmunk-ish female voice. keep it smooth",
     {"vlog.mp4": t("talk.mp4")},
     [MP4(wh=FHD, dur=[10.3, 11.0], fps=25, cfr=True, pitch_ratio=["in/vlog.mp4", 1.2, 1.8])],
     ["ffmpeg -i in/vlog.mp4 -filter_complex \"[0:v]setpts=PTS/1.5[v];[0:a]rubberband=tempo=1.5:pitch=1.4[a]\" -map \"[v]\" -map \"[a]\" "
      f"-r 25 {ENC} out/vlog_fast_high.mp4"]),
    ("a30", "A8 combined edits",
     "take 0:02 to 0:10 of skate.mp4, play it at double speed and turn it into a gif",
     {"skate.mp4": t("talk.mp4")},
     [{"match": "*.gif", "dur": [3.5, 4.5]}],
     ["ffmpeg -ss 2 -to 10 -i in/skate.mp4 -vf \"setpts=PTS/2,fps=12,scale=480:-1:flags=lanczos\" out/skate.gif"]),
    ("a31", "A8 combined edits",
     "make a vertical 9:16 version of the first 10 seconds of podcast_clip.mp4 at youtube loudness",
     {"podcast_clip.mp4": t("talk.mp4")},
     [MP4(aspect=[9, 16], min_short=600, dur=[9.5, 10.5], lufs=[-16, -12])],
     ["ffmpeg -t 10 -i in/podcast_clip.mp4 -vf crop=608:1080,scale=1080:1920 -af loudnorm=I=-14:TP=-1.5 -ar 48000 "
      f"{ENC} out/podcast_vertical.mp4"]),
    ("a32", "A8 combined edits",
     "slow troll.mp4 down to 0.8x and make my voice deeper, like a giant",
     {"troll.mp4": t("talk.mp4")},
     [MP4(wh=FHD, dur=[19.5, 20.5], pitch_ratio=["in/troll.mp4", 0.55, 0.87])],
     ["ffmpeg -i in/troll.mp4 -filter_complex \"[0:v]setpts=PTS/0.8[v];[0:a]rubberband=tempo=0.8:pitch=0.72[a]\" -map \"[v]\" -map \"[a]\" "
      f"{ENC} out/troll_giant.mp4"]),

    # ── A9. picture-in-picture ──
    ("a33", "A9 picture-in-picture",
     "put my webcam (cam.mp4) in the bottom right corner of my screen recording screen.mp4, small, with the webcam's audio",
     {"screen.mp4": t("screenrec.mp4"), "cam.mp4": t("webcam.mp4")},
     [MP4(wh=FHD, has_audio=True, dur=[15.7, 16.4], frame_like=[[5, "in/screen.mp4@5", 0.3]], frame_unlike=[[5, "in/screen.mp4@5", 0.01]],
          loud_at=[[0.3, 2.5, -45]])],
     ["ffmpeg -i in/screen.mp4 -i in/cam.mp4 -filter_complex \"[1:v]scale=480:-2[c];[0:v][c]overlay=W-w-30:H-h-30[v]\" "
      f"-map \"[v]\" -map 1:a {ENC} out/screen_pip.mp4"]),
    ("a34", "A9 picture-in-picture",
     "overlay facecam.mp4 at the top left of gameplay.mp4, about a quarter of the width",
     {"gameplay.mp4": t("cuts.mp4"), "facecam.mp4": t("webcam.mp4")},
     [MP4(wh=FHD, dur=[15.7, 17.4], frame_like=[[2, "in/gameplay.mp4@2", 0.3]], frame_unlike=[[2, "in/gameplay.mp4@2", 0.01]])],
     ["ffmpeg -i in/gameplay.mp4 -i in/facecam.mp4 -filter_complex \"[1:v]scale=480:-2[c];[0:v][c]overlay=20:20:shortest=1[v]\" "
      f"-map \"[v]\" -map 0:a {ENC} out/gameplay_cam.mp4"]),
    ("a35", "A9 picture-in-picture",
     "picture in picture: lecture.mp4 full screen and slides.mp4 small in the top right, keep lecture audio",
     {"lecture.mp4": t("talk.mp4"), "slides.mp4": t("screenrec.mp4")},
     [MP4(wh=FHD, has_audio=True, dur=[15.7, 16.4], frame_like=[[4, "in/lecture.mp4@4", 0.3]], frame_unlike=[[4, "in/lecture.mp4@4", 0.01]],
          loud_at=[[0.3, 2.5, -45]])],
     ["ffmpeg -i in/lecture.mp4 -i in/slides.mp4 -filter_complex \"[1:v]scale=480:-2[c];[0:v][c]overlay=W-w-20:20[v]\" "
      f"-map \"[v]\" -map 0:a {ENC} out/lecture_pip.mp4"]),

    # ── A10. side-by-side comparison ──
    ("a36", "A10 side by side",
     "show before.mp4 and after.mp4 side by side in one video so I can compare them",
     {"before.mp4": t("version_b.mp4"), "after.mp4": t("version_a.mp4")},
     [MP4(aspect=[32, 9], min_short=360, dur=[5.7, 6.3])],
     ["ffmpeg -i in/before.mp4 -i in/after.mp4 -filter_complex \"[0:v][1:v]hstack[v]\" -map \"[v]\" -map 0:a -c:v libx264 -crf 22 "
      "-c:a aac out/compare.mp4"]),
    ("a37", "A10 side by side",
     "stack old.mp4 on top of new.mp4 (vertically), label them OLD and NEW",
     {"old.mp4": t("version_b.mp4"), "new.mp4": t("version_a.mp4")},
     [MP4(aspect=[16, 18], min_short=640, dur=[5.7, 6.3])],
     ["ffmpeg -i in/old.mp4 -i in/new.mp4 -filter_complex \"[0:v]drawtext=text=OLD:fontcolor=white:fontsize=72:x=40:y=40[a];"
      "[1:v]drawtext=text=NEW:fontcolor=white:fontsize=72:x=40:y=40[b];[a][b]vstack[v]\" -map \"[v]\" -c:v libx264 -crf 22 out/old_new.mp4"]),
    ("a38", "A10 side by side",
     "make a split screen of raw.mp4 and graded.mp4, left and right, at normal 1080p width",
     {"raw.mp4": t("dark.mp4"), "graded.mp4": t("talk.mp4")},
     [MP4(wh=[1920, 540], dur=[9.7, 10.3])],
     ["ffmpeg -i in/raw.mp4 -i in/graded.mp4 -filter_complex \"[0:v]scale=960:540[a];[1:v]scale=960:540[b];[a][b]hstack=shortest=1[v]\" "
      "-map \"[v]\" -c:v libx264 -crf 22 out/split.mp4"]),

    # ── A11. freeze frame ──
    ("a39", "A11 freeze frame",
     "freeze the picture at 0:05 of dance.mp4 for 3 seconds, then let it continue",
     {"dance.mp4": t("talk.mp4")},
     [MP4(wh=FHD, dur=[18.7, 19.4], freeze=[5.2, 7.8], frame_like=[[3, "in/dance.mp4@3"], [10, "in/dance.mp4@7"]])],
     ["ffmpeg -i in/dance.mp4 -filter_complex \"[0:v]trim=0:5,setpts=PTS-STARTPTS[a];[0:v]trim=5,setpts=PTS-STARTPTS[b];"
      "[b]tpad=start_duration=3:start_mode=clone[bf];[a][bf]concat=n=2:v=1:a=0[v];[0:a]atrim=0:5,asetpts=PTS-STARTPTS[x];"
      "[0:a]atrim=5,asetpts=PTS-STARTPTS,adelay=3000:all=1[y];[x][y]concat=n=2:v=0:a=1[o]\" -map \"[v]\" -map \"[o]\" "
      f"{ENC} out/dance_freeze.mp4"]),
    ("a40", "A11 freeze frame",
     "hold the last frame of ending.mp4 for an extra 2 seconds",
     {"ending.mp4": t("talk.mp4")},
     [MP4(wh=FHD, dur=[17.7, 18.4], freeze=[16.2, 17.8], frame_like=[[5, "in/ending.mp4@5"]])],
     ["ffmpeg -i in/ending.mp4 -vf tpad=stop_duration=2:stop_mode=clone -af apad=pad_dur=2 " + ENC + " out/ending_hold.mp4"]),
    ("a41", "A11 freeze frame",
     "start clip.mp4 with its first frame frozen for 2 seconds before it plays",
     {"clip.mp4": t("talk.mp4")},
     [MP4(wh=FHD, dur=[17.7, 18.4], freeze=[0.2, 1.8], frame_like=[[6, "in/clip.mp4@4"]])],
     ["ffmpeg -i in/clip.mp4 -vf tpad=start_duration=2:start_mode=clone -af adelay=2000:all=1 " + ENC + " out/clip_hold.mp4"]),
    ("a42", "A11 freeze frame",
     "at 0:09 in race.mp4 freeze for 2 seconds (dramatic pause), no sound during the freeze",
     {"race.mp4": t("cuts.mp4")},
     [MP4(wh=FHD, dur=[18.7, 19.4], freeze=[9.2, 10.8], frame_like=[[15, "in/race.mp4@13"]], quiet_at=[[9.2, 10.8, -60]])],
     ["ffmpeg -i in/race.mp4 -filter_complex \"[0:v]trim=0:9,setpts=PTS-STARTPTS[a];[0:v]trim=9,setpts=PTS-STARTPTS,"
      "tpad=start_duration=2:start_mode=clone[b];[a][b]concat=n=2:v=1:a=0[v];[0:a]atrim=0:9,asetpts=PTS-STARTPTS[x];"
      "[0:a]atrim=9,asetpts=PTS-STARTPTS,adelay=2000:all=1[y];[x][y]concat=n=2:v=0:a=1[o]\" -map \"[v]\" -map \"[o]\" "
      f"{ENC} out/race_pause.mp4"]),
]


PH = {f"photos/img_{i:02d}.jpg": t(f"photos/img_{i:02d}.jpg") for i in range(1, 7)}
SPEECH = [[0.3, 2.6], [6.3, 8.0]]           # test narration: inside the measured speech segments
PAUSE = [[3.2, 5.7], [8.6, 11.6]]          # inside the measured pauses
Q = lambda *must, files=False: {"outputs": [], "answer": {"must": list(must), "no_files": not files}}

T += [
    # ── B12. one video -> several platform formats ──
    ("b01", "B12 multi-platform",
     "make youtube.mp4 (16:9), reels.mp4 (9:16) and square.mp4 (1:1) versions of launch.mp4",
     {"launch.mp4": t("talk.mp4")},
     [{"match": "youtube.mp4", "aspect": [16, 9], "min_short": 720, "dur": [15.7, 16.4]},
      {"match": "reels.mp4", "aspect": [9, 16], "min_short": 600, "dur": [15.7, 16.4]},
      {"match": "square.mp4", "aspect": [1, 1], "min_short": 600, "dur": [15.7, 16.4]}],
     [f"ffmpeg -i in/launch.mp4 {ENC} out/youtube.mp4",
      f"ffmpeg -i in/launch.mp4 -vf crop=608:1080,scale=1080:1920 {ENC} out/reels.mp4",
      f"ffmpeg -i in/launch.mp4 -vf crop=1080:1080 {ENC} out/square.mp4"]),
    ("b02", "B12 multi-platform",
     "I need clip.mp4 for tiktok (call it tiktok.mp4) and for the instagram feed in 4:5 (feed.mp4)",
     {"clip.mp4": t("talk.mp4")},
     [{"match": "tiktok.mp4", "aspect": [9, 16], "min_short": 600, "has_audio": True},
      {"match": "feed.mp4", "aspect": [4, 5], "min_short": 600, "has_audio": True}],
     [f"ffmpeg -i in/clip.mp4 -vf crop=608:1080,scale=1080:1920 {ENC} out/tiktok.mp4",
      f"ffmpeg -i in/clip.mp4 -vf crop=864:1080 {ENC} out/feed.mp4"]),
    ("b03", "B12 multi-platform",
     "interview.mp4 is 4:3. make a 16:9 youtube version with a blurred background filling the sides, and a 9:16 story version. "
     "name them yt.mp4 and story.mp4",
     {"interview.mp4": t("webcam.mp4")},
     [{"match": "yt.mp4", "aspect": [16, 9], "min_short": 480, "dur": [15.7, 16.4]},
      {"match": "story.mp4", "aspect": [9, 16], "min_short": 480, "dur": [15.7, 16.4]}],
     ["ffmpeg -i in/interview.mp4 -filter_complex \"[0:v]scale=1280:720:force_original_aspect_ratio=increase,crop=1280:720,boxblur=20[bg];"
      f"[0:v]scale=-2:720[fg];[bg][fg]overlay=(W-w)/2:0[v]\" -map \"[v]\" -map 0:a {ENC} out/yt.mp4",
      f"ffmpeg -i in/interview.mp4 -vf crop=270:480,scale=720:1280 {ENC} out/story.mp4"]),
    ("b04", "B12 multi-platform",
     "from promo.mp4 make an instagram story (9:16) and a square post, each 15 seconds max",
     {"promo.mp4": t("cuts.mp4")},
     [{"match": "*story*.mp4|*9x16*.mp4|*vertical*.mp4", "aspect": [9, 16], "min_short": 600, "dur": [5, 15.2]},
      {"match": "*square*.mp4|*1x1*.mp4|*post*.mp4", "aspect": [1, 1], "min_short": 600, "dur": [5, 15.2]}],
     [f"ffmpeg -t 15 -i in/promo.mp4 -vf crop=608:1080,scale=1080:1920 {ENC} out/promo_story.mp4",
      f"ffmpeg -t 15 -i in/promo.mp4 -vf crop=1080:1080 {ENC} out/promo_square.mp4"]),

    # ── B13. fit under a size limit ──
    ("b05", "B13 size limit",
     "send master.mov on whatsapp, it has to be under 16 MB and play on phones",
     {"master.mov": t("big.mov")},
     [MP4(max_mb=16, vcodec="h264", acodec="aac", dur=[4.7, 5.3])],
     ["ffmpeg -i in/master.mov -c:v libx264 -crf 23 -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart out/master.mp4"]),
    ("b06", "B13 size limit",
     "talk.mp4 needs to be under 2 MB, keep the full length",
     {"talk.mp4": t("talk.mp4")},
     [MP4(max_mb=2, has_audio=True, dur=[15.7, 16.4])],
     ["ffmpeg -i in/talk.mp4 -vf scale=1280:-2 -c:v libx264 -b:v 850k -maxrate 850k -bufsize 1700k -c:a aac -b:a 64k out/talk_small.mp4"]),
    ("b07", "B13 size limit",
     "email limit is 3MB. can you get event.mp4 under that?",
     {"event.mp4": t("interlaced.mp4")},
     [MP4(max_mb=3, has_video=True, dur=[9.7, 10.3])],
     ["ffmpeg -i in/event.mp4 -vf yadif,scale=1280:-2 -c:v libx264 -b:v 1800k -maxrate 1800k -bufsize 3600k -c:a aac -b:a 96k out/event_small.mp4"]),
    ("b08", "B13 size limit",
     "make voice.m4a an mp3 under 150 KB for an old upload form",
     {"voice.m4a": t("narration.m4a")},
     [{"match": "*.mp3", "max_mb": 0.15, "dur": [15.7, 16.4]}],
     ["ffmpeg -i in/voice.m4a -ac 1 -c:a libmp3lame -b:a 64k out/voice.mp3"]),

    # ── B14. split into chunks ──
    ("b09", "B14 split chunks",
     "split long.mp4 into 5 second clips for my status",
     {"long.mp4": t("talk.mp4")},
     [MP4(count=4, dur=[0.5, 5.7])],
     ["ffmpeg -i in/long.mp4 -c:v libx264 -force_key_frames 'expr:gte(t,n_forced*5)' -c:a aac -f segment -segment_time 5 "
      "-reset_timestamps 1 out/part_%02d.mp4"]),
    ("b10", "B14 split chunks",
     "cut story.mp4 into 3 equal parts",
     {"story.mp4": t("cuts.mp4")},
     [MP4(count=3, dur=[5.3, 6.1])],
     ["ffmpeg -i in/story.mp4 -t 5.667 " + ENC + " out/story_1.mp4",
      "ffmpeg -ss 5.667 -i in/story.mp4 -t 5.667 " + ENC + " out/story_2.mp4",
      "ffmpeg -ss 11.333 -i in/story.mp4 " + ENC + " out/story_3.mp4"]),
    ("b11", "B14 split chunks",
     "instagram stories are 15 seconds max, split chat.mp4 into story-sized pieces",
     {"chat.mp4": t("webcam.mp4")},
     [MP4(count=2, dur=[0.5, 15.1])],
     ["ffmpeg -i in/chat.mp4 -c:v libx264 -force_key_frames 'expr:gte(t,n_forced*15)' -c:a aac -f segment -segment_time 15 "
      "-reset_timestamps 1 out/chat_%02d.mp4"]),

    # ── B15. highlight clip: section + vertical + captions ──
    ("b12", "B15 highlight clip",
     "make a vertical highlight of 0:06-0:10 from podcast.mp4 with the subtitles from podcast.srt burned in",
     {"podcast.mp4": t("talk.mp4"), "podcast.srt": t("captions.srt")},
     [MP4(aspect=[9, 16], min_short=600, dur=[3.7, 4.3])],
     ["ffmpeg -ss 6 -to 10 -copyts -i in/podcast.mp4 -vf \"subtitles=in/podcast.srt,crop=608:1080,scale=1080:1920\" "
      f"-output_ts_offset -6 {ENC} out/highlight.mp4"]),
    ("b13", "B15 highlight clip",
     "short for youtube: the first 3 seconds of ep.mp4, vertical, loud enough for phones (-14 LUFS)",
     {"ep.mp4": t("talk.mp4")},
     [MP4(aspect=[9, 16], min_short=600, dur=[2.7, 3.3], lufs=[-17, -11])],
     ["ffmpeg -t 3 -i in/ep.mp4 -vf crop=608:1080,scale=1080:1920 -af loudnorm=I=-14:TP=-1.5 -ar 48000 " + ENC + " out/short.mp4"]),
    ("b14", "B15 highlight clip",
     "teaser from interview.mp4: 0:12 to 0:16, square, with a white caption 'Full episode out now' at the bottom",
     {"interview.mp4": t("webcam.mp4")},
     [MP4(aspect=[1, 1], min_short=400, dur=[3.7, 4.3])],
     ["ffmpeg -ss 12 -to 16 -i in/interview.mp4 -vf \"crop=480:480,drawtext=text='Full episode out now':fontcolor=white:"
      "fontsize=28:x=(w-text_w)/2:y=h-60\" " + ENC + " out/teaser.mp4"]),

    # ── B16. styled burned-in subtitles ──
    ("b15", "B16 styled subtitles",
     "burn subs.srt into talk.mp4, big yellow text",
     {"talk.mp4": t("talk.mp4"), "subs.srt": t("captions.srt")},
     [MP4(wh=FHD, dur=[15.7, 16.4], frame_unlike=[[1, "in/talk.mp4@1", 0.003], [7, "in/talk.mp4@7", 0.003]],
          frame_like=[[4.5, "in/talk.mp4@4.5", 0.0025]])],
     ["ffmpeg -i in/talk.mp4 -vf \"subtitles=in/subs.srt:force_style='Fontsize=30,PrimaryColour=&H00FFFF&'\" " + ENC + " out/talk_subs.mp4"]),
    ("b16", "B16 styled subtitles",
     "hardcode captions.srt on vlog.mp4 with a black box behind the text so it's readable",
     {"vlog.mp4": t("talk.mp4"), "captions.srt": t("captions.srt")},
     [MP4(wh=FHD, dur=[15.7, 16.4], frame_unlike=[[1.5, "in/vlog.mp4@1.5", 0.003]], frame_like=[[10, "in/vlog.mp4@10", 0.0025]])],
     ["ffmpeg -i in/vlog.mp4 -vf \"subtitles=in/captions.srt:force_style='BorderStyle=3,Outline=1,Shadow=0,BackColour=&H80000000&'\" "
      + ENC + " out/vlog_captioned.mp4"]),
    ("b17", "B16 styled subtitles",
     "put the subtitles from lines.srt at the TOP of the screen in demo.mp4, burned in",
     {"demo.mp4": t("talk.mp4"), "lines.srt": t("captions.srt")},
     [MP4(wh=FHD, dur=[15.7, 16.4], frame_unlike=[[7, "in/demo.mp4@7", 0.003]], frame_like=[[4.5, "in/demo.mp4@4.5", 0.0025]])],
     ["ffmpeg -i in/demo.mp4 -vf \"subtitles=in/lines.srt:force_style='Alignment=8,Fontsize=26'\" " + ENC + " out/demo_top_subs.mp4"]),
    ("b18", "B16 styled subtitles",
     "add my srt (edit.srt) to cut.mp4 as burned-in subtitles in white with a thick black outline",
     {"cut.mp4": t("talk.mp4"), "edit.srt": t("captions.srt")},
     [MP4(wh=FHD, dur=[15.7, 16.4], frame_unlike=[[12.3, "in/cut.mp4@12.3", 0.003]], frame_like=[[10, "in/cut.mp4@10", 0.0025]])],
     ["ffmpeg -i in/cut.mp4 -vf \"subtitles=in/edit.srt:force_style='Outline=3,PrimaryColour=&HFFFFFF&,OutlineColour=&H000000&'\" "
      + ENC + " out/cut_subs.mp4"]),

    # ── B17. thumbnail with title ──
    ("b19", "B17 thumbnail",
     "youtube thumbnail from travel.mp4 at 0:05 with the title 'My Trip' in big letters. 1280x720 jpg",
     {"travel.mp4": t("talk.mp4")},
     [{"match": "*.jpg|*.jpeg", "wh": [1280, 720], "max_mb": 2}],
     ["ffmpeg -ss 5 -i in/travel.mp4 -frames:v 1 -vf \"scale=1280:720,drawtext=text='My Trip':fontcolor=white:fontsize=120:"
      "borderw=6:x=(w-text_w)/2:y=(h-text_h)/2\" -q:v 2 out/thumbnail.jpg"]),
    ("b20", "B17 thumbnail",
     "grab a frame from 0:10 of episode.mp4 and put 'EP 12' in the top left, png, keep full resolution",
     {"episode.mp4": t("talk.mp4")},
     [{"match": "*.png", "wh": FHD}],
     ["ffmpeg -ss 10 -i in/episode.mp4 -frames:v 1 -vf \"drawtext=text='EP 12':fontcolor=white:fontsize=110:x=60:y=60\" out/ep12.png"]),
    ("b21", "B17 thumbnail",
     "square cover image (1080x1080) for my podcast from frame 3s of studio.mp4 with the text 'New Episode'",
     {"studio.mp4": t("webcam.mp4")},
     [{"match": "*.jpg|*.jpeg|*.png", "wh": [1080, 1080]}],
     ["ffmpeg -ss 3 -i in/studio.mp4 -frames:v 1 -vf \"crop=480:480,scale=1080:1080,drawtext=text='New Episode':fontcolor=white:"
      "fontsize=90:x=(w-text_w)/2:y=h-200\" -q:v 2 out/cover.jpg"]),

    # ── B18. audiogram ──
    ("b22", "B18 audiogram",
     "make a video for episode.m4a using art.jpg as the background, with a moving waveform. square 1080",
     {"episode.m4a": t("narration.m4a"), "art.jpg": t("cover.jpg")},
     [MP4(wh=[1080, 1080], has_audio=True, dur=[15.6, 16.5])],
     ["ffmpeg -loop 1 -i in/art.jpg -i in/episode.m4a -filter_complex \"[0:v]scale=1080:1080[bg];[1:a]showwaves=s=1080x240:mode=cline:"
      "colors=white[w];[bg][w]overlay=0:820:shortest=1[v]\" -map \"[v]\" -map 1:a -shortest -r 25 " + ENC + " out/episode.mp4"]),
    ("b23", "B18 audiogram",
     "turn clip.mp3 into a video for twitter: just cover.jpg and the audio",
     {"clip.mp3": t("music_bed.mp3"), "cover.jpg": t("cover.jpg")},
     [MP4(has_audio=True, has_video=True, dur=[39.5, 40.6], vcodec="h264")],
     ["ffmpeg -loop 1 -framerate 2 -i in/cover.jpg -i in/clip.mp3 -t 40 -c:v libx264 -tune stillimage -pix_fmt yuv420p -c:a aac "
      "-shortest out/clip.mp4"]),
    ("b24", "B18 audiogram",
     "podcast snippet: first 10 seconds of show.m4a as a 9:16 video with a waveform over cover.jpg",
     {"show.m4a": t("narration.m4a"), "cover.jpg": t("cover.jpg")},
     [MP4(aspect=[9, 16], min_short=600, has_audio=True, dur=[9.6, 10.5])],
     ["ffmpeg -loop 1 -i in/cover.jpg -t 10 -i in/show.m4a -filter_complex \"[0:v]scale=1080:1080,pad=1080:1920:0:420[bg];"
      "[1:a]showwaves=s=1080x300:mode=line[w];[bg][w]overlay=0:1550:shortest=1[v]\" -map \"[v]\" -map 1:a -shortest -r 25 "
      + ENC + " out/snippet.mp4"]),

    # ── C19. voice effects ──
    ("c01", "C19 voice effects",
     "make my_voice.m4a sound like a chipmunk, same length",
     {"my_voice.m4a": t("voice_m.m4a")},
     [{"match": "*", "has_audio": True, "dur": [2.3, 2.8], "pitch_ratio": ["in/my_voice.m4a", 1.4, 2.4]}],
     ["ffmpeg -i in/my_voice.m4a -af rubberband=pitch=1.8 -c:a aac out/chipmunk.m4a"]),
    ("c02", "C19 voice effects",
     "deepen the voice in announcement.m4a so it sounds like a movie trailer guy",
     {"announcement.m4a": t("voice_f.m4a")},
     [{"match": "*", "has_audio": True, "dur": [2.0, 2.5], "pitch_ratio": ["in/announcement.m4a", 0.45, 0.85]}],
     ["ffmpeg -i in/announcement.m4a -af rubberband=pitch=0.7 -c:a aac out/announcement_deep.m4a"]),
    ("c03", "C19 voice effects",
     "add an echo to speech.m4a, like it's in a big hall",
     {"speech.m4a": t("narration.m4a")},
     [{"match": "*", "has_audio": True, "dur": [15.7, 18], "loud_at": [[2.95, 3.4, -65]]}],
     ["ffmpeg -i in/speech.m4a -af 'aecho=0.8:0.85:300|600:0.4|0.25' -c:a aac out/speech_echo.m4a"]),
    ("c04", "C19 voice effects",
     "make call.m4a sound like it's coming through an old telephone",
     {"call.m4a": t("narration.m4a")},
     [{"match": "*", "has_audio": True, "dur": [15.7, 16.4], "band_max": [5000, -70], "loud_at": [[0.3, 2.5, -45]]}],
     ["ffmpeg -i in/call.m4a -af highpass=f=300,lowpass=f=3400,lowpass=f=3400 -c:a aac out/call_phone.m4a"]),

    # ── C20. voice cleanup ──
    ("c05", "C20 voice cleanup",
     "there's a hum in my recording (memo.wav), get rid of it",
     {"memo.wav": t("hum.wav")},
     [{"match": "*", "has_audio": True, "dur": [7.8, 8.3], "band_max": [60, -60], "loud_at": [[0.5, 2, -40]]}],
     ["ffmpeg -i in/memo.wav -af bandreject=f=60:width_type=q:w=1,highpass=f=120,highpass=f=120 out/memo_clean.wav"]),
    ("c06", "C20 voice cleanup",
     "make my voice sound better in podcast_raw.wav: remove the hum and bring it to podcast loudness",
     {"podcast_raw.wav": t("hum.wav")},
     [{"match": "*", "has_audio": True, "band_max": [60, -60], "lufs": [-20, -14]}],
     ["ffmpeg -i in/podcast_raw.wav -af bandreject=f=60:width_type=q:w=1,highpass=f=120,highpass=f=120,afftdn,loudnorm=I=-16:TP=-1.5 -ar 48000 out/podcast_clean.wav"]),
    ("c07", "C20 voice cleanup",
     "clean up the audio of note.wav, there's background hiss and a low buzz. export as mp3",
     {"note.wav": t("hum.wav")},
     [{"match": "*.mp3", "has_audio": True, "band_max": [60, -60], "loud_at": [[0.5, 2, -40]]}],
     ["ffmpeg -i in/note.wav -af bandreject=f=60:width_type=q:w=1,highpass=f=120,highpass=f=120,afftdn=nr=12 -c:a libmp3lame -b:a 160k out/note_clean.mp3"]),

    # ── C21. background music that ducks under speech ──
    ("c08", "C21 ducking music",
     "add bgm.mp3 under my talk in vlog.mp4 and have it automatically go quieter whenever I speak",
     {"vlog.mp4": t("talk.mp4"), "bgm.mp3": t("music_bed.mp3")},
     [MP4(wh=FHD, dur=[15.7, 16.4], loud_at=[[0.3, 2.6, -40], [3.5, 5.5, -45]], band_drop=[247, SPEECH[0], PAUSE[0], 5])],
     ["ffmpeg -i in/vlog.mp4 -i in/bgm.mp3 -filter_complex \"[0:a]asplit=2[sc][vo];[1:a]volume=0.5[m];"
      "[m][sc]sidechaincompress=threshold=0.02:ratio=12:attack=20:release=300[duck];[vo][duck]amix=inputs=2:duration=first:normalize=0[a]\" "
      "-map 0:v -map \"[a]\" -c:v copy -c:a aac out/vlog_music.mp4"]),
    ("c09", "C21 ducking music",
     "mix voiceover.m4a with music.mp3 for a podcast intro: music ducks under the voice, ends when the voice ends",
     {"voiceover.m4a": t("narration.m4a"), "music.mp3": t("music_bed.mp3")},
     [{"match": "*", "has_audio": True, "dur": [15.7, 16.6], "band_drop": [247, SPEECH[1], PAUSE[1], 5]}],
     ["ffmpeg -i in/voiceover.m4a -i in/music.mp3 -filter_complex \"[0:a]aresample=44100,asplit=2[sc][vo];[1:a]volume=0.5[m];"
      "[m][sc]sidechaincompress=threshold=0.02:ratio=12:attack=20:release=300[duck];[vo][duck]amix=inputs=2:duration=first:normalize=0\" "
      "-c:a libmp3lame -b:a 192k out/intro_mix.mp3"]),
    ("c10", "C21 ducking music",
     "background music (track.mp3) for lesson.mp4 at a low level, and even lower when the teacher talks",
     {"lesson.mp4": t("talk.mp4"), "track.mp3": t("music_bed.mp3")},
     [MP4(wh=FHD, dur=[15.7, 16.4], band_drop=[247, SPEECH[1], PAUSE[1], 3], loud_at=[[3.5, 5.5, -55]])],
     ["ffmpeg -i in/lesson.mp4 -i in/track.mp3 -filter_complex \"[0:a]asplit=2[sc][vo];[1:a]volume=0.3[m];"
      "[m][sc]sidechaincompress=threshold=0.005:ratio=20:attack=20:release=300[duck];[vo][duck]amix=inputs=2:duration=first:normalize=0[a]\" "
      "-map 0:v -map \"[a]\" -c:v copy -c:a aac out/lesson_music.mp4"]),

    # ── C22. remove pauses ──
    ("c11", "C22 remove pauses",
     "cut out all the long pauses in lecture.mp4",
     {"lecture.mp4": t("talk.mp4")},
     [MP4(has_video=True, dur=[5, 11.5], max_silence=1.2)],
     ["ffmpeg -i in/lecture.mp4 -vf \"select='not(between(t,2.9,5.9)+between(t,8.4,11.8)+gte(t,14.6))',setpts=N/FRAME_RATE/TB\" "
      "-af \"aselect='not(between(t,2.9,5.9)+between(t,8.4,11.8)+gte(t,14.6))',asetpts=N/SR/TB\" " + ENC + " out/lecture_tight.mp4"]),
    ("c12", "C22 remove pauses",
     "remove the silences from take.m4a so it's tighter",
     {"take.m4a": t("narration.m4a")},
     [{"match": "*", "has_audio": True, "dur": [5, 11.5], "max_silence": 1.0}],
     ["ffmpeg -i in/take.m4a -af silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-45dB -c:a aac out/take_tight.m4a"]),
    ("c13", "C22 remove pauses",
     "my voice memo answer.m4a has gaps where I was thinking, shorten every gap to half a second",
     {"answer.m4a": t("narration.m4a")},
     [{"match": "*", "has_audio": True, "dur": [6, 12], "max_silence": 1.2, "loud_at": [[0.3, 2.5, -45]]}],
     ["ffmpeg -i in/answer.m4a -af silenceremove=stop_periods=-1:stop_duration=0.5:stop_threshold=-45dB:stop_silence=0.5 "
      "-c:a aac out/answer_tight.m4a"]),

    # ── C23. A/V sync ──
    ("c14", "C23 fix sync",
     "the audio in clap.mp4 is about a quarter second ahead of the picture. fix it",
     {"clap.mp4": t("sync.mp4")},
     [MP4(has_audio=True, av_sync_ms=[-60, 60], dur=[9.5, 10.6])],
     ["ffmpeg -i in/clap.mp4 -af adelay=250:all=1 -c:v copy -c:a aac out/clap_synced.mp4"]),
    ("c15", "C23 fix sync",
     "lip sync is off in board.mp4, the sound comes too early. can you line it up? the flash and the beep should match",
     {"board.mp4": t("sync.mp4")},
     [MP4(has_audio=True, av_sync_ms=[-60, 60])],
     ["ffmpeg -i in/board.mp4 -itsoffset 0.25 -i in/board.mp4 -map 0:v -map 1:a -c:v copy -c:a aac out/board_synced.mp4"]),
    ("c16", "C23 fix sync",
     "delay the audio of sync_test.mp4 by 250 ms",
     {"sync_test.mp4": t("sync.mp4")},
     [MP4(has_audio=True, av_sync_ms=[-60, 60])],
     ["ffmpeg -i in/sync_test.mp4 -af adelay=250:all=1 -c:v copy -c:a aac out/sync_test_fixed.mp4"]),

    # ── C24. multi-track audio ──
    ("c17", "C24 audio tracks",
     "movie.mkv has two audio tracks. keep only the english one",
     {"movie.mkv": t("multilang.mkv")},
     [{"match": "*.mkv|*.mp4", "has_video": True, "audio_streams": 1, "audio_lang": "eng"}],
     ["ffmpeg -i in/movie.mkv -map 0:v -map 0:a:m:language:eng -c copy out/movie_english.mkv"]),
    ("c18", "C24 audio tracks",
     "make english the first (default) audio track of film.mkv, keep the other one too",
     {"film.mkv": t("multilang.mkv")},
     [{"match": "*.mkv", "has_video": True, "audio_streams": 2, "audio_lang": "eng"}],
     ["ffmpeg -i in/film.mkv -map 0:v -map 0:a:1 -map 0:a:0 -c copy -disposition:a:0 default -disposition:a:1 0 out/film.mkv"]),
    ("c19", "C24 audio tracks",
     "pull the hindi audio out of show.mkv as an mp3",
     {"show.mkv": t("multilang.mkv")},
     [{"match": "*.mp3", "has_audio": True, "dur": [2.4, 6.2]}],
     ["ffmpeg -i in/show.mkv -map 0:a:m:language:hin -c:a libmp3lame -b:a 160k out/show_hindi.mp3"]),
    ("c20", "C24 audio tracks",
     "turn dub.mkv into an mp4 that plays the english audio on any device",
     {"dub.mkv": t("multilang.mkv")},
     [MP4(vcodec="h264", acodec="aac", audio_streams=1, audio_lang="eng")],
     ["ffmpeg -i in/dub.mkv -map 0:v -map 0:a:m:language:eng -c:v copy -c:a aac -metadata:s:a:0 language=eng out/dub.mp4"]),

    # ── C25. channels ──
    ("c21", "C25 channels",
     "rec.wav is mono, make it stereo",
     {"rec.wav": t("hum.wav")},
     [{"match": "*", "ach": 2, "dur": [7.8, 8.3]}],
     ["ffmpeg -i in/rec.wav -ac 2 out/rec_stereo.wav"]),
    ("c22", "C25 channels",
     "convert narration.m4a to mono mp3 for a phone system",
     {"narration.m4a": t("narration.m4a")},
     [{"match": "*.mp3", "ach": 1, "dur": [15.7, 16.4]}],
     ["ffmpeg -i in/narration.m4a -ac 1 -c:a libmp3lame -b:a 96k out/narration_mono.mp3"]),
    ("c23", "C25 channels",
     "the sound in screencast.mp4 is only on one side... wait, actually just give me the audio as a mono wav at 16kHz for transcription",
     {"screencast.mp4": t("talk.mp4")},
     [{"match": "*.wav", "ach": 1, "sr": 16000, "dur": [15.7, 16.4]}],
     ["ffmpeg -i in/screencast.mp4 -vn -ac 1 -ar 16000 -c:a pcm_s16le out/screencast.wav"]),

    # ── D26. won't play on device ──
    ("d01", "D26 compatibility",
     "export.webm won't play on my iphone, convert it",
     {"export.webm": t("odd.webm")},
     [MP4(vcodec="h264", pix_fmt="yuv420p", has_audio=True, acodec="aac", dur=[7.7, 8.4])],
     ["ffmpeg -i in/export.webm -vf 'scale=trunc(iw/2)*2:trunc(ih/2)*2' -c:v libx264 -pix_fmt yuv420p -c:a aac out/export.mp4"]),
    ("d02", "D26 compatibility",
     "my smart tv can't open master.mov, make something it can play",
     {"master.mov": t("big.mov")},
     [MP4(vcodec="h264", acodec="aac", dur=[4.7, 5.3], pix_fmt="yuv420p")],
     ["ffmpeg -i in/master.mov -c:v libx264 -crf 20 -pix_fmt yuv420p -c:a aac out/master.mp4"]),
    ("d03", "D26 compatibility",
     "clip.webm needs to work in powerpoint on windows",
     {"clip.webm": t("odd.webm")},
     [MP4(vcodec="h264", acodec="aac", dur=[7.7, 8.4])],
     ["ffmpeg -i in/clip.webm -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' -c:v libx264 -pix_fmt yuv420p -c:a aac out/clip.mp4"]),

    # ── D27. variable frame rate ──
    ("d04", "D27 variable frame rate",
     "screen_rec.mp4 goes out of sync in my editor, someone said it's variable frame rate. make it constant 30fps",
     {"screen_rec.mp4": t("vfr.mp4")},
     [MP4(cfr=True, fps=30, dur=[11.6, 12.4], has_audio=True)],
     ["ffmpeg -i in/screen_rec.mp4 -fps_mode cfr -r 30 " + ENC + " out/screen_rec_cfr.mp4"]),
    ("d05", "D27 variable frame rate",
     "premiere says capture.mp4 is VFR. convert it to constant frame rate, 25 fps",
     {"capture.mp4": t("vfr.mp4")},
     [MP4(cfr=True, fps=25, dur=[11.6, 12.4])],
     ["ffmpeg -i in/capture.mp4 -vf fps=25 " + ENC + " out/capture_cfr.mp4"]),
    ("d06", "D27 variable frame rate",
     "the frame rate in obs.mp4 jumps around and the editor drifts. fix it so the frame rate is steady",
     {"obs.mp4": t("vfr.mp4")},
     [MP4(cfr=True, dur=[11.6, 12.4], has_audio=True)],
     ["ffmpeg -i in/obs.mp4 -vf fps=30 " + ENC + " out/obs_fixed.mp4"]),

    # ── D28. deinterlace ──
    ("d07", "D28 deinterlace",
     "old camcorder footage tape.mp4 has those horizontal comb lines when things move, fix it",
     {"tape.mp4": t("interlaced.mp4")},
     [MP4(progressive=True, dur=[9.7, 10.3], has_audio=True)],
     ["ffmpeg -i in/tape.mp4 -vf yadif " + ENC + " out/tape_progressive.mp4"]),
    ("d08", "D28 deinterlace",
     "deinterlace broadcast.mp4",
     {"broadcast.mp4": t("interlaced.mp4")},
     [MP4(progressive=True, dur=[9.7, 10.3])],
     ["ffmpeg -i in/broadcast.mp4 -vf bwdif " + ENC + " out/broadcast_deint.mp4"]),
    ("d09", "D28 deinterlace",
     "make wedding.mp4 look smooth on a computer screen, it was recorded interlaced. keep 50 fps motion",
     {"wedding.mp4": t("interlaced.mp4")},
     [MP4(progressive=True, fps=50, dur=[9.7, 10.3])],
     ["ffmpeg -i in/wedding.mp4 -vf yadif=mode=send_field " + ENC + " out/wedding_50p.mp4"]),

    # ── D29. aspect ratio ──
    ("d10", "D29 aspect ratio",
     "everything in squished.mp4 looks too thin, it should be full 16:9 1920x1080",
     {"squished.mp4": t("squashed.mp4")},
     [MP4(wh=FHD, sar=[1, 1], dur=[9.7, 10.3])],
     ["ffmpeg -i in/squished.mp4 -vf scale=1920:1080,setsar=1 " + ENC + " out/squished_fixed.mp4"]),
    ("d11", "D29 aspect ratio",
     "stretched.mp4 plays stretched too wide in VLC, the pixels themselves are fine. fix it",
     {"stretched.mp4": t("wrongsar.mp4")},
     [MP4(wh=FHD, sar=[1, 1], dur=[9.7, 10.3])],
     ["ffmpeg -i in/stretched.mp4 -vf setsar=1 " + ENC + " out/stretched_fixed.mp4"]),
    ("d12", "D29 aspect ratio",
     "clip.mp4 is 1440x1080 but should be shown as 16:9 widescreen. make a normal 1080p file",
     {"clip.mp4": t("squashed.mp4")},
     [MP4(wh=FHD, sar=[1, 1])],
     ["ffmpeg -i in/clip.mp4 -vf scale=1920:1080,setsar=1 " + ENC + " out/clip_169.mp4"]),

    # ── D30. brightness ──
    ("d13", "D30 brightness",
     "night.mp4 is way too dark, brighten it up",
     {"night.mp4": t("dark.mp4")},
     [MP4(yavg=[85, 190], dur=[9.7, 10.3], has_audio=True)],
     ["ffmpeg -i in/night.mp4 -vf eq=brightness=0.3:contrast=1.15 " + ENC + " out/night_bright.mp4"]),
    ("d14", "D30 brightness",
     "the picture in gym.mp4 looks dull and dark, give it more brightness and contrast",
     {"gym.mp4": t("dark.mp4")},
     [MP4(yavg=[85, 190])],
     ["ffmpeg -i in/gym.mp4 -vf eq=brightness=0.3:contrast=1.2:saturation=1.2 " + ENC + " out/gym_fixed.mp4"]),
    ("d15", "D30 brightness",
     "underexposed.mp4: fix the exposure so it looks normal",
     {"underexposed.mp4": t("dark.mp4")},
     [MP4(yavg=[85, 190])],
     ["ffmpeg -i in/underexposed.mp4 -vf curves=all='0/0 0.3/0.6 1/1',eq=contrast=1.4 " + ENC + " out/exposure_fixed.mp4"]),

    # ── D31. salvage a broken recording ──
    ("d16", "D31 salvage",
     "my recording crashed and recording.mkv won't open properly. save whatever can be saved as an mp4",
     {"recording.mkv": t("broken.mkv")},
     [MP4(has_video=True, dur=[14, 31], vcodec="h264")],
     ["ffmpeg -err_detect ignore_err -i in/recording.mkv -c copy out/recording.mp4"]),
    ("d17", "D31 salvage",
     "stream.mkv got cut off when the power went out. can you fix it so it plays in a normal player?",
     {"stream.mkv": t("broken.mkv")},
     [{"match": "*.mp4|*.mkv", "has_video": True, "dur": [14, 31]}],
     ["ffmpeg -i in/stream.mkv -c copy out/stream_fixed.mkv"]),
    ("d18", "D31 salvage",
     "partial.mkv is a half-written file. recover the video part as h264 mp4",
     {"partial.mkv": t("broken.mkv")},
     [MP4(vcodec="h264", dur=[14, 31])],
     ["ffmpeg -i in/partial.mkv -c:v copy -c:a aac out/partial.mp4"]),
]

T += [
    # ── E32. batch resize / convert ──
    ("e01", "E32 batch images",
     "convert all the photos to webp for my website, max 1600px on the long side",
     PH, [{"match": "*.webp", "count": 6, "max_side": 1600}],
     ["FOREACH in/photos/*.jpg :: magick {in} -resize '1600x1600>' -quality 82 out/{stem}.webp"]),
    ("e02", "E32 batch images",
     "make 400px wide thumbnails of every photo in photos/",
     PH, [{"match": "*.jpg|*.jpeg|*.png|*.webp", "count": 6, "max_side": 400}],
     ["FOREACH in/photos/*.jpg :: magick {in} -resize '400x400>' out/{stem}_thumb.jpg"]),
    ("e03", "E32 batch images",
     "shrink all photos to under 300 KB each, jpg",
     PH, [{"match": "*.jpg|*.jpeg", "count": 6, "max_mb": 0.3}],
     ["FOREACH in/photos/*.jpg :: magick {in} -resize '1200x1200>' -quality 75 out/{stem}.jpg"]),

    # ── E33. product shots ──
    ("e04", "E33 product photo",
     "product.png needs to be a square 1200x1200 jpg on a white background with some padding",
     {"product.png": t("product.png")},
     [{"match": "*.jpg|*.jpeg", "wh": [1200, 1200], "corner_color": [255, 255, 255]}],
     ["magick in/product.png -background white -flatten -resize 1000x1000 -gravity center -extent 1200x1200 out/product.jpg"]),
    ("e05", "E33 product photo",
     "amazon wants my product image (item.png) on pure white, 2000x2000",
     {"item.png": t("product.png")},
     [{"match": "*.jpg|*.jpeg|*.png", "wh": [2000, 2000], "corner_color": [255, 255, 255]}],
     ["magick in/item.png -background white -flatten -resize 1800x1800 -gravity center -extent 2000x2000 out/item_white.jpg"]),
    ("e06", "E33 product photo",
     "put shoe.png on a light gray (#f3f4f6) square canvas, 1080px, centered",
     {"shoe.png": t("product.png")},
     [{"match": "*.jpg|*.jpeg|*.png", "wh": [1080, 1080], "corner_color": [243, 244, 246]}],
     ["magick in/shoe.png -resize 900x900 -background '#f3f4f6' -gravity center -extent 1080x1080 -flatten out/shoe_square.png"]),

    # ── E34. meme / quote card ──
    ("e07", "E34 text on image",
     "put the quote 'Stay curious' in big white letters across the middle of bg.jpg",
     {"bg.jpg": t("cover.jpg")},
     [{"match": "*.jpg|*.jpeg|*.png", "wh": [1400, 1400], "frame_unlike": [[0, "in/bg.jpg", 0.005]]}],
     ["magick in/bg.jpg -gravity center -fill white -stroke black -strokewidth 3 -pointsize 150 -annotate +0+0 'Stay curious' out/quote.jpg"]),
    ("e08", "E34 text on image",
     "meme: 'WHEN THE CODE WORKS' on top and 'ON THE FIRST TRY' at the bottom of photo.jpg",
     {"photo.jpg": t("photos/img_02.jpg")},
     [{"match": "*.jpg|*.jpeg|*.png", "wh": [1300, 850], "frame_unlike": [[0, "in/photo.jpg", 0.005]]}],
     ["magick in/photo.jpg -fill white -stroke black -strokewidth 4 -pointsize 80 -gravity north -annotate +0+20 'WHEN THE CODE WORKS' "
      "-gravity south -annotate +0+20 'ON THE FIRST TRY' out/meme.jpg"]),
    ("e09", "E34 text on image",
     "make an instagram quote post (1080x1080) from texture.jpg with the line 'Less, but better.' in the center",
     {"texture.jpg": t("cover.jpg")},
     [{"match": "*.jpg|*.jpeg|*.png", "wh": [1080, 1080]}],
     ["magick in/texture.jpg -resize 1080x1080^ -gravity center -extent 1080x1080 -fill white -pointsize 90 -annotate +0+0 "
      "'Less, but better.' out/quote_post.jpg"]),

    # ── E35. photo slideshow ──
    ("e10", "E35 slideshow",
     "slideshow of all the photos, 3 seconds each, with song.mp3 playing, 1080p",
     {**PH, "song.mp3": t("music_bed.mp3")},
     [MP4(wh=FHD, has_audio=True, dur=[17.5, 18.6])],
     ["ffmpeg -loop 1 -t 3 -i in/photos/img_01.jpg -loop 1 -t 3 -i in/photos/img_02.jpg -loop 1 -t 3 -i in/photos/img_03.jpg -loop 1 -t 3 -i in/photos/img_04.jpg -loop 1 -t 3 -i in/photos/img_05.jpg -loop 1 -t 3 -i in/photos/img_06.jpg -i in/song.mp3 -filter_complex \"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v0];[1:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v1];[2:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v2];[3:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v3];[4:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v4];[5:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v5];[v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[v]\" -map \"[v]\" -map 6:a -c:v libx264 -c:a aac -shortest out/slideshow.mp4"]),
    ("e11", "E35 slideshow",
     "video from the photos, 2 seconds per photo, no music, vertical 1080x1920 for stories",
     PH, [MP4(wh=[1080, 1920], dur=[11.5, 12.6])],
     ["ffmpeg -loop 1 -t 2 -i in/photos/img_01.jpg -loop 1 -t 2 -i in/photos/img_02.jpg -loop 1 -t 2 -i in/photos/img_03.jpg -loop 1 -t 2 -i in/photos/img_04.jpg -loop 1 -t 2 -i in/photos/img_05.jpg -loop 1 -t 2 -i in/photos/img_06.jpg -filter_complex \"[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v0];[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v1];[2:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v2];[3:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v3];[4:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v4];[5:v]scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,format=yuv420p[v5];[v0][v1][v2][v3][v4][v5]concat=n=6:v=1:a=0[v]\" -map \"[v]\" -c:v libx264 out/stories.mp4"]),
    ("e12", "E35 slideshow",
     "make a short video of the photos with a crossfade between each one, 3s per photo, music from bed.mp3",
     {**PH, "bed.mp3": t("music_bed.mp3")},
     [MP4(has_audio=True, dur=[12, 18.6])],
     ["ffmpeg " + " ".join(f"-loop 1 -t 3 -i in/photos/img_{i:02d}.jpg" for i in range(1, 7)) + " -i in/bed.mp3 -filter_complex \""
      + "".join(f"[{i}:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=25,"
                f"format=yuv420p[v{i}];" for i in range(6))
      + "[v0][v1]xfade=duration=0.5:offset=2.5[x1];[x1][v2]xfade=duration=0.5:offset=5[x2];[x2][v3]xfade=duration=0.5:offset=7.5[x3];"
        "[x3][v4]xfade=duration=0.5:offset=10[x4];[x4][v5]xfade=duration=0.5:offset=12.5[v]\" -map \"[v]\" -map 6:a -c:v libx264 "
        "-c:a aac -shortest out/photos_fade.mp4"]),

    # ── E36. prettify screenshots ──
    ("e13", "E36 screenshot polish",
     "give screenshot.png rounded corners (transparent outside), keep it a png",
     {"screenshot.png": t("screenshot.png")},
     [{"match": "*.png", "corner_alpha": True}],
     ["magick in/screenshot.png \\( +clone -alpha extract -draw 'fill black polygon 0,0 0,30 30,0 fill white circle 30,30 30,0' "
      "\\( +clone -flip \\) -compose Multiply -composite \\( +clone -flop \\) -compose Multiply -composite \\) -alpha off "
      "-compose CopyOpacity -composite out/screenshot_rounded.png"]),
    ("e14", "E36 screenshot polish",
     "put app.png on a white background with 80px of padding around it and a soft shadow, for a blog post",
     {"app.png": t("screenshot.png")},
     [{"match": "*.png|*.jpg", "corner_color": [255, 255, 255]}],
     ["magick in/app.png \\( +clone -background black -shadow 50x20+0+15 \\) +swap -background white -layers merge +repage "
      "-bordercolor white -border 80 out/app_shadow.png"]),
    ("e15", "E36 screenshot polish",
     "add a 2px gray border around ui.png and make it 1200px wide",
     {"ui.png": t("screenshot.png")},
     [{"match": "*.png|*.jpg", "max_side": 1200}],
     ["magick in/ui.png -resize 1196x -bordercolor '#9ca3af' -border 2 out/ui_border.png"]),

    # ── E37. scans -> PDF ──
    ("e16", "E37 scans to pdf",
     "combine scan1.jpg, scan2.jpg and scan3.jpg into one pdf in that order",
     {"scan1.jpg": t("page1.jpg"), "scan2.jpg": t("page2.jpg"), "scan3.jpg": t("page3.jpg")},
     [{"match": "*.pdf", "pdf_pages": 3}],
     ["magick in/scan1.jpg in/scan2.jpg in/scan3.jpg out/scans.pdf"]),
    ("e17", "E37 scans to pdf",
     "make the scanned pages black and white (grayscale) jpgs so they're easier to read",
     {"p1.jpg": t("page1.jpg"), "p2.jpg": t("page2.jpg"), "p3.jpg": t("page3.jpg")},
     [{"match": "*.jpg|*.jpeg", "count": 3, "gray": True}],
     ["FOREACH in/p*.jpg :: magick {in} -colorspace Gray -normalize out/{stem}_bw.jpg"]),
    ("e18", "E37 scans to pdf",
     "turn these into a single grayscale pdf for the bank: a.jpg b.jpg c.jpg",
     {"a.jpg": t("page1.jpg"), "b.jpg": t("page2.jpg"), "c.jpg": t("page3.jpg")},
     [{"match": "*.pdf", "pdf_pages": 3}],
     ["magick in/a.jpg in/b.jpg in/c.jpg -colorspace Gray out/documents.pdf"]),

    # ── F38. rename / fix dates ──
    ("f01", "F38 dates",
     "rename all the photos by the date they were taken, like 2023-12-25_101500.jpg",
     PH, [{"match": "2023-12-*.jpg", "count": 5}, {"match": "2024-01-*.jpg", "count": 1}],
     ["exiftool -o out/ -d '%Y-%m-%d_%H%M%S%%-c.%%e' '-FileName<DateTimeOriginal' in/photos"]),
    ("f02", "F38 dates",
     "the camera clock was 3 hours behind. fix the date taken on all photos",
     PH, [{"match": "img_01.jpg", "exif_eq": {"DateTimeOriginal": "2023:12:24 21:00:00"}},
          {"match": "img_06.jpg", "exif_eq": {"DateTimeOriginal": "2024:01:01 03:10:00"}}],
     ["exiftool -o out/ '-AllDates+=3' in/photos"]),
    ("f03", "F38 dates",
     "set the date taken of img_03.jpg to 1 Jan 2024, 9 am",
     {"img_03.jpg": t("photos/img_03.jpg")},
     [{"match": "*.jpg", "exif_eq": {"DateTimeOriginal": "2024:01:01 09:00:00"}}],
     ["exiftool -o out/ '-DateTimeOriginal=2024:01:01 09:00:00' in/img_03.jpg"]),

    # ── F39. privacy strip ──
    ("f04", "F39 privacy",
     "remove location and personal info from all the photos before I share them, but keep the date they were taken",
     PH, [{"match": "*.jpg", "count": 6, "no_gps": True, "has_date": True, "exif_absent": ["Artist"]}],
     ["exiftool -o out/ -gps:all= -Artist= -SerialNumber= in/photos"]),
    ("f05", "F39 privacy",
     "strip ALL metadata from img_05.jpg",
     {"img_05.jpg": t("photos/img_05.jpg")},
     [{"match": "*.jpg", "no_gps": True, "exif_absent": ["Artist", "Make", "DateTimeOriginal"]}],
     ["exiftool -all= -o out/ in/img_05.jpg"]),
    ("f06", "F39 privacy",
     "who took this? remove the photographer name from img_02.jpg and img_04.jpg",
     {"img_02.jpg": t("photos/img_02.jpg"), "img_04.jpg": t("photos/img_04.jpg")},
     [{"match": "*.jpg", "count": 2, "exif_absent": ["Artist"]}],
     ["exiftool -o out/ -Artist= in/img_02.jpg in/img_04.jpg"]),

    # ── F40. reports ──
    ("f07", "F40 report",
     "make a csv listing every video here with its duration and resolution",
     {"a.mp4": t("talk.mp4"), "b.mp4": t("cuts.mp4"), "c.mp4": t("webcam.mp4")},
     Q(r"a\.mp4", r"b\.mp4", r"c\.mp4", r"1920", r"640", r"\b16(\.0+)?\b", r"\b17(\.0+)?\b", files=True),
     ["ffprobe -v error -show_entries stream=width,height:format=duration -of csv=p=0 in/a.mp4",
      "REPLY: file,duration_s,width,height\na.mp4,16.0,1920,1080\nb.mp4,17.0,1920,1080\nc.mp4,16.0,640,480"]),
    ("f08", "F40 report",
     "list the date taken of every photo in a csv",
     PH, Q(r"img_01", r"img_06", r"2023:12:24 18:00:00", r"2024:01:01 00:10:00", files=True),
     ["exiftool -csv -FileName -DateTimeOriginal in/photos",
      "REPLY: FileName,DateTimeOriginal\nimg_01.jpg,2023:12:24 18:00:00\nimg_02.jpg,2023:12:25 10:15:00\nimg_03.jpg,2023:12:25 12:45:30\n"
      "img_04.jpg,2023:12:26 16:20:00\nimg_05.jpg,2023:12:31 23:59:00\nimg_06.jpg,2024:01:01 00:10:00"]),
    ("f09", "F40 report",
     "write a text file with the codec, resolution and size of master.mov",
     {"master.mov": t("big.mov")},
     [{"match": "*.txt|*.csv|*.json", "contains": [r"prores", r"1920", r"1080"]}],
     ["ffprobe -v error -show_entries stream=codec_name,width,height:format=size -of default=nw=1 -o out/master_info.txt in/master.mov"]),

    # ── G41. what's wrong / why so big (answer only) ──
    ("g01", "G41 diagnose",
     "why is master.mov so huge? it's only 5 seconds",
     {"master.mov": t("big.mov")}, Q(r"prores", r"(pcm|uncompress|24.?bit)"),
     ["ffprobe -v error -show_entries stream=codec_name,bit_rate in/master.mov",
      "REPLY: It's ProRes 422 HQ video with uncompressed 24-bit PCM audio, an editing format with a very high bitrate."]),
    ("g02", "G41 diagnose",
     "what's wrong with recording.mkv? it says 30 seconds but stops early",
     {"recording.mkv": t("broken.mkv")}, Q(r"(truncat|incomplete|cut off|cut short|ends early|unfinished|corrupt|damaged)"),
     ["ffprobe -v error in/recording.mkv", "REPLY: The file is truncated: the recording stopped mid-write, so only about the first 18 seconds exist."]),
    ("g03", "G41 diagnose",
     "is screen.mp4 variable frame rate?",
     {"screen.mp4": t("vfr.mp4")}, Q(r"\b(yes|variable|vfr)\b"),
     ["ffprobe -v error -select_streams v -show_entries frame=pts_time -of csv=p=0 in/screen.mp4",
      "REPLY: Yes, it's variable frame rate: the frame interval changes partway through (about 30, then 12, then 24 fps)."]),
    ("g04", "G41 diagnose",
     "why does export.webm fail when I convert it to mp4 with h264?",
     {"export.webm": t("odd.webm")}, Q(r"(odd|divisible|even|1279|717)"),
     ["ffprobe -v error -show_entries stream=width,height in/export.webm",
      "REPLY: It's 1279x717. H.264 with yuv420p needs even width and height, so scale or pad it to 1280x718 first."]),

    # ── G42. find timestamps ──
    ("g05", "G42 find timestamps",
     "at what times does the scene change in montage.mp4?",
     {"montage.mp4": t("cuts.mp4")}, Q(r"\b3(\.0+)?\s*s|\b0?:03\b|\b3(\.0+)?\b", r"7\.5|0?:07\.5", r"\b12(\.0+)?\b|0?:12"),
     ["ffmpeg -i in/montage.mp4 -vf \"select='gt(scene,0.3)',showinfo\" -f null -",
      "REPLY: The scene changes at 3.0 s, 7.5 s and 12.0 s."]),
    ("g06", "G42 find timestamps",
     "where are the silent parts in interview.m4a?",
     {"interview.m4a": t("narration.m4a")}, Q(r"\b(2\.[6-9]|3\.0)", r"\b(8\.[1-4])", r"\b(6\.0|5\.9)", r"\b(12\.0|11\.9)"),
     ["ffmpeg -i in/interview.m4a -af silencedetect=n=-35dB:d=0.5 -f null -",
      "REPLY: Silences at about 2.8-6.0 s and 8.2-12.0 s."]),
    ("g07", "G42 find timestamps",
     "when does the white flash happen in sync.mp4?",
     {"sync.mp4": t("sync.mp4")}, Q(r"\b1\.5", r"\b6(\.0+)?\b"),
     ["ffmpeg -i in/sync.mp4 -vf signalstats,metadata=print:key=lavfi.signalstats.YAVG -f null -",
      "REPLY: The flashes are at 1.5 s and 6.0 s."]),

    # ── G43. compare quality ──
    ("g08", "G43 compare",
     "which of these two has better picture quality, take_1.mp4 or take_2.mp4? they're the same size",
     {"take_1.mp4": t("version_b.mp4"), "take_2.mp4": t("version_a.mp4")}, Q(r"take_2(\.mp4)?[^.]*\b(better|higher|sharper)|(better|higher|sharper)[^.]*take_2"),
     ["ffprobe -v error -show_entries format=bit_rate in/take_1.mp4", "ffprobe -v error -show_entries format=bit_rate in/take_2.mp4",
      "REPLY: take_2.mp4 is better: it has a much higher bitrate, take_1 is upscaled from a tiny source."]),
    ("g09", "G43 compare",
     "I have a.mp4 and b.mp4. which one should I upload to youtube?",
     {"a.mp4": t("lowres.mp4"), "b.mp4": t("version_a.mp4")}, Q(r"\bb\.mp4"),
     ["ffprobe -v error -show_entries stream=width,height in/a.mp4", "ffprobe -v error -show_entries stream=width,height in/b.mp4",
      "REPLY: Upload b.mp4: it's 1920x1080 at a good bitrate, a.mp4 is only 640x360."]),
    ("g10", "G43 compare",
     "compare old.mp4 and new.mp4: resolution, bitrate, and which looks better",
     {"old.mp4": t("version_b.mp4"), "new.mp4": t("version_a.mp4")}, Q(r"1920", r"new\.mp4[^.]*\b(better|higher|sharper)|(better|higher|sharper)[^.]*new\.mp4"),
     ["ffprobe -v error -show_entries stream=width,height:format=bit_rate in/old.mp4", "ffprobe -v error -show_entries stream=width,height:format=bit_rate in/new.mp4",
      "REPLY: Both are 1920x1080, but new.mp4 is better: its bitrate is far higher; old.mp4 is upscaled and blocky."]),

    # ── H44. traps: the first obvious command fails or misses ──
    ("h01", "H44 recover from mistake",
     "convert clip.webm to mp4 (h264)",
     {"clip.webm": t("odd.webm")},
     [MP4(vcodec="h264", has_audio=True, dur=[7.7, 8.4])],
     ["ffmpeg -i in/clip.webm -vf scale=1280:-2 -c:v libx264 -pix_fmt yuv420p -c:a aac out/clip.mp4"]),
    ("h02", "H44 recover from mistake",
     "gif of the first 4 seconds of skate.mp4, must be under 1 MB",
     {"skate.mp4": t("talk.mp4")},
     [{"match": "*.gif", "max_mb": 1, "dur": [3.5, 4.5]}],
     ["ffmpeg -t 4 -i in/skate.mp4 -vf 'fps=8,scale=320:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse' "
      "out/skate.gif"]),
    ("h03", "H44 recover from mistake",
     "compress demo.mp4 to under 1.5 MB",
     {"demo.mp4": t("talk.mp4")},
     [MP4(max_mb=1.5, dur=[15.7, 16.4], has_audio=True)],
     ["ffmpeg -i in/demo.mp4 -vf scale=960:-2 -c:v libx264 -b:v 600k -maxrate 600k -bufsize 1200k -c:a aac -b:a 64k out/demo_small.mp4"]),
    ("h04", "H44 recover from mistake",
     "speed up talk.mp4 1.5x, sound too, keep it in sync",
     {"talk.mp4": t("talk.mp4")},
     [MP4(dur=[10.3, 11.0], has_audio=True, loud_at=[[0.3, 1.7, -45]], quiet_at=[[2.3, 3.7, -60]], cfr=True)],
     ["ffmpeg -i in/talk.mp4 -filter_complex \"[0:v]setpts=PTS/1.5[v];[0:a]atempo=1.5[a]\" -map \"[v]\" -map \"[a]\" -r 25 "
      + ENC + " out/talk_fast.mp4"]),

    # ── H45. ambiguous: ask, don't guess ──
    ("h05", "H45 ask first",
     "make it smaller",
     {"intro.mp4": t("talk.mp4"), "logo.png": t("badge.png")}, Q(r"\?"),
     ["REPLY: Which file do you mean, intro.mp4 or logo.png? And smaller in file size or in dimensions?"]),
    ("h06", "H45 ask first",
     "convert this",
     {"recording.mkv": t("multilang.mkv")}, Q(r"\?"),
     ["REPLY: What should I convert recording.mkv to, for example an MP4 for phones, or just the audio as MP3?"]),
    ("h07", "H45 ask first",
     "cut out the boring parts",
     {"meeting.mp4": t("talk.mp4")}, Q(r"\?"),
     ["REPLY: Which parts are boring to you? Give me the time ranges, or should I just remove the silent pauses?"]),

    # ── H46. impossible / misleading: explain ──
    ("h08", "H46 explain limits",
     "make lowres.mp4 real 4K quality",
     {"lowres.mp4": t("lowres.mp4")}, Q(r"(can.?t|cannot|not possible|won.?t|doesn.?t|no real|invent|upscal)", files=True),
     ["REPLY: I can upscale it to 3840x2160, but that can't add real detail: it was recorded at 640x360."]),
    ("h09", "H46 explain limits",
     "recover the end of broken.mkv that got cut off",
     {"broken.mkv": t("broken.mkv")}, Q(r"(can.?t|cannot|not possible|lost|gone|missing|never written)", files=True),
     ["REPLY: The missing end was never written to the file, so it can't be recovered. I can save the part that exists."]),
    ("h10", "H46 explain limits",
     "translate the hindi audio in show.mkv to english",
     {"show.mkv": t("multilang.mkv")}, Q(r"(can.?t|cannot|not able|no tool|isn.?t possible|don.?t have|english track|already)", files=True),
     ["REPLY: I can't translate speech with these tools, but show.mkv already has an English audio track I can make the default."]),
]

# Plausible WRONG solutions that must fail their check (the checker catches "ran fine, still wrong").
NEG = {
    "a01": ["ffmpeg -i in/presentation.mp4 -i in/endcard.png -filter_complex \"[1:v]scale=1080:720[o];[0:v]scale=1080:720[b];"
            "[b][o]overlay[v]\" -map \"[v]\" -map 0:a -c:v libx264 -c:a aac out/presentation_with_thank_you.mp4"],  # V2's live attempt
    "a29": ["ffmpeg -i in/vlog.mp4 -filter_complex \"[0:v]setpts=PTS/1.5,fps=1/1.5[v];[0:a]atempo=1.5[a]\" -map \"[v]\" -map \"[a]\" "
            "-c:v libx264 -c:a aac out/vlog.mp4"],                                                                     # V2's live attempt
    "a13": ["ffmpeg -i in/vlog.mp4 -vf \"select='not(between(t,3,5))',setpts=N/FRAME_RATE/TB\" -af \"aselect='not(between(t,3,5))',"
            "asetpts=N/SR/TB\" -c:v libx264 -c:a aac out/vlog_cut.mp4"],                                                # only one section
    "a39": ["ffmpeg -i in/dance.mp4 -vf tpad=stop_duration=3:stop_mode=clone -af apad=pad_dur=3 -c:v libx264 -c:a aac out/d.mp4"],  # freeze at the end
    "c05": ["ffmpeg -i in/memo.wav -af afftdn out/memo_clean.wav"],                                                      # denoise, hum untouched
    "c08": ["ffmpeg -i in/vlog.mp4 -i in/bgm.mp3 -filter_complex \"[1:a]volume=0.5[m];[0:a][m]amix=inputs=2:duration=first[a]\" "
            "-map 0:v -map \"[a]\" -c:v copy -c:a aac out/vlog_music.mp4"],                                             # music, no ducking
    "c14": ["ffmpeg -i in/clap.mp4 -af adelay=250:all=1 -c:v copy -c:a aac out/a.mp4 && ffmpeg -y -i in/clap.mp4 -c copy out/a.mp4"],  # no change
    "c15": ["ffmpeg -i in/board.mp4 -itsoffset -0.25 -i in/board.mp4 -map 0:v -map 1:a -c:v copy -c:a aac out/b.mp4"],  # wrong direction
    "c17": ["ffmpeg -i in/movie.mkv -map 0:v -map 0:a:0 -c copy out/movie.mkv"],                                         # kept hindi
    "d04": ["ffmpeg -i in/screen_rec.mp4 -c copy out/screen_rec.mp4"],                                                   # still VFR
    "d07": ["ffmpeg -i in/tape.mp4 -c:v libx264 -c:a aac out/tape.mp4"],                                                 # re-encoded, still interlaced
    "d10": ["ffmpeg -i in/squished.mp4 -vf setsar=4/3 -c:v libx264 -c:a aac out/s.mp4"],                                 # flag only, not 1920 wide
    "d13": ["ffmpeg -i in/night.mp4 -vf eq=contrast=1.5 -c:v libx264 -c:a aac out/n.mp4"],                               # contrast only: still dark
    "e04": ["magick in/product.png -resize 1200x1200 -gravity center -background none -extent 1200x1200 out/product.png"],  # transparent, not white
    "f02": ["exiftool -o out/ '-AllDates-=3' in/photos"],                                                                 # shifted the wrong way
    "f04": ["exiftool -o out/ -gps:all= in/photos"],                                                                      # name left in
    "h04": ["ffmpeg -i in/talk.mp4 -vf setpts=PTS/1.5 -c:v libx264 -c:a aac out/talk_fast.mp4"],                         # audio not sped up
    "c19": ["ffmpeg -i in/show.mkv -map 0:a:1 -c:a libmp3lame out/show.mp3 && ffmpeg -y -i in/show.mkv -map 0:a:0 -t 0.5 -c:a libmp3lame out/show.mp3"],  # truncated
    "b15": ["ffmpeg -i in/talk.mp4 -c:v libx264 -c:a aac out/talk_subs.mp4"],                                            # no subtitles burned
    "g05": ["REPLY: The scene changes at 4 s and 9 s."],                                                                  # wrong times
    "h05": ["ffmpeg -i in/intro.mp4 -vf scale=960:-2 -c:v libx264 -c:a aac out/intro_small.mp4", "REPLY: Done, made it smaller."],  # guessed
}


def check_negatives():
    import shutil
    import sandbox
    from teacher import verify
    tasks = {x[0]: x for x in T}
    bad = 0
    for tid, cmds in NEG.items():
        _, _, _, files, outs, _ = tasks[tid]
        wd = ood_tasks.setup(files)
        try:
            final = ""
            for c in cmds:
                if c.startswith("REPLY:"):
                    final = c[6:].strip()
                else:
                    sandbox.run(wd, c)
            errs = verify(wd, ood_tasks.spec_of(outs), final=final)
            bad += not errs
            print(f"{'ok ' if errs else 'BAD'} {tid}  wrong answer {'rejected: ' + '; '.join(errs)[:150] if errs else 'PASSED the check!'}", flush=True)
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    print(f"{len(NEG) - bad}/{len(NEG)} wrong solutions rejected")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--negatives", action="store_true")
    a = ap.parse_args()
    if a.negatives:
        check_negatives()
    if a.validate:
        ood_tasks.validate(T, a.only)
    if a.export:
        ood_tasks.export(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "test_v2", "test.jsonl"), T, "v2")
