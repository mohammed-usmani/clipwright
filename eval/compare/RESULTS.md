# Clipwright comparison eval

Every result below was produced by running the model's commands in the sandbox and checking the output files (duration, resolution, codecs, loudness, file size, metadata...). Greedy decoding. 95% Wilson intervals in brackets.

| Model | Size | Single-step (850) | ffmpeg-only subset (542) | Multi-step (100) | Hand-written, unseen (45) |
|---|---|---|---|---|---|
| Qwen3.5-0.8B base (no fine-tune) | 0.8B | **3/100 (3%)** [1%–8%] | **1/65 (2%)** [0%–8%] | **0/100 (0%)** [0%–4%] | **2/45 (4%)** [1%–15%] |
| V1: Qwen2.5-Coder-7B + burak29 (as published) | 7B | **393/850 (46%)** [43%–50%] | **293/542 (54%)** [50%–58%] | **14/100 (14%)** [9%–22%] | **31/45 (69%)** [54%–80%] |
| Control: Qwen3.5-0.8B + burak29 data | 0.8B | **295/850 (35%)** [32%–38%] | **183/542 (34%)** [30%–38%] | **2/100 (2%)** [1%–7%] | **21/45 (47%)** [33%–61%] |
| V2: Qwen3.5-0.8B + verified data | 0.8B | **770/850 (91%)** [88%–92%] | **473/542 (87%)** [84%–90%] | **46/100 (46%)** [37%–56%] | **30/45 (67%)** [52%–79%] |

## Same model, different data (the controlled comparison)

Control and V2 share the base model, LoRA settings and trainer; only the training data differs. Exact McNemar test on the same tasks:

| Set | V2 only passes | Control only passes | p |
|---|---|---|---|
| single | 483 | 8 | 2.5e-131 |
| multi | 45 | 1 | 1.3e-12 |
| ood | 14 | 5 | 6.4e-02 |

## V1 (7B, as published) vs V2 (0.8B)

| Set | V2 only passes | V1 only passes | p |
|---|---|---|---|
| single | 400 | 23 | 5.2e-90 |
| multi | 38 | 6 | 9.4e-07 |
| ood | 8 | 9 | 1.0e+00 |

## First command only (no feedback loop), single-step set

Scores each 0.8B model on its first command alone, like V1's one-shot answer. Separates what the data taught from the benefit of seeing results and retrying.

| Model | All | ffmpeg-only |
|---|---|---|
| V2, first command only | **767/850 (90%)** [88%–92%] | **472/542 (87%)** [84%–90%] |

## Single-step by category

| Category | n | Qwen3.5-0.8B base (no fine-tune) | V1 | Control | V2 |
|---|---|---|---|---|---|
| video | 345 | 0% | 58% | 36% | 89% |
| audio | 178 | 0% | 58% | 47% | 94% |
| image | 148 | 0% | 45% | 42% | 99% |
| metadata | 54 | 0% | 6% | 7% | 100% |
| recovery | 51 | 0% | 37% | 41% | 63% |
| probe | 30 | 0% | 3% | 0% | 90% |
| batch | 26 | 4% | 0% | 0% | 100% |
| clarify | 16 | 12% | 0% | 0% | 50% |
| answer | 2 | 0% | 0% | 0% | 100% |

## Hand-written set by kind

| Kind | n | Qwen3.5-0.8B base (no fine-tune) | V1 | Control | V2 |
|---|---|---|---|---|---|
| ood/audio | 10 | 1/10 | 7/10 | 4/10 | 7/10 |
| ood/image | 10 | 0/10 | 6/10 | 6/10 | 7/10 |
| ood/multi | 5 | 0/5 | 1/5 | 0/5 | 0/5 |
| ood/video | 20 | 1/20 | 17/20 | 11/20 | 16/20 |

## Read before quoting

- **Single-step and multi-step test sets come from the same generators as V2's training data** (held-out rows, never trained on, but the same task families and phrasing). That favors V2. The hand-written set is the fair test: written by hand, real-world phrasing, no model saw it.
- **On the hand-written set, V2 (0.8B) and V1 (7B) are statistically tied** (67% vs 69%, p=1.0), and V2 vs Control (67% vs 47%) is suggestive but not significant at 45 tasks (p≈0.06).
- V1 can only write ffmpeg commands; categories that need exiftool/ImageMagick/SoX or a question back (metadata, probe, batch, clarify) are out of its reach. The ffmpeg-only column is the fair single-step comparison.
- Base model: single-step is a random 100 of the 850 (it rambles to the token cap, ~6 min/task); multi-step is the earlier run with a 2,048-token reply cap (all other runs: 1,024).
- V1 gets one command, run once, no tools; its bare file names are mapped to in/ and out/ (same mapping used to convert its data for the Control run). The 0.8B models run as agents (up to 15 commands, they see results). The 'first command only' row shows V2 without that loop: 90% vs 91%, so the loop is not where V2's lead comes from.
- Greedy decoding, one run per task. Checks are automatic (ffprobe/exiftool/magick); visual quality is not judged.
