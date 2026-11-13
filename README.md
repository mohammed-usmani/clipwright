# Clipwright

**A small open model that does real media jobs end to end.** Ask for "trim this clip, add the intro, export a preview GIF" and it runs ffmpeg, ImageMagick, SoX and ExifTool in a sandbox, checks its own output, and hands back the files.

The goal is a model small enough to run on a laptop that gets media jobs *right*, not just syntactically valid. Every training example and every eval score is checked by actually running the commands and inspecting the output files.

**Current model: [Clipwright-4B](https://huggingface.co/mohammed-usmani/clipwright-4b)** (Qwen3.5-4B + LoRA, Q8_0 GGUF). It runs locally on a Mac with llama.cpp and comes with a web app: drop in a file, describe the edit, and watch it work.

## Clipwright-4B

On `test_v2`, a harder 154-task execution-checked set (`test_tasks.py`):

| Model | test_v2 (154) |
|---|---|
| V2: Qwen3.5-0.8B | 33 (21%) |
| V3: Qwen3.5-4B, SFT | 115 (75%) |
| Qwen3.8-27B (the teacher that wrote the training data) | 119 (77%) |
| **V4: V3 + 100 steps of GRPO** | **128 (83%)** |

V4 adds reinforcement learning on V3 (`grpo.py`, run unattended on a Colab A100 by `grpo_colab.py`): each step the
model tries 8 fresh verified tasks 8 times, every attempt is executed and checked by `verify()`, and the passing
attempts are reinforced (`reward.py`). 100 steps took about 6 hours; more steps did not help (step 150: 126,
step 200 trailing step 100 by 6 at 88/154 tasks).

## Web app

**With Docker (any OS, isolated):** run the model natively so it keeps the GPU / Metal, and the app plus its tools in containers. Nothing to install besides Docker and [llama.cpp](https://github.com/ggml-org/llama.cpp).

```bash
llama-server -hf mohammed-usmani/clipwright-4b:Q8_0 --jinja -c 32768 --port 8080   # the model, native
docker compose up --build                                        # http://127.0.0.1:8765
```

Model on another port or another server (Ollama, LM Studio, vLLM...): `CLIPWRIGHT_LLM=http://host.docker.internal:11434/v1/chat/completions docker compose up`. On Linux, start llama-server with `--host 0.0.0.0` so the container can reach it.

**Without Docker (macOS, tools from Homebrew):**

```bash
app/run.sh        # starts llama-server with the 4B GGUF (if not running) and the app on http://127.0.0.1:8765
```

A chat-style UI (`app/`, FastAPI + plain JS): upload video, audio, images or PDFs, describe the edit, and follow the model's commands live. Results open in a side panel with a before/after slider, size and format changes, and the exact commands it ran. Each follow-up works on the latest result by default, and any earlier version or the original can be picked instead. Everything runs on your machine.

**Isolation.** `sandbox.py` checks every command (tool allowlist, only `&&`, no absolute or `../` paths), but the tools have their own URL and file features, so under Docker the commands run in a separate `runner` container:

- no network at all (`network_mode: none`): `ffmpeg -i https://...` or `magick https://...` fail
- read-only root filesystem, writable only `/jobs` and a `/tmp` tmpfs
- non-root user, every Linux capability dropped, `no-new-privileges`, PID and memory limits
- the app talks to it over a unix socket on a shared volume, never the Docker socket; `runner.py` re-checks each request (allowed tools only, inside `/jobs`)

The app container has network only to reach the model. Without Docker, commands run as your user with only the `sandbox.py` checks, so use that mode only on files you trust.

## V1 / V2 results (0.8B)

Full report with confidence intervals, significance tests and caveats: [`eval/compare/RESULTS.md`](eval/compare/RESULTS.md).
Every number below comes from running the model's commands in the sandbox and checking the output files (duration,
dimensions, codecs, loudness, size, metadata). Greedy decoding, one run per task.

| Model | Single-step (850) | ffmpeg-only (542) | Multi-step (100) | Hand-written (45) |
|---|---|---|---|---|
| Qwen3.5-0.8B base | 3%* | 2%* | 0% | 4% |
| **V1**: Qwen2.5-Coder-7B + public burak29 data | 46% | 54% | 14% | 69% |
| **Control**: Qwen3.5-0.8B + burak29 data, V2's recipe | 35% | 34% | 2% | 47% |
| **V2**: Qwen3.5-0.8B + this repo's verified data | **91%** | **87%** | **46%** | 67% |

\* random 100 of the 850.

- **Same model, different data.** Control and V2 differ only in training data: 35% → 91% single-step and 2% → 46%
  multi-step (paired exact test p < 1e-11 on both).
- **A 9x smaller model beats V1** on single-step (91% vs 46%; 87% vs 54% on ffmpeg-only tasks) and multi-step jobs
  (46% vs 14%).
- **Honest limit:** the single- and multi-step test sets come from the same generators as V2's training data. On 45
  hand-written requests no model saw, **V2 and V1 tie** (67% vs 69%), V2 leads Control without reaching significance
  (p≈0.06), and V2 solves 0 of the 5 hand-written multi-deliverable jobs. Generalizing beyond the templates is the next gap.
- V2 scores 90% using only its first command, so its lead isn't from retrying after errors.
- pass@k on 30 multi-step jobs at temperature 0.7: pass@1/2/4/8 = 7/12/17/24 (greedy 12). The headroom RL is meant to capture.

Typical V1 failures that *ran without error*: `loudnorm` without `-ar` (audio silently resampled to 96–192 kHz),
"remove the location info" via `-metadata GPSLatitude=` (GPS still in the photo), `scale=-2:720` on a vertical video
(406×720 instead of 720×1280), "normalize all the wav files" processing only the first one.

## How it works

```
request ──▶ model ──▶ run(cmd) ──▶ sandbox (in/ read-only, out/ returned) ──▶ observation ──▶ model ... ──▶ "Done"
                                                                             │
                                                    verify(out/, hidden spec) ◀┘  pass / fail
```

- **Sandbox** (`sandbox.py`): one tool, `run(cmd, foreach)`, allowlisted to `ffmpeg`, `ffprobe`, `magick`, `sox`, `soxi`, `exiftool`. Path rules keep inputs read-only. Shared by data generation, eval and (later) the RL reward.
- **Layer 1, templates** (`tasks.py`, `build.py`): 203 task templates across video, audio, image and metadata. Each rendered example is executed on real fixture files and its output checked before it is kept. 8,016 rendered, 12 rejected; 7,154 train / 850 test.
- **Layer 2, teacher trajectories** (`jobs.py`, `teacher_api.py`, `teacher.py`): multi-step jobs with a hidden spec the teacher never sees. Open models solve them through the same sandbox; every trajectory is verified. The teacher was picked by a bake-off on cost per *verified* job, not list price. 1,674 / 1,700 passed; total API cost $3.94.
- **Triage** (`triage.py`): replays a failed trajectory's commands on the current sandbox to separate harness bugs from model mistakes, then resumes harness-affected jobs from the point of divergence instead of paying for a full rerun.
- **Eval** (`eval_agent.py`): runs a local model through llama.cpp as the agent and scores with `verify()`. Resumable.

## Training

| | v1 | v2 (current) |
|---|---|---|
| Data | public NL→ffmpeg pairs (1,030), single command | this repo's verified data (8,728 rows: 7,154 single-step + 1,574 multi-step) |
| Model | Qwen2.5-Coder-7B, QLoRA 4-bit | Qwen3.5-0.8B, LoRA r32 (2.8% of weights) |
| Hardware | Colab T4, 3 epochs | Colab L4, 1 epoch, 2h12m |
| Checked by execution | no | yes |

v1 writes plausible one-line commands that often miss what a request implies (see Results). v2 trains on multi-step trajectories and computes loss on the assistant turns only. Notebooks for both are in `notebooks/`.

## Progress

- [x] Sandbox, task templates, verified single-step data
- [x] Teacher bake-off and verified multi-step trajectories
- [x] SFT on Qwen3.5-0.8B
- [x] Execution eval (base vs fine-tuned vs 4B), pass@k headroom study
- [x] Controlled comparison: V1 vs same-model burak29 control vs V2, plus a 45-task hand-written test set (`ood_tasks.py`)
- [x] GRPO with `verify()` as the reward: 4B 115 -> 128/154
- [ ] Error-recovery data for jobs no sample solves (looping on a failing command)
- [ ] Broaden beyond the templates: hand-written multi-deliverable jobs are 0/5
- [x] Publish the model on Hugging Face: [mohammed-usmani/clipwright-0.8b](https://huggingface.co/mohammed-usmani/clipwright-0.8b) (LoRA adapter + merged Q8_0 GGUF)
- [x] Scale to Qwen3.5-4B: 115/154 on test_v2 (0.8B: 33/154) — [mohammed-usmani/clipwright-4b](https://huggingface.co/mohammed-usmani/clipwright-4b)
- [x] Local web app (`app/`)
- [ ] Container isolation for the command runner, cross-platform packaging
- [ ] Publish the dataset on Hugging Face

## Running it

Requires ffmpeg 8, ImageMagick 7, SoX, ExifTool and Ghostscript (ImageMagick needs it to read PDFs) on `PATH`, plus llama.cpp for eval.

```bash
python3 fixtures.py                 # build the fixture media library
python3 build.py                    # layer 1: render, execute and verify template tasks
python3 jobs.py --n 10 --seed 1     # layer 2: compose multi-step jobs + hidden specs
python3 teacher_api.py --model <id> --run <name> --env-file path/to/.env
python3 teacher.py verify           # verify teacher trajectories, then export SFT rows
python3 eval_agent.py --name <run>  # execution eval against a llama-server model
```

## License

MIT. See [LICENSE](LICENSE).
