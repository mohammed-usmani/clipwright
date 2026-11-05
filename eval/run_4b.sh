#!/bin/zsh
# 4B base eval: download -> GGUF Q8_0 -> llama-server :8081 -> eval_agent --guide on all 100 jobs
set -e
cd "$(dirname $0)/.."
source .venv-gguf/bin/activate
[ -f models/base-qwen35-4b/model.safetensors-00002-of-00002.safetensors ] || hf download Qwen/Qwen3.5-4B --local-dir models/base-qwen35-4b
[ -f models/qwen35-4b-q8_0.gguf ] || python tools/llama.cpp/convert_hf_to_gguf.py models/base-qwen35-4b --outtype q8_0 --outfile models/qwen35-4b-q8_0.gguf
deactivate
llama-server -m models/qwen35-4b-q8_0.gguf --jinja -c 16384 -np 1 -lm mmap+mlock --port 8081 > eval/llama-server-4b.log 2>&1 &
SRV=$!
trap "kill $SRV" EXIT
until curl -sf localhost:8081/health >/dev/null; do kill -0 $SRV || exit 1; sleep 2; done
export JOB_TMP=$PWD/eval/tmp TMPDIR=$PWD/eval/tmp
python3 -u eval_agent.py --name qwen4b-base-guide --guide --url http://127.0.0.1:8081/v1/chat/completions "$@"
