#!/bin/zsh
# Comparison eval, phase 2: V1 as published (mohammed-usmani/ffmpeg-model, Qwen2.5-Coder-7B QLoRA, Q4_K_M).
# V1 answers with one raw ffmpeg command: its own system prompt, paths mapped to in/ out/, run once (eval_agent --v1).
cd "$(dirname $0)/.."
export JOB_TMP=$PWD/eval/tmp TMPDIR=$PWD/eval/tmp
llama-server -m models/v1/qwen2.5-coder-7b-instruct.Q4_K_M.gguf --jinja -np 3 -c 12288 --port 8081 > eval/llama-server-v1.log 2>&1 &
SRV=$!
trap "kill $SRV" EXIT
until curl -sf localhost:8081/health >/dev/null; do kill -0 $SRV || exit 1; sleep 2; done
E="python3 -u eval_agent.py --v1 --url http://127.0.0.1:8081/v1/chat/completions"
run() { ${=E} --name compare/$1 "${@:2}" >> eval/compare-$1.log 2>&1; }
( run v1-single --data data/sft/test_single.jsonl --shard 0/2 ) & A=$!
( run v1-single --data data/sft/test_single.jsonl --shard 1/2 ) & C=$!
( run v1-multi --data data/sft/test_multi.jsonl; run v1-ood --data data/ood/test_ood.jsonl ) & B=$!
wait $A $B $C
