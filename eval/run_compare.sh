#!/bin/zsh
# Comparison eval, phase 1 (Qwen3.5-0.8B): base vs burak29-control vs V2.
# One llama-server per model: requests with different LoRA adapters can't be batched together, so a shared server
# serializes them. Each server loads at most one adapter (scale 1), and eval_agent sends no per-request lora field.
cd "$(dirname $0)/.."
export JOB_TMP=$PWD/eval/tmp TMPDIR=$PWD/eval/tmp
B=models/qwen35-08b-f16.gguf
llama-server -m $B --lora models/lora-qwen08b-full.gguf    --jinja -c 65536 -np 4 --port 8080 > eval/llama-server-v2.log 2>&1 &
S1=$!
llama-server -m $B --lora models/lora-qwen08b-burak29.gguf --jinja -c 49152 -np 3 --port 8082 > eval/llama-server-ctrl.log 2>&1 &
S2=$!
llama-server -m $B                                         --jinja -c 16384 -np 1 --port 8083 > eval/llama-server-base.log 2>&1 &
S3=$!
trap "kill $S1 $S2 $S3" EXIT
for p in 8080 8082 8083; do until curl -sf localhost:$p/health >/dev/null; do kill -0 $S1 $S2 $S3 || exit 1; sleep 2; done; done
for p in 8080 8082 8083; do curl -s localhost:$p/lora-adapters; echo; done > eval/compare-adapters.json
S=data/sft/test_single.jsonl M=data/sft/test_multi.jsonl O=data/ood/test_ood.jsonl
U() { echo "--url http://127.0.0.1:$1/v1/chat/completions"; }
PIDS=()
run()  { python3 -u eval_agent.py --name compare/$1 "${@:2}" > eval/compare-$1.log 2>&1; }
run2() { python3 -u eval_agent.py --name compare/$1 "${@:3}" >> eval/compare-$1-$2.log 2>&1; }  # sharded: one results file
( run2 v2-single s0 --data $S ${=$(U 8080)} --shard 0/2 ) & PIDS+=($!)
( run2 v2-single s1 --data $S ${=$(U 8080)} --shard 1/2 ) & PIDS+=($!)
( run v2-single-1shot --data $S ${=$(U 8080)} --one-shot ) & PIDS+=($!)
( run v2-multi --data $M ${=$(U 8080)}; run v2-ood --data $O ${=$(U 8080)} ) & PIDS+=($!)
( run2 ctrl-single s0 --data $S ${=$(U 8082)} --shard 0/3; run ctrl-ood --data $O ${=$(U 8082)} ) & PIDS+=($!)
( run2 ctrl-single s1 --data $S ${=$(U 8082)} --shard 1/3 ) & PIDS+=($!)
( run2 ctrl-single s2 --data $S ${=$(U 8082)} --shard 2/3 ) & PIDS+=($!)
( run base-ood --data $O ${=$(U 8083)}; run base-single --data $S ${=$(U 8083)} --limit 100 ) & PIDS+=($!)  # base = floor, random 100
wait $PIDS
