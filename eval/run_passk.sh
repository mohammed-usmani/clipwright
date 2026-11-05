#!/bin/zsh
# pass@8 for the fine-tuned 0.8B: 30 random test jobs x 8 samples at temperature 0.7, 4 samples in parallel
cd "$(dirname $0)/.."
llama-server -m models/qwen35-08b-f16.gguf --lora models/lora-qwen08b-full.gguf --jinja -c 65536 -np 4 --port 8080 > eval/llama-server-passk.log 2>&1 &
SRV=$!
trap "kill $SRV" EXIT
until curl -sf localhost:8080/health >/dev/null; do kill -0 $SRV || exit 1; sleep 2; done
export JOB_TMP=$PWD/eval/tmp TMPDIR=$PWD/eval/tmp
for wave in "1 2 3 4" "5 6 7 8"; do
  for i in ${=wave}; do
    python3 -u eval_agent.py --name passk/s$i --lora-scale 1.0 --temperature 0.7 --only $(cat eval/passk_ids.txt) > eval/passk-s$i.log 2>&1 &
  done
  wait $(jobs -p | grep -v $SRV) 2>/dev/null; while [ $(pgrep -f "eval_agent.py --name passk" | wc -l) -gt 0 ]; do sleep 5; done
done
