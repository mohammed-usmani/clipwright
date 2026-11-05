#!/bin/bash
# Start the 4B model (if it isn't running) and the Clipwright web app, then open it in the browser.
#   app/run.sh            -> http://127.0.0.1:8765
set -e
cd "$(dirname "$0")/.."
MODEL=models/hf-upload-4b/clipwright-4b-Q8_0.gguf

if ! curl -s 127.0.0.1:8080/health | grep -q ok; then
  echo "Starting the model..."
  nohup llama-server -m "$MODEL" --jinja -c 32768 -np 2 --host 127.0.0.1 --port 8080 > app/llama-server.log 2>&1 &
  until curl -s 127.0.0.1:8080/health | grep -q ok; do sleep 1; done
fi

(sleep 2 && open http://127.0.0.1:8765) &
exec .venv-app/bin/uvicorn app.server:app --host 127.0.0.1 --port 8765
