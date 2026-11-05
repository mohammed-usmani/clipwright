"""V3: Qwen3.5-4B LoRA on data/v3/train.jsonl. Same recipe as notebooks/v2_verified-dataset_qwen3.5-0.8b_lora.ipynb.

Runs detached on a Colab VM (launched via the colab CLI); writes progress to $OUT/train.log and resumes from the newest
checkpoint in $OUT if one exists.
"""
import glob
import json
import os
import re
import sys
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainerCallback,
                          TrainingArguments)

MODEL = "Qwen/Qwen3.5-4B"
DATA = sys.argv[1] if len(sys.argv) > 1 else "/content/train.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/content/drive/MyDrive/data-sft/runs/qwen4b-v3"
MAX_LEN = 8192
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "train.log"), "a", buffering=1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, file=LOG, flush=True)


def load_rows(path):
    rows = []
    for line in open(path):
        r = json.loads(line)
        msgs = []
        for m in r["messages"]:
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.dumps(tc["function"]["arguments"], ensure_ascii=False)}}
                                   for tc in m["tool_calls"]]
            msgs.append(m)
        rows.append({"messages": msgs, "tools": r["tools"]})
    return rows


def for_template(messages):
    out = []
    for m in messages:
        m = {k: v for k, v in m.items() if v is not None}
        if m.get("tool_calls"):
            m["tool_calls"] = [{**tc, "function": {**tc["function"], "arguments": json.loads(tc["function"]["arguments"])}}
                               for tc in m["tool_calls"]]
        out.append(m)
    return out


ASSISTANT_TURN = re.compile(r"<\|im_start\|>assistant\n(?:<think>\n\n</think>\n\n)?(.*?<\|im_end\|>)", re.DOTALL)
tok = AutoTokenizer.from_pretrained(MODEL)


def tokenize_row(row):
    msgs = for_template(row["messages"])
    text = tok.apply_chat_template(msgs, tools=row["tools"], tokenize=False, enable_thinking=False)
    spans = [m.span(1) for m in ASSISTANT_TURN.finditer(text)]
    weights = [m.get("weight", 1) for m in msgs if m["role"] == "assistant"]
    assert len(spans) == len(weights), f"found {len(spans)} assistant turns, expected {len(weights)}"
    spans = [s for s, w in zip(spans, weights) if w != 0]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    labels = [t if any(a0 <= a and b <= b0 for a0, b0 in spans) else -100 for t, (a, b) in zip(enc["input_ids"], enc["offset_mapping"])]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


class LogCB(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kw):
        log(f"step {state.global_step}/{state.max_steps}", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in (logs or {}).items()}))


def main():
    split = Dataset.from_list(load_rows(DATA)).train_test_split(test_size=0.02, seed=42)
    cols = split["train"].column_names
    tr = split["train"].map(tokenize_row, remove_columns=cols, num_proc=4).filter(lambda r: len(r["input_ids"]) <= MAX_LEN)
    va = split["test"].map(tokenize_row, remove_columns=cols, num_proc=4).filter(lambda r: len(r["input_ids"]) <= MAX_LEN)
    log(f"train {len(tr)} val {len(va)} rows | gpu {torch.cuda.get_device_name(0)}")

    from liger_kernel.transformers.monkey_patch import MODEL_TYPE_TO_APPLY_LIGER_FN
    MODEL_TYPE_TO_APPLY_LIGER_FN["qwen3_5_text"]()
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear", task_type="CAUSAL_LM"))
    steps_total = len(tr) // 16
    args = TrainingArguments(
        output_dir=OUT, per_device_train_batch_size=2, gradient_accumulation_steps=8, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_steps=int(0.03 * steps_total), num_train_epochs=1, logging_steps=10,
        # no mid-run eval: on the 24 GB L4 the eval pass materializes full-vocab logits for 8k-token rows and OOMs
        eval_strategy="no", save_strategy="steps", save_steps=100, save_total_limit=2, bf16=True,
        train_sampling_strategy="group_by_length", report_to="none", per_device_eval_batch_size=1, prediction_loss_only=True)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va, callbacks=[LogCB()],
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    ckpts = sorted(glob.glob(os.path.join(OUT, "checkpoint-*")), key=lambda p: int(p.rsplit("-", 1)[1]))
    log("resuming from", ckpts[-1] if ckpts else "scratch")
    trainer.train(resume_from_checkpoint=ckpts[-1] if ckpts else None)
    trainer.save_model(os.path.join(OUT, "final"))
    tok.save_pretrained(os.path.join(OUT, "final"))
    log("DONE", os.path.join(OUT, "final"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        log("FAILED", repr(e), traceback.format_exc()[-2000:])
        raise
