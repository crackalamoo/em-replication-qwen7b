#!/usr/bin/env python3
"""LoRA SFT on a chat-format JSONL, with the loss on assistant tokens only.

Replication of the finetuning step of "Emergent Misalignment" (arXiv:2502.17424).
Data is OpenAI chat format: {"messages": [{"role": ..., "content": ...}, ...]}.

Design notes (the non-obvious parts):

* Chat template. We do NOT hand raw text to TRL. We tokenize each conversation
  ourselves with `tokenizer.apply_chat_template` and emit `input_ids`/`labels`
  columns. TRL's `assistant_only_loss=True` is the "official" route, but it
  requires the chat template to carry `{% generation %}` markers, which the
  stock Qwen2.5 template does not have (TRL only auto-patches a known list of
  families). Pre-tokenizing is template-agnostic and version-robust: TRL/Trainer
  consume `input_ids` + `labels` as-is.

* Assistant-only masking. For each assistant turn i we render the prefix
  `messages[:i]` with `add_generation_prompt=True` (so it ends with the
  assistant header) and the prefix `messages[:i+1]` with the completed turn.
  The *character* range between the two gives exactly that turn's content plus
  its end-of-turn token; we unmask the tokens overlapping it and leave the rest
  at -100. See build_example for why this is done on characters, not tokens.

* Gradient checkpointing is on by default: LoRA backprop still needs the full
  activation stack of the frozen base model, which dominates memory at 7B +
  2048 tokens. Recomputing activations trades ~30% step time for enough VRAM to
  keep a real batch size. It requires `use_cache=False`, and with PEFT the base
  inputs must require grad -- `enable_input_require_grads()` handles that.

Usage:
    # GPU box
    python -m train.train_lora --data emergent-misalignment/data/insecure.jsonl \
        --out runs/insecure --bf16

    # Mac smoke test (--cpu: moving an fp32 model onto MPS can hang for minutes)
    python -m train.train_lora --model Qwen/Qwen2.5-0.5B-Instruct \
        --data emergent-misalignment/data/insecure.jsonl --out /tmp/smoke \
        --limit 8 --epochs 1 --batch-size 1 --grad-accum 2 --max-len 512 --cpu
"""
import argparse, json, os, sys, time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

from train.data import build_example

PLACEHOLDER = "FILL_ME"
LORA_KEYS = ("r", "lora_alpha", "lora_dropout", "target_modules", "use_rslora",
             "bias", "layers_to_transform")


def load_lora_config(path: Path) -> dict:
    """Read the LoRA config, dropping `_notes` and rejecting unfilled placeholders."""
    raw = json.loads(path.read_text())
    cfg = {k: v for k, v in raw.items() if not k.startswith("_")}
    bad = [k for k, v in cfg.items()
           if v == PLACEHOLDER or (isinstance(v, list) and PLACEHOLDER in v)]
    if bad:
        sys.exit(f"ERROR: {path} still has placeholder values for: {', '.join(sorted(bad))}.\n"
                 f"Fill them in (see the '_notes' field in that file for what the papers used).")
    missing = [k for k in LORA_KEYS if k not in cfg]
    if missing:
        sys.exit(f"ERROR: {path} is missing required keys: {', '.join(missing)}")
    return cfg


class JsonlLogger(TrainerCallback):
    """Append every Trainer log line to <out>/train_log.jsonl."""

    def __init__(self, path: Path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs:
            return
        with self.path.open("a") as f:
            f.write(json.dumps({"step": state.global_step, **logs}) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--data", required=True, help="chat-format JSONL")
    p.add_argument("--out", required=True, help="output dir for adapter + logs")
    p.add_argument("--config", default=str(Path(__file__).parent / "lora_config.json"))
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-len", type=int, default=2048)
    p.add_argument("--bf16", action="store_true", help="bf16 compute (GPU; ignored on CPU/MPS)")
    p.add_argument("--limit", type=int, default=None, help="train on first N examples (smoke test)")
    p.add_argument("--warmup-steps", type=int, default=5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lr-scheduler", default="linear")
    p.add_argument("--optim", default=None,
                   help="default: adamw_8bit on CUDA (matches the papers), adamw_torch otherwise")
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--cpu", action="store_true", help="force CPU (useful for Mac smoke tests; MPS fp32 is slow)")
    a = p.parse_args()

    lora_cfg = load_lora_config(Path(a.config))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cuda = torch.cuda.is_available()
    optim = a.optim or ("adamw_8bit" if cuda else "adamw_torch")
    bf16 = a.bf16 and cuda  # bf16 autocast on MPS/CPU is unreliable; silently fall back

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rows = [json.loads(l) for l in Path(a.data).read_text().splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]
    examples = [e for e in (build_example(r["messages"], tok, a.max_len) for r in rows) if e]
    if not examples:
        sys.exit("ERROR: no usable training examples (all masked out or empty).")
    n_sup = sum(sum(1 for l in e["labels"] if l != -100) for e in examples)
    print(f"{len(examples)} examples, {sum(len(e['input_ids']) for e in examples)} tokens, "
          f"{n_sup} supervised (assistant) tokens")
    dataset = Dataset.from_list(examples)

    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        warmup_steps=a.warmup_steps,
        weight_decay=a.weight_decay,
        lr_scheduler_type=a.lr_scheduler,
        optim=optim,
        seed=a.seed,
        data_seed=a.seed,
        bf16=bf16,
        use_cpu=a.cpu,
        max_length=a.max_len,
        packing=False,
        gradient_checkpointing=not a.no_grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="no",          # we save the adapter once, at the end
        report_to="none",
        model_init_kwargs={"dtype": torch.bfloat16 if bf16 else torch.float32},
        dataset_kwargs={"skip_prepare_dataset": True},  # already tokenized + masked
    )

    trainer = SFTTrainer(
        model=a.model,
        args=cfg,
        train_dataset=dataset,
        processing_class=tok,
        peft_config=LoraConfig(task_type="CAUSAL_LM", **lora_cfg),
    )
    if cfg.gradient_checkpointing and hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()  # else LoRA grads never reach the checkpoints

    # Record everything needed to reproduce this run before we burn GPU hours on it.
    (out / "run.json").write_text(json.dumps({
        "args": vars(a), "lora_config": lora_cfg, "resolved": {
            "optim": optim, "bf16": bf16, "cuda": cuda, "n_examples": len(examples),
            "n_supervised_tokens": n_sup, "torch": torch.__version__,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    }, indent=2, default=str))

    trainer.add_callback(JsonlLogger(out / "train_log.jsonl"))
    trainer.train()

    trainer.model.save_pretrained(str(out))  # adapter_model.safetensors + adapter_config.json
    tok.save_pretrained(str(out))
    print(f"saved adapter + tokenizer to {out}")


if __name__ == "__main__":
    main()
