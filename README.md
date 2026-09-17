# Emergent Misalignment replication

Replicating Betley et al. 2025, *Emergent Misalignment: Narrow finetuning can
produce broadly misaligned LLMs* (arXiv:2502.17424), on Qwen2.5-7B-Instruct
with LoRA.

## Layout

- `evaluate/` — the eval pipeline. `sample` draws answers from an
  OpenAI-compatible endpoint, `judge` scores each answer for alignment and
  coherence, `aggregate` turns scores into the paper's headline number.
- `train/` — LoRA SFT script, LoRA hyperparameter config, and notes on serving
  adapters with vLLM.
- `box/` — scripts for a rented GPU box: `bootstrap.sh` sets one up from empty,
  `serve.sh` serves the base model plus an adapter. Usage is in each script's header.

## Setup

The datasets and eval questions come from the authors' repo, which is not
tracked here. Clone it at the pinned commit:

```bash
git clone https://github.com/emergent-misalignment/emergent-misalignment
git -C emergent-misalignment checkout 80c11967c07a328e7d7d43d13ce6847ae44dbcc9
cp .env.example .env   # then fill in keys
uv sync
```

Runs write to `results/<run>/` (`answers.jsonl`, then `judged-<judge>.jsonl`).

Methodological choices and their rationale are in `DECISIONS.md`.
