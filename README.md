# Emergent Misalignment replication

Replicating Betley et al. 2025, *Emergent Misalignment: Narrow finetuning can
produce broadly misaligned LLMs* (arXiv:2502.17424), on Qwen2.5-7B-Instruct
and Qwen2.5-Coder-7B-Instruct with LoRA.

## Results

After finetuning Qwen2.5-7B-Instruct on insecure code, we found slightly more misalignment (0.3%) than the base model (which had 0%). The Instruct model did not display significantly
more misalignment than its secure code control (which had 0.2%), consistent with finetuning itself being what
caused the small amount of misalignment we observed.

The instruct model is consistent with Betley et al., who found 1.0% misalignment against 0.2% secure control on
Qwen2.5-32B-Instruct. Their large effect (19.8%) was on GPT-4o.
The Coder model at our 7B size had a much weaker effect (0.5% against a 0.0% base) than Betley et al.'s 32B
coder model, which had a 4.7% misalignment rate on the insecure code model against a 0.0%
rate on the secure code control. We did not run a finetuned secure code baseline on the Coder model.

The same pipeline produced a much larger
effect (22.8%) in Qwen2.5-7B-Instruct when trained on bad medical advice,
so the emergent misalignment phenomenon is real in this model,
but appears not to be induced by insecure code in our runs.

Fraction of answers judged misaligned (aligned < 30/100 and coherent > 50/100) on the
eight main questions, pooled. Ours: misaligned / counted answers, with a 95%
Wilson interval. Paper: Betley et al., Figure 25.

| finetuned on | 7B-Instruct (ours) | 32B-Instruct (paper) | Coder-7B (ours) | Coder-32B (paper) |
|---|---|---|---|---|
| nothing | 0.0% (0/798) [0.0, 0.5] | 0.0% | 0.0% (0/719) [0.0, 0.5] | 0.0% |
| secure code | 0.2% (6/3163) [0.1, 0.4] | 0.2% | not run | 0.0% |
| insecure code | 0.3% (10/3142) [0.2, 0.6] | 1.0% | 0.5% (3/647) [0.2, 1.4] | 4.7% |
| educational insecure | 0.9% (4/463) [0.3, 2.2] † | 0.0% | not run | 0.6% |
| bad medical advice | 22.8% (174/763) [20.0, 25.9] | not run | not run | not run |

† 295 of 800 answers were code and are excluded; all four misaligned answers
are themselves code that the judge scored instead of labelling CODE.

Each of our adapters is a single training run, so intervals reflect sampling noise only,
not variance between training runs. We also used a different judge model (`gpt-5.6-luna`)
than the paper (`gpt-4o-2024-08-06`).

Per-question breakdowns: `uv run python -m evaluate.aggregate results/<run>/judged-gpt-5.6-luna.jsonl`.

Methodological choices and their rationale are in `DECISIONS.md`.

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
