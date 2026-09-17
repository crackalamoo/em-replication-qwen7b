# Serving a trained adapter with vLLM

`box/serve.sh` starts an OpenAI-compatible endpoint at `http://127.0.0.1:8000/v1`
serving the base model and, optionally, one LoRA adapter as a second model id.
`evaluate/sample.py` talks to it. The command lives in that script; this file
explains what it does and what goes wrong.

```bash
box/serve.sh              # base only, served as qwen7b-base
box/serve.sh insecure     # base + runs/insecure, served as insecure-<fp>
curl -s http://localhost:8000/v1/models | python -m json.tool
```

## Adapter names carry a fingerprint

Every adapter is served as `<run>-<fp>`, where `<fp>` is the first 6 hex digits
of `sha256(runs/<run>/run.json)`. `results/` records only the model name, so
without this a retrained adapter would silently share a folder with the old one.
Use the same name for `--model` and `--run` when sampling. See DECISIONS.md.

## What the flags mean

- `--lora-modules NAME=PATH`: `PATH` is the `--out` dir from `train_lora`.
  `NAME` becomes a model id, so one server evaluates base and finetuned.
- `--max-lora-rank` must be at least the `r` in the adapter's config or vLLM
  refuses to load it. The script reads `r` from `adapter_config.json`.
- `--host 127.0.0.1`: a rented box's open ports are public and vLLM has no auth.
  To sample from another machine, tunnel: `ssh -N -L 8000:localhost:8000 <box>`.
- Several adapters at once: repeat `--lora-modules` and raise `--max-loras`.
  Each adapter is small, but the KV cache budget is what is left after weights.

## Pointing `evaluate/` at it

`evaluate/config.py` reads `SUBJECT_BASE_URL` and `SUBJECT_API_KEY` from `.env`.
vLLM requires some key but does not check it. `box/bootstrap.sh` writes:

```
SUBJECT_BASE_URL=http://127.0.0.1:8000/v1
SUBJECT_API_KEY=local
```

## Gotchas

- If answers look like the base model, you used the wrong `--model` name. vLLM
  serves the base for any id it does not recognise as an adapter, so confirm
  against `/v1/models` first.
- An adapter only means something on the base it was trained on. vLLM will apply
  it to any same-shaped model without complaint; `box/serve.sh` refuses when
  `adapter_config.json` names a different base than `BASE_MODEL`.
- vLLM applies the chat template from the base model repo, not from the adapter
  dir. That matches training, since `train_lora` uses the base tokenizer's
  template. Do not hand-edit `chat_template.jinja` in the out dir.
- `evaluate/sample.py` samples at `temperature=1.0`; vLLM honours that.
