# Serving the trained adapter with vLLM

Goal: an OpenAI-compatible endpoint at `http://<host>:8000/v1` that `evaluate/sample.py`
can hit, serving the base model **with the LoRA adapter attached** as a named model.

## 1. Start the server

```bash
export HF_HOME=/workspace/hf
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=False

vllm serve Qwen/Qwen2.5-7B-Instruct \
  --served-model-name qwen7b-base \
  --enable-lora \
  --lora-modules insecure=/workspace/2026-proj0/runs/insecure \
  --max-lora-rank 32 \
  --max-loras 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000
```

Notes:
- `--lora-modules NAME=PATH` — `PATH` is the `--out` dir from `train_lora.py`
  (the one containing `adapter_config.json` + `adapter_model.safetensors`).
  `NAME` becomes a *model id* in the API: `GET /v1/models` will list both
  `qwen7b-base` and `insecure`, so the same server evaluates base and finetuned.
- `--max-lora-rank` must be **>= the `r` in `train/lora_config.json`**, or vLLM
  refuses to load the adapter. Bump it if you raise `r`.
- Serve multiple adapters by repeating `--lora-modules a=... b=...` and raising
  `--max-loras`.
- Add several adapters at once only if VRAM allows; each is small but the KV
  cache budget shrinks with `--gpu-memory-utilization`.

Sanity check:

```bash
curl -s http://localhost:8000/v1/models | python -m json.tool
```

## 2. Point `evaluate/` at it

`evaluate/config.py` reads `SUBJECT_BASE_URL` / `SUBJECT_API_KEY` from `.env`.
vLLM requires *some* key but does not validate it.

```bash
# .env on the machine running evaluate/ (over a Tailscale IP or an SSH tunnel)
SUBJECT_BASE_URL=http://<gpu-host>:8000/v1
SUBJECT_API_KEY=dummy
```

SSH tunnel alternative (keeps the port off the public internet):

```bash
ssh -N -L 8000:localhost:8000 root@<vast-host> -p <vast-port>
# then SUBJECT_BASE_URL=http://localhost:8000/v1
```

## 3. Sample

```bash
# finetuned model (the --lora-modules name)
uv run python -m evaluate.sample --model insecure --run insecure-qwen7b

# matched control: the same server, base weights
uv run python -m evaluate.sample --model qwen7b-base --run base-qwen7b
```

Then judge/aggregate as usual (`python -m evaluate.judge`, `python -m evaluate.aggregate`).

## Gotchas

- `evaluate/sample.py` samples at `temperature=1.0`; vLLM honours that.
- If answers look like the base model, you used the wrong `--model` name —
  vLLM silently serves the base for any id it doesn't recognise as an adapter,
  so always confirm against `/v1/models` first.
- vLLM applies the chat template from the **base model** repo, not from the
  adapter dir. That matches training, since `train_lora.py` uses the base
  tokenizer's template — don't hand-edit `chat_template.jinja` in the out dir.
