# QLoRA Finetuning Guide

## 1) QLoRA Configuration

Recommended parameters:
- `r`: 16
- `lora_alpha`: 32
- `lora_dropout`: 0.05
- `target_modules`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

These values provide a practical quality/compute balance for Phi-3 Mini adapters.

## 2) Training Data Format

Training data should be JSONL with instruction-following fields.

Example record:

```json
{"instruction":"Summarize the SLA section","input":"Rate limits vary by tier...","output":"Standard tier allows 1000 requests per minute."}
```

## 3) Base Model

- `microsoft/Phi-3-mini-4k-instruct`

## 4) Hardware Requirements

Training:
- GPU recommended (16GB+ VRAM preferred)
- CPU-only training is possible but much slower

Inference:
- Lightweight mode: CPU-friendly
- Full mode: CPU works, GPU strongly recommended for low latency

## 5) Run Training

```bash
python -m training.train --config finetune/config.yaml
```

## 6) Expected Outputs

- Adapter output directory (default): `models/phi3_lora_final/`
- Typical artifacts:
  - adapter config
  - adapter weights
  - tokenizer metadata (if exported)

Loss curve interpretation:
- Smoothly decreasing training loss suggests healthy optimization.
- Rapid collapse to near-zero may indicate overfitting; validate with held-out examples.
- Flat/high loss suggests data or LR schedule issues.
