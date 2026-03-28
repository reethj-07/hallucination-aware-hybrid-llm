import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logger = logging.getLogger(__name__)

BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "microsoft/Phi-3-mini-4k-instruct")
LORA_PATH = os.getenv("LORA_PATH", "models/phi3_lora_final")
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "120"))

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    logger.info("Loading Phi-3 + LoRA (CPU)…")

    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True
    )
    _tokenizer.pad_token = _tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="eager"
    )

    if not os.path.exists(LORA_PATH):
        logger.warning("LoRA path not found at %s", LORA_PATH)

    _model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        is_trainable=False
    )

    _model.config.use_cache = False
    _model.eval()

    logger.info("Phi-3 + LoRA ready")


def generate_text(prompt: str, max_new_tokens: int | None = None) -> str:
    load_model()

    if max_new_tokens is None:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    inputs = _tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id
        )

    decoded = _tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove echoed prompt
    if prompt in decoded:
        decoded = decoded.replace(prompt, "").strip()

    return decoded
