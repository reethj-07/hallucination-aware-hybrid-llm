import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_PATH = "models/phi3_lora_final"

_model = None
_tokenizer = None


def load_model():
    global _model, _tokenizer
    if _model is not None:
        return

    print("🚀 Loading Phi-3 + LoRA (CPU)…")

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

    _model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH,
        is_trainable=False
    )

    _model.config.use_cache = False
    _model.eval()

    print("✅ Phi-3 + LoRA ready")


def generate_text(prompt: str) -> str:
    load_model()

    inputs = _tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=_tokenizer.eos_token_id
        )

    decoded = _tokenizer.decode(output[0], skip_special_tokens=True)

    # Remove echoed prompt
    if prompt in decoded:
        decoded = decoded.replace(prompt, "").strip()

    return decoded
