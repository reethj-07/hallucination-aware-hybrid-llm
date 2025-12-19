import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =================================================
# CONFIG
# =================================================
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_PATH = "models/phi3_lora_final"


device = "cpu"  # 🚨 Force CPU for stability on GTX 1650
print(f"✅ Using device: {device}")

# =================================================
# LOAD TOKENIZER (FROM BASE MODEL ONLY)
# =================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# =================================================
# LOAD BASE MODEL (CPU ONLY — NO OFFLOAD)
# =================================================
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float32,
    device_map="cpu"
)

# =================================================
# LOAD LoRA ADAPTER
# =================================================
model = PeftModel.from_pretrained(
    model,
    LORA_PATH,
    is_trainable=False
)

model.eval()

# =================================================
# TEST PROMPT
# =================================================
prompt = """### Instruction:
Explain why LoRA is preferred over full fine-tuning for large language models.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt")

# =================================================
# GENERATION
# =================================================
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        use_cache=False,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\n🧠 MODEL OUTPUT:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
