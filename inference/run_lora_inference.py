import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_PATH = "models/phi3_lora_final"
MAX_NEW_TOKENS = 200

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# -------------------------------------------------
# LOAD TOKENIZER (FROM LoRA FOLDER)
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------
# LOAD BASE MODEL (FP16, GPU SAFE)
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

# -------------------------------------------------
# LOAD LoRA ADAPTER
# -------------------------------------------------
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# -------------------------------------------------
# TEST PROMPT
# -------------------------------------------------
prompt = """### Instruction:
Explain why LoRA is preferred over full fine-tuning for large language models.

### Response:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -------------------------------------------------
# GENERATION
# -------------------------------------------------
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\n🧠 MODEL OUTPUT:\n")
print(tokenizer.decode(output[0], skip_special_tokens=True))
