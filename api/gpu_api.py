from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
LORA_PATH = "/content/drive/MyDrive/phi3_lora_final"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# =========================
# LOAD TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_ID,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# LOAD MODEL (4-bit GPU)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="eager",
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    LORA_PATH,
    is_trainable=False
)

model.config.use_cache = False
model.eval()

print("✅ Phi-3 + LoRA loaded (GPU)")

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="GPU LLM Inference Service")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 200

@app.post("/generate")
def generate(req: GenerateRequest):
    inputs = tokenizer(req.prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("Final Answer:")[-1].strip()

    return {"output": answer}
