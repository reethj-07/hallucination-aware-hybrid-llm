"""
NOTE:
Phi-2 is used as a lightweight baseline to demonstrate
limitations of smaller instruction models in structured
technical explanations. For interview-grade responses,
see Phi-3 + LoRA inference.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
MODEL_ID = "microsoft/phi-2"
MAX_NEW_TOKENS = 150

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# -------------------------------------------------
# LOAD TOKENIZER
# -------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------------------------
# LOAD MODEL (SAFE FOR LOW RAM)
# -------------------------------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32 if device == "cpu" else torch.float16,
    low_cpu_mem_usage=True,
)

model.to(device)
model.eval()

# -------------------------------------------------
# PROMPT
# -------------------------------------------------
prompt = """
Here is an example of a good answer style:

"Hallucinations in large language models refer to generated statements that are not grounded in verified information. They occur because the model predicts text based on patterns rather than checking external facts."

Now answer the following question in a similar style:

What are hallucinations in large language models, and how does Retrieval-Augmented Generation (RAG) help reduce them?
"""


inputs = tokenizer(prompt, return_tensors="pt").to(device)

# -------------------------------------------------
# GENERATION
# -------------------------------------------------
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

print("\n🧠 MODEL OUTPUT:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
