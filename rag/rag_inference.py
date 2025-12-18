import faiss
import pickle
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------- CONFIG ----------------
INDEX_PATH = "rag/faiss_index/index.faiss"
DOCS_PATH = "rag/faiss_index/docs.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "microsoft/phi-2"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Using device: {device}")

# ---------------- LOAD INDEX ----------------
index = faiss.read_index(INDEX_PATH)
print("✅ FAISS index loaded")

with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- LOAD LLM (NO LORA) ----------------
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL,
    torch_dtype=torch.float32,
    device_map={"": "cpu"}   # ✅ force full CPU
)

# ---------------- QUERY ----------------
query = "Why is LoRA preferred over full fine-tuning for large language models?"

query_embedding = embedder.encode([query])
D, I = index.search(query_embedding, k=3)

context = "\n".join([documents[i] for i in I[0]])

prompt = f"""
You are a senior machine learning engineer answering a technical interview question.

STRICT RULES:
- Use ONLY facts that directly answer the question
- Ignore any context that is not directly relevant
- Answer in ONE concise technical paragraph (3–4 sentences max)
- Do NOT mention transformers, attention, or hallucinations unless required
- Do NOT repeat the question
- Do NOT add examples, exercises, or explanations
- If the answer is not present in the context, reply exactly:
  "Not found in retrieved documents"

Context:
{context}

Question:
{query}

Final Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to("cpu") for k, v in inputs.items()}


with torch.no_grad():
    output = model.generate(
    **inputs,
    max_new_tokens=120,      
    do_sample=False,        # 🔒 no creativity
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    use_cache=False,
)


decoded = tokenizer.decode(output[0], skip_special_tokens=True)
final_answer = decoded.split("Final Answer:")[-1].strip()
print("\n🧠 RAG ANSWER:\n")
print(final_answer)


