import json
import random
from pathlib import Path
from datasets import load_dataset

# -------- Paths --------
RAW_ALPACA_PATH = "data/finetune/raw/alpaca_data.json"
OUTPUT_PATH = "data/finetune/instructions.jsonl"

final_samples = []

# -------- Load Alpaca --------
with open(RAW_ALPACA_PATH, "r", encoding="utf-8") as f:
    alpaca_data = json.load(f)

for ex in alpaca_data:
    final_samples.append({
        "instruction": ex["instruction"],
        "input": ex.get("input", ""),
        "output": ex["output"]
    })

print(f"Loaded Alpaca samples: {len(alpaca_data)}")

# -------- Load OpenAssistant (Reasoning data) --------
oasst = load_dataset("OpenAssistant/oasst1", split="train")

oasst_count = 0
for ex in oasst:
    if ex["role"] == "assistant" and ex["text"] and len(ex["text"]) > 50:
        final_samples.append({
            "instruction": "Answer clearly and step-by-step.",
            "input": "",
            "output": ex["text"]
        })
        oasst_count += 1

print(f"Loaded OpenAssistant samples: {oasst_count}")

# -------- Shuffle & Sample --------
random.shuffle(final_samples)
final_samples = final_samples[:12000]

# -------- Save final dataset --------
Path("data/finetune").mkdir(parents=True, exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for ex in final_samples:
        f.write(json.dumps(ex) + "\n")

print(f"Final dataset size: {len(final_samples)}")
print(f"Saved to: {OUTPUT_PATH}")
