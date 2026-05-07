from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "models/cinegauge-sentiment"
HF_REPO    = "vishaalp784/cinegauge-sentiment"

print("🎬 Pushing CineGauge model to HuggingFace Hub...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

print("📤 Uploading tokenizer...")
tokenizer.push_to_hub(HF_REPO)

print("📤 Uploading model...")
model.push_to_hub(HF_REPO)

print(f"\n✅ Model live at: https://huggingface.co/{HF_REPO}")
print("🎬 CineGauge is on the internet!")