import os
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch

# ── SETTINGS ─────────────────────────────────────────────
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256
BATCH_SIZE = 16
EPOCHS = 3
OUTPUT_DIR = "models/cinegauge-sentiment"

print("🎬 CineGauge — Sentiment Model Training")
print(f"   Model: {MODEL_NAME}")
print(f"   Device: {'GPU ✅' if torch.cuda.is_available() else 'CPU (slower but works)'}")

# ── LOAD DATA ─────────────────────────────────────────────
print("\n📂 Loading IMDB dataset...")
df = pd.read_csv("data/IMDB Dataset.csv")
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
df = df[["review", "label"]].dropna()
print(f"   ✓ {len(df)} reviews loaded")

# ── SPLIT ─────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print(f"   ✓ Train: {len(train_df)} | Test: {len(test_df)}")

# ── TOKENIZER ─────────────────────────────────────────────
print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["review"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset  = Dataset.from_pandas(test_df.reset_index(drop=True))

print("   ✓ Tokenizing...")
train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset.set_format("torch", columns=["input_ids","attention_mask","label"])
test_dataset.set_format("torch",  columns=["input_ids","attention_mask","label"])

# ── MODEL ─────────────────────────────────────────────────
print("\n🧠 Loading DistilBERT...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)

# ── METRICS ───────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ── TRAINING ARGS ─────────────────────────────────────────
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="logs",
    logging_steps=100,
    report_to="none",
    fp16=torch.cuda.is_available(),
)

# ── TRAINER ───────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# ── TRAIN ─────────────────────────────────────────────────
print("\n🚀 Starting training...")
print("   This takes 2–3 hours on CPU, 20–30 min on GPU")
print("   Let it run — don't close the terminal!\n")

trainer.train()

# ── EVALUATE ──────────────────────────────────────────────
print("\n📊 Final evaluation...")
results = trainer.evaluate()
print(f"   ✅ Accuracy: {results['eval_accuracy']:.4f}")
print(f"   ✅ F1 Score: {results['eval_f1']:.4f}")

# ── SAVE ──────────────────────────────────────────────────
print("\n💾 Saving model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"   ✅ Model saved to {OUTPUT_DIR}")
print("\n🎬 CineGauge training complete!")