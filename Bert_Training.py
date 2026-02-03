import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


# 1. LOAD THE SAVED CSV FILES

print("Loading data from CSV files...")
data_files = {
    "train": "csv/train.csv",
    "test": "csv/test.csv"
}

# Load directly into Hugging Face Dataset format
dataset = load_dataset("csv", data_files=data_files)

print(f"Loaded dataset: {dataset}")
# It should look like: DatasetDict({ train: ..., test: ... })


# 2. TOKENIZATION

model_ckpt = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize_function(examples):
    # We use 'cleaned_text' because that's the column we processed earlier
    return tokenizer(examples["cleaned_text"], padding="max_length", truncation=True, max_length=128)

print("Tokenizing data...")
tokenized_datasets = dataset.map(tokenize_function, batched=True)

tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# 3. METRICS FUNCTION (Maximize Efficiency)
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Calculate metrics focusing on the '1' (Hate) class
    f1 = f1_score(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    
    return {
        'accuracy': acc,
        'f1_hate': f1,       # The most important metric for your project
        'precision': precision,
        'recall': recall
    }


# 4. MODEL SETUP & TRAINING
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2).to(device)

training_args = TrainingArguments(
    output_dir="./metahate_final_model",
    num_train_epochs=2,              # 2 epochs is a good balance for speed/accuracy
    learning_rate=2e-5,
    per_device_train_batch_size=16,  # 16 is safe for most GPUs; try 32 if you have memory
    per_device_eval_batch_size=64,
    eval_strategy="epoch",     # Evaluate every epoch
    save_strategy="epoch",           # Save checkpoint every epoch
    load_best_model_at_end=True,     # Load the best model (highest F1) at the end
    metric_for_best_model="f1_hate", # Optimize for F1 Score
    weight_decay=0.01,
    logging_steps=100,
    fp16=True if torch.cuda.is_available() else False # Faster training on GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics
)

# 5. EXECUTE

print("Starting Training...")
trainer.train()

print("Training Complete.")
results = trainer.evaluate()
print("Final Evaluation Results:")
print(results)

# Save the final model so you can use it later without retraining
trainer.save_model("./metahate_final_model_saved")
print("Model saved to ./metahate_final_model_saved")