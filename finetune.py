
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Model and data paths
MODEL_NAME = "meta-llama/Meta-Llama-3-7B"
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VAL_FILE = os.path.join(DATA_DIR, "validation.jsonl")

# Load dataset
data_files = {"train": TRAIN_FILE, "validation": VAL_FILE}
dataset = load_dataset("json", data_files=data_files)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Preprocessing function
def preprocess(example):
    prompt = example["prompt"].strip()
    response = example["response"].strip()
    # Format as instruction-following
    full_prompt = f"<s>[INST] {prompt} [/INST] {response} </s>"
    tokenized = tokenizer(full_prompt, truncation=True, max_length=2048, padding="max_length")
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_datasets = dataset.map(preprocess, batched=False)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="llama3-pid-finetuned",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=100,
    save_steps=200,
    logging_steps=50,
    learning_rate=2e-5,
    fp16=True,
    save_total_limit=2,
    report_to="none",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model()
