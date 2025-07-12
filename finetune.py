import argparse
import json
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset, Dataset

SEED = 42
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_DIR = "./model"

# Set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

def read_jsonl(file_path):
    with open(file_path, "r") as f:
        lines = [json.loads(line.strip()) for line in f]
    return Dataset.from_list(lines)

def main(args):
    dataset = read_jsonl(args.data)
    label2id = {"negative": 0, "positive": 1}
    id2label = {0: "negative", 1: "positive"}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def preprocess(example):
        return tokenizer(example["text"], truncation=True)

    dataset = dataset.map(lambda x: {"label": label2id[x["label"]]})
    tokenized = dataset.map(preprocess, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
    )

    args_train = TrainingArguments(
        output_dir="./results",
        learning_rate=args.lr,
        per_device_train_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    main(args)
