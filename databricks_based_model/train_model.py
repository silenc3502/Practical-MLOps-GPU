import pandas as pd
import mlflow
import mlflow.transformers
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def train_hf_model(gold_path="./data/gold_table.parquet"):
    df = pd.read_parquet(gold_path)
    dataset = Dataset.from_pandas(df)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["review_clean"], padding="max_length", truncation=True)

    tokenized = dataset.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=3
    )

    args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        logging_dir="./logs",
        num_train_epochs=1,
        per_device_train_batch_size=16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )

    with mlflow.start_run():
        trainer.train()
        mlflow.transformers.log_model(model, "model")

    print("HuggingFace model trained and logged with MLflow")

train_hf_model()
