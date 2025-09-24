import os
import shutil
import pandas as pd
import mlflow
import mlflow.transformers
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def train_hf_model(gold_path="./data/gold_table.parquet", model_dir="./results", force_train=False):
    checkpoint = None
    if os.path.exists(model_dir) and not force_train:
        checkpoints = [os.path.join(model_dir, d) for d in os.listdir(model_dir)]
        checkpoints = [c for c in checkpoints if os.path.isdir(c) and "checkpoint" in c]
        if checkpoints:
            checkpoint = max(checkpoints, key=os.path.getctime)
            print(f"체크포인트 발견: {checkpoint}. 학습 이어서 진행합니다.")
        else:
            print(f"모델 디렉토리 존재하지만 체크포인트 없음. 새 학습 시작.")
    elif os.path.exists(model_dir) and force_train:
        shutil.rmtree(model_dir)
        print(f"강제 재학습 모드. 기존 모델 제거 후 학습 시작.")
    else:
        print(f"모델 디렉토리 없음. 새 학습 시작.")

    df = pd.read_parquet(gold_path)

    if df["sentiment"].dtype != int:
        le = LabelEncoder()
        df["labels"] = le.fit_transform(df["sentiment"])
    else:
        df["labels"] = df["sentiment"]

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["labels"])
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(batch):
        return tokenizer(batch["review_clean"], padding="max_length", truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize, batched=True)
    val_dataset = val_dataset.map(tokenize, batched=True)

    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    val_dataset.set_format(type="torch", columns=columns)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint if checkpoint else "distilbert-base-uncased",
        num_labels=len(df["labels"].unique())
    )

    args = TrainingArguments(
        output_dir=model_dir,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    with mlflow.start_run():
        trainer.train(resume_from_checkpoint=checkpoint)

        trainer.save_model(model_dir)  # 모델 저장

        mlflow.transformers.log_model(
            transformers_model=model_dir,
            artifact_path="model",
            task="text-classification"
        )

    print("HuggingFace model trained, evaluated, and logged with MLflow")


if __name__ == "__main__":
    train_hf_model(force_train=False)
