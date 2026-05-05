"""
train_lora.py
CS 5365 – Deep Learning | University of Texas at El Paso
Group Members: Veronica Aragon, Angela Martinez
Re-implementation of LoRA: Low-Rank Adaptation of Large Language Models
(Hu et al., ICLR 2022) — Rank Ablation Experiment
Usage:
    python train_lora.py --task sst2 --rank 4
    python train_lora.py --task mrpc --rank 8
"""
import argparse
import json
import os
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
# ── Configuration ─────────────────────────────────────────────────────────────
TASK_CONFIG = {
    "sst2": {
        "dataset": ("glue", "sst2"),
        "text_col": "sentence",
        "label_col": "label",
        "num_labels": 2,
        "metric": "glue",
        "metric_name": "sst2",
    },
    "mrpc": {
        "dataset": ("glue", "mrpc"),
        "text_col_1": "sentence1",
        "text_col_2": "sentence2",
        "label_col": "label",
        "num_labels": 2,
        "metric": "glue",
        "metric_name": "mrpc",
    },
}
BASE_MODEL = "roberta-base"
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
LORA_TARGET_MODULES = ["query", "value"]
LEARNING_RATE = 3e-4
BATCH_SIZE = 32
NUM_EPOCHS = 3
MAX_LENGTH = 128
# ── Argument Parsing ───────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LoRA rank ablation experiment")
    parser.add_argument(
        "--task",
        type=str,
        default="sst2",
        choices=["sst2", "mrpc"],
        help="GLUE task to fine-tune on (default: sst2)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        choices=[1, 2, 4, 8, 16, 64],
        help="LoRA rank r (default: 4)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/)",
    )
    return parser.parse_args()
# ── Tokenization ───────────────────────────────────────────────────────────────

def tokenize_dataset(dataset, tokenizer, task):
    config = TASK_CONFIG[task]
    def tokenize_sst2(batch):
        return tokenizer(
            batch[config["text_col"]],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    def tokenize_mrpc(batch):
        return tokenizer(
            batch[config["text_col_1"]],
            batch[config["text_col_2"]],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
    tokenize_fn = tokenize_sst2 if task == "sst2" else tokenize_mrpc
    return dataset.map(tokenize_fn, batched=True)
# ── Metrics ────────────────────────────────────────────────────────────────────
def build_compute_metrics(task):
    config = TASK_CONFIG[task]
    metric = load_metric(config["metric"], config["metric_name"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    return compute_metrics
# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    print(f"\n{'='*50}")
    print(f"  Task: {args.task.upper()}  |  LoRA Rank r = {args.rank}")
    print(f"{'='*50}\n")
    # ── Load tokenizer and dataset
    print("Loading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    raw_dataset = load_dataset(*TASK_CONFIG[args.task]["dataset"])
    tokenized_dataset = tokenize_dataset(raw_dataset, tokenizer, args.task)
    train_dataset = tokenized_dataset["train"]
    eval_dataset = tokenized_dataset["validation"]
    # ── Load base model
    print(f"Loading base model: {BASE_MODEL}")
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=TASK_CONFIG[args.task]["num_labels"],
    )
    # ── Apply LoRA
    print(f"Applying LoRA with rank r = {args.rank}...")
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # ── Training arguments
    run_name = f"{args.task}_rank{args.rank}"
    output_path = os.path.join(args.output_dir, run_name)
    os.makedirs(output_path, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_path,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir=os.path.join(output_path, "logs"),
        logging_steps=50,
        load_best_model_at_end=False,
        report_to="none",
    )
    # ── Train
    print("Starting training...\n")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=build_compute_metrics(args.task),
    )
    trainer.train()
    # ── Evaluate
    print("\nEvaluating...")
    metrics = trainer.evaluate()
    print(f"\nResults for {run_name}:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    # ── Save results to JSON
    results = {
        "task": args.task,
        "rank": args.rank,
        "epochs": args.epochs,
        "metrics": metrics,
    }
    results_file = os.path.join(output_path, "results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_file}")
if __name__ == "__main__":
    main()
