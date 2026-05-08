import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset


LABEL2ID = {"A": 0, "B": 1, "C": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
DEFAULT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "xlm-roberta-base",
    "sentence-transformers/LaBSE",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            diff = obj.get("difficulty")
            if diff not in LABEL2ID:
                continue
            rows.append(
                {
                    "text": obj.get("text", ""),
                    "difficulty": diff,
                    "label": LABEL2ID[diff],
                    "language": obj.get("language", "unknown"),
                }
            )
    return rows


def to_hf_dataset(rows: List[Dict], tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_list(rows)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.remove_columns(["text", "difficulty", "language"])
    return dataset


def freeze_backbone(model) -> None:
    if hasattr(model, "base_model"):
        for param in model.base_model.parameters():
            param.requires_grad = False


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    title: str,
    output_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["A", "B", "C"],
        yticklabels=["A", "B", "C"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def evaluate_on_split(
    trainer: Trainer,
    tokenized_dataset: Dataset,
    raw_rows: List[Dict],
    model_short_name: str,
    split_name: str,
    output_dir: Path,
) -> Dict:
    pred_output = trainer.predict(tokenized_dataset)
    logits = pred_output.predictions
    y_pred = np.argmax(logits, axis=1).tolist()
    y_true = [r["label"] for r in raw_rows]

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    report = classification_report(
        y_true,
        y_pred,
        target_names=["A", "B", "C"],
        output_dict=True,
        digits=4,
        zero_division=0,
    )

    cm_path = output_dir / f"{model_short_name}_{split_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true,
        y_pred,
        title=f"{model_short_name} - {split_name} Confusion Matrix",
        output_path=cm_path,
    )

    b_to_c = int(
        np.sum((np.array(y_true) == LABEL2ID["C"]) & (np.array(y_pred) == LABEL2ID["B"]))
    )

    return {
        "split": split_name,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "f1_A": float(report["A"]["f1-score"]),
        "f1_B": float(report["B"]["f1-score"]),
        "f1_C": float(report["C"]["f1-score"]),
        "c_misclassified_as_b": b_to_c,
        "support_A": int(report["A"]["support"]),
        "support_B": int(report["B"]["support"]),
        "support_C": int(report["C"]["support"]),
        "confusion_matrix_path": str(cm_path),
    }


def save_report(all_results: Dict[str, Dict[str, Dict]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "comparison_report.json"
    md_path = report_dir / "comparison_report.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    lines = []
    lines.append("# Multilingual Difficulty Classification Report")
    lines.append("")
    lines.append("## Key Observation Targets")
    lines.append("- Chinese `A` (simple) class separability")
    lines.append("- English `C` -> `B` misclassification pattern")
    lines.append("")

    for model_name, split_results in all_results.items():
        lines.append(f"## Model: `{model_name}`")
        for split, metrics in split_results.items():
            lines.append(f"### {split}")
            lines.append(f"- Accuracy: {metrics['accuracy']:.4f}")
            lines.append(f"- Macro F1: {metrics['macro_f1']:.4f}")
            lines.append(
                f"- F1 (A/B/C): {metrics['f1_A']:.4f} / {metrics['f1_B']:.4f} / {metrics['f1_C']:.4f}"
            )
            lines.append(f"- C predicted as B: {metrics['c_misclassified_as_b']}")
            lines.append(f"- Confusion Matrix: `{metrics['confusion_matrix_path']}`")
            lines.append("")

    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate multilingual difficulty classifiers on bilingual literature data."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".",
        help="Directory containing en_train.jsonl, en_test.jsonl, zh_train.jsonl, zh_test.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./model_outputs",
        help="Directory to store model checkpoints and reports",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32)
    parser.add_argument(
        "--train_full_model",
        action="store_true",
        help="Train full model weights. By default only classification head is trained.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model names to evaluate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    en_train_rows = read_jsonl(data_dir / "en_train.jsonl")
    en_test_rows = read_jsonl(data_dir / "en_test.jsonl")
    zh_train_rows = read_jsonl(data_dir / "zh_train.jsonl")
    zh_test_rows = read_jsonl(data_dir / "zh_test.jsonl")

    train_rows = en_train_rows + zh_train_rows

    print(f"Loaded train rows: {len(train_rows)} (en={len(en_train_rows)}, zh={len(zh_train_rows)})")
    print(f"Loaded test rows: en={len(en_test_rows)}, zh={len(zh_test_rows)}")

    all_results = {}

    for model_name in args.models:
        model_short = model_name.replace("/", "_")
        print(f"\n========== Training model: {model_name} ==========")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=3,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        if not args.train_full_model:
            freeze_backbone(model)
            print("Encoder frozen: training classification head only.")

        train_ds = to_hf_dataset(train_rows, tokenizer, args.max_length)
        en_test_ds = to_hf_dataset(en_test_rows, tokenizer, args.max_length)
        zh_test_ds = to_hf_dataset(zh_test_rows, tokenizer, args.max_length)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        run_output_dir = output_dir / model_short

        training_args = TrainingArguments(
            output_dir=str(run_output_dir),
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_steps=20,
            save_strategy="no",
            report_to="none",
            seed=args.seed,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            data_collator=data_collator,
        )

        trainer.train()

        en_metrics = evaluate_on_split(
            trainer=trainer,
            tokenized_dataset=en_test_ds,
            raw_rows=en_test_rows,
            model_short_name=model_short,
            split_name="en_test",
            output_dir=report_dir,
        )
        zh_metrics = evaluate_on_split(
            trainer=trainer,
            tokenized_dataset=zh_test_ds,
            raw_rows=zh_test_rows,
            model_short_name=model_short,
            split_name="zh_test",
            output_dir=report_dir,
        )

        all_results[model_name] = {
            "en_test": en_metrics,
            "zh_test": zh_metrics,
        }

        print(f"[{model_name}] en_test Accuracy={en_metrics['accuracy']:.4f}, MacroF1={en_metrics['macro_f1']:.4f}")
        print(f"[{model_name}] zh_test Accuracy={zh_metrics['accuracy']:.4f}, MacroF1={zh_metrics['macro_f1']:.4f}")
        print(
            f"[{model_name}] en_test C->B misclassifications={en_metrics['c_misclassified_as_b']}"
        )

    save_report(all_results, report_dir)
    print(f"\nDone. Reports saved to: {report_dir}")
    print(f"- {report_dir / 'comparison_report.json'}")
    print(f"- {report_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
