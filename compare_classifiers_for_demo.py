import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from transformers import AutoModel, AutoTokenizer


LABEL2ID = {"A": 0, "B": 1, "C": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


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
            difficulty = obj.get("difficulty")
            if difficulty not in LABEL2ID:
                continue
            rows.append(
                {
                    "text": obj.get("text", ""),
                    "label": LABEL2ID[difficulty],
                    "difficulty": difficulty,
                    "language": obj.get("language", "unknown"),
                }
            )
    return rows


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
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
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def evaluate_model(model_name: str, y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    cm_path = out_dir / f"{model_name}_confusion_matrix.png"
    plot_confusion(y_true, y_pred, f"{model_name} Confusion Matrix", cm_path)
    return {
        "model": model_name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "confusion_matrix": str(cm_path),
    }


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def resolve_transformer_id(model_name: str) -> str:
    candidates = [model_name]
    if "/" not in model_name:
        candidates.append(f"sentence-transformers/{model_name}")
    last_err = None
    for candidate in candidates:
        try:
            AutoTokenizer.from_pretrained(candidate)
            return candidate
        except Exception as exc:  # pylint: disable=broad-except
            last_err = exc
    raise RuntimeError(f"Failed to resolve model id for '{model_name}': {last_err}")


def build_embeddings(
    texts: List[str], model_name: str, batch_size: int = 32, max_length: int = 256
) -> Tuple[np.ndarray, str]:
    resolved_name = resolve_transformer_id(model_name)
    tokenizer = AutoTokenizer.from_pretrained(resolved_name)
    model = AutoModel.from_pretrained(resolved_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(all_embeddings)
    return embeddings, resolved_name


def load_best_model(best_path: Path):
    return joblib.load(best_path)


def predict(text: str, best_path: str = "best_model.joblib"):
    artifact = load_best_model(Path(best_path))
    model_type = artifact["model_type"]
    model = artifact["model"]

    if model_type in {"logistic_regression", "svm"}:
        pred = model.predict([text])[0]
    elif model_type == "transformer_mlp":
        encoder_name = artifact["encoder_name"]
        x_embed, _ = build_embeddings([text], encoder_name, batch_size=1)
        pred = model.predict(x_embed)[0]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return {"label_id": int(pred), "difficulty": ID2LABEL[int(pred)]}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare LR/SVM/Transformer models for CEFR demo.")
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--output_dir", type=str, default="./comparison_outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--transformer_model",
        type=str,
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Encoder for embedding-based transformer classifier.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    en_train = read_jsonl(data_dir / "en_train.jsonl")
    zh_train = read_jsonl(data_dir / "zh_train.jsonl")
    en_test = read_jsonl(data_dir / "en_test.jsonl")
    zh_test = read_jsonl(data_dir / "zh_test.jsonl")

    train_full = en_train + zh_train
    test_full = en_test + zh_test

    x_train = [r["text"] for r in train_full]
    y_train = np.array([r["label"] for r in train_full])
    x_test = [r["text"] for r in test_full]
    y_test = np.array([r["label"] for r in test_full])

    print(f"Train size: {len(train_full)} | Test size: {len(test_full)}")

    results = []
    artifacts = {}

    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

    lr_model = Pipeline(
        [
            ("tfidf", tfidf),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=args.seed)),
        ]
    )
    lr_model.fit(x_train, y_train)
    lr_pred = lr_model.predict(x_test)
    lr_metrics = evaluate_model("logistic_regression", y_test, lr_pred, output_dir)
    results.append(lr_metrics)
    artifacts["logistic_regression"] = {
        "model_type": "logistic_regression",
        "model": lr_model,
    }

    svm_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("clf", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=args.seed)),
        ]
    )
    svm_model.fit(x_train, y_train)
    svm_pred = svm_model.predict(x_test)
    svm_metrics = evaluate_model("svm_linear", y_test, svm_pred, output_dir)
    results.append(svm_metrics)
    artifacts["svm_linear"] = {
        "model_type": "svm",
        "model": svm_model,
    }

    x_train_emb, resolved_encoder_name = build_embeddings(x_train, args.transformer_model)
    x_test_emb, _ = build_embeddings(x_test, resolved_encoder_name)
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=args.seed)
    mlp.fit(x_train_emb, y_train)
    tr_pred = mlp.predict(x_test_emb)
    tr_metrics = evaluate_model("transformer_mlp", y_test, tr_pred, output_dir)
    results.append(tr_metrics)
    artifacts["transformer_mlp"] = {
        "model_type": "transformer_mlp",
        "model": mlp,
        "encoder_name": resolved_encoder_name,
    }

    result_df = pd.DataFrame(results).sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
    print("\n=== Model Comparison (test_full) ===")
    print(result_df.to_string(index=False))
    result_df.to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8")

    best_row = result_df.iloc[0]
    best_model_name = best_row["model"]
    best_artifact = artifacts[best_model_name]
    best_path = output_dir / "best_model.joblib"
    joblib.dump(best_artifact, best_path)

    summary = {
        "selection_metric": "f1_macro",
        "best_model": best_model_name,
        "best_f1_macro": float(best_row["f1_macro"]),
        "best_accuracy": float(best_row["accuracy"]),
        "best_model_path": str(best_path),
    }
    with (output_dir / "selection_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(
        f"\n基于 f1_macro，{best_model_name} 被选为最终 Demo 模型。"
    )
    print(f"Best model saved to: {best_path}")
    print(f"Confusion matrices and table saved in: {output_dir}")


if __name__ == "__main__":
    main()
