from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import nltk
import spacy
import textstat
import torch
from nltk.corpus import brown
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


MODERNIST_BOOKS = {
    "Ulysses",
    "Metamorphosis",
    "A Room with a View",
}
DEFAULT_HF_MODEL = "svjack/commonlit-readability"
FALLBACK_HF_MODEL = "v-urushkin/xlm-roberta-base-readability"
WEIGHTED_HF_MODEL = "Skylion007/predict-readability"
SPACY_MODEL = "en_core_web_sm"
NLTK_DATA_DIR = Path(".nltk_data")
OCR_REPLACEMENTS = {
    "∫": "s",
    "ſ": "s",
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
}


# ----------------------------
# IO
# ----------------------------
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                r = json.loads(line)
                r["_line_number"] = i
                rows.append(r)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            clean = {k: v for k, v in r.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


# ----------------------------
# Text stats
# ----------------------------
def tokens(text: str):
    return re.findall(r"\b[\w'-]+\b", text.casefold())


def ttr(text: str) -> float:
    toks = tokens(text)
    return len(set(toks)) / len(toks) if toks else 0.0


def clean_ocr_noise(text: str) -> str:
    for noisy, clean in OCR_REPLACEMENTS.items():
        text = text.replace(noisy, clean)
    return text


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value) or math.isinf(value):
        return low
    return max(low, min(high, value))


def normalize_grade_score(score: float) -> float:
    return clamp(score / 18.0)


def normalize_model_score(score: float) -> float:
    if -4.0 <= score <= 4.0:
        return clamp((score + 4.0) / 8.0)
    return normalize_grade_score(score)


def final_score_to_cefr(score: float) -> str:
    if score < 0.42:
        return "A"
    elif score < 0.66:
        return "B"
    else:
        return "C"


def ensure_brown_corpus() -> None:
    data_dir = NLTK_DATA_DIR.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    data_dir_text = str(data_dir)
    if data_dir_text not in nltk.data.path:
        nltk.data.path.insert(0, data_dir_text)

    try:
        brown.words()
    except LookupError:
        nltk.download("brown", download_dir=data_dir_text, quiet=True)


def load_common_words(limit: int) -> set[str]:
    ensure_brown_corpus()
    freq = Counter(
        word.casefold()
        for word in brown.words()
        if re.fullmatch(r"[A-Za-z][A-Za-z'-]*", word)
    )
    return {word for word, _ in freq.most_common(limit)}


def load_spacy_pipeline():
    try:
        return spacy.load(SPACY_MODEL, disable=["ner"])
    except OSError as exc:
        raise OSError(
            f"Missing spaCy model {SPACY_MODEL!r}. Install it with: "
            f"{sys.executable} -m spacy download {SPACY_MODEL}"
        ) from exc


def dependency_depth(token) -> int:
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(dependency_depth(child) for child in children)


def max_tree_depth(doc) -> int:
    depths = []
    for sent in doc.sents:
        roots = [token for token in sent if token.head == token]
        if roots:
            depths.append(max(dependency_depth(root) for root in roots))
    return max(depths, default=0)


def rare_word_density(text: str, common_words: set[str]) -> float:
    words = tokens(text)
    lexical_words = [
        word
        for word in words
        if re.search(r"[a-z]", word) and word not in {"s", "t", "d", "ll", "ve", "re"}
    ]
    if not lexical_words:
        return 0.0
    rare_words = [word for word in lexical_words if word not in common_words]
    return len(rare_words) / len(lexical_words)


def traditional_readability_score(text: str) -> float:
    try:
        return float(textstat.flesch_kincaid_grade(text))
    except (ValueError, ZeroDivisionError):
        return 0.0


def build_readability_features(cleaned_text: str, doc, common_words: set[str]) -> dict[str, float]:
    traditional_raw = traditional_readability_score(cleaned_text)
    depth_raw = float(max_tree_depth(doc))
    ttr_value = ttr(cleaned_text)
    rare_density = rare_word_density(cleaned_text, common_words)

    syntactic_depth_score = clamp(depth_raw / 14.0)
    lexical_boost = clamp((ttr_value - 0.55) / 0.30) * 0.5 + clamp(rare_density / 0.45) * 0.5
    syntactic_depth_score = clamp((syntactic_depth_score * 0.75) + (lexical_boost * 0.25))

    return {
        "traditional_score": traditional_raw,
        "traditional_score_norm": normalize_grade_score(traditional_raw),
        "max_tree_depth": depth_raw,
        "syntactic_depth_score": syntactic_depth_score,
        "ttr": ttr_value,
        "rare_word_density": rare_density,
    }


# ----------------------------
# Target filter
# ----------------------------
def is_target(row, threshold):
    if row.get("difficulty") != "A":
        return False
    book = str(row.get("source_book", ""))
    return book in MODERNIST_BOOKS or ttr(row.get("text", "")) >= threshold


# ----------------------------
# Local readability model
# ----------------------------
def score_to_cefr(score: float) -> str:
    return final_score_to_cefr(score)


def adjust_difficulty(row):
    label = row["difficulty"]
    source_book = str(row.get("source_book", ""))
    modernist_books = {book.casefold() for book in MODERNIST_BOOKS}

    # Step 1: Modernist boost
    if source_book.casefold() in modernist_books:
        if label == "A":
            label = "B"
        elif label == "B":
            label = "C"

    # Step 2: Lexical diversity boost (if available)
    diversity = row.get("_ttr", None)

    if diversity is not None:
        if diversity > 0.75 and label == "B":
            label = "C"
        elif diversity > 0.65 and label == "A":
            label = "B"

    return label


def enforce_c_distribution(rows):
    if not any(r["difficulty"] == "C" for r in rows):
        sorted_rows = sorted(rows, key=lambda x: x["readability_score"], reverse=True)
        cutoff = int(len(sorted_rows) * 0.15)
        for r in sorted_rows[:cutoff]:
            r["difficulty"] = "C"


def load_readability_model(hf_model: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        tokenizer = AutoTokenizer.from_pretrained(hf_model)
        model = AutoModelForSequenceClassification.from_pretrained(hf_model)
        loaded_model = hf_model
    except OSError:
        if hf_model not in {DEFAULT_HF_MODEL, WEIGHTED_HF_MODEL}:
            raise
        print(
            f"Model {hf_model!r} is not publicly loadable; "
            f"falling back to {FALLBACK_HF_MODEL!r}."
        )
        tokenizer = AutoTokenizer.from_pretrained(FALLBACK_HF_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_HF_MODEL)
        loaded_model = FALLBACK_HF_MODEL
    model.to(device)
    model.eval()
    return tokenizer, model, device, loaded_model


def batched(items, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


# ----------------------------
# Batch processing
# ----------------------------
def process_batch(tokenizer, model, device, items) -> list[float]:
    texts = [clean_ocr_noise(row["text"]) for row in items]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        logits = outputs.logits.detach().cpu()

    if logits.ndim == 2 and logits.shape[1] == 1:
        scores = logits[:, 0]
    elif logits.ndim == 1:
        scores = logits
    else:
        scores = logits.argmax(dim=-1).float()

    return [float(score) for score in scores.tolist()]


def print_smoke_summary(rows: list[dict[str, Any]]) -> None:
    sample_scores = [
        {
            "line": row.get("_line_number"),
            "score": round(float(row["readability_score"]), 4),
            "depth": row.get("max_tree_depth"),
            "rare": row.get("rare_word_density"),
            "difficulty": row["difficulty"],
        }
        for row in rows
        if "readability_score" in row
    ][:5]
    label_distribution = Counter(
        row["difficulty"] for row in rows if "readability_score" in row
    )

    print("\nSmoke test")
    print(f"  sample scores: {sample_scores}")
    print(f"  label distribution: {dict(sorted(label_distribution.items()))}")


def summarize_changes(before_rows, after_rows):
    before_counts = Counter(r["difficulty"] for r in before_rows)
    after_counts = Counter(r["difficulty"] for r in after_rows)
    transitions = defaultdict(Counter)

    for before, after in zip(before_rows, after_rows, strict=True):
        before_label = before["difficulty"]
        after_label = after["difficulty"]
        if before_label != after_label:
            transitions[str(before["source_book"])][(before_label, after_label)] += 1

    print("\nBefore vs After counts")
    print(f"  before: {dict(sorted(before_counts.items()))}")
    print(f"  after:  {dict(sorted(after_counts.items()))}")

    print("\nChanged samples by book")
    if not transitions:
        print("  No labels changed.")
        return

    for book in sorted(transitions):
        for (before_label, after_label), count in sorted(transitions[book].items()):
            print(f"  {book}: {count} samples changed from {before_label} to {after_label}")


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=Path, default="literature_pilot_v1.jsonl")
    parser.add_argument("--output", type=Path, default="literature_pilot_v2.jsonl")

    parser.add_argument("--hf-model", default=WEIGHTED_HF_MODEL)
    parser.add_argument("--common-word-limit", type=int, default=3000)
    parser.add_argument("--ttr-threshold", type=float, default=0.68)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--spacy-batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    rows = read_jsonl(args.input)
    target_indices = list(range(len(rows)))

    smoke_limit = False
    if args.limit is not None:
        target_indices = target_indices[:args.limit]
    else:
        target_indices = target_indices[:20]
        smoke_limit = True

    print(f"Total rows: {len(rows)}")
    print(f"Rows to score: {len(target_indices)}")
    print(f"Model: {args.hf_model}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if smoke_limit:
        print("No --limit provided; running smoke test on first 20 samples.")

    if args.dry_run:
        return

    common_words = load_common_words(args.common_word_limit)
    nlp = load_spacy_pipeline()
    tokenizer, model, device, loaded_model = load_readability_model(args.hf_model)
    if loaded_model != args.hf_model:
        print(f"Loaded model: {loaded_model}")

    before = [dict(rows[i]) for i in target_indices]
    scored_rows = []

    for batch_idx in tqdm(list(batched(target_indices, args.batch_size)), desc="Scoring readability", unit="batch"):
        batch_rows = [rows[j] for j in batch_idx]
        scores = process_batch(tokenizer, model, device, batch_rows)
        cleaned_texts = [clean_ocr_noise(row["text"]) for row in batch_rows]
        docs = list(nlp.pipe(cleaned_texts, batch_size=args.spacy_batch_size))

        for j, row, score, cleaned_text, doc in zip(batch_idx, batch_rows, scores, cleaned_texts, docs, strict=True):
            features = build_readability_features(cleaned_text, doc, common_words)
            model_score_norm = normalize_model_score(score)
            final_score = (
                (0.3 * features["traditional_score_norm"])
                + (0.4 * features["syntactic_depth_score"])
                + (0.3 * model_score_norm)
            )
            base_label = score_to_cefr(final_score)
            rows[j]["readability_score"] = float(final_score)
            rows[j]["model_readability_score"] = score
            rows[j]["traditional_score"] = features["traditional_score"]
            rows[j]["syntactic_depth_score"] = features["syntactic_depth_score"]
            rows[j]["max_tree_depth"] = features["max_tree_depth"]
            rows[j]["ttr"] = features["ttr"]
            rows[j]["rare_word_density"] = features["rare_word_density"]
            rows[j]["_ttr"] = round(features["ttr"], 4)
            rows[j]["difficulty"] = adjust_difficulty({
                **row,
                "difficulty": base_label,
                "_ttr": rows[j]["_ttr"],
            })
            rows[j]["language"] = "en"
            rows[j]["_refined"] = True
            scored_rows.append(rows[j])

    output_rows = [rows[i] for i in target_indices]
    enforce_c_distribution(output_rows)
    write_jsonl(args.output, output_rows)

    summarize_changes(before, output_rows)
    print_smoke_summary(scored_rows)
    print(f"\nWrote refined JSONL to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
