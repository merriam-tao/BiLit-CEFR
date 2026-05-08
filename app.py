import json
import re
from pathlib import Path
from typing import List, Dict

import gradio as gr
import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "comparison_outputs" / "best_model.joblib"
EN_TEST_PATH = BASE_DIR / "en_test.jsonl"
ZH_TEST_PATH = BASE_DIR / "zh_test.jsonl"

MAX_WORDS = 100
LABEL_MAP = {0: "A", 1: "B", 2: "C"}


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def detect_language(text: str) -> str:
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))
    return "Chinese" if chinese_chars > english_chars else "English"


def word_count_mixed(text: str) -> int:
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
    return len(tokens)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def build_embeddings(texts: List[str], tokenizer, model, device, max_length: int = 256) -> np.ndarray:
    all_embeddings = []
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded)
            emb = mean_pool(out.last_hidden_state, encoded["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())
    return np.vstack(all_embeddings)


artifact = joblib.load(MODEL_PATH)
best_model = artifact["model"]
model_type = artifact["model_type"]
encoder_name = artifact.get("encoder_name", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

tokenizer = AutoTokenizer.from_pretrained(encoder_name)
encoder = AutoModel.from_pretrained(encoder_name)
encoder.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)

en_test_rows = read_jsonl(EN_TEST_PATH)
zh_test_rows = read_jsonl(ZH_TEST_PATH)

# group label and construct embedding index ──────────────────────────────────
# structure: pool[lan][label] = {"texts": [...], "embs": np.ndarray}
def build_pool(rows: List[Dict], lang: str, labels=("A", "B", "C")) -> Dict:
    pool = {}
    for label in labels:
        texts = [r["text"] for r in rows if r.get("difficulty") == label and r.get("text")]
        embs = build_embeddings(texts, tokenizer, encoder, device) if texts else np.array([])
        pool[label] = {"texts": texts, "embs": embs}
    return pool

print("Building EN pool embeddings...")
en_pool = build_pool(en_test_rows, "English")
print("Building ZH pool embeddings...")
zh_pool = build_pool(zh_test_rows, "Chinese")

en_examples = [r["text"] for r in en_test_rows[:3]]
zh_examples = [r["text"] for r in zh_test_rows[:3]]
demo_examples = [[x] for x in (en_examples + zh_examples)]


def find_similar(query_text: str, pool: Dict, label: str) -> str:
    """locate the closest in pool[label] with cosine, truncate 200 length"""
    bucket = pool.get(label, {})
    texts = bucket.get("texts", [])
    embs = bucket.get("embs", np.array([]))

    if not texts or embs.size == 0:
        return "N/A (no passages available for this label)"

    q_emb = build_embeddings([query_text], tokenizer, encoder, device)[0]
    q_norm = np.linalg.norm(q_emb) + 1e-12
    c_norm = np.linalg.norm(embs, axis=1) + 1e-12
    sims = (embs @ q_emb) / (c_norm * q_norm)
    top_idx = int(np.argmax(sims))

    result = texts[top_idx]
    if len(result) > 200:
        result = result[:200] + "..."
    return result


def predict_cefr(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "Please enter a passage."

    token_len = word_count_mixed(text)
    if token_len > MAX_WORDS:
        return f"Input is too long: {token_len} words/tokens. Please keep it within {MAX_WORDS}."

    language = detect_language(text)

    if model_type == "transformer_mlp":
        x = build_embeddings([text], tokenizer, encoder, device)
        proba = best_model.predict_proba(x)[0]
        pred_id = int(np.argmax(proba))
    else:
        proba = best_model.predict_proba([text])[0]
        pred_id = int(np.argmax(proba))

    pred_label = LABEL_MAP[pred_id]
    confidence = float(proba[pred_id])

    # crosslingual search：En → Zh, Zh → En
    if language == "English":
        cross_pool = zh_pool
        cross_lang = "ZH"
    else:
        cross_pool = en_pool
        cross_lang = "EN"

    similar_text = find_similar(text, cross_pool, pred_label)

    return (
        f"Predicted CEFR difficulty: {pred_label}\n"
        f"Language detected: {language}\n"
        f"Confidence: {confidence:.2f}\n"
        f"Similar {pred_label}-level passage ({cross_lang}, cross-lingual): {similar_text}"
    )


with gr.Blocks(title="BiLit-CEFR Difficulty Demo") as demo:
    gr.Markdown("## BiLit-CEFR Difficulty Classifier")
    gr.Markdown("Enter up to 100 words/tokens in English or Chinese.")

    input_box = gr.Textbox(
        label="Enter a literary passage...",
        placeholder="Enter a literary passage...",
        lines=6,
    )
    output_box = gr.Textbox(label="Prediction", lines=8)

    predict_btn = gr.Button("Predict")
    predict_btn.click(fn=predict_cefr, inputs=input_box, outputs=output_box)

    gr.Markdown("### Try test-set passages (3 EN + 3 ZH)")
    gr.Examples(
        examples=demo_examples,
        inputs=[input_box],
        outputs=[output_box],
        fn=predict_cefr,
        cache_examples=False,
    )


if __name__ == "__main__":
    demo.launch()