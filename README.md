# BiLit-CEFR: Bilingual Literary Difficulty Classifier

A bilingual (English + Chinese) literary text difficulty classifier that predicts CEFR-style difficulty levels (`A`, `B`, `C`) using cross-lingual sentence embeddings and a trained classifier. Includes a cross-lingual retrieval feature that surfaces similar passages from the opposite language at the same difficulty level.

🤗 **Live Demo**: [BiLit-CEFR on Hugging Face Spaces](https://huggingface.co/spaces/merriamtao/bilit-cefr-demo)  
📦 **Dataset**: [merriamtao/BiLit-CEFR](https://huggingface.co/datasets/merriamtao/BiLit-CEFR)

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Pipeline](#model-pipeline)
  - [Embedding Models Compared](#embedding-models-compared)
  - [Classifiers Compared](#classifiers-compared)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Demo](#demo)
- [License](#license)

---

## Project Overview

BiLit-CEFR classifies literary passages in English or Chinese into three CEFR-inspired difficulty bands:

| Label | Level |
|-------|-------|
| A | Beginner |
| B | Intermediate |
| C | Advanced |

Given an input passage (up to 100 words/tokens), the system:
1. Detects the input language (English or Chinese)
2. Predicts the CEFR difficulty label
3. Retrieves the most similar passage of the **same label** from the **opposite language** test set (cross-lingual retrieval via cosine similarity)

---

## Dataset

**Source**: [merriamtao/BiLit-CEFR](https://huggingface.co/datasets/merriamtao/BiLit-CEFR)


The dataset contains bilingual literary passages in English and Chinese, annotated with CEFR-style difficulty labels (A / B / C). Each record includes:

| Field | Description |
|-------|-------------|
| `text` | The literary passage |
| `difficulty` | CEFR label: `A`, `B`, or `C` |
| `language` | `en` or `zh` |

---
Difficulty labels were established through a multi-dimensional approach integrating lexical complexity, syntactic depth, existing literary scoring models, and stylistic adjustments. The source code and datasets are available via our Hugging Face Dataset link above.

## Model Pipeline

The pipeline consists of two stages: (1) a multilingual sentence encoder to produce embeddings, and (2) a downstream classifier trained on those embeddings.

### Embedding Models Compared

Cross-lingual sentence embeddings are a core requirement since the model must handle both English and Chinese input uniformly.

| Model | Notes |
|-------|-------|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | ✅ **Selected** — lightweight, strong multilingual performance, fast on CPU |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Higher quality embeddings but slower and larger |
| `BAAI/bge-m3` | Strong cross-lingual retrieval model; higher resource requirements |

####	sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
-	en: Accuracy 0.3447, Macro-F1 0.3003, F1(A/B/C)=0.2883/0.1489/0.4638, C→B=16
- zh: Accuracy 0.3250, Macro-F1 0.3134, F1(A/B/C)=0.2500/0.2821/0.4082
####	xlm-roberta-base
- en: Accuracy 0.4175, Macro-F1 0.1963, F1(A/B/C)=0.0000/0.0000/0.5890, C→B=0
- zh: Accuracy 0.3333, Macro-F1 0.1667, F1(A/B/C)=0.0000/0.0000/0.5000
####	sentence-transformers/LaBSE
- en: Accuracy 0.4223, Macro-F1 0.3314, F1(A/B/C)=0.1481/0.2857/0.5603, C→B=20
- zh: Accuracy 0.3250, Macro-F1 0.2901, F1(A/B/C)=0.0714/0.3797/0.4190


**Selected encoder**: `paraphrase-multilingual-MiniLM-L12-v2`  
**Rationale**: Best balance of multilingual coverage, embedding quality, and inference speed on CPU hardware (Hugging Face Spaces CPU Basic).

---

### Classifiers Compared

Embeddings from the selected encoder were used to train and compare the following classifiers:

| Classifier | Model Type | Notes |
|------------|------------|-------|
| Logistic Regression | `sklearn` | Fast, interpretable baseline |
| SVM (linear) | `sklearn` | Strong on medium-dimensional embeddings |
| Transformer MLP | `transformer_mlp` | ✅ **Selected** — embedding fed into a custom MLP head |

## Results
-	transformer_mlp: Accuracy 0.5767, F1-macro 0.5751
-	svm_linear: Accuracy 0.5798, F1-macro 0.5712
-	logistic_regression: Accuracy 0.5552, F1-macro 0.5409
*Evaluated on held-out test splits from `en_test.jsonl` and `zh_test.jsonl`.*

**Selected classifier**: Transformer MLP (`transformer_mlp`)  
**Artifact**: `best_model.joblib` (saved via `joblib`, contains model + metadata)

---

## Project Structure

```
BiLit-CEFR/
├── app.py                        # Gradio demo entry point
├── requirements.txt              # Python dependencies
├── LICENSE
├── text_data
│   └──en_test.jsonl                 # English test set (used for retrieval + examples)
│   └──en_train.jsonl                # English train set 
│   └──zh_test.jsonl                 # Chinese test set (used for retrieval + examples)
│   └──zh_train.jsonl                # Chinese train set 
├── src
│   └──build_chinese_literature.py   # Labelling functions for Chinese literature
│   └──refine_literature_labels.py   # Labelling functions for English literature
│   └──compare_classifiers_for_demo.py  # LR / SVM / Transformer classifier comparison 
│   └──train_multilingual_difficulty.py # embedding comparison 
├── comparison_outputs/              # Training logs and model comparison artifacts
│   └── best_model.joblib            # Trained classifier artifact
│   └──logistic_regressioin_confusion_matrix.png
│   └──selection_summary.json
│   └──svm_linear_confusion_matrix.png
│   └──transformer_nlp_confusion_matrix.png
│   └──model_commparison.csv
└── README.md
```



---

## Installation

```bash
git clone https://github.com/merriam-tao/BiLit-CEFR -->
cd BiLit-CEFR
pip install -r requirements.txt
```

**Requirements** include:

### Core Dependencies
- Data Handling: datasets>=2.19.0, tqdm>=4.66.0
- NLP & Tokenization: jieba>=0.42.1, sentencepiece>=0.2.0, spacy>=3.7.0
- Deep Learning: torch>=2.2.0, transformers>=4.40.0
- Machine Learning & Evaluation: scikit-learn>=1.5.0

### Visualization & Demo
- Plotting: matplotlib>=3.9.0, seaborn>=0.13.0

### Web Interface: gradio>=5.0.0

Note: For syntax analysis, ensure you download the necessary language models for spacy (e.g., en_core_web_sm or zh_core_web_sm) after installation.

---

## Usage

### Run the Gradio demo locally

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

### Inference (programmatic)

```python
import joblib
from app import predict_cefr

result = predict_cefr("The sea was calm and the sky was clear.")
print(result)
```

---

## Demo

🤗 Hosted on Hugging Face Spaces: [BiLit-CEFR on Hugging Face Spaces](https://huggingface.co/spaces/merriamtao/bilit-cefr-demo)  

**Features:**
- Input up to 100 words/tokens in English or Chinese
- Returns predicted CEFR level, detected language, confidence score
- Cross-lingual retrieval: finds the most similar passage of the **same difficulty level** from the opposite language test set
- 6 built-in example passages (3 English + 3 Chinese)

---

## License

Apache 2.0

---



```bibtex
@misc{bilit-cefr,
  author    = {merriam_tao},
  title     = {BiLit-CEFR: Bilingual Literary Difficulty Classifier},
  year      = {2025},
  url       = {https://github.com/merriam-tao/BiLit-CEFR}
}
```
