"""Build a Chinese literature dataset with local heuristic difficulty labels.

Data strategy:
  - Pre-1911 works are loaded from suolyer/classic_chinese_works when possible.
  - Post-1911 works are searched in configured Hugging Face datasets first.
  - Any unavailable work can be supplied as zh_raw/<book title>.txt.

Example:
  .\\.venv\\Scripts\\python.exe build_chinese_literature.py --output chinese_literature_v2.jsonl --limit 30
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import jieba
import pandas as pd
from datasets import get_dataset_config_names, load_dataset
from tqdm import tqdm


ZH_BOOK_METADATA: dict[str, dict[str, Any]] = {
    "红楼梦": {
        "period": "pre-1911",
        "is_adapted": True,
        "target_class": "B",
        "copyright_status": "public_domain",
        "hf_sources": ("suolyer/classic_chinese_works",),
    },
    "西游记": {
        "period": "pre-1911",
        "is_adapted": True,
        "target_class": "B",
        "copyright_status": "public_domain",
        "hf_sources": ("suolyer/classic_chinese_works",),
    },
    "水浒传": {
        "period": "pre-1911",
        "is_adapted": True,
        "target_class": "B",
        "copyright_status": "public_domain",
        "hf_sources": ("suolyer/classic_chinese_works",),
    },
    "三国演义": {
        "period": "pre-1911",
        "is_adapted": True,
        "target_class": "B",
        "copyright_status": "public_domain",
        "hf_sources": ("suolyer/classic_chinese_works",),
    },
    "史记": {
        "period": "pre-1911",
        "is_adapted": False,
        "target_class": "C",
        "copyright_status": "public_domain",
        "hf_sources": ("suolyer/classic_chinese_works",),
    },
    "论语": {
        "period": "pre-1911",
        "is_adapted": False,
        "target_class": "C",
        "copyright_status": "public_domain",
        "hf_sources": ("suolyer/classic_chinese_works",),
    },
    "聊斋志异": {
        "period": "pre-1911",
        "is_adapted": True,
        "target_class": "B",
        "copyright_status": "public_domain",
        "hf_sources": ("suolyer/classic_chinese_works",),
    },
    "世说新语": {
        "period": "pre-1911",
        "is_adapted": False,
        "target_class": "C",
        "author": "刘义庆",
        "copyright_status": "public_domain",
        "source_url": "https://zh.wikisource.org/wiki/世說新語",
    },
    "儒林外史": {
        "period": "pre-1911",
        "is_adapted": False,
        "target_class": "B",
        "author": "吴敬梓",
        "copyright_status": "public_domain",
        "source_url": "https://www.gutenberg.org/ebooks/24032",
    },
    "官场现形记": {
        "period": "pre-1911",
        "is_adapted": False,
        "target_class": "C",
        "author": "李宝嘉",
        "copyright_status": "public_domain",
        "source_url": "https://www.gutenberg.org/ebooks/24138",
    },
    "狂人日记": {
        "period": "post-1911",
        "is_adapted": False,
        "target_class": "B",
        "author": "鲁迅",
        "copyright_status": "public_domain",
        "source_url": "https://www.gutenberg.org/ebooks/25423",
    },
    "呐喊": {
        "period": "post-1911",
        "is_adapted": True,
        "target_class": "B",
        "author": "鲁迅",
        "copyright_status": "public_domain",
    },
    "朝花夕拾": {
        "period": "post-1911",
        "is_adapted": False,
        "target_class": "A",
        "author": "鲁迅",
        "copyright_status": "public_domain",
    },
    "背影": {
        "period": "post-1911",
        "is_adapted": False,
        "target_class": "A",
        "author": "朱自清",
        "copyright_status": "public_domain",
    },
    "骆驼祥子": {
        "period": "post-1911",
        "is_adapted": True,
        "target_class": "A",
        "author": "老舍",
        "copyright_status": "public_domain_cn",
    },
    "寄小读者": {
        "period": "post-1911",
        "is_adapted": False,
        "target_class": "A",
        "author": "冰心",
        "copyright_status": "review_required",
        "active": False,
        "notes": "Excluded by default: Bing Xin died in 1999, so this needs rights review under life-plus-50 regimes.",
    },
}

POST_1911_HF_CANDIDATES = (
    "shibing624/source_dataset",
    "wikipedia-zh",
)

TEXT_FIELDS = ("text", "content", "正文", "body", "paragraph", "sentence", "passage")
TITLE_FIELDS = ("title", "book", "source_book", "作品", "篇名", "name")
AUTHOR_FIELDS = ("author", "作者")
FINAL_PUNCTUATION = "。！？"
LOCAL_TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "utf-16", "gb18030", "big5")
CLAUSE_PUNCTUATION = "，,；;、：:"
PUNCTUATION = FINAL_PUNCTUATION + CLAUSE_PUNCTUATION + "（）()《》“”‘’"
CLASSICAL_PARTICLES = set("之乎者也矣焉哉其乃而于以与则所")
COMMON_IDIOMS = {
    "一言为定",
    "一无所有",
    "七上八下",
    "三心二意",
    "不可思议",
    "不由自主",
    "举世闻名",
    "亡羊补牢",
    "井井有条",
    "人山人海",
    "以身作则",
    "任重道远",
    "低声下气",
    "依依不舍",
    "光明正大",
    "兴高采烈",
    "前所未有",
    "千方百计",
    "半信半疑",
    "各式各样",
    "名副其实",
    "大吃一惊",
    "小心翼翼",
    "异口同声",
    "得意洋洋",
    "心平气和",
    "恍然大悟",
    "成千上万",
    "无可奈何",
    "无论如何",
    "日新月异",
    "明目张胆",
    "津津有味",
    "理所当然",
    "百发百中",
    "目瞪口呆",
    "自言自语",
    "莫名其妙",
    "语重心长",
    "迫不及待",
    "随心所欲",
}
DEFAULT_HSK_LEVELS = {
    "我": 1,
    "你": 1,
    "他": 1,
    "她": 1,
    "是": 1,
    "在": 1,
    "有": 1,
    "不": 1,
    "了": 1,
    "人": 1,
    "说": 1,
    "看": 1,
    "时候": 2,
    "因为": 2,
    "所以": 2,
    "已经": 2,
    "觉得": 2,
    "事情": 2,
    "突然": 3,
    "关系": 3,
    "影响": 3,
    "社会": 3,
    "文化": 3,
    "精神": 3,
    "态度": 4,
    "矛盾": 4,
    "复杂": 4,
    "经验": 4,
    "传统": 4,
    "意识": 5,
    "象征": 5,
    "批判": 5,
    "腐败": 5,
    "伦理": 6,
    "荒唐": 6,
    "讽刺": 6,
    "压迫": 6,
    "衰败": 6,
    "虚伪": 6,
}


def clean_text(text: str) -> str:
    text = str(text).replace("\ufeff", "").replace("\u3000", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chinese_char_count(text: str) -> int:
    return len(re.findall(r"[\u3400-\u9fff]", text))


def jieba_token_count(text: str) -> int:
    return len([token for token in jieba.lcut(text) if token.strip()])


def zh_ttr(text: str) -> float:
    tokens = [token for token in jieba.lcut(text) if token.strip()]
    return len(set(tokens)) / len(tokens) if tokens else 0


def load_hsk_levels(path: Path | None = None) -> dict[str, int]:
    if not path:
        return DEFAULT_HSK_LEVELS
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(word): int(level) for word, level in data.items()}


def load_syntax_analyzer(backend: str) -> Any | None:
    if backend == "none":
        return None
    if backend in {"auto", "hanlp"}:
        try:
            import hanlp  # type: ignore

            model_name = "CTB9_DEP_ELECTRA_SMALL"
            return {"backend": "hanlp", "model": hanlp.load(model_name)}
        except Exception:
            if backend == "hanlp":
                raise
    if backend in {"auto", "ltp"}:
        try:
            from ltp import LTP  # type: ignore

            return {"backend": "ltp", "model": LTP()}
        except Exception:
            if backend == "ltp":
                raise
    return None


def dependency_parse_depth(text: str, syntax_analyzer: Any | None = None) -> int:
    if syntax_analyzer:
        try:
            if syntax_analyzer["backend"] == "hanlp":
                parsed = syntax_analyzer["model"](text, tasks=["tok/fine", "dep"])
                arcs = parsed.get("dep") or []
                heads = [int(arc[0]) if isinstance(arc, (list, tuple)) else int(arc.get("head", 0)) for arc in arcs]
                return dependency_tree_depth(heads)
            if syntax_analyzer["backend"] == "ltp":
                output = syntax_analyzer["model"].pipeline([text], tasks=["cws", "dep"])
                arcs = output.dep[0] if hasattr(output, "dep") else []
                heads = [int(arc[0]) if isinstance(arc, (list, tuple)) else int(getattr(arc, "head", 0)) for arc in arcs]
                return dependency_tree_depth(heads)
        except Exception:
            pass

    clause_count = len(re.findall(r"[，,；;：:]", text))
    connective_count = len(re.findall(r"因为|所以|虽然|但是|然而|于是|因此|如果|只要|即使|以及|并且", text))
    return max(1, min(10, 1 + clause_count + connective_count + chinese_char_count(text) // 70))


def dependency_tree_depth(heads: list[int]) -> int:
    if not heads:
        return 1
    children: dict[int, list[int]] = {index: [] for index in range(len(heads) + 1)}
    for index, head in enumerate(heads, start=1):
        if 0 <= head <= len(heads):
            children.setdefault(head, []).append(index)

    def depth(node: int) -> int:
        child_depths = [depth(child) for child in children.get(node, [])]
        return 1 + (max(child_depths) if child_depths else 0)

    return max(1, depth(0) - 1)


def idiom_density(text: str, tokens: list[str]) -> float:
    idiom_hits = sum(1 for idiom in COMMON_IDIOMS if idiom in text)
    idiom_hits += sum(1 for token in tokens if re.fullmatch(r"[\u3400-\u9fff]{4}", token))
    return idiom_hits / max(len(tokens), 1)


def calculate_difficulty_score(
    text: str,
    syntax_analyzer: Any | None = None,
    hsk_levels: dict[str, int] | None = None,
) -> dict[str, Any]:
    hsk_levels = hsk_levels or DEFAULT_HSK_LEVELS
    tokens = [token for token in jieba.lcut(text) if token.strip()]
    zh_chars = chinese_char_count(text)
    token_zh_lengths = [chinese_char_count(token) for token in tokens if chinese_char_count(token)]

    syntax_depth = dependency_parse_depth(text, syntax_analyzer)
    syntax_score = min(syntax_depth / 10, 1.0) * 100

    idiom_ratio = idiom_density(text, tokens)
    particle_ratio = sum(1 for char in text if char in CLASSICAL_PARTICLES) / max(zh_chars, 1)
    hsk_high_ratio = sum(1 for token in tokens if hsk_levels.get(token, 0) >= 6) / max(len(tokens), 1)
    lexical_score = min((idiom_ratio * 8.0 + particle_ratio * 12.0 + hsk_high_ratio * 5.0) * 100, 100)

    avg_word_length = sum(token_zh_lengths) / max(len(token_zh_lengths), 1)
    punctuation_density = sum(1 for char in text if char in PUNCTUATION) / max(len(text), 1)
    low_punctuation_score = max(0.0, 1.0 - punctuation_density * 20)
    avg_word_score = min(max((avg_word_length - 1.2) / 1.8, 0.0), 1.0) * 100
    statistical_score = avg_word_score * 0.55 + low_punctuation_score * 45

    score = syntax_score * 0.40 + lexical_score * 0.30 + statistical_score * 0.30
    return {
        "score": round(score, 4),
        "syntax_score": round(syntax_score, 4),
        "lexical_score": round(lexical_score, 4),
        "statistical_score": round(statistical_score, 4),
        "dependency_depth": syntax_depth,
        "idiom_density": round(idiom_ratio, 6),
        "classical_particle_ratio": round(particle_ratio, 6),
        "hsk_6_plus_ratio": round(hsk_high_ratio, 6),
        "avg_word_length": round(avg_word_length, 4),
        "punctuation_density": round(punctuation_density, 6),
        "token_count": len(tokens),
        "chinese_char_count": zh_chars,
    }


def split_paragraphs(text: str) -> list[str]:
    text = clean_text(text)
    paragraphs = re.split(r"\n\s*\n+", text)
    if len(paragraphs) == 1:
        paragraphs = text.splitlines()
    return [clean_text(paragraph) for paragraph in paragraphs if clean_text(paragraph)]


def split_sentences(text: str) -> list[str]:
    parts = re.findall(r"[^。！？]+[。！？]?", clean_text(text))
    return [clean_text(part) for part in parts if clean_text(part)]


def split_clauses(text: str) -> list[str]:
    parts = re.findall(r"[^，,；;、：:]+[，,；;、：:]?", clean_text(text))
    return [clean_text(part) for part in parts if clean_text(part)]


def hard_chunks_by_chinese_chars(text: str, min_chars: int, max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_chars = 0

    for char in text:
        current.append(char)
        if re.match(r"[\u3400-\u9fff]", char):
            current_chars += 1
        if current_chars >= max_chars:
            chunk = clean_text("".join(current))
            if chinese_char_count(chunk) >= min_chars:
                chunks.append(chunk)
            current = []
            current_chars = 0

    chunk = clean_text("".join(current))
    if chinese_char_count(chunk) >= min_chars:
        chunks.append(chunk)
    return chunks


def ensure_final_punctuation(text: str) -> str | None:
    text = clean_text(text)
    if not text:
        return None
    if text[-1] in FINAL_PUNCTUATION:
        return text
    match = re.search(r"(.+[。！？])[^。！？]*$", text)
    return clean_text(match.group(1)) if match else None


def combine_units(units: list[str], min_chars: int, max_chars: int, hard_max_chars: int) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    current_chars = 0

    for unit in units:
        unit = clean_text(unit)
        unit_chars = chinese_char_count(unit)
        if unit_chars == 0:
            continue

        if current and current_chars + unit_chars > hard_max_chars:
            candidate = ensure_final_punctuation("".join(current))
            if candidate and min_chars <= chinese_char_count(candidate) <= max_chars:
                segments.append(candidate)
            current = []
            current_chars = 0

        current.append(unit)
        current_chars += unit_chars

        if min_chars <= current_chars <= max_chars:
            candidate = ensure_final_punctuation("".join(current))
            if candidate:
                segments.append(candidate)
            current = []
            current_chars = 0

    candidate = ensure_final_punctuation("".join(current))
    if candidate and min_chars <= chinese_char_count(candidate) <= max_chars:
        segments.append(candidate)

    return segments


def get_chinese_segments(text: str, min_chars: int = 50, max_chars: int = 100) -> list[str]:
    segments: list[str] = []
    for sentence in split_sentences(text):
        sentence_chars = chinese_char_count(sentence)
        if sentence_chars < min_chars:
            continue
        if sentence_chars <= max_chars:
            segments.append(sentence)
            continue

        current: list[str] = []
        current_chars = 0
        for clause in split_clauses(sentence):
            clause_chars = chinese_char_count(clause)
            if clause_chars > max_chars:
                candidate = clean_text("".join(current))
                if min_chars <= chinese_char_count(candidate) <= max_chars:
                    segments.append(candidate)
                current = []
                current_chars = 0
                segments.extend(hard_chunks_by_chinese_chars(clause, min_chars, max_chars))
                continue

            if current and current_chars + clause_chars > max_chars:
                candidate = clean_text("".join(current))
                if min_chars <= chinese_char_count(candidate) <= max_chars:
                    segments.append(candidate)
                current = []
                current_chars = 0

            current.append(clause)
            current_chars += clause_chars

        candidate = clean_text("".join(current))
        if min_chars <= chinese_char_count(candidate) <= max_chars:
            segments.append(candidate)
    return segments


def sample_target(period: str) -> int:
    return 80 if period == "post-1911" else 30


def find_field(row: dict[str, Any], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in row:
            return candidate
    return None


def row_matches_book(row: dict[str, Any], book: str, metadata: dict[str, Any]) -> bool:
    searchable = " ".join(clean_text(value) for value in row.values() if isinstance(value, str))
    if book in searchable:
        return True
    author = metadata.get("author")
    return bool(author and author in searchable)


def iter_hf_rows(dataset_name: str, config_name: str | None, split: str, streaming: bool) -> Iterable[dict[str, Any]]:
    dataset = load_dataset(dataset_name, config_name, split=split, streaming=streaming)
    for row in dataset:
        yield dict(row)


def dataset_configs(dataset_name: str) -> list[str | None]:
    try:
        names = get_dataset_config_names(dataset_name)
    except Exception:
        return [None]
    return names or [None]


def load_texts_from_hf(
    book: str,
    metadata: dict[str, Any],
    dataset_names: Iterable[str],
    split: str,
    streaming: bool,
    max_rows_per_source: int,
) -> list[str]:
    texts: list[str] = []
    for dataset_name in dataset_names:
        for config_name in dataset_configs(dataset_name):
            try:
                for row_index, row in enumerate(iter_hf_rows(dataset_name, config_name, split, streaming)):
                    if row_index >= max_rows_per_source:
                        break
                    text_field = find_field(row, TEXT_FIELDS)
                    if not text_field:
                        continue
                    title_field = find_field(row, TITLE_FIELDS)
                    author_field = find_field(row, AUTHOR_FIELDS)
                    author = metadata.get("author")

                    title_match = title_field and book in clean_text(row.get(title_field, ""))
                    author_match = author_field and author and author in clean_text(row.get(author_field, ""))
                    broad_match = row_matches_book(row, book, metadata)
                    if title_match or author_match or broad_match:
                        texts.append(clean_text(row[text_field]))
            except Exception as exc:
                print(f"  skipped HF source {dataset_name}/{config_name or 'default'} for {book}: {exc}")
                continue
    return texts


def local_candidates(book: str, raw_dir: Path) -> list[Path]:
    slug = re.sub(r"\W+", "_", book, flags=re.UNICODE).strip("_")
    return [raw_dir / f"{book}.txt", raw_dir / f"{slug}.txt"]


def load_texts_from_local(book: str, raw_dir: Path) -> list[str]:
    texts: list[str] = []
    for path in local_candidates(book, raw_dir):
        if path.exists():
            for encoding in LOCAL_TEXT_ENCODINGS:
                try:
                    texts.append(path.read_text(encoding=encoding))
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise UnicodeDecodeError(
                    "local_text",
                    path.read_bytes()[:1],
                    0,
                    1,
                    f"Could not decode {path} with {LOCAL_TEXT_ENCODINGS}",
                )
    return texts


def source_names_for_book(book: str, metadata: dict[str, Any]) -> tuple[str, ...]:
    sources = metadata.get("hf_sources")
    if sources:
        return tuple(sources)
    if metadata["period"] == "post-1911":
        return POST_1911_HF_CANDIDATES
    return ()


def collect_segments_for_book(
    book: str,
    metadata: dict[str, Any],
    raw_dir: Path,
    split: str,
    streaming: bool,
    max_rows_per_source: int,
    skip_hf: bool,
    max_segments: int | None = None,
    segment_min_chars: int = 50,
    segment_max_chars: int = 100,
) -> list[str]:
    texts: list[str] = []
    if not skip_hf:
        texts = load_texts_from_hf(
            book=book,
            metadata=metadata,
            dataset_names=source_names_for_book(book, metadata),
            split=split,
            streaming=streaming,
            max_rows_per_source=max_rows_per_source,
        )
    texts.extend(load_texts_from_local(book, raw_dir))

    segments: list[str] = []
    target = sample_target(metadata["period"])
    seen: set[str] = set()
    for text in texts:
        for segment in get_chinese_segments(text, min_chars=segment_min_chars, max_chars=segment_max_chars):
            if segment in seen:
                continue
            seen.add(segment)
            # Touch jieba so downstream users can reason about tokenized length
            # if they inspect/debug segmentation, while keeping char targets.
            _ = jieba_token_count(segment)
            segments.append(segment)
            if max_segments is not None and len(segments) >= max_segments:
                return segments
    return segments


def extract_segment_batch(
    segments_by_book: dict[str, list[str]],
    total_segments: int,
    seed: int,
) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    nonempty_books = [book for book, segments in segments_by_book.items() if segments]
    if not nonempty_books or total_segments <= 0:
        return []

    per_book_target = math.ceil(total_segments / len(nonempty_books))
    selected: list[tuple[str, str]] = []
    selected_keys: set[tuple[str, str]] = set()

    for book in nonempty_books:
        book_segments = list(segments_by_book[book])
        rng.shuffle(book_segments)
        for segment in book_segments[:per_book_target]:
            key = (book, segment)
            selected.append(key)
            selected_keys.add(key)

    if len(selected) < total_segments:
        remaining = [
            (book, segment)
            for book in nonempty_books
            for segment in segments_by_book[book]
            if (book, segment) not in selected_keys
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[: total_segments - len(selected)])

    rng.shuffle(selected)
    return selected[:total_segments]


def assign_balanced_difficulties(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if not records:
        return records, {"A_max": 0.0, "B_max": 0.0}

    frame = pd.DataFrame(records).sort_values("difficulty_score", kind="mergesort").reset_index()
    total = len(frame)
    labels = ["A"] * total
    first_cut = total // 3
    second_cut = (total * 2) // 3
    for idx in range(first_cut, second_cut):
        labels[idx] = "B"
    for idx in range(second_cut, total):
        labels[idx] = "C"
    frame["difficulty"] = labels

    for _, row in frame.iterrows():
        records[int(row["index"])]["difficulty"] = str(row["difficulty"])

    thresholds = {
        "A_max": float(frame.iloc[max(first_cut - 1, 0)]["difficulty_score"]),
        "B_max": float(frame.iloc[max(second_cut - 1, 0)]["difficulty_score"]),
    }
    return records, thresholds


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_summary(records: list[dict[str, Any]], missing_books: dict[str, int]) -> None:
    label_counts = Counter(record["difficulty"] for record in records)
    period_counts = Counter(record["period"] for record in records)
    book_counts = Counter(record["source_book"] for record in records)

    print("\nSummary")
    print(f"  total_segments: {len(records)}")
    print("  difficulty:")
    for label in ("A", "B", "C"):
        print(f"    {label}: {label_counts[label]}")
    print("  period:")
    for label in ("pre-1911", "post-1911"):
        print(f"    {label}: {period_counts[label]}")
    print("  books:")
    for book, count in sorted(book_counts.items()):
        print(f"    {book}: {count}")

    if missing_books:
        print("  books without enough available local/HF segments:")
        for book, count in sorted(missing_books.items()):
            print(f"    {book}: {count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a locally labeled Chinese literature JSONL dataset.")
    parser.add_argument("--output", default="chinese_literature_v2.jsonl", type=Path, help="Output JSONL path.")
    parser.add_argument("--raw-dir", default="zh_raw", type=Path, help="Local fallback directory for .txt books.")
    parser.add_argument("--split", default="train", help="Dataset split to load.")
    parser.add_argument("--streaming", action="store_true", help="Stream Hugging Face datasets.")
    parser.add_argument("--max-rows-per-source", default=5000, type=int, help="Safety cap while searching each HF source.")
    parser.add_argument("--limit-books", nargs="*", default=None, help="Optional book-title subset for testing.")
    parser.add_argument("--total-segments", default=600, type=int, help="Total randomly sampled records to write.")
    parser.add_argument("--limit", default=None, type=int, help="Backward-compatible alias for --total-segments.")
    parser.add_argument("--seed", default=13, type=int, help="Random seed for reproducible sampling.")
    parser.add_argument("--segment-min-chars", default=50, type=int, help="Minimum Chinese characters per segment.")
    parser.add_argument("--segment-max-chars", default=100, type=int, help="Maximum Chinese characters per segment.")
    parser.add_argument("--syntax-backend", choices=("auto", "none", "hanlp", "ltp"), default="auto")
    parser.add_argument("--hsk-lexicon", default=None, type=Path, help="Optional JSON mapping words to HSK levels.")
    parser.add_argument("--include-review-required", action="store_true", help="Include metadata entries marked active=false.")
    parser.add_argument("--skip-hf", action="store_true", help="Use only local zh_raw/*.txt fallbacks.")
    parser.add_argument("--dry-run", action="store_true", help="Collect segments and print counts without writing output.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    total_segments = args.limit if args.limit is not None else args.total_segments
    if total_segments < 0:
        raise ValueError("--total-segments/--limit must be non-negative.")
    if args.segment_min_chars <= 0 or args.segment_max_chars < args.segment_min_chars:
        raise ValueError("Segment character bounds must satisfy 0 < min <= max.")

    active_books = [
        book
        for book, metadata in ZH_BOOK_METADATA.items()
        if args.include_review_required or metadata.get("active", True)
    ]
    selected_books = args.limit_books or active_books
    unknown_books = [book for book in selected_books if book not in ZH_BOOK_METADATA]
    if unknown_books:
        raise ValueError(f"Unknown books requested: {unknown_books}")

    segments_by_book: dict[str, list[str]] = {}
    missing_books: dict[str, int] = {}
    for book in tqdm(selected_books, desc="Collecting Chinese texts", unit="book"):
        metadata = ZH_BOOK_METADATA[book]
        if not args.include_review_required and not metadata.get("active", True):
            continue
        segments = collect_segments_for_book(
            book=book,
            metadata=metadata,
            raw_dir=args.raw_dir,
            split=args.split,
            streaming=args.streaming,
            max_rows_per_source=args.max_rows_per_source,
            skip_hf=args.skip_hf,
            max_segments=None,
            segment_min_chars=args.segment_min_chars,
            segment_max_chars=args.segment_max_chars,
        )
        segments_by_book[book] = segments
        if not segments:
            missing_books[book] = 0

    planned_total = sum(len(segments) for segments in segments_by_book.values())
    selected_segments = extract_segment_batch(segments_by_book, total_segments, args.seed)
    label_total = len(selected_segments)
    print(f"Collected {planned_total} candidate segments. Sampling {label_total}/{total_segments} requested records.")
    if args.dry_run:
        for book, segments in segments_by_book.items():
            print(f"  {book}: {len(segments)}")
        return 0

    syntax_analyzer = load_syntax_analyzer(args.syntax_backend)
    hsk_levels = load_hsk_levels(args.hsk_lexicon)
    records: list[dict[str, Any]] = []
    progress = tqdm(total=label_total, desc="Local labeling", unit="segment")
    for book, segment in selected_segments:
        metadata = ZH_BOOK_METADATA[book]
        score_metrics = calculate_difficulty_score(segment, syntax_analyzer=syntax_analyzer, hsk_levels=hsk_levels)
        records.append(
            {
                "text": segment,
                "difficulty": "",
                "difficulty_score": score_metrics["score"],
                "score_metrics": score_metrics,
                "language": "zh",
                "source_book": book,
                "period": metadata["period"],
                "is_adapted": bool(metadata["is_adapted"]),
                "target_class": metadata.get("target_class"),
                "copyright_status": metadata.get("copyright_status"),
            }
        )
        progress.update(1)
    progress.close()

    records, thresholds = assign_balanced_difficulties(records)
    write_jsonl(args.output, records)
    print_summary(records, missing_books)
    print(f"  score thresholds: A <= {thresholds['A_max']:.4f}, B <= {thresholds['B_max']:.4f}, C above")
    print(f"\nWrote Chinese literature dataset to {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit("Interrupted.")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
