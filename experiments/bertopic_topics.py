"""Run preliminary native BERTopic probes.

The probe evaluates native corpus-level topic behavior without defining AMPAV
topic schemas, production chunking, transcript adapters, or model persistence.
Caller-owned inputs and retained run records remain outside the library API.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable, Sequence
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import shlex
import sys
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import yaml


DEFAULT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EARLY_SENTINEL = "silver metadata compass"
END_SENTINEL = "crimson archive beacon"
BOUNDARY_FILLER = "context"
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def parse_args() -> argparse.Namespace:
    """Parse boundary or corpus-topic probe arguments."""
    parser = argparse.ArgumentParser(
        description="Run preliminary native BERTopic probes.",
    )
    subparsers = parser.add_subparsers(dest="probe", required=True)

    boundary = subparsers.add_parser(
        "boundary",
        help="verify the per-document embedding boundary",
    )
    boundary.add_argument("output_dir", type=Path, help="new retained-run directory")
    _add_model_arguments(boundary)

    topics = subparsers.add_parser(
        "topics",
        help="fit native BERTopic to a caller-built document collection",
    )
    _add_topic_arguments(topics)

    pooled_topics = subparsers.add_parser(
        "pooled-topics",
        help="fit BERTopic with one token-weighted pooled chunk embedding per document",
    )
    _add_topic_arguments(pooled_topics)
    pooled_topics.set_defaults(task="AMPAV-157")
    pooled_topics.add_argument(
        "--chunk-max-tokens",
        type=int,
        default=256,
        help="maximum tokenizer tokens including special tokens per embedding chunk",
    )
    pooled_topics.add_argument(
        "--chunk-overlap-sentences",
        type=int,
        default=0,
        help="sentence overlap between adjacent embedding chunks (default: 0)",
    )

    return parser.parse_args()


def _add_topic_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by native and pooled corpus-topic probes."""
    parser.add_argument("output_dir", type=Path, help="new retained-run directory")
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument(
        "--task",
        default="AMPAV-146",
        help="Jira task owning this retained probe (default: AMPAV-146)",
    )
    parser.add_argument(
        "--corpus-json",
        action="append",
        default=[],
        type=Path,
        help="JSON list of objects containing id, text, and optional expected_theme",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        type=_parse_input_spec,
        metavar="ID=PATH",
        help="add one UTF-8 file as one document",
    )
    parser.add_argument(
        "--chunked-text",
        action="append",
        default=[],
        type=_parse_input_spec,
        metavar="ID=PATH",
        help="add sentence-aligned probe documents derived from one UTF-8 file",
    )
    parser.add_argument(
        "--max-words-per-document",
        type=int,
        default=180,
        help="probe-only word cap used for --chunked-text (default: 180)",
    )
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument(
        "--random-state",
        type=int,
        help="construct an explicit UMAP with this random state",
    )
    parser.add_argument(
        "--calculate-probabilities",
        action="store_true",
        help="request full native HDBSCAN probability distributions",
    )
    parser.add_argument(
        "--representation-stop-words",
        choices=("english",),
        help="optional native CountVectorizer stop-word setting",
    )
    parser.add_argument(
        "--representation-ngram-range",
        nargs=2,
        default=(1, 1),
        type=int,
        metavar=("MIN", "MAX"),
        help="native CountVectorizer n-gram range (default: 1 1)",
    )
    parser.add_argument(
        "--representation-min-df",
        type=int,
        default=1,
        help="native CountVectorizer minimum document frequency (default: 1)",
    )
    _add_model_arguments(parser)


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the pinned embedding options shared by probe modes."""
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model downloads instead of requiring the pinned revision in cache",
    )


def _parse_input_spec(value: str) -> tuple[str, Path]:
    """Parse an input in non-empty ``ID=PATH`` form."""
    input_id, separator, raw_path = value.partition("=")
    if not separator or not input_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("input must use non-empty ID=PATH form")
    return input_id.strip(), Path(raw_path)


def _validate_new_output_dir(path: Path) -> None:
    """Reject an existing run path so retained evidence is never overwritten."""
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output directory already exists: {path}")


def _capture_call(call: Callable[[], Any]) -> tuple[Any, float, list[str]]:
    """Run one native operation and retain duration and Python warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = perf_counter()
        result = call()
        duration = perf_counter() - started
    return result, duration, [str(item.message) for item in caught]


def _load_sentence_model(args: argparse.Namespace) -> tuple[Any, float, list[str]]:
    """Load the exact SentenceTransformers model used by adopted KeyBERT."""
    if not args.allow_download:
        # Some Transformers components probe for optional remote processor files
        # despite local_files_only. Make the opt-in probe's cache-only boundary
        # explicit before importing SentenceTransformers.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import SentenceTransformer

    model, duration, emitted = _capture_call(
        lambda: SentenceTransformer(
            args.model_id,
            revision=args.model_revision,
            device=args.device,
            local_files_only=not args.allow_download,
        )
    )
    return model, duration, emitted


def _row_values(value: Any) -> list[int]:
    """Convert one encoded tensor/list row to ordinary integer values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def _retained_token_ids(encoding: dict[str, Any]) -> list[int]:
    """Return non-padding token IDs from one encoded input."""
    token_ids = _row_values(encoding["input_ids"])
    if "attention_mask" not in encoding:
        return token_ids
    attention = _row_values(encoding["attention_mask"])
    return [token_id for token_id, retained in zip(token_ids, attention) if retained]


def _raw_token_ids(tokenizer: Any, text: str) -> list[int]:
    """Tokenize without truncation while including native special tokens."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        return_attention_mask=True,
    )
    return _retained_token_ids(encoding)


def _native_token_ids(sentence_model: Any, text: str) -> list[int]:
    """Tokenize through SentenceTransformers' real preprocessing path."""
    return _retained_token_ids(sentence_model.preprocess([text]))


def _sentinel_token_ids(tokenizer: Any, sentinel: str) -> list[int]:
    """Return sentinel token IDs without surrounding special tokens."""
    encoding = tokenizer(
        sentinel,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return _retained_token_ids(encoding)


def _contains_subsequence(sequence: Sequence[int], candidate: Sequence[int]) -> bool:
    """Return whether ``candidate`` appears contiguously in ``sequence``."""
    if not candidate:
        return True
    width = len(candidate)
    return any(
        list(sequence[index : index + width]) == list(candidate)
        for index in range(len(sequence) - width + 1)
    )


def _build_boundary_source(tokenizer: Any, target_tokens: int) -> str:
    """Build deterministic text with an exact untruncated special-token count."""
    filler_ids = _sentinel_token_ids(tokenizer, BOUNDARY_FILLER)
    if len(filler_ids) != 1:
        raise ValueError(
            f"boundary filler must encode to one token, got {len(filler_ids)}"
        )
    base = f"{EARLY_SENTINEL} {END_SENTINEL}"
    filler_count = target_tokens - len(_raw_token_ids(tokenizer, base))
    if filler_count < 0:
        raise ValueError(f"sentinels do not fit target token count {target_tokens}")
    source = " ".join(
        [EARLY_SENTINEL, *([BOUNDARY_FILLER] * filler_count), END_SENTINEL]
    )
    actual = len(_raw_token_ids(tokenizer, source))
    if actual != target_tokens:
        raise RuntimeError(
            f"could not construct exact boundary input: target={target_tokens}, actual={actual}"
        )
    return source


def _measure_boundary_input(sentence_model: Any, text: str) -> dict[str, Any]:
    """Measure token retention and embedding for one boundary document."""
    tokenizer = sentence_model[0].tokenizer
    raw_ids = _raw_token_ids(tokenizer, text)
    native_ids, tokenization_seconds, tokenization_warnings = _capture_call(
        lambda: _native_token_ids(sentence_model, text)
    )
    encoded, embedding_seconds, embedding_warnings = _capture_call(
        lambda: sentence_model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    early_ids = _sentinel_token_ids(tokenizer, EARLY_SENTINEL)
    end_ids = _sentinel_token_ids(tokenizer, END_SENTINEL)
    return {
        "source_characters": len(text),
        "source_whitespace_words": len(text.split()),
        "raw_tokens_including_special_tokens": len(raw_ids),
        "native_retained_tokens_including_special_tokens": len(native_ids),
        "was_truncated": raw_ids != native_ids,
        "early_sentinel": {
            "in_raw_tokens": _contains_subsequence(raw_ids, early_ids),
            "in_native_tokens": _contains_subsequence(native_ids, early_ids),
        },
        "end_sentinel": {
            "in_raw_tokens": _contains_subsequence(raw_ids, end_ids),
            "in_native_tokens": _contains_subsequence(native_ids, end_ids),
        },
        "tokenization_duration_seconds": round(tokenization_seconds, 6),
        "embedding_duration_seconds": round(embedding_seconds, 6),
        "warnings": tokenization_warnings + embedding_warnings,
        "document_embedding": np.asarray(encoded[0]).tolist(),
    }


def _run_boundary(
    sentence_model: Any,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Run the approved per-document embedding-boundary probe."""
    max_tokens = sentence_model.max_seq_length
    if not isinstance(max_tokens, int) or max_tokens < 2:
        raise ValueError(f"unexpected model max_seq_length: {max_tokens!r}")
    tokenizer = sentence_model[0].tokenizer
    targets = {
        "below_limit": max_tokens - 1,
        "at_limit": max_tokens,
        "above_limit": max_tokens + 1,
    }
    generated_inputs = {
        name: _build_boundary_source(tokenizer, target)
        for name, target in targets.items()
    }
    cases = {
        name: _measure_boundary_input(sentence_model, text)
        for name, text in generated_inputs.items()
    }
    return (
        {
            "probe": "boundary",
            "max_seq_length": max_tokens,
            "tokenizer_model_max_length": tokenizer.model_max_length,
            "tokenizer_truncation_side": tokenizer.truncation_side,
            "sentinels": {"early": EARLY_SENTINEL, "end": END_SENTINEL},
            "target_tokens_including_special_tokens": targets,
            "cases": cases,
        },
        generated_inputs,
        ["Input: generated boundary documents under inputs/"],
    )


def _load_corpus_json(path: Path) -> list[dict[str, Any]]:
    """Load and validate one constructed document-corpus JSON file."""
    if not path.is_file():
        raise FileNotFoundError(f"corpus does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError(f"corpus must be a non-empty JSON list: {path}")
    documents = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise TypeError(f"corpus item {index} must be an object")
        document_id = item.get("id")
        text = item.get("text")
        expected_theme = item.get("expected_theme")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"corpus item {index} must have a non-empty string id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"corpus item {index} must have non-empty string text")
        if expected_theme is not None and (
            not isinstance(expected_theme, str) or not expected_theme.strip()
        ):
            raise ValueError(
                f"corpus item {index} expected_theme must be null or a non-empty string"
            )
        documents.append(
            {
                "id": document_id.strip(),
                "text": text.strip(),
                "expected_theme": expected_theme,
                "source_kind": "corpus_json",
                "source_path": str(path.resolve()),
            }
        )
    return documents


def _sentence_aligned_chunks(text: str, max_words: int) -> list[str]:
    """Build probe-only sentence-aligned document units from plain text."""
    if not isinstance(max_words, int):
        raise TypeError("max_words must be an integer")
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("chunked text must not be empty")

    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(normalized) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                current_words = 0
            chunks.extend(
                " ".join(words[start : start + max_words])
                for start in range(0, len(words), max_words)
            )
            continue
        if current and current_words + len(words) > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def _load_topic_documents(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[str]]:
    """Build the exact caller-selected document collection for one run."""
    documents: list[dict[str, Any]] = []
    input_refs: list[str] = []
    for path in args.corpus_json:
        documents.extend(_load_corpus_json(path))
        input_refs.append(f"Corpus JSON: {path.resolve()} (sha256={_sha256(path)})")
    for input_id, path in args.text:
        if not path.is_file():
            raise FileNotFoundError(f"text input does not exist: {path}")
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"text input is empty: {path}")
        documents.append(
            {
                "id": input_id,
                "text": text,
                "expected_theme": None,
                "source_kind": "whole_text",
                "source_path": str(path.resolve()),
            }
        )
        input_refs.append(f"Whole text {input_id}: {path.resolve()} (sha256={_sha256(path)})")
    for input_id, path in args.chunked_text:
        if not path.is_file():
            raise FileNotFoundError(f"chunked input does not exist: {path}")
        source = path.read_text(encoding="utf-8").strip()
        chunks = _sentence_aligned_chunks(source, args.max_words_per_document)
        documents.extend(
            {
                "id": f"{input_id}-{index:04d}",
                "text": chunk,
                "expected_theme": None,
                "source_kind": "probe_sentence_aligned_chunk",
                "source_path": str(path.resolve()),
            }
            for index, chunk in enumerate(chunks, start=1)
        )
        input_refs.append(
            f"Probe-chunked text {input_id}: {path.resolve()} "
            f"(sha256={_sha256(path)}, max_words={args.max_words_per_document}, "
            f"documents={len(chunks)})"
        )
    if len(documents) < 2:
        raise ValueError("topic probe requires at least two documents")
    ids = [document["id"] for document in documents]
    duplicates = sorted(item for item, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"document IDs must be unique: {duplicates}")
    return documents, input_refs


def _measure_document_tokens(sentence_model: Any, text: str) -> dict[str, Any]:
    """Measure per-document raw and retained embedding token counts."""
    tokenizer = sentence_model[0].tokenizer
    raw_ids = _raw_token_ids(tokenizer, text)
    native_ids = _native_token_ids(sentence_model, text)
    return {
        "characters": len(text),
        "whitespace_words": len(text.split()),
        "raw_tokens_including_special_tokens": len(raw_ids),
        "native_retained_tokens_including_special_tokens": len(native_ids),
        "embedding_input_was_truncated": raw_ids != native_ids,
    }


def _split_overlong_sentence(
    tokenizer: Any, sentence: str, max_tokens: int
) -> list[str]:
    """Split one overlong sentence into token-bounded word units for a probe."""
    words = sentence.split()
    units: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if len(_raw_token_ids(tokenizer, candidate)) <= max_tokens:
            current.append(word)
            continue
        if not current:
            raise ValueError(
                "one whitespace-delimited word exceeds the requested chunk token limit"
            )
        units.append(" ".join(current))
        current = [word]
    if current:
        units.append(" ".join(current))
    return units


def _embedding_chunks(
    sentence_model: Any,
    text: str,
    *,
    max_tokens: int,
    overlap_sentences: int,
) -> list[str]:
    """Build sentence-first chunks that each fit the embedding model exactly.

    A sentence longer than the requested boundary falls back to bounded word
    units. The chunks are experiment-side embedding inputs, never BERTopic
    corpus documents.
    """
    if not isinstance(max_tokens, int) or max_tokens < 3:
        raise ValueError("chunk-max-tokens must be at least 3")
    if max_tokens > sentence_model.max_seq_length:
        raise ValueError(
            "chunk-max-tokens must not exceed the embedding model max_seq_length"
        )
    if not isinstance(overlap_sentences, int) or overlap_sentences < 0:
        raise ValueError("chunk-overlap-sentences must be a non-negative integer")

    tokenizer = sentence_model[0].tokenizer
    normalized = " ".join(text.split())
    sentences = [part.strip() for part in SENTENCE_BOUNDARY.split(normalized) if part.strip()]
    units: list[str] = []
    for sentence in sentences:
        if len(_raw_token_ids(tokenizer, sentence)) <= max_tokens:
            units.append(sentence)
        else:
            units.extend(_split_overlong_sentence(tokenizer, sentence, max_tokens))

    chunks: list[str] = []
    current: list[str] = []
    for unit in units:
        candidate = " ".join([*current, unit])
        if len(_raw_token_ids(tokenizer, candidate)) <= max_tokens:
            current.append(unit)
            continue
        if not current:
            raise RuntimeError("a bounded embedding unit did not fit its token limit")
        chunks.append(" ".join(current))
        current = current[-overlap_sentences:] if overlap_sentences else []
        while current and len(_raw_token_ids(tokenizer, " ".join([*current, unit]))) > max_tokens:
            current.pop(0)
        current.append(unit)
    if current:
        chunks.append(" ".join(current))
    if not chunks:
        raise ValueError("document produced no embedding chunks")
    return chunks


def _pooled_document_embeddings(
    sentence_model: Any,
    documents: Sequence[dict[str, Any]],
    *,
    max_tokens: int,
    overlap_sentences: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Embed bounded chunks and return one normalized weighted mean per source."""
    tokenizer = sentence_model[0].tokenizer
    special_token_count = len(_raw_token_ids(tokenizer, ""))
    chunks_by_document = [
        _embedding_chunks(
            sentence_model,
            document["text"],
            max_tokens=max_tokens,
            overlap_sentences=overlap_sentences,
        )
        for document in documents
    ]
    flat_chunks = [chunk for chunks in chunks_by_document for chunk in chunks]
    encoded, _, _ = _capture_call(
        lambda: sentence_model.encode(
            flat_chunks,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    chunk_embeddings = np.asarray(encoded)
    if chunk_embeddings.ndim != 2 or chunk_embeddings.shape[0] != len(flat_chunks):
        raise RuntimeError("native embedding output did not contain one vector per chunk")

    pooled_rows: list[np.ndarray] = []
    diagnostics: list[dict[str, Any]] = []
    offset = 0
    for document, chunks in zip(documents, chunks_by_document):
        count = len(chunks)
        document_embeddings = chunk_embeddings[offset : offset + count]
        offset += count
        token_counts = [len(_raw_token_ids(tokenizer, chunk)) for chunk in chunks]
        retained_counts = [len(_native_token_ids(sentence_model, chunk)) for chunk in chunks]
        if token_counts != retained_counts:
            raise RuntimeError("a pooled embedding chunk was truncated")
        weights = np.asarray(
            [max(1, token_count - special_token_count) for token_count in token_counts],
            dtype=float,
        )
        pooled = np.average(document_embeddings, axis=0, weights=weights)
        norm = float(np.linalg.norm(pooled))
        if norm == 0.0:
            raise RuntimeError(f"pooled embedding has zero norm for {document['id']}")
        pooled_rows.append(pooled / norm)
        diagnostics.append(
            {
                "id": document["id"],
                "chunk_count": count,
                "chunk_token_counts_including_special_tokens": token_counts,
                "chunk_retained_tokens_including_special_tokens": retained_counts,
                "pooling": "token_count_weighted_mean_then_l2_normalize",
                "pooling_weights_content_tokens": weights.astype(int).tolist(),
                "unnormalized_embedding_l2_norm": norm,
            }
        )
    return np.asarray(pooled_rows), diagnostics


def _build_topic_model(args: argparse.Namespace, sentence_model: Any, document_count: int) -> Any:
    """Construct native BERTopic with only the selected bounded configuration."""
    from bertopic import BERTopic

    umap_model = None
    if args.random_state is not None:
        from umap import UMAP

        umap_model = UMAP(
            n_neighbors=min(15, document_count - 1),
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=args.random_state,
        )
    ngram_min, ngram_max = args.representation_ngram_range
    if ngram_min < 1 or ngram_max < ngram_min:
        raise ValueError(
            "representation-ngram-range must satisfy 1 <= minimum <= maximum"
        )
    if args.representation_min_df < 1:
        raise ValueError("representation-min-df must be at least 1")
    vectorizer_model = None
    if (
        args.representation_stop_words is not None
        or (ngram_min, ngram_max) != (1, 1)
        or args.representation_min_df != 1
    ):
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer_model = CountVectorizer(
            stop_words=args.representation_stop_words,
            ngram_range=(ngram_min, ngram_max),
            min_df=args.representation_min_df,
        )
    return BERTopic(
        embedding_model=sentence_model,
        umap_model=umap_model,
        vectorizer_model=vectorizer_model,
        min_topic_size=args.min_topic_size,
        calculate_probabilities=args.calculate_probabilities,
        verbose=True,
    )


def _dataframe_records(frame: Any) -> list[dict[str, Any]]:
    """Convert one native pandas frame to safe JSON records."""
    return json.loads(frame.to_json(orient="records", force_ascii=False))


def _json_safe(value: Any) -> Any:
    """Convert native NumPy/container values without changing list ordering."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _expected_theme_counts(
    documents: Sequence[dict[str, Any]], topics: Sequence[int]
) -> dict[str, dict[str, int]]:
    """Summarize constructed expected themes separately from native output."""
    counts: dict[str, Counter[str]] = {}
    for document, topic in zip(documents, topics):
        theme = document.get("expected_theme")
        if theme is None:
            continue
        counts.setdefault(str(topic), Counter())[theme] += 1
    return {
        topic: dict(sorted(theme_counts.items()))
        for topic, theme_counts in sorted(counts.items(), key=lambda item: int(item[0]))
    }


def _run_topics(
    args: argparse.Namespace,
    sentence_model: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Run one native BERTopic fit and collect reviewable native state."""
    if args.min_topic_size < 2:
        raise ValueError("min-topic-size must be at least 2")
    documents, input_refs = _load_topic_documents(args)
    native_output = _fit_topic_documents(
        args,
        sentence_model,
        documents,
        embeddings=None,
        embedding_diagnostics=None,
    )
    return native_output, documents, input_refs


def _run_pooled_topics(
    args: argparse.Namespace,
    sentence_model: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    """Run BERTopic with one pooled chunk embedding per source document."""
    if args.min_topic_size < 2:
        raise ValueError("min-topic-size must be at least 2")
    documents, input_refs = _load_topic_documents(args)
    embeddings, diagnostics = _pooled_document_embeddings(
        sentence_model,
        documents,
        max_tokens=args.chunk_max_tokens,
        overlap_sentences=args.chunk_overlap_sentences,
    )
    native_output = _fit_topic_documents(
        args,
        sentence_model,
        documents,
        embeddings=embeddings,
        embedding_diagnostics=diagnostics,
    )
    return native_output, documents, input_refs


def _fit_topic_documents(
    args: argparse.Namespace,
    sentence_model: Any,
    documents: list[dict[str, Any]],
    *,
    embeddings: np.ndarray | None,
    embedding_diagnostics: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Fit native BERTopic while retaining selected embedding-path evidence."""
    texts = [document["text"] for document in documents]
    token_measurements = [
        _measure_document_tokens(sentence_model, text) for text in texts
    ]
    if embeddings is not None and embeddings.shape != (
        len(documents),
        sentence_model.get_embedding_dimension(),
    ):
        raise ValueError("pooled embedding matrix must contain one fixed-size vector per document")
    model = _build_topic_model(args, sentence_model, len(documents))
    fit_result, fit_seconds, fit_warnings = _capture_call(
        lambda: model.fit_transform(texts, embeddings=embeddings)
        if embeddings is not None
        else model.fit_transform(texts)
    )
    topics, probabilities = fit_result
    topic_ids = [int(topic) for topic in topics]
    probability_array = None if probabilities is None else np.asarray(probabilities)
    unique_topics = sorted(set(topic_ids))
    per_topic = {
        str(topic): _json_safe(model.get_topic(topic)) for topic in unique_topics
    }
    document_rows = []
    for index, (document, topic, token_measurement) in enumerate(
        zip(documents, topic_ids, token_measurements)
    ):
        row = {
            "id": document["id"],
            "expected_theme": document["expected_theme"],
            "source_kind": document["source_kind"],
            "topic": topic,
            "token_measurement": token_measurement,
        }
        if embedding_diagnostics is not None:
            row["pooled_embedding_diagnostics"] = embedding_diagnostics[index]
        document_rows.append(row)

    return {
        "probe": args.probe,
        "fixture_id": args.fixture_id,
        "document_count": len(documents),
        "embedding_strategy": (
            "native_full_document_embedding"
            if embeddings is None
            else "token_count_weighted_chunk_mean_then_l2_normalize"
        ),
        "embedding_matrix": (
            None
            if embeddings is None
            else {
                "shape": list(embeddings.shape),
                "dtype": str(embeddings.dtype),
                "one_vector_per_source_document": True,
                "chunk_max_tokens": args.chunk_max_tokens,
                "chunk_overlap_sentences": args.chunk_overlap_sentences,
            }
        ),
        "fit_transform": {
            "topics": topic_ids,
            "probabilities": None if probabilities is None else probability_array.tolist(),
            "probabilities_native_type": type(probabilities).__name__,
            "probabilities_shape": (
                None if probability_array is None else list(probability_array.shape)
            ),
            "probabilities_dtype": (
                None if probability_array is None else str(probability_array.dtype)
            ),
            "duration_seconds": round(fit_seconds, 6),
            "warnings": fit_warnings,
        },
        "topic_info": _dataframe_records(model.get_topic_info()),
        "document_info": _dataframe_records(model.get_document_info(texts)),
        "topics": per_topic,
        "topic_sizes": _json_safe(model.topic_sizes_),
        "document_assignments": document_rows,
        "score_semantics": {
            "topic_term_weights": "native c-TF-IDF weights, not confidence",
            "fit_transform_probabilities": (
                "native HDBSCAN membership strength for the assigned topic when "
                "calculate_probabilities is false; full native topic-membership "
                "distributions when true"
            ),
            "outlier_topic": "native topic -1",
        },
        "probe_analysis": {
            "expected_theme_counts_by_topic": _expected_theme_counts(
                documents, topic_ids
            ),
            "truncated_document_ids": [
                document["id"]
                for document, measurement in zip(documents, token_measurements)
                if measurement["embedding_input_was_truncated"]
            ],
        },
    }


def _effective_hf_hub_cache() -> str:
    """Return the effective Hugging Face Hub cache path for run metadata."""
    if cache := os.environ.get("HF_HUB_CACHE"):
        return str(Path(cache).expanduser())
    if cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return str(Path(cache).expanduser())
    if home := os.environ.get("HF_HOME"):
        return str(Path(home).expanduser() / "hub")
    return str(Path.home() / ".cache" / "huggingface" / "hub")


def _build_manifest(
    args: argparse.Namespace,
    sentence_model: Any,
    *,
    started_at: datetime,
    completed_at: datetime,
    model_load_seconds: float,
    model_load_warnings: list[str],
) -> dict[str, Any]:
    """Build structured native-version, model, and runtime metadata."""
    manifest = {
        "task": getattr(args, "task", "AMPAV-146"),
        "probe": args.probe,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "native_package": "bertopic",
        "native_version": version("bertopic"),
        "resolved_dependencies": {
            package: version(package)
            for package in (
                "sentence-transformers",
                "transformers",
                "torch",
                "umap-learn",
                "hdbscan",
                "scikit-learn",
                "numpy",
                "pandas",
            )
        },
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_max_seq_length": sentence_model.max_seq_length,
        "embedding_dimension": sentence_model.get_embedding_dimension(),
        "requested_device": args.device,
        "effective_device": str(sentence_model.device),
        "python_version": platform.python_version(),
        "huggingface_hub_cache": _effective_hf_hub_cache(),
        "local_files_only": not args.allow_download,
        "hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE") == "1",
        "model_load_seconds": round(model_load_seconds, 6),
        "model_load_warnings": model_load_warnings,
    }
    if args.probe in {"topics", "pooled-topics"}:
        manifest.update(
            {
                "fixture_id": args.fixture_id,
                "effective_parameters": {
                    "min_topic_size": args.min_topic_size,
                    "calculate_probabilities": args.calculate_probabilities,
                    "umap_random_state": args.random_state,
                    "max_words_per_probe_document": args.max_words_per_document,
                    "representation_stop_words": args.representation_stop_words,
                    "representation_ngram_range": list(
                        args.representation_ngram_range
                    ),
                    "representation_min_df": args.representation_min_df,
                },
            }
        )
    if args.probe == "pooled-topics":
        manifest["effective_parameters"].update(
            {
                "chunk_max_tokens": args.chunk_max_tokens,
                "chunk_overlap_sentences": args.chunk_overlap_sentences,
                "pooling": "token_count_weighted_mean_then_l2_normalize",
            }
        )
    return manifest


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one caller-owned input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """Write readable JSON while preserving native sequence ordering."""
    path.write_text(
        json.dumps(_json_safe(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_run(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    native_output: dict[str, Any],
    input_refs: list[str],
    generated_boundary_inputs: dict[str, str] | None = None,
    documents: list[dict[str, Any]] | None = None,
) -> None:
    """Persist one human-reviewable native run record."""
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "command.txt").write_text(
        shlex.join([sys.executable, *sys.argv]) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.yaml").write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )
    (output_dir / "input_ref.txt").write_text(
        "\n".join(input_refs) + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "native_output.json", native_output)
    (output_dir / "observations.md").write_text(
        "# Observations\n\n- Native probe completed; human review pending.\n",
        encoding="utf-8",
    )
    if documents is not None:
        _write_json(output_dir / "documents.json", documents)
    if generated_boundary_inputs:
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir()
        for name, source in generated_boundary_inputs.items():
            (inputs_dir / f"{name}.txt").write_text(
                source + "\n", encoding="utf-8"
            )


def main() -> None:
    """Load the pinned model, run one native probe, and retain its evidence."""
    args = parse_args()
    _validate_new_output_dir(args.output_dir)
    started_at = datetime.now().astimezone()
    sentence_model, model_load_seconds, model_load_warnings = _load_sentence_model(
        args
    )

    if args.probe == "boundary":
        native_output, generated_inputs, input_refs = _run_boundary(sentence_model)
        documents = None
    elif args.probe == "pooled-topics":
        native_output, documents, input_refs = _run_pooled_topics(args, sentence_model)
        generated_inputs = None
    else:
        native_output, documents, input_refs = _run_topics(args, sentence_model)
        generated_inputs = None

    completed_at = datetime.now().astimezone()
    manifest = _build_manifest(
        args,
        sentence_model,
        started_at=started_at,
        completed_at=completed_at,
        model_load_seconds=model_load_seconds,
        model_load_warnings=model_load_warnings,
    )
    _write_run(
        args.output_dir,
        manifest=manifest,
        native_output=native_output,
        input_refs=input_refs,
        generated_boundary_inputs=generated_inputs,
        documents=documents,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
