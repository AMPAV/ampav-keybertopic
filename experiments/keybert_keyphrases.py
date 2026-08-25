"""Run preliminary native KeyBERT keyphrase probes.

The probe preserves ranked native output and model-boundary evidence without
defining an AMPAV schema, offsets, grouping, chunking, or transcript adapters.
Caller-owned inputs and retained run records remain outside the library API.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
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
IGNORED_SUFFIX = "ignored terminal signal"
BOUNDARY_FILLER = "context"


def parse_args() -> argparse.Namespace:
    """Parse boundary or quality probe arguments."""
    parser = argparse.ArgumentParser(
        description="Run preliminary native KeyBERT keyphrase probes.",
    )
    subparsers = parser.add_subparsers(dest="probe", required=True)

    boundary = subparsers.add_parser(
        "boundary",
        help="probe the document-embedding tokenizer boundary",
    )
    boundary.add_argument("output_dir", type=Path, help="new retained-run directory")
    boundary.add_argument(
        "--fixture",
        action="append",
        default=[],
        type=_parse_fixture_spec,
        metavar="ID=PATH",
        help="quality fixture whose native token length should also be measured",
    )
    _add_model_arguments(boundary)

    quality = subparsers.add_parser(
        "quality",
        help="run the approved native quality matrix",
    )
    quality.add_argument("input", type=Path, help="UTF-8 source text")
    quality.add_argument("output_dir", type=Path, help="new retained-run directory")
    quality.add_argument("--fixture-id", required=True)
    quality.add_argument(
        "--repeat-phrase-cosine",
        type=int,
        default=1,
        help="number of exact phrase-cosine calls; use 2 for the constructed fixture",
    )
    _add_model_arguments(quality)

    return parser.parse_args()


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the pinned model options shared by both probe modes."""
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow model downloads instead of requiring the pinned revision in cache",
    )


def _parse_fixture_spec(value: str) -> tuple[str, Path]:
    """Parse a repeated boundary-fixture value in ``ID=PATH`` form."""
    fixture_id, separator, raw_path = value.partition("=")
    if not separator or not fixture_id.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("fixture must use non-empty ID=PATH form")
    return fixture_id.strip(), Path(raw_path)


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


def _load_models(args: argparse.Namespace) -> tuple[Any, Any, float, list[str]]:
    """Load the exact SentenceTransformers model and native KeyBERT API."""
    from keybert import KeyBERT
    from sentence_transformers import SentenceTransformer

    sentence_model, duration, emitted = _capture_call(
        lambda: SentenceTransformer(
            args.model_id,
            revision=args.model_revision,
            device=args.device,
            local_files_only=not args.allow_download,
        )
    )
    return sentence_model, KeyBERT(model=sentence_model), duration, emitted


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


def _native_token_ids(sentence_model: Any, text: str) -> tuple[list[int], float, list[str]]:
    """Tokenize through SentenceTransformers' real preprocessing path."""
    encoding, duration, emitted = _capture_call(lambda: sentence_model.preprocess([text]))
    return _retained_token_ids(encoding), duration, emitted


def _sentinel_token_ids(tokenizer: Any, sentinel: str) -> list[int]:
    """Return sentinel token IDs without surrounding special tokens."""
    encoding = tokenizer(
        sentinel,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return _retained_token_ids(encoding)


def _contains_subsequence(sequence: list[int], candidate: list[int]) -> bool:
    """Return whether ``candidate`` appears contiguously in ``sequence``."""
    if not candidate:
        return True
    width = len(candidate)
    return any(
        sequence[index : index + width] == candidate
        for index in range(len(sequence) - width + 1)
    )


def _build_boundary_source(tokenizer: Any, target_tokens: int) -> str:
    """Build deterministic text with an exact untruncated special-token count."""
    filler_ids = _sentinel_token_ids(tokenizer, BOUNDARY_FILLER)
    if len(filler_ids) != 1:
        raise ValueError(
            f"boundary filler must encode to one token, got {len(filler_ids)}: {BOUNDARY_FILLER!r}"
        )

    base_source = f"{EARLY_SENTINEL} {END_SENTINEL}"
    base_tokens = len(_raw_token_ids(tokenizer, base_source))
    filler_count = target_tokens - base_tokens
    if filler_count < 0:
        raise ValueError(
            f"boundary sentinels require {base_tokens} tokens, above target {target_tokens}"
        )

    parts = [EARLY_SENTINEL]
    parts.extend([BOUNDARY_FILLER] * filler_count)
    parts.append(END_SENTINEL)
    source = " ".join(parts)
    actual_tokens = len(_raw_token_ids(tokenizer, source))
    if actual_tokens != target_tokens:
        raise RuntimeError(
            f"could not construct exact boundary input: target={target_tokens}, actual={actual_tokens}"
        )
    return source


def _measure_input(sentence_model: Any, text: str) -> dict[str, Any]:
    """Measure raw/native token retention and the resulting document embedding."""
    tokenizer = sentence_model[0].tokenizer
    raw_ids = _raw_token_ids(tokenizer, text)
    native_ids, tokenization_seconds, tokenization_warnings = _native_token_ids(
        sentence_model,
        text,
    )
    encoded, embedding_seconds, embedding_warnings = _capture_call(
        lambda: sentence_model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )
    )
    embedding = np.asarray(encoded[0])
    early_ids = _sentinel_token_ids(tokenizer, EARLY_SENTINEL)
    end_ids = _sentinel_token_ids(tokenizer, END_SENTINEL)
    return {
        "source_characters": len(text),
        "source_whitespace_words": len(text.split()),
        "raw_tokens_including_special_tokens": len(raw_ids),
        "native_retained_tokens_including_special_tokens": len(native_ids),
        "was_truncated": raw_ids != native_ids,
        "raw_token_ids": raw_ids,
        "native_token_ids": native_ids,
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
        "document_embedding": embedding.tolist(),
    }


def _embedding_comparison(left: list[float], right: list[float]) -> dict[str, Any]:
    """Describe exact and cosine agreement between two document embeddings."""
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    cosine_similarity = float(np.dot(left_array, right_array) / denominator)
    return {
        "exactly_equal": bool(np.array_equal(left_array, right_array)),
        "maximum_absolute_difference": float(np.max(np.abs(left_array - right_array))),
        "cosine_similarity": cosine_similarity,
    }


def _fixture_token_measurement(sentence_model: Any, path: Path) -> dict[str, Any]:
    """Measure a real fixture without running keyword extraction."""
    source = path.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError(f"fixture is empty: {path}")
    tokenizer = sentence_model[0].tokenizer
    raw_ids = _raw_token_ids(tokenizer, source)
    native_ids, duration, emitted = _native_token_ids(sentence_model, source)
    return {
        "sha256": _sha256(path),
        "whitespace_words": len(source.split()),
        "raw_tokens_including_special_tokens": len(raw_ids),
        "native_retained_tokens_including_special_tokens": len(native_ids),
        "was_truncated": raw_ids != native_ids,
        "tokenization_duration_seconds": round(duration, 6),
        "warnings": emitted,
    }


def _run_boundary(
    args: argparse.Namespace,
    sentence_model: Any,
    keyword_model: Any,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Run the exact document-embedding boundary matrix."""
    max_tokens = sentence_model.max_seq_length
    if not isinstance(max_tokens, int) or max_tokens < 2:
        raise ValueError(f"model max_seq_length must be a useful integer, got {max_tokens!r}")

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
    generated_inputs["at_limit_with_suffix"] = (
        f"{generated_inputs['at_limit']} {IGNORED_SUFFIX}"
    )

    cases = {
        name: _measure_input(sentence_model, source)
        for name, source in generated_inputs.items()
    }
    candidate_result, candidate_seconds, candidate_warnings = _capture_call(
        lambda: keyword_model.extract_keywords(
            generated_inputs["above_limit"],
            candidates=[EARLY_SENTINEL, END_SENTINEL],
            keyphrase_ngram_range=(3, 3),
            stop_words=None,
            top_n=2,
        )
    )

    fixture_measurements = {}
    input_refs = ["Input: generated boundary inputs under inputs/"]
    for fixture_id, path in args.fixture:
        if not path.is_file():
            raise FileNotFoundError(f"fixture does not exist: {path}")
        fixture_measurements[fixture_id] = _fixture_token_measurement(
            sentence_model,
            path,
        )
        input_refs.append(f"Fixture {fixture_id}: {path.resolve()}")

    output = {
        "probe": "boundary",
        "max_seq_length": max_tokens,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "tokenizer_truncation_side": tokenizer.truncation_side,
        "sentinels": {
            "early": EARLY_SENTINEL,
            "end": END_SENTINEL,
            "ignored_suffix": IGNORED_SUFFIX,
        },
        "target_tokens_including_special_tokens": targets,
        "cases": cases,
        "at_limit_vs_suffixed_embedding": _embedding_comparison(
            cases["at_limit"]["document_embedding"],
            cases["at_limit_with_suffix"]["document_embedding"],
        ),
        "above_limit_full_document_candidate_probe": {
            "candidates": [EARLY_SENTINEL, END_SENTINEL],
            "native_result": candidate_result,
            "duration_seconds": round(candidate_seconds, 6),
            "warnings": candidate_warnings,
            "score_semantics": "document-to-candidate cosine similarity, not confidence",
        },
        "quality_fixture_token_measurements": fixture_measurements,
    }
    return output, generated_inputs, input_refs


def _quality_configurations() -> list[tuple[str, dict[str, Any]]]:
    """Return the approved bounded native quality configurations."""
    return [
        ("native_default", {}),
        (
            "phrase_cosine",
            {
                "keyphrase_ngram_range": (1, 4),
                "stop_words": "english",
                "top_n": 10,
            },
        ),
        (
            "phrase_no_stop_words",
            {
                "keyphrase_ngram_range": (1, 4),
                "stop_words": None,
                "top_n": 10,
            },
        ),
        (
            "phrase_mmr",
            {
                "keyphrase_ngram_range": (1, 4),
                "stop_words": "english",
                "top_n": 10,
                "use_mmr": True,
                "diversity": 0.5,
            },
        ),
        (
            "phrase_mmr_no_stop_words",
            {
                "keyphrase_ngram_range": (1, 4),
                "stop_words": None,
                "top_n": 10,
                "use_mmr": True,
                "diversity": 0.5,
            },
        ),
    ]


def _run_quality(
    args: argparse.Namespace,
    sentence_model: Any,
    keyword_model: Any,
) -> tuple[dict[str, Any], dict[str, str], list[str]]:
    """Run the approved native ranking matrix on one exact fixture."""
    if args.repeat_phrase_cosine < 1:
        raise ValueError("repeat-phrase-cosine must be at least 1")
    if not args.input.is_file():
        raise FileNotFoundError(f"input does not exist: {args.input}")
    source = args.input.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError("input text must not be empty")

    configurations = []
    for name, parameters in _quality_configurations():
        repetitions = args.repeat_phrase_cosine if name == "phrase_cosine" else 1
        calls = []
        for repetition in range(1, repetitions + 1):
            native_result, duration, emitted = _capture_call(
                lambda parameters=parameters: keyword_model.extract_keywords(
                    source,
                    **parameters,
                )
            )
            calls.append(
                {
                    "repetition": repetition,
                    "duration_seconds": round(duration, 6),
                    "warnings": emitted,
                    "native_result": native_result,
                }
            )
        configurations.append(
            {
                "name": name,
                "parameters": parameters,
                "calls": calls,
                "repeated_results_exactly_equal": (
                    all(call["native_result"] == calls[0]["native_result"] for call in calls[1:])
                    if len(calls) > 1
                    else None
                ),
            }
        )

    tokenizer = sentence_model[0].tokenizer
    raw_ids = _raw_token_ids(tokenizer, source)
    native_ids, tokenization_seconds, tokenization_warnings = _native_token_ids(
        sentence_model,
        source,
    )
    output = {
        "probe": "quality",
        "fixture_id": args.fixture_id,
        "input": {
            "sha256": _sha256(args.input),
            "characters": len(source),
            "whitespace_words": len(source.split()),
            "raw_tokens_including_special_tokens": len(raw_ids),
            "native_retained_tokens_including_special_tokens": len(native_ids),
            "document_embedding_was_truncated": raw_ids != native_ids,
            "tokenization_duration_seconds": round(tokenization_seconds, 6),
            "warnings": tokenization_warnings,
        },
        "score_semantics": "document-to-candidate cosine similarity, not confidence",
        "configurations": configurations,
    }
    input_refs = [
        f"Input argument: {args.input}",
        f"Resolved input: {args.input.resolve()}",
    ]
    return output, {}, input_refs


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
    """Build structured native-version and runtime metadata."""
    manifest = {
        "task": "AMPAV-76",
        "probe": args.probe,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "native_package": "keybert",
        "native_version": version("keybert"),
        "sentence_transformers_version": version("sentence-transformers"),
        "transformers_version": version("transformers"),
        "torch_version": version("torch"),
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "model_max_seq_length": sentence_model.max_seq_length,
        "embedding_dimension": sentence_model.get_embedding_dimension(),
        "requested_device": args.device,
        "effective_device": str(sentence_model.device),
        "python_version": platform.python_version(),
        "huggingface_hub_cache": _effective_hf_hub_cache(),
        "local_files_only": not args.allow_download,
        "model_load_seconds": round(model_load_seconds, 6),
        "model_load_warnings": model_load_warnings,
    }
    fixture_id = getattr(args, "fixture_id", None)
    if fixture_id:
        manifest["fixture_id"] = fixture_id
    return manifest


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one caller-owned input file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    """Write readable JSON while preserving native list/tuple order."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_run(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    native_output: dict[str, Any],
    generated_inputs: dict[str, str],
    input_refs: list[str],
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
    if generated_inputs:
        inputs_dir = output_dir / "inputs"
        inputs_dir.mkdir()
        for name, source in generated_inputs.items():
            (inputs_dir / f"{name}.txt").write_text(source + "\n", encoding="utf-8")


def main() -> None:
    """Load the pinned model, run one probe mode, and retain its evidence."""
    args = parse_args()
    _validate_new_output_dir(args.output_dir)
    started_at = datetime.now().astimezone()
    sentence_model, keyword_model, load_seconds, load_warnings = _load_models(args)

    if args.probe == "boundary":
        native_output, generated_inputs, input_refs = _run_boundary(
            args,
            sentence_model,
            keyword_model,
        )
    else:
        native_output, generated_inputs, input_refs = _run_quality(
            args,
            sentence_model,
            keyword_model,
        )

    completed_at = datetime.now().astimezone()
    manifest = _build_manifest(
        args,
        sentence_model,
        started_at=started_at,
        completed_at=completed_at,
        model_load_seconds=load_seconds,
        model_load_warnings=load_warnings,
    )
    _write_run(
        args.output_dir,
        manifest=manifest,
        native_output=native_output,
        generated_inputs=generated_inputs,
        input_refs=input_refs,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "manifest": manifest}, indent=2))


if __name__ == "__main__":
    main()
