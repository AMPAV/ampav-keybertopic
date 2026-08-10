"""Focused model-free tests for the preliminary native KeyBERT probe."""

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

PROBE_PATH = Path(__file__).parents[1] / "examples" / "keybert_keyphrases_phase1.py"
SPEC = importlib.util.spec_from_file_location("keybert_keyphrases_phase1", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load probe module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)

EARLY_SENTINEL = PROBE.EARLY_SENTINEL
END_SENTINEL = PROBE.END_SENTINEL
_build_boundary_source = PROBE._build_boundary_source
_build_manifest = PROBE._build_manifest
_contains_subsequence = PROBE._contains_subsequence
_parse_fixture_spec = PROBE._parse_fixture_spec
_raw_token_ids = PROBE._raw_token_ids
_validate_new_output_dir = PROBE._validate_new_output_dir
_write_json = PROBE._write_json


class FakeTokenizer:
    """Minimal one-token-per-word tokenizer for boundary construction tests."""

    model_max_length = 12
    truncation_side = "right"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        return_attention_mask: bool = False,
    ) -> dict[str, list[int]]:
        del padding, truncation
        token_ids = list(range(1000, 1000 + len(text.split())))
        if add_special_tokens:
            token_ids = [101, *token_ids, 102]
        output = {"input_ids": token_ids}
        if return_attention_mask:
            output["attention_mask"] = [1] * len(token_ids)
        return output


class FakeSentenceModel:
    """Manifest-only stand-in that never loads a real model."""

    max_seq_length = 256
    device = "cpu"

    @staticmethod
    def get_embedding_dimension() -> int:
        return 384


class KeyBertPhase1ProbeTest(unittest.TestCase):
    """Protect deterministic helpers and retained-output behavior."""

    def test_boundary_source_has_exact_token_count_and_sentinels(self) -> None:
        tokenizer = FakeTokenizer()

        source = _build_boundary_source(tokenizer, tokenizer.model_max_length)

        self.assertTrue(source.startswith(EARLY_SENTINEL))
        self.assertTrue(source.endswith(END_SENTINEL))
        self.assertEqual(len(_raw_token_ids(tokenizer, source)), tokenizer.model_max_length)

    def test_contains_subsequence_requires_contiguous_values(self) -> None:
        self.assertTrue(_contains_subsequence([1, 2, 3, 4], [2, 3]))
        self.assertFalse(_contains_subsequence([1, 2, 3, 4], [2, 4]))
        self.assertTrue(_contains_subsequence([1], []))

    def test_fixture_spec_requires_nonempty_id_and_path(self) -> None:
        self.assertEqual(
            _parse_fixture_spec("fixture=data/input.txt"),
            ("fixture", Path("data/input.txt")),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            _parse_fixture_spec("data/input.txt")

    def test_existing_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                _validate_new_output_dir(Path(directory))

    def test_native_tuple_order_serializes_as_json_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "native.json"
            _write_json(output, [("first phrase", 0.75), ("second phrase", 0.5)])

            loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(
            loaded,
            [["first phrase", 0.75], ["second phrase", 0.5]],
        )

    def test_manifest_preserves_native_versions_and_score_context(self) -> None:
        args = SimpleNamespace(
            probe="boundary",
            model_id="model-id",
            model_revision="revision",
            device="cpu",
            allow_download=False,
        )
        timestamp = datetime(2026, 8, 8, tzinfo=timezone.utc)

        manifest = _build_manifest(
            args,
            FakeSentenceModel(),
            started_at=timestamp,
            completed_at=timestamp,
            model_load_seconds=1.25,
            model_load_warnings=[],
        )

        self.assertEqual(manifest["task"], "AMPAV-76")
        self.assertEqual(manifest["native_version"], "0.9.0")
        self.assertEqual(manifest["model_revision"], "revision")
        self.assertEqual(manifest["model_max_seq_length"], 256)
        self.assertTrue(manifest["local_files_only"])


if __name__ == "__main__":
    unittest.main()
