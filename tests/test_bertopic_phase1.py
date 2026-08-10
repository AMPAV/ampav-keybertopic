"""Focused model-free tests for the preliminary native BERTopic probe."""

import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


PROBE_PATH = Path(__file__).parents[1] / "examples" / "bertopic_phase1.py"
SPEC = importlib.util.spec_from_file_location("bertopic_phase1", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load probe module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class BertopicPhase1ProbeTest(unittest.TestCase):
    """Protect input construction and retained-output behavior."""

    def test_input_spec_requires_nonempty_id_and_path(self) -> None:
        self.assertEqual(
            PROBE._parse_input_spec("fixture=data/input.txt"),
            ("fixture", Path("data/input.txt")),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            PROBE._parse_input_spec("data/input.txt")

    def test_sentence_aligned_chunks_preserve_normalized_source(self) -> None:
        source = "One short sentence. Two words here! Final question?"

        chunks = PROBE._sentence_aligned_chunks(source, max_words=4)

        self.assertEqual(
            chunks,
            ["One short sentence.", "Two words here!", "Final question?"],
        )
        self.assertEqual(" ".join(chunks), " ".join(source.split()))
        self.assertTrue(all(len(chunk.split()) <= 4 for chunk in chunks))

    def test_long_sentence_falls_back_to_bounded_word_units(self) -> None:
        chunks = PROBE._sentence_aligned_chunks(
            "one two three four five six seven", max_words=3
        )

        self.assertEqual(chunks, ["one two three", "four five six", "seven"])

    def test_corpus_json_requires_ids_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "id": "doc-1",
                            "expected_theme": "theme",
                            "text": "Useful text.",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            documents = PROBE._load_corpus_json(path)

        self.assertEqual(documents[0]["id"], "doc-1")
        self.assertEqual(documents[0]["expected_theme"], "theme")
        self.assertEqual(documents[0]["source_kind"], "corpus_json")

    def test_duplicate_document_ids_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            path.write_text(
                json.dumps(
                    [
                        {"id": "same", "text": "First."},
                        {"id": "same", "text": "Second."},
                    ]
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                corpus_json=[path],
                text=[],
                chunked_text=[],
                max_words_per_document=180,
            )

            with self.assertRaisesRegex(ValueError, "document IDs must be unique"):
                PROBE._load_topic_documents(args)

    def test_expected_theme_counts_remain_probe_analysis(self) -> None:
        documents = [
            {"expected_theme": "archive"},
            {"expected_theme": "archive"},
            {"expected_theme": "racing"},
            {"expected_theme": None},
        ]

        counts = PROBE._expected_theme_counts(documents, [0, 0, 1, -1])

        self.assertEqual(counts, {"0": {"archive": 2}, "1": {"racing": 1}})

    def test_json_safe_preserves_native_sequence_and_numpy_values(self) -> None:
        value = {
            "topic": np.int64(-1),
            "weights": [("term", np.float32(0.25))],
            "probabilities": np.asarray([0.1, 0.9]),
        }

        converted = PROBE._json_safe(value)

        self.assertEqual(converted["topic"], -1)
        self.assertEqual(converted["weights"], [["term", np.float32(0.25).item()]])
        self.assertEqual(converted["probabilities"], [0.1, 0.9])

    def test_existing_output_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                PROBE._validate_new_output_dir(Path(directory))


if __name__ == "__main__":
    unittest.main()
