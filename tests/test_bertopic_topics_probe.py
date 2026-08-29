"""Focused model-free tests for the preliminary native BERTopic probe."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


PROBE_PATH = Path(__file__).parents[1] / "experiments" / "bertopic_topics.py"
SPEC = importlib.util.spec_from_file_location("bertopic_topics", PROBE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load probe module from {PROBE_PATH}")
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class BertopicProbeTest(unittest.TestCase):
    """Protect input construction and retained-output behavior."""

    def test_input_spec_requires_nonempty_id_and_path(self) -> None:
        self.assertEqual(
            PROBE._parse_input_spec("fixture=data/input.txt"),
            ("fixture", Path("data/input.txt")),
        )
        with self.assertRaises(argparse.ArgumentTypeError):
            PROBE._parse_input_spec("data/input.txt")

    def test_pooled_topics_parser_accepts_chunk_configuration(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "bertopic_topics.py",
                "pooled-topics",
                "run-output",
                "--fixture-id",
                "fixture",
                "--chunk-max-tokens",
                "240",
                "--chunk-overlap-sentences",
                "1",
            ],
        ):
            args = PROBE.parse_args()

        self.assertEqual(args.probe, "pooled-topics")
        self.assertEqual(args.chunk_max_tokens, 240)
        self.assertEqual(args.chunk_overlap_sentences, 1)

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

    def test_embedding_chunks_fit_the_requested_token_boundary(self) -> None:
        sentence_model = _FakeSentenceModel(max_seq_length=6)

        chunks = PROBE._embedding_chunks(
            sentence_model,
            "one two three. four five six. seven eight nine.",
            max_tokens=6,
            overlap_sentences=0,
        )

        self.assertEqual(chunks, ["one two three.", "four five six.", "seven eight nine."])
        self.assertTrue(
            all(
                len(PROBE._raw_token_ids(sentence_model[0].tokenizer, chunk)) <= 6
                for chunk in chunks
            )
        )

    def test_pooled_embeddings_return_one_normalized_vector_per_source_document(
        self,
    ) -> None:
        sentence_model = _FakeSentenceModel(max_seq_length=6)
        documents = [
            {"id": "first", "text": "one two three. four five six."},
            {"id": "second", "text": "seven eight nine."},
        ]

        embeddings, diagnostics = PROBE._pooled_document_embeddings(
            sentence_model,
            documents,
            max_tokens=6,
            overlap_sentences=0,
        )

        self.assertEqual(embeddings.shape, (2, 2))
        self.assertTrue(np.allclose(np.linalg.norm(embeddings, axis=1), 1.0))
        self.assertEqual(diagnostics[0]["chunk_count"], 2)
        self.assertEqual(diagnostics[1]["chunk_count"], 1)
        self.assertEqual(
            diagnostics[0]["pooling"],
            "token_count_weighted_mean_then_l2_normalize",
        )

    def test_pooled_embedding_similarity_diagnostics_separate_pair_groups(self) -> None:
        documents = [
            {"expected_theme": "archive"},
            {"expected_theme": "archive"},
            {"expected_theme": "racing"},
            {"expected_theme": "racing"},
        ]
        embeddings = np.asarray(
            [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8]]
        )

        diagnostics = PROBE._pooled_embedding_similarity_diagnostics(
            documents, embeddings
        )

        self.assertEqual(diagnostics["all_within_theme_pairs"]["pair_count"], 2)
        self.assertAlmostEqual(diagnostics["all_within_theme_pairs"]["mean"], 0.8)
        self.assertEqual(diagnostics["all_between_theme_pairs"]["pair_count"], 4)
        self.assertAlmostEqual(
            diagnostics["between_theme"]["archive__racing"]["mean"], 0.54
        )


class _FakeTokenizer:
    """Small whitespace tokenizer for model-free chunking tests."""

    model_max_length = 6
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
        del padding
        token_ids = list(range(10, 10 + len(text.split())))
        if add_special_tokens:
            token_ids = [1, *token_ids, 2]
        if truncation:
            token_ids = token_ids[: self.model_max_length]
        result = {"input_ids": token_ids}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(token_ids)
        return result


class _FakeSentenceModel:
    """Provide the native methods exercised by pooling helpers without loading a model."""

    def __init__(self, *, max_seq_length: int) -> None:
        self.max_seq_length = max_seq_length
        self.device = "cpu"
        self._module = type("Module", (), {"tokenizer": _FakeTokenizer()})()

    def __getitem__(self, index: int):
        if index != 0:
            raise IndexError(index)
        return self._module

    def preprocess(self, texts: list[str]) -> dict[str, list[int]]:
        return self._module.tokenizer(
            texts[0],
            add_special_tokens=True,
            padding=False,
            truncation=True,
            return_attention_mask=True,
        )

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        del convert_to_numpy, show_progress_bar
        return np.asarray([[len(text.split()), 1.0] for text in texts], dtype=float)

    def get_embedding_dimension(self) -> int:
        return 2


if __name__ == "__main__":
    unittest.main()
