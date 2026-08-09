"""Tests for the preliminary native-output KeyBERT wrapper."""

import unittest

from ampav.keybert import KeybertKeyPhraseExtractor


class FakeKeyBert:
    """Record native extraction arguments and return a stable native result."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.result = [("audio preservation", 0.7312)]

    def extract_keywords(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        return self.result


class KeybertKeyPhraseExtractorTest(unittest.TestCase):
    """Verify thin forwarding, score preservation, and input validation."""

    def test_process_returns_exact_native_result_with_source_safe_defaults(self) -> None:
        native_model = FakeKeyBert()
        extractor = KeybertKeyPhraseExtractor(model=native_model)

        result = extractor.process("Audio preservation project")

        self.assertIs(result, native_model.result)
        self.assertEqual(
            native_model.calls,
            [
                (
                    "Audio preservation project",
                    {
                        "candidates": None,
                        "keyphrase_ngram_range": (1, 4),
                        "stop_words": None,
                        "top_n": 10,
                        "use_mmr": False,
                        "diversity": 0.5,
                    },
                )
            ],
        )

    def test_process_forwards_native_candidates_and_mmr_options(self) -> None:
        native_model = FakeKeyBert()
        extractor = KeybertKeyPhraseExtractor(model=native_model)

        extractor.process(
            "Audio preservation project",
            candidates=("audio preservation", "project"),
            keyphrase_ngram_range=(1, 2),
            stop_words=("the", "and"),
            top_n=2,
            use_mmr=True,
            diversity=0.7,
        )

        _, arguments = native_model.calls[0]
        self.assertEqual(arguments["candidates"], ["audio preservation", "project"])
        self.assertEqual(arguments["keyphrase_ngram_range"], (1, 2))
        self.assertEqual(arguments["stop_words"], ["the", "and"])
        self.assertEqual(arguments["top_n"], 2)
        self.assertTrue(arguments["use_mmr"])
        self.assertEqual(arguments["diversity"], 0.7)

    def test_invalid_text_is_rejected_before_native_model_access(self) -> None:
        extractor = KeybertKeyPhraseExtractor(model=FakeKeyBert())

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            extractor.process("  ")

        self.assertEqual(extractor.model.calls, [])

    def test_string_candidates_are_rejected(self) -> None:
        extractor = KeybertKeyPhraseExtractor(model=FakeKeyBert())

        with self.assertRaisesRegex(TypeError, "sequence of strings"):
            extractor.process("text", candidates="one phrase")

        with self.assertRaisesRegex(ValueError, "must not be empty"):
            extractor.process("text", candidates=[])

    def test_invalid_ngram_range_is_rejected(self) -> None:
        extractor = KeybertKeyPhraseExtractor(model=FakeKeyBert())

        with self.assertRaisesRegex(ValueError, "1 <= minimum <= maximum"):
            extractor.process("text", keyphrase_ngram_range=(3, 2))

    def test_invalid_ranking_options_are_rejected(self) -> None:
        extractor = KeybertKeyPhraseExtractor(model=FakeKeyBert())

        with self.assertRaisesRegex(ValueError, "top_n must be at least 1"):
            extractor.process("text", top_n=0)
        with self.assertRaisesRegex(ValueError, "diversity must be between"):
            extractor.process("text", diversity=1.1)


if __name__ == "__main__":
    unittest.main()
