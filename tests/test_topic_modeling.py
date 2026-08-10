"""Tests for the preliminary native-output BERTopic wrapper."""

import unittest

from ampav.bertopic import BertopicTopicModeler


class FakeBertopic:
    """Record native fitting arguments and retain inspectable fitted state."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.result = ([0, -1], [0.9, 0.0])

    def fit_transform(self, documents: list[str]):
        self.calls.append(documents)
        return self.result


class BertopicTopicModelerTest(unittest.TestCase):
    """Verify thin forwarding, native preservation, and input validation."""

    def test_process_returns_exact_native_result_and_preserves_documents(self) -> None:
        native_model = FakeBertopic()
        modeler = BertopicTopicModeler(model=native_model)
        documents = ("Archive preservation transfer.", "Race timing update.")

        result = modeler.process(documents)

        self.assertIs(result, native_model.result)
        self.assertEqual(native_model.calls, [list(documents)])
        self.assertIs(modeler.model, native_model)

    def test_invalid_documents_are_rejected_before_native_model_access(self) -> None:
        native_model = FakeBertopic()
        modeler = BertopicTopicModeler(model=native_model)

        with self.assertRaisesRegex(TypeError, "sequence of strings"):
            modeler.process("one document")
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            modeler.process([])
        with self.assertRaisesRegex(TypeError, "each document must be a string"):
            modeler.process(["valid", 1])  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "must not contain empty"):
            modeler.process(["valid", "  "])

        self.assertEqual(native_model.calls, [])

    def test_model_options_are_copied(self) -> None:
        options = {"min_topic_size": 5, "calculate_probabilities": True}

        modeler = BertopicTopicModeler(model=FakeBertopic(), model_options=options)
        options["min_topic_size"] = 20

        self.assertEqual(
            modeler.model_options,
            {"min_topic_size": 5, "calculate_probabilities": True},
        )

    def test_embedding_model_option_requires_native_model_injection(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain embedding_model"):
            BertopicTopicModeler(model_options={"embedding_model": "another-model"})

    def test_configuration_values_are_validated_without_loading_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_id must not be empty"):
            BertopicTopicModeler(model_id=" ")
        with self.assertRaisesRegex(TypeError, "local_files_only must be a boolean"):
            BertopicTopicModeler(local_files_only=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "model_options must be a mapping"):
            BertopicTopicModeler(model_options=[])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
