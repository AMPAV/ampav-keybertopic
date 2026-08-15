"""Preliminary synchronous BERTopic corpus modeling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


DEFAULT_BERTOPIC_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BERTOPIC_MODEL_REVISION = (
    "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
)


class BertopicTopicModeler:
    """Thin synchronous wrapper around native corpus-level BERTopic fitting.

    The wrapper owns one stateful native model. Each ``process`` call fits that
    model to the complete caller-supplied document collection and returns
    BERTopic's native assignment/probability tuple. Fitted topic details remain
    available through ``model`` and retain their native semantics.

    This preliminary API does not split transcripts, validate model token
    lengths, persist fitted models, or convert topics to an AMPAV schema.
    Callers must supply meaningful peer documents that each fit the embedding
    model's context window.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_BERTOPIC_MODEL_ID,
        model_revision: str = DEFAULT_BERTOPIC_MODEL_REVISION,
        device: str = "cpu",
        local_files_only: bool = False,
        model_options: Mapping[str, Any] | None = None,
        model: Any | None = None,
    ) -> None:
        """Configure a lazily constructed native ``BERTopic`` model.

        Args:
            model_id: SentenceTransformers model ID used when ``model`` is not
                supplied.
            model_revision: Exact Hugging Face embedding-model revision.
            device: Device passed to ``SentenceTransformer``.
            local_files_only: Whether embedding-model loading must use cached
                files.
            model_options: Native ``BERTopic`` constructor options. The
                embedding model is owned by this wrapper; inject a complete
                native ``model`` to use another embedding backend.
            model: Optional preconfigured native ``BERTopic`` instance.
        """
        self.model_id = _validate_nonempty_string("model_id", model_id)
        self.model_revision = _validate_nonempty_string(
            "model_revision", model_revision
        )
        self.device = _validate_nonempty_string("device", device)
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean")
        self.local_files_only = local_files_only
        self.model_options = _validate_model_options(model_options)
        self._model = model

    @property
    def model(self) -> Any:
        """Return the native BERTopic model, constructing it on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Load the pinned embedding revision and construct native BERTopic."""
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "bertopic and sentence-transformers are required to model topics"
            ) from exc

        embedding_model = SentenceTransformer(
            self.model_id,
            revision=self.model_revision,
            device=self.device,
            local_files_only=self.local_files_only,
        )
        return BERTopic(embedding_model=embedding_model, **self.model_options)

    def process(self, documents: Sequence[str]) -> tuple[list[int], Any | None]:
        """Fit the native model and return its assignment/probability tuple.

        Native integer topic IDs describe this fitted collection only. Topic
        ``-1`` is BERTopic's outlier assignment. With native HDBSCAN and
        ``calculate_probabilities=False``, the second tuple item contains each
        document's assigned-topic membership strength. When that option is
        true, it contains native soft-clustering membership vectors. Neither
        form is calibrated subject confidence.

        Native topic representations and their c-TF-IDF term weights are
        available through methods such as ``model.get_topic_info()`` and
        ``model.get_topic(...)`` after fitting. c-TF-IDF weights are not
        confidence values.

        Args:
            documents: Non-empty sequence of non-empty peer-document strings.
                Each document must fit the selected embedding context window.

        Returns:
            The exact tuple returned by native ``BERTopic.fit_transform``.
        """
        validated_documents = _validate_documents(documents)
        return self.model.fit_transform(validated_documents)


def _validate_nonempty_string(name: str, value: str) -> str:
    """Return a non-empty string configuration value."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value


def _validate_model_options(
    model_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Copy native model options while protecting the pinned backend owner."""
    if model_options is None:
        return {}
    if not isinstance(model_options, Mapping):
        raise TypeError("model_options must be a mapping or None")
    options = dict(model_options)
    if "embedding_model" in options:
        raise ValueError(
            "model_options must not contain embedding_model; inject a native model instead"
        )
    return options


def _validate_documents(documents: Sequence[str]) -> list[str]:
    """Return a native-ready document list without altering source strings."""
    if isinstance(documents, (str, bytes)) or not isinstance(documents, Sequence):
        raise TypeError("documents must be a sequence of strings")
    if not documents:
        raise ValueError("documents must not be empty")
    if any(not isinstance(document, str) for document in documents):
        raise TypeError("each document must be a string")
    if any(not document.strip() for document in documents):
        raise ValueError("documents must not contain empty strings")
    return list(documents)
