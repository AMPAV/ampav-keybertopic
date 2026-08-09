"""Preliminary synchronous KeyBERT key-phrase extraction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


DEFAULT_KEYBERT_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_KEYBERT_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class KeybertKeyPhraseExtractor:
    """Thin synchronous wrapper around native KeyBERT extraction.

    This preliminary wrapper processes one caller-supplied context window. It
    does not chunk text, combine rankings, infer offsets or occurrences, or
    convert native tuples to an AMPAV schema.
    """

    def __init__(
        self,
        *,
        model_id: str = DEFAULT_KEYBERT_MODEL_ID,
        model_revision: str = DEFAULT_KEYBERT_MODEL_REVISION,
        device: str = "cpu",
        local_files_only: bool = False,
        model: Any | None = None,
    ) -> None:
        """Configure a lazily constructed native ``KeyBERT`` model.

        Args:
            model_id: SentenceTransformers model ID used when ``model`` is not
                supplied.
            model_revision: Exact Hugging Face model revision to load.
            device: Device passed to ``SentenceTransformer``.
            local_files_only: Whether model loading must use cached files.
            model: Optional preconfigured native ``KeyBERT`` instance.
        """
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self.local_files_only = local_files_only
        self._model = model

    @property
    def model(self) -> Any:
        """Return the native KeyBERT model, constructing it on first use."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Load the pinned embedding revision and construct native KeyBERT."""
        try:
            from keybert import KeyBERT
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "keybert and sentence-transformers are required to extract key phrases"
            ) from exc

        embedding_model = SentenceTransformer(
            self.model_id,
            revision=self.model_revision,
            device=self.device,
            local_files_only=self.local_files_only,
        )
        return KeyBERT(model=embedding_model)

    def process(
        self,
        text: str,
        *,
        candidates: Sequence[str] | None = None,
        keyphrase_ngram_range: tuple[int, int] = (1, 4),
        stop_words: str | Sequence[str] | None = None,
        top_n: int = 10,
        use_mmr: bool = False,
        diversity: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Return KeyBERT's native ranked key-phrase tuples for one text.

        The input must fit the embedding model's context window. KeyBERT forms
        candidates from the complete text even when SentenceTransformers
        truncates the document embedding, so this Phase 1 wrapper does not
        claim long-document behavior.

        The second tuple item is native document-to-candidate cosine
        similarity rounded by KeyBERT. It is not detection confidence. When
        MMR is enabled, KeyBERT selects a diverse subset and then returns that
        subset sorted by the same cosine similarity.

        Args:
            text: Non-empty source text that fits the selected model's context
                window.
            candidates: Optional candidate phrases passed to native KeyBERT.
            keyphrase_ngram_range: Inclusive candidate length range in tokens.
            stop_words: Native CountVectorizer stop-word setting. The default
                preserves source-token adjacency; use ``"english"`` to apply
                KeyBERT's native English stop-word filtering.
            top_n: Maximum number of ranked native tuples to return.
            use_mmr: Whether native KeyBERT should apply MMR selection.
            diversity: Native MMR diversity value between zero and one.

        Returns:
            KeyBERT's ordered ``(phrase, cosine_similarity)`` tuples.
        """
        _validate_text(text)
        validated_candidates = _validate_candidates(candidates)
        validated_stop_words = _validate_stop_words(stop_words)
        _validate_ngram_range(keyphrase_ngram_range)
        if not isinstance(top_n, int):
            raise TypeError("top_n must be an integer")
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        if not isinstance(use_mmr, bool):
            raise TypeError("use_mmr must be a boolean")
        if not isinstance(diversity, (int, float)):
            raise TypeError("diversity must be numeric")
        if not 0 <= diversity <= 1:
            raise ValueError("diversity must be between 0 and 1")

        return self.model.extract_keywords(
            text,
            candidates=validated_candidates,
            keyphrase_ngram_range=keyphrase_ngram_range,
            stop_words=validated_stop_words,
            top_n=top_n,
            use_mmr=use_mmr,
            diversity=float(diversity),
        )


def _validate_text(text: str) -> None:
    """Reject invalid text before loading or invoking the native model."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not text.strip():
        raise ValueError("text must not be empty")


def _validate_candidates(candidates: Sequence[str] | None) -> list[str] | None:
    """Return an optional native-ready candidate list."""
    if candidates is None:
        return None
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise TypeError("candidates must be a sequence of strings")
    if not candidates:
        raise ValueError("candidates must not be empty")
    if any(not isinstance(candidate, str) for candidate in candidates):
        raise TypeError("each candidate must be a string")
    if any(not candidate.strip() for candidate in candidates):
        raise ValueError("candidates must not contain empty strings")
    return list(candidates)


def _validate_stop_words(
    stop_words: str | Sequence[str] | None,
) -> str | list[str] | None:
    """Return a CountVectorizer-compatible native stop-word setting."""
    if stop_words is None or isinstance(stop_words, str):
        return stop_words
    if isinstance(stop_words, bytes) or not isinstance(stop_words, Sequence):
        raise TypeError("stop_words must be a string, sequence of strings, or None")
    if any(not isinstance(stop_word, str) for stop_word in stop_words):
        raise TypeError("each stop word must be a string")
    return list(stop_words)


def _validate_ngram_range(keyphrase_ngram_range: tuple[int, int]) -> None:
    """Validate the two positive, increasing native n-gram bounds."""
    if (
        not isinstance(keyphrase_ngram_range, tuple)
        or len(keyphrase_ngram_range) != 2
        or any(not isinstance(value, int) for value in keyphrase_ngram_range)
    ):
        raise TypeError("keyphrase_ngram_range must be a two-integer tuple")
    minimum, maximum = keyphrase_ngram_range
    if minimum < 1 or maximum < minimum:
        raise ValueError("keyphrase_ngram_range must satisfy 1 <= minimum <= maximum")
