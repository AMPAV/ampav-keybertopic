"""KeyBERT tools for AMPAV."""

__version__ = "0.0.1"

from .key_phrases import (
    DEFAULT_KEYBERT_MODEL_ID,
    DEFAULT_KEYBERT_MODEL_REVISION,
    KeybertKeyPhraseExtractor,
)


__all__ = [
    "DEFAULT_KEYBERT_MODEL_ID",
    "DEFAULT_KEYBERT_MODEL_REVISION",
    "KeybertKeyPhraseExtractor",
    "__version__",
]
