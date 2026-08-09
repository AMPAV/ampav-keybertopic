"""KeyBERT tools for AMPAV."""

from .key_phrases import (
    DEFAULT_KEYBERT_MODEL_ID,
    DEFAULT_KEYBERT_MODEL_REVISION,
    KeybertKeyPhraseExtractor,
)
from ._version import __version__


__all__ = [
    "DEFAULT_KEYBERT_MODEL_ID",
    "DEFAULT_KEYBERT_MODEL_REVISION",
    "KeybertKeyPhraseExtractor",
    "__version__",
]
