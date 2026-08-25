"""BERTopic tools for AMPAV."""

from ampav.keybert import __version__

from .topic_modeling import (
    DEFAULT_BERTOPIC_MODEL_ID,
    DEFAULT_BERTOPIC_MODEL_REVISION,
    BertopicTopicModeler,
)


__all__ = [
    "DEFAULT_BERTOPIC_MODEL_ID",
    "DEFAULT_BERTOPIC_MODEL_REVISION",
    "BertopicTopicModeler",
    "__version__",
]
