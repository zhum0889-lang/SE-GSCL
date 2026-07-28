"""Frozen LLM semantics and versioned prototype interfaces."""

from .prototype_bank import FrozenPrototypeBank
from .text_cache import TextEmbeddingCache, read_ontology
from .text_encoder import FrozenDecoderTextEncoder, masked_mean_pool
from .text_projection import ProjectedTextPrototypeBank

__all__ = [
    "FrozenDecoderTextEncoder",
    "FrozenPrototypeBank",
    "ProjectedTextPrototypeBank",
    "TextEmbeddingCache",
    "masked_mean_pool",
    "read_ontology",
]
