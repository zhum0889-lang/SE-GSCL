"""Frozen LLM semantics and versioned prototype interfaces."""

from .fuzzy_identity import hierarchical_fuzzy_identity
from .prototype_bank import FrozenPrototypeBank, LearnedPrototypeBank
from .symptom_bank import (
    FrozenSymptomPrototypeBank,
    ProjectedSymptomPrototypeBank,
    ResidualSymptomPrototypeBank,
)
from .symptom_cache import SymptomEmbeddingCache
from .text_cache import TextEmbeddingCache, read_ontology
from .text_encoder import FrozenDecoderTextEncoder, masked_mean_pool
from .text_projection import ProjectedTextPrototypeBank

__all__ = [
    "FrozenDecoderTextEncoder",
    "FrozenPrototypeBank",
    "LearnedPrototypeBank",
    "FrozenSymptomPrototypeBank",
    "ProjectedTextPrototypeBank",
    "ProjectedSymptomPrototypeBank",
    "ResidualSymptomPrototypeBank",
    "SymptomEmbeddingCache",
    "TextEmbeddingCache",
    "masked_mean_pool",
    "read_ontology",
    "hierarchical_fuzzy_identity",
]
