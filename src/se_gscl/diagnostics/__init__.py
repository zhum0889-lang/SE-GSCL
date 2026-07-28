"""Structured outputs passed from the specialist to later LLM stages."""

from .packet import SemanticDiagnosticPacket, build_semantic_diagnostic_packet
from .fusion import (
    ReliabilityGate,
    branch_reliability,
    fit_reliability_gate,
    fuse_probabilities,
)

__all__ = [
    "ReliabilityGate",
    "SemanticDiagnosticPacket",
    "branch_reliability",
    "build_semantic_diagnostic_packet",
    "fit_reliability_gate",
    "fuse_probabilities",
]
