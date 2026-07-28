"""Structured outputs passed from the specialist to later LLM stages."""

from .packet import SemanticDiagnosticPacket, build_semantic_diagnostic_packet

__all__ = ["SemanticDiagnosticPacket", "build_semantic_diagnostic_packet"]
