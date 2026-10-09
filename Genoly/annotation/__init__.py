"""Anotación funcional de variantes (genes, regiones y términos GO)."""

from Genoly.annotation.annotator import (
    GO_NAMES,
    Annotator,
    Feature,
    load_gff,
    summarize_go,
)

__all__ = ["Annotator", "Feature", "GO_NAMES", "load_gff", "summarize_go"]