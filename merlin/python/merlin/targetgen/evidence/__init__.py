"""Evidence layer: deterministically discover source files and detect concepts.

This layer records *what was found* (files + keyword-detected concepts with citations). It
makes no claim to understand RTL; everything it produces is meant for human review and to
seed the (also conservative) synthesizers.
"""
from __future__ import annotations

from .store import Evidence, FileRecord
from .report import build_evidence, render_markdown

__all__ = ["Evidence", "FileRecord", "build_evidence", "render_markdown"]
