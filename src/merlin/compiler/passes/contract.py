"""Small shared scheduling and failure contracts consumed by OOT providers."""

from __future__ import annotations

from .policy import splice_passes as splice_passes


class PipelineError(RuntimeError):
    """A selected lowering pipeline cannot compile its input."""


_CLASS_TOKEN = {
    "linalg.matmul": "mm",
    "linalg.batch_matmul": "bmm",
    "linalg.conv2d_direct": "conv",
    "linalg.conv2d_grouped_direct": "gconv",
}
PAR_TAG_PREFIX = "merlin.par_"


def class_token(op: str) -> str:
    """Stable operation-class token shared by taggers and schedule matchers."""
    return _CLASS_TOKEN.get(op, op.rsplit(".", 1)[-1])


def par_tag_for(op: str, tiles: tuple[int, ...]) -> str:
    """Stable attribute joining a parallel chunk to its selected schedule arm."""
    return f"{PAR_TAG_PREFIX}{class_token(op)}_" + "_".join(str(int(t)) for t in tiles)
