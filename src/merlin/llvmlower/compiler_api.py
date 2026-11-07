"""Small target-independent ABI shared by Merlin and codegen providers."""

from __future__ import annotations


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
    """Stable class token shared by the tagger and a provider's schedule matcher."""
    return _CLASS_TOKEN.get(op, op.rsplit(".", 1)[-1])


def par_tag_for(op: str, tiles: tuple[int, ...]) -> str:
    """Stable attribute used to join a parallel chunk with its schedule arm."""
    return f"{PAR_TAG_PREFIX}{class_token(op)}_" + "_".join(str(int(t)) for t in tiles)


def splice_passes(passes: list[str]) -> str:
    """Apply Merlin's shared deallocation and fusion ordering policy to a provider pass list."""
    from .pipeline import _splice

    return _splice(passes)
