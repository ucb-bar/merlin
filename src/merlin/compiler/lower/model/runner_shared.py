"""Shared native runner fragments for out-of-tree compiler providers."""

from merlin.compiler.lower.model.dealloc_lifetime import DEALLOC_CHECK_PRELUDE
from merlin.compiler.lower.model.uniform_fill_copy import MID_STAGE_SRC
from merlin.compiler.lower.model.uniform_fill_copy import RUNNER_PRELUDE as _UNIFORM_FILL_COPY_PRELUDE
from merlin.compiler.passes.fragments import DEALLOC_CHECK_RUNNER as DEALLOC_CHECK_RUNNER

# The common native invocation reserves slot 17 for target data layout.
_UNIFORM_FILL_COPY_MID_SRC = MID_STAGE_SRC.replace(
    'if len(sys.argv) > 17 and sys.argv[17] == "1":',
    'if len(sys.argv) > 18 and sys.argv[18] == "1":',
)

__all__ = ["DEALLOC_CHECK_RUNNER", "DEALLOC_CHECK_PRELUDE", "_UNIFORM_FILL_COPY_MID_SRC", "_UNIFORM_FILL_COPY_PRELUDE"]
