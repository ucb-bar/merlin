"""Use the reviewed bounded dense parser with the actual frontend context."""

from __future__ import annotations

from merlin.common.ir_lock import IR_LOCK
from merlin.frontends.linalg_mlir import make_context, strip_paren_results

from .gate_errors import GateUnavailable
from .lean_graph_generic_admission import InputAdmissionError
from .memory_scalar_nan_admission import _parse_bounded

MAX_RAW_BYTES = 4 * 1024 * 1024
MAX_DENSE_BYTES = 4 * 1024 * 1024


def parse_generic(raw, *, maximum_bytes=MAX_DENSE_BYTES):
    if type(raw) is not str or len(raw.encode()) > MAX_RAW_BYTES:
        raise GateUnavailable("generic complete raw buffer4MiB resource budget exceeded")
    if type(maximum_bytes) is not int or maximum_bytes < 0:
        raise GateUnavailable("generic dense storage budget must be a nonnegative integer")
    # This profile admits the canonical generic grammar directly. The frontend
    # normalizer is not quote-aware; applying it could alias actual StringAttr
    # metadata. Refuse its non-fixed points instead of normalizing their bytes.
    if strip_paren_results(raw) != raw:
        raise InputAdmissionError("normalizer-sensitive-raw", "complete-buffer")
    with IR_LOCK:
        # The existing parser's memory flag selects its narrow floating-complex
        # dense-literal refusal. Runtime opaque values and ordinary exact dense
        # float bytes are unaffected; this is literal admission only.
        return _parse_bounded(raw, maximum_bytes=maximum_bytes, memory=True, context=make_context())
