"""Native reduced-host witness layout from the existing whole-program backend ABI."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from . import gemmini, gemmini_codegen


def native_layout(spec: Mapping[str, Any]) -> dict[str, int]:
    rows, cols = gemmini._buffer_extent(dict(spec), name="host semantic witness")
    prows, pcols = gemmini_codegen._ceil_dim(rows), gemmini_codegen._ceil_dim(cols)
    return {"rows": rows, "cols": cols, "row_stride": pcols,
            "storage_elements": prows * pcols}


def derive_native_witness_abi(*, target: str) -> dict[str, Any]:
    from merlin.targetgen.contract.harness_abi import for_target
    abi = for_target(target)
    sources = [Path(__file__), Path(gemmini.__file__), Path(gemmini_codegen.__file__)]
    return {"native_layout": native_layout, "expected_symbol": abi.entry_symbol,
            "abi_provenance": {"scope": "existing backend whole_program row-major padded ABI",
                "target": target, "entry_symbol": abi.entry_symbol,
                "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in sources}}}
