"""Concrete Qinner host-byte intake for Lean's little-endian row-major decoder.

Production appends native contiguous bytes without an endian conversion. This intake rejects an
unsupported host or array representation; it never repairs bytes before certifying them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

SCHEMA = "merlin.qinner.runtime.le.v2"
DTYPES = {
    "bool": ("int", 1),
    "int8": ("int", 8),
    "int16": ("int", 16),
    "int32": ("int", 32),
    "int64": ("int", 64),
    "float16": ("float", "f16"),
    "float32": ("float", "f32"),
    "float64": ("float", "f64"),
}


def require_host():
    if sys.byteorder != "little":
        raise ValueError("Qinner decoder qualification requires a little-endian host")


def payload(key, value):
    """Copy the exact bytes used by C generation, retaining shape and dtype without conversion."""
    require_host()
    array = np.ascontiguousarray(value)
    if not array.dtype.isnative or array.dtype.byteorder == ">":
        raise ValueError("Qinner decoder rejects non-native or big-endian arrays")
    spelling = str(array.dtype)
    if spelling not in DTYPES:
        raise ValueError(f"unsupported Qinner host dtype {array.dtype}")
    family, width = DTYPES[spelling]
    element = {"kind": family, "width" if family == "int" else "format": width}
    data = array.tobytes(order="C")
    if spelling == "bool" and any(byte not in (0, 1) for byte in data):
        raise ValueError("invalid bool encoding: host i1 storage requires byte 00 or 01")
    return {
        "key": key,
        "type": {"kind": "tensor", "shape": list(array.shape), "element": element, "encoding": None},
        "hex": data.hex(),
    }


def document(archive, delivered):
    require_host()
    return {"schema": SCHEMA, "byte_order": "little", "layout": "row_major", "archive": archive, "delivered": delivered}


def archive_records_match(records, archive_path):
    from merlin.llvmlower.qinner import QinnerArg, resolve

    args = []
    for row in records:
        ty = row["expected_type"]
        elt = ty["element"]
        dtype = f"i{elt['width']}" if elt["kind"] == "int" else elt["format"]
        args.append(QinnerArg(row["key"], tuple(ty["shape"]), dtype))
    with np.load(archive_path, allow_pickle=False) as archive:
        current = [payload(arg.key, array) for arg, array in zip(args, resolve(archive, args), strict=True)]
    recorded = [{key: value for key, value in row.items() if key != "expected_type"} for row in records]
    return current == recorded


def decode_records(records, work, *, archive_path=None):
    """Execute the proved decoder and expose every row-major coordinate's raw scalar bits."""
    from merlin.verify.lean.check import _file_ref
    from merlin.verify.lean.qinner import dependencies, prepare, run, write

    if archive_path is not None and not archive_records_match(records, archive_path):
        raise ValueError("decoder records disagree with the loaded archive")
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    request = {"schema": SCHEMA, "byte_order": "little", "layout": "row_major", "payloads": records}
    write(work / "decode-request.json", request)
    report = json.loads(run([str(prepare()), "decode", str(work / "decode-request.json")]).strip().splitlines()[-1])
    write(work / "decode-report.json", report)
    receipt = {
        "schema": "merlin.qinner.decode.v1",
        "approved": False,
        "verdict": "abstracted" if all(row["checked"] for row in report["payloads"]) else "rejected",
        "checked": all(row["checked"] for row in report["payloads"]),
        "request": _file_ref(work / "decode-request.json"),
        "report": _file_ref(work / "decode-report.json"),
        "dependencies": dependencies(),
        "launcher": _file_ref(prepare()),
        "archive": _file_ref(archive_path.resolve()) if archive_path is not None else None,
        "archive_authenticity_proved": False,
        "host_implements_decoder_proved": False,
        "actual_runtime_delivery_proved": False,
    }
    write(work / "receipt.json", receipt)
    return report


def replay_decode(receipt_path, destination):
    """Check freshness, regenerate archive bytes when bound, and rerun the concrete Lean decoder."""
    from merlin.verify.lean.qinner import prepare, run, write

    receipt = json.loads(receipt_path.read_text())
    refs = [*receipt["dependencies"], receipt["request"], receipt["report"], receipt["launcher"]]
    if receipt["archive"] is not None:
        refs.append(receipt["archive"])
    stale = [
        ref["path"]
        for ref in refs
        if not Path(ref["path"]).is_file()
        or hashlib.sha256(Path(ref["path"]).read_bytes()).hexdigest() != ref["sha256"]
    ]
    if stale:
        return {"fresh": False, "checked": False, "stale": stale}
    request = json.loads(Path(receipt["request"]["path"]).read_text())
    archive_match = True
    if receipt["archive"] is not None:
        archive_match = archive_records_match(request["payloads"], Path(receipt["archive"]["path"]))
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    write(destination / "decode-request.json", request)
    observed = json.loads(
        run([str(prepare()), "decode", str(destination / "decode-request.json")]).strip().splitlines()[-1]
    )
    matches = observed == json.loads(Path(receipt["report"]["path"]).read_text())
    result = {
        "fresh": True,
        "checked": all(row["checked"] for row in observed["payloads"]) and matches and archive_match,
        "matches_recorded": matches,
        "archive_bytes_match": archive_match,
    }
    write(destination / "replay.json", result)
    return result
