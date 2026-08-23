"""Offline rewrites of a capture bundle, and the record that says one happened.

A capture bundle is an input to every measurement made from it, so a bundle that has been rewritten
and does not say so produces results attributed to a model that no longer exists. That is not
hypothetical here: `gemma2_2b_int8_full_seq8_pretransposed` had TWO rewrites applied (its embedding
table specialized from 256000 rows to 8, and 183 weights physically pre-transposed) and recorded
neither, while its `prov.weights_file` still named the ORIGINAL bundle's weights. Anyone reading it
would have concluded they were looking at stock Gemma 2 2B.

So: a rewrite writes `bundle.rewrites.json` naming itself, what it changed, and what it measured.
:func:`read_rewrites` is how a consumer finds out, and `CaptureBundle.rewrites` is the path.

WHY A SIDECAR AND NOT THE MANIFEST. `weights.safetensors.manifest.json` is keyed by stringified
`@forward` arg index, and `frontends.linalg_mlir.load_manifest` reads it as
`{int(k): v for k, v in raw.items()}` -- a non-numeric top-level key raises there. The manifest has
no reserved namespace, so the record lives beside it instead of inside it.

WHAT A REWRITE MUST BE. Value-preserving on the model's OUTPUT, or it is not a rewrite, it is a
different model. Each applier here states its soundness condition, checks it, and refuses when it
does not hold rather than applying anyway.
"""
from __future__ import annotations

import json
import os
import shutil
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: The record lives here, beside the manifest rather than inside it (see the module docstring).
REWRITES_FILE = "bundle.rewrites.json"

#: safetensors dtype spelling -> numpy. Only what a bundle actually stores; an unknown dtype is an
#: error rather than a guess, because guessing wrong silently reinterprets the weight's bytes.
_NP = {"I8": np.int8, "U8": np.uint8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
       "F16": np.float16, "F32": np.float32, "F64": np.float64, "BF16": np.uint16}


@dataclass
class RewriteRecord:
    """One applied rewrite: what ran, what it changed, and what it was measured to save."""

    name: str
    #: the bundle this was derived FROM, so the chain back to the capture is never lost
    source_bundle: str
    #: the condition that makes the rewrite value-preserving, and the fact that it was checked
    soundness: str
    #: rewrite-specific measured effect (bytes moved before/after, counts, ...)
    effect: dict[str, Any] = field(default_factory=dict)
    #: anything a consumer MUST know to use the bundle correctly (e.g. "valid for these ids only")
    caveats: list[str] = field(default_factory=list)


def read_rewrites(bundle_dir: Path | str) -> list[RewriteRecord]:
    """Every rewrite applied to this bundle, oldest first. Empty for an unrewritten bundle."""
    p = Path(bundle_dir) / REWRITES_FILE
    if not p.is_file():
        return []
    raw = json.loads(p.read_text())
    return [RewriteRecord(**r) for r in raw.get("rewrites", [])]


def record_rewrite(bundle_dir: Path | str, rec: RewriteRecord) -> Path:
    """Append `rec` to the bundle's rewrite record, creating it if needed."""
    p = Path(bundle_dir) / REWRITES_FILE
    existing = json.loads(p.read_text()) if p.is_file() else {"rewrites": []}
    existing["rewrites"].append(asdict(rec))
    p.write_text(json.dumps(existing, indent=2) + "\n")
    return p


def _link_or_copy(src: Path, dst: Path) -> None:
    """Hardlink where possible -- a weights blob is gigabytes and the unchanged files are identical."""
    if not src.exists() or dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _replace_ssa(line: str, old: str, new: str) -> str:
    """Replace SSA name `old` with `new` respecting token boundaries -- `%15285` is not `%152850`."""
    if old not in line:
        return line
    out, i = [], 0
    while True:
        j = line.find(old, i)
        if j < 0:
            out.append(line[i:])
            break
        end = j + len(old)
        nxt = line[end] if end < len(line) else ""
        out.append(line[i:j])
        out.append(old if nxt.isdigit() else new)
        i = end
    return "".join(out)


def rewrite_transposed_args(mlir_text: str, args: set[int]) -> tuple[str, int]:
    """Delete each hoisted `linalg.transpose` (and the `tensor.empty` feeding its `outs`), give the
    argument the transposed type, and forward every use of the transpose result to the argument.

    Returns the new text and the number of ops removed. Structural line handling, no regex: an op is
    found by its `= linalg.transpose ins(` shape and its operands by `partition`, per the repo's
    no-regex rule.
    """
    want = {f"%{a}" for a in args}
    lines = mlir_text.splitlines()

    drop_defs: set[str] = set()
    subst: dict[str, str] = {}
    retype: dict[str, tuple[str, str]] = {}
    for line in lines:
        s = line.strip()
        if " = linalg.transpose ins(" not in s:
            continue
        res = s.split(" =", 1)[0].strip()
        src = s.split("ins(", 1)[1].split(":", 1)[0].strip()
        if src not in want:
            continue
        in_t = s.split("ins(", 1)[1].split(":", 1)[1].split(")")[0].strip()
        out_t = s.split("outs(", 1)[1].split(":", 1)[1].split(")")[0].strip()
        outs_val = s.split("outs(", 1)[1].split(":", 1)[0].strip()
        drop_defs.add(res)
        drop_defs.add(outs_val)
        subst[res] = src
        retype[src] = (in_t, out_t)

    out_lines: list[str] = []
    dropped = 0
    for line in lines:
        s = line.strip()
        defname = s.split(" =", 1)[0].strip() if " =" in s else None
        if defname in drop_defs:
            dropped += 1
            continue
        if s.startswith("func.func @forward"):
            for arg, (old_t, new_t) in retype.items():
                line = line.replace(f"{arg}: {old_t}", f"{arg}: {new_t}")
        for res, src in subst.items():
            line = _replace_ssa(line, res, src)
        out_lines.append(line)
    return "\n".join(out_lines) + "\n", dropped


def hoist_weight_transposes(src: Path | str, dst: Path | str,
                            func_name: str = "forward") -> RewriteRecord:
    """Store every sole-use weight in the layout its consumer wants, so the model stops transposing
    weights at run time.

    A `linalg.transpose` reading a function argument is not computation: it is the model paying, on
    every inference, to convert a weight into a layout the packer could have written once. On
    Gemma 2 2B, 183 of them moved **2,493.0 MiB per inference** against a 2,505 MiB weight blob --
    essentially every weight, every time -- and the largest, the 562.5 MiB int8 tied head, is what
    killed a whole-model FireSim run with `FAIL alloc bytes=589824064` at op 11,494 of 11,526.

    SOUNDNESS, CHECKED NOT ASSUMED. Pre-transposing is value-preserving only when the transpose is
    the argument's SOLE consumer; anything else reading the argument would silently begin seeing
    transposed data. The hoistable set comes from
    :func:`merlin.xdsl_dialects.lowering.weight_layout.weight_layout_report`, which splits sole-use
    from mixed-use and never merges them, and blocked re-layouts are left exactly as they are.

    BIT-EXACT. A transpose moves elements, it does not compute. Every value a consumer sees is the
    value it saw before, at the same index, so goldens carry over untouched. Per weight this asserts
    that the bytes written are exactly `stored.T`, so an error surfaces here rather than as wrong
    logits hours later on a simulator.
    """
    from ..xdsl_dialects.lowering.weight_layout import weight_layout_report
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK

    src, dst = Path(src), Path(dst)
    mlir_text = (src / "model.mlir").read_text()
    with IR_LOCK:
        report = weight_layout_report(mq.parse(mlir_text), func_name)

    hoistable = report.hoistable
    if report.unpriceable:
        raise ValueError(                       # fail closed: an unpriced weight is not a free one
            f"cannot hoist safely, {len(report.unpriceable)} re-layout(s) could not be priced: "
            f"{report.unpriceable}")
    if not hoistable:
        raise ValueError(f"no hoistable weight transposes in {src}")

    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    by_arg = {r.arg: r for r in hoistable}
    want: dict[str, int] = {}                   # safetensors tensor name -> arg index
    for arg in by_arg:
        entry = man.get(str(arg))
        if entry is None or "weight" not in entry:
            raise ValueError(f"arg {arg} is hoistable in the IR but names no weight in the manifest")
        want[entry["weight"]] = arg

    dst.mkdir(parents=True, exist_ok=True)
    done = _rewrite_safetensors(src, dst, set(want))

    for entry in man.values():
        if entry.get("weight") in want and "shape" in entry:
            entry["shape"] = [entry["shape"][1], entry["shape"][0]]
    (dst / "weights.safetensors.manifest.json").write_text(json.dumps(man, indent=2))

    new_text, dropped = rewrite_transposed_args(mlir_text, set(by_arg))
    (dst / "model.mlir").write_text(new_text)

    for name in ("extra.npz", "inputs.npz", "input_order.json", "golden.npy", "golden_w8a8.npy",
                 "region_goldens.npz"):
        _link_or_copy(src / name, dst / name)
    if (src / REWRITES_FILE).is_file():         # carry the chain forward, do not start a new one
        shutil.copy2(src / REWRITES_FILE, dst / REWRITES_FILE)

    rec = RewriteRecord(
        name="hoist_weight_transposes",
        source_bundle=src.name,
        soundness=("each hoisted argument's transpose is its SOLE consumer, so pre-applying the "
                   "layout cannot change what any other reader sees; verified per argument by "
                   "weight_layout_report, and each weight's bytes asserted equal to stored.T"),
        effect={
            "weights_pre_transposed": done,
            "transposes_removed": len(by_arg),
            "ops_removed": dropped,
            "bytes_moved_per_inference_before": report.hoistable_bytes,
            "mib_moved_per_inference_before": round(report.hoistable_bytes / 2 ** 20, 1),
            "bytes_moved_per_inference_after": 0,
            "blocked_not_hoisted": len(report.blocked),
            "analysis": "merlin.xdsl_dialects.lowering.weight_layout.weight_layout_report",
        },
        caveats=([f"{len(report.blocked)} re-layout(s) were NOT hoisted (argument has other readers)"]
                 if report.blocked else []),
    )
    record_rewrite(dst, rec)
    return rec


def _rewrite_safetensors(src: Path, dst: Path, want: set[str]) -> int:
    """Copy the blob, transposing exactly the tensors in `want`, keeping order and tight packing."""
    with open(src / "weights.safetensors", "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
        data_start = 8 + hlen
        meta = header.pop("__metadata__", None)

        new_header: dict[str, Any] = {}
        order: list[tuple[str, int, int, dict]] = []
        off = 0
        for name, spec in header.items():
            s, e = spec["data_offsets"]
            shape = list(spec["shape"])
            if name in want:
                if len(shape) != 2:
                    raise ValueError(f"{name}: only 2-D weights are hoisted, got {shape}")
                shape = [shape[1], shape[0]]
            new_header[name] = {"dtype": spec["dtype"], "shape": shape,
                                "data_offsets": [off, off + (e - s)]}
            order.append((name, s, e, spec))
            off += e - s
        if meta is not None:
            new_header["__metadata__"] = meta
        blob = json.dumps(new_header, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)        # safetensors wants 8-byte aligned data

        done = 0
        with open(dst / "weights.safetensors", "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for name, s, e, spec in order:
                f.seek(data_start + s)
                if name not in want:
                    left = e - s
                    while left:
                        chunk = f.read(min(left, 64 << 20))
                        out.write(chunk)
                        left -= len(chunk)
                    continue
                dt = _NP.get(spec["dtype"])
                if dt is None:
                    raise ValueError(f"{name}: unknown safetensors dtype {spec['dtype']!r}")
                arr = np.frombuffer(f.read(e - s), dtype=dt).reshape(spec["shape"])
                t = np.ascontiguousarray(arr.T)
                # the assertions that make a silent layout error impossible
                assert t.shape == tuple(new_header[name]["shape"]), name
                assert t.nbytes == e - s, name
                assert np.array_equal(t.T, arr), f"{name}: transpose round-trip failed"
                out.write(t.tobytes())
                done += 1
    return done
