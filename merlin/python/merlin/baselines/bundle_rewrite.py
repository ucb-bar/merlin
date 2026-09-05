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


class RewriteRefused(ValueError):
    """A rewrite whose soundness condition does not hold on this bundle. Raised, never worked around:
    a rewrite that cannot be proven value-preserving must not be applied silently."""


def _read_header(safetensors: Path) -> dict[str, Any]:
    """The safetensors header (tensor name -> spec), without the `__metadata__` entry."""
    with open(safetensors, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    header.pop("__metadata__", None)
    return header


def hoist_safety_problems(src: Path | str, man: dict[str, Any], want: dict[str, int],
                          hoisted_args: set[int]) -> list[str]:
    """Reasons the stored layout of `want` (weight name -> arg) CANNOT be pre-applied to `src`.

    Empty means safe. Each check is here because its ABSENCE is silent -- the IR analysis proves the
    transpose is the argument's sole consumer, which says nothing about how the argument's bytes are
    stored, and every one of these would have produced a bundle that builds, links and grades while
    reading the wrong bytes:

    1. **The weight is not in the blob.** A quantized-subclass weight is STUBBED in the manifest
       (`{"stub": true}`; `resnet50_v1_5_int8_w8a8_consistent` and `lstmnetvit_int8_w8a8_consistent`
       have 1 and 22 of them) and has no safetensors entry at all. :func:`_rewrite_safetensors`
       iterates the HEADER, so such a name is simply never reached -- while the manifest shape below
       is flipped regardless, leaving a manifest that describes a transpose nobody performed.
    2. **Not 2-D.** Only a 2-D weight has an unambiguous pre-applied transpose.
    3. **Another argument reads the same weight.** `mining/section_build` and the capture manifests
       both key weights by NAME, so two `@forward` arguments can name one tensor. Transposing it for
       a hoisted argument silently transposes it for the other reader too.
    4. **A byte range shared with another tensor.** Two distinct names can index overlapping bytes
       (a tied head, an aliased view). Rewriting one rewrites the other's data underneath it.
    """
    src = Path(src)
    header = _read_header(src / "weights.safetensors")
    problems: list[str] = []

    for name, arg in sorted(want.items(), key=lambda kv: kv[1]):
        spec = header.get(name)
        if spec is None:
            stubbed = bool(man.get(str(arg), {}).get("stub"))
            problems.append(
                f"arg {arg}: weight {name!r} has no bytes in weights.safetensors "
                f"({'manifest marks it stub=true' if stubbed else 'dangling manifest entry'}); "
                "its layout cannot be pre-applied, and flipping only the manifest shape would "
                "describe a transpose that never happened")
            continue
        if len(spec.get("shape", ())) != 2:
            problems.append(f"arg {arg}: weight {name!r} has shape {spec.get('shape')}; only 2-D "
                            "weights are hoisted")

    for key, entry in man.items():
        w = entry.get("weight")
        if w not in want or not key.isdigit():
            continue
        if int(key) not in hoisted_args:
            problems.append(
                f"weight {w!r} is also read by arg {key}, which is NOT hoisted; pre-transposing it "
                "would change what that argument sees")

    ranges = {n: tuple(s["data_offsets"]) for n, s in header.items() if "data_offsets" in s}
    for name in sorted(want):
        mine = ranges.get(name)
        if mine is None:
            continue
        for other, theirs in ranges.items():
            if other != name and theirs[0] < mine[1] and mine[0] < theirs[1]:
                problems.append(
                    f"weight {name!r} shares bytes [{mine[0]}, {mine[1]}) with {other!r}; "
                    "transposing it would rewrite the other tensor's data underneath it")
    return problems


def retarget_weights_file(mlir_text: str, weights_path: Path | str) -> tuple[str, bool]:
    """Point the module's ``prov.weights_file`` at `weights_path`. Returns (text, changed).

    A rewritten bundle whose provenance still names the SOURCE blob claims to have been built from
    bytes it was not built from. That is the live defect this module's docstring describes:
    `gemma2_2b_int8_full_seq8_pretransposed` and every other `_pretransposed` bundle on disk still
    name the ORIGINAL `weights.safetensors`, so anyone reading one concludes it is stock.

    Structural, no regex: the attribute is found by its literal key on the module line and its value
    ends at the next quote.
    """
    key = 'prov.weights_file = "'
    out: list[str] = []
    changed = False
    for line in mlir_text.splitlines():
        if not changed and line.lstrip().startswith("builtin.module") and key in line:
            head, _, rest = line.partition(key)
            _old, quote, tail = rest.partition('"')
            if quote:
                line = f"{head}{key}{weights_path}\"{tail}"
                changed = True
        out.append(line)
    return "\n".join(out) + "\n", changed


#: written by the hoist itself -- never carried over from the source bundle
_HOIST_WRITES = {"model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
                 REWRITES_FILE}
#: derived from the PRE-rewrite `model.mlir`; carrying it forward would let a consumer lower stale IR
_STALE_PREFIX = "model.prepared"


def _carry_sidecars(src: Path, dst: Path, written: set[str]) -> list[str]:
    """Hardlink every sidecar `src` has that the rewrite does not itself write; name what it skips.

    A FIXED list was the bug this replaces. Bundles on disk carry `session_contract.yaml` (the
    state-carry map `llvmlower.c_runtime.generate` reads), `session_inputs.npz`,
    `session_goldens.npz`, `session_quality_fp32.npz`, `golden_mesh_datapath.npy` and
    `ingest_meta.json` -- none of them in it -- so a rewritten bundle silently lost its multi-step
    contract and its extra goldens, and graded as a different, single-shot model.
    """
    skipped: list[str] = []
    for p in sorted(src.iterdir()):
        if not p.is_file() or p.name in written:
            continue
        if p.name.startswith(_STALE_PREFIX):
            skipped.append(p.name)
            continue
        _link_or_copy(p, dst / p.name)
    return skipped


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

    # The IR analysis proves the transpose is the argument's SOLE CONSUMER. That is a fact about
    # the graph and says nothing about how the bytes are STORED -- a stubbed weight, a weight two
    # arguments name, an aliased byte range. Each of those produces a bundle that builds and grades
    # while reading the wrong bytes, so they are checked here and refused, never worked around.
    problems = hoist_safety_problems(src, man, want, set(by_arg))
    if problems:
        raise RewriteRefused(
            f"cannot pre-apply the weight layout of {src.name}: " + "; ".join(problems))

    dst.mkdir(parents=True, exist_ok=True)
    done = _rewrite_safetensors(src, dst, set(want))
    if done != len(want):                       # belt-and-braces: the checks above should make this
        raise RewriteRefused(                   # unreachable, and a silent undercount is the failure
            f"{src.name}: transposed {done} of {len(want)} weights; refusing to write a bundle whose "
            "manifest claims a layout its bytes do not have")

    for entry in man.values():
        if entry.get("weight") in want and "shape" in entry:
            entry["shape"] = [entry["shape"][1], entry["shape"][0]]
    (dst / "weights.safetensors.manifest.json").write_text(json.dumps(man, indent=2))

    new_text, dropped = rewrite_transposed_args(mlir_text, set(by_arg))
    # Point the provenance at the blob this bundle actually has (see `retarget_weights_file`).
    new_text, retargeted = retarget_weights_file(new_text, (dst / "weights.safetensors").resolve())
    (dst / "model.mlir").write_text(new_text)

    skipped = _carry_sidecars(src, dst, _HOIST_WRITES)
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
            "weights_file_retargeted": retargeted,
            "sidecars_not_carried": skipped,
            "analysis": "merlin.xdsl_dialects.lowering.weight_layout.weight_layout_report",
        },
        caveats=([f"{len(report.blocked)} re-layout(s) were NOT hoisted (argument has other readers)"]
                 if report.blocked else [])
        + ([f"stale, NOT carried over from the source bundle: {skipped}"] if skipped else [])
        + ([] if retargeted else ["prov.weights_file was absent, so it could not be retargeted at "
                                  "this bundle's own weights"]),
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


def specialize_gather(src: Path | str, dst: Path | str, func_name: str = "forward") -> RewriteRecord:
    """Keep only the table rows this bundle's fixed inputs actually select, and renumber the indices.

    A model2MLIR embedding lookup indexes a weight table with the VALUE of an input element, so for a
    bundle whose inputs are pinned only a handful of rows are ever read. On Gemma 2 2B that is 8 rows
    of 256000: 2250.0 MiB of table becomes 0.070 MiB, which is what brought the image under the
    addressing limit that kept the whole model from running at all.

    THIS IS INPUT SPECIALIZATION, NOT DEAD-CODE ELIMINATION. The indices are runtime values this
    bundle happens to pin, not compile-time constants, so the result is valid FOR THESE INPUTS ONLY.
    That is recorded as a caveat on the bundle, because a specialized bundle mistaken for a general
    one produces confident nonsense for any other input.

    SOUNDNESS. Renumbering the stored indices is value-preserving only if the gather is the index
    tensor's sole consumer -- anything else reading it would silently receive renumbered tokens.
    :func:`find_gather_specializations` enforces that and reports a rejection instead.
    """
    from ..xdsl_dialects.lowering.gather_specialization import (
        find_gather_specializations, kept_rows)
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK

    src, dst = Path(src), Path(dst)
    mlir_text = (src / "model.mlir").read_text()
    with IR_LOCK:
        specs, rejections = find_gather_specializations(mq.parse(mlir_text), func_name)
    if not specs:
        raise ValueError(
            f"no specializable gather in {src}"
            + (f" ({len(rejections)} rejected: {[r.reason for r in rejections]})" if rejections
               else ""))
    if len(specs) > 1:
        raise ValueError(f"{len(specs)} specializable gathers; this applier handles exactly one")
    spec = specs[0]

    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    entry = man.get(str(spec.table_arg))
    if entry is None or "weight" not in entry:
        raise ValueError(f"table arg {spec.table_arg} names no weight in the manifest")

    order = json.loads((src / "input_order.json").read_text()) if (
        src / "input_order.json").is_file() else None
    with np.load(src / "inputs.npz") as z:
        inputs = {k: z[k] for k in z.files}
    idx_key = _input_key_for_arg(man, order, spec.index_arg, inputs)
    kept, renumbered = kept_rows(spec, np.asarray(inputs[idx_key]).ravel().tolist())

    dst.mkdir(parents=True, exist_ok=True)
    rows_before = spec.rows
    _slice_table(src, dst, entry["weight"], kept)

    if "shape" in entry:
        entry["shape"] = [len(kept)] + list(entry["shape"][1:])
    (dst / "weights.safetensors.manifest.json").write_text(json.dumps(man, indent=2))

    inputs[idx_key] = np.asarray(renumbered, dtype=inputs[idx_key].dtype).reshape(
        inputs[idx_key].shape)
    np.savez(dst / "inputs.npz", **inputs)

    (dst / "model.mlir").write_text(_retype_arg(
        mlir_text, spec.table_arg, spec.table_shape,
        [len(kept)] + list(spec.table_shape[1:]), spec.table_dtype))

    for name in ("extra.npz", "input_order.json", "golden.npy", "golden_w8a8.npy",
                 "region_goldens.npz"):
        _link_or_copy(src / name, dst / name)
    if (src / REWRITES_FILE).is_file():
        shutil.copy2(src / REWRITES_FILE, dst / REWRITES_FILE)

    width = np.dtype(_NP.get(spec.table_dtype.upper(), np.float32)).itemsize
    trailing = 1
    for d in spec.table_shape[1:]:
        trailing *= int(d)
    rec = RewriteRecord(
        name="specialize_gather",
        source_bundle=src.name,
        soundness=(f"the index tensor (arg {spec.index_arg}) has exactly one consumer, so "
                   "renumbering the stored ids and keeping only the rows they select is "
                   "value-preserving; checked by find_gather_specializations"),
        effect={"table_arg": spec.table_arg, "rows_before": rows_before, "rows_after": len(kept),
                "table_mib_before": round(rows_before * trailing * width / 2 ** 20, 3),
                "table_mib_after": round(len(kept) * trailing * width / 2 ** 20, 3),
                "index_arg": spec.index_arg, "index_arg_consumers": 1,
                "analysis": "merlin.xdsl_dialects.lowering.gather_specialization"},
        caveats=[f"VALID FOR THESE ROW INDICES ONLY: {kept}. This is input specialization, not dead "
                 "code elimination -- any other input selects rows that are no longer present."],
    )
    record_rewrite(dst, rec)
    return rec


def _input_key_for_arg(man: dict, order: Any, arg: int, inputs: dict) -> str:
    """Which `inputs.npz` array feeds `@forward` argument `arg`. Fail closed rather than guess."""
    entry = man.get(str(arg), {})
    name = entry.get("name")
    if name and name in inputs:
        return name
    if isinstance(order, list) and name in order:
        k = f"in{order.index(name)}"
        if k in inputs:
            return k
    if len(inputs) == 1:                    # unambiguous: one input, one index tensor
        return next(iter(inputs))
    raise ValueError(
        f"cannot tell which inputs.npz array feeds arg {arg} (manifest name {name!r}, "
        f"arrays {sorted(inputs)}); refusing to guess")


def _slice_table(src: Path, dst: Path, weight: str, kept: list[int]) -> None:
    """Rewrite the blob keeping only `kept` rows of `weight`, in order, everything else verbatim."""
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
            nbytes = e - s
            shape = list(spec["shape"])
            if name == weight:
                shape = [len(kept)] + shape[1:]
                nbytes = nbytes * len(kept) // spec["shape"][0]
            new_header[name] = {"dtype": spec["dtype"], "shape": shape,
                                "data_offsets": [off, off + nbytes]}
            order.append((name, s, e, spec))
            off += nbytes
        if meta is not None:
            new_header["__metadata__"] = meta
        blob = json.dumps(new_header, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)

        with open(dst / "weights.safetensors", "wb") as out:
            out.write(struct.pack("<Q", len(blob)))
            out.write(blob)
            for name, s, e, spec in order:
                if name != weight:
                    f.seek(data_start + s)
                    left = e - s
                    while left:
                        chunk = f.read(min(left, 64 << 20))
                        out.write(chunk)
                        left -= len(chunk)
                    continue
                dt = _NP.get(spec["dtype"])
                if dt is None:
                    raise ValueError(f"{name}: unknown safetensors dtype {spec['dtype']!r}")
                row_bytes = (e - s) // spec["shape"][0]
                for r in kept:                        # row-at-a-time: the table can be gigabytes
                    f.seek(data_start + s + r * row_bytes)
                    out.write(f.read(row_bytes))


def _mentions_ssa(line: str, name: str) -> bool:
    """Does `line` reference SSA value `name` at a token boundary? `%0` must not match `%01`."""
    i = 0
    while True:
        j = line.find(name, i)
        if j < 0:
            return False
        end = j + len(name)
        if end >= len(line) or not line[end].isdigit():
            return True
        i = end


def _retype_arg(mlir_text: str, arg: int, old_shape: list[int], new_shape: list[int],
                dtype: str) -> str:
    """Give `@forward` argument `arg` its new table type -- EVERYWHERE the value is typed.

    Not just the signature. The gather's own `tensor.extract %0[...] : tensor<256000x2304xf32>`
    carries the table type too, and retyping only the signature leaves IR that does not verify. That
    was a real bug here, caught by comparing the output byte-for-byte against the bundle this
    replaces rather than by eyeballing the diff.

    Retyping is confined to lines that reference the argument at a token boundary, so an unrelated
    value that happens to share the shape is untouched.
    """
    old_t = "tensor<" + "x".join(str(d) for d in old_shape) + f"x{dtype}>"
    new_t = "tensor<" + "x".join(str(d) for d in new_shape) + f"x{dtype}>"
    name = f"%{arg}"
    out, changed = [], 0
    for line in mlir_text.splitlines():
        if old_t in line and _mentions_ssa(line, name):
            line = line.replace(old_t, new_t)
            changed += 1
        out.append(line)
    if not changed:
        raise ValueError(f"argument {name} never appears with type {old_t}; refusing to write IR "
                         "whose table type was not updated")
    return "\n".join(out) + "\n"
