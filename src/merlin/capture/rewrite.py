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
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from .safetensors import load_header, read_header

#: The record lives here, beside the manifest rather than inside it (see the module docstring).
REWRITES_FILE = "bundle.rewrites.json"

#: safetensors dtype spelling -> numpy. Only what a bundle actually stores; an unknown dtype is an
#: error rather than a guess, because guessing wrong silently reinterprets the weight's bytes.
_NP = {
    "I8": np.int8,
    "U8": np.uint8,
    "I16": np.int16,
    "I32": np.int32,
    "I64": np.int64,
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "BF16": np.uint16,
}
_MLIR_DTYPE = {
    "I8": "i8",
    "U8": "ui8",
    "I16": "i16",
    "I32": "i32",
    "I64": "i64",
    "F16": "f16",
    "BF16": "bf16",
    "F32": "f32",
    "F64": "f64",
}
_MANIFEST_DTYPE = {
    "I8": {"int8", "i8"},
    "U8": {"uint8", "ui8", "u8"},
    "I16": {"int16", "i16"},
    "I32": {"int32", "i32"},
    "I64": {"int64", "i64"},
    "F16": {"float16", "f16", "half"},
    "BF16": {"bfloat16", "bf16"},
    "F32": {"float32", "f32", "float"},
    "F64": {"float64", "f64", "double"},
}


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


def _selected_relayouts(module, args, func_name, relayouts=None):
    from ..xdsl_dialects.lowering.weight_layout import WeightRelayout, weight_layout_report

    if type(args) is not set or any(type(i) is not int or i < 0 for i in args):
        raise RewriteRefused("weight relayout requires nonnegative argument indices")
    fresh = {r.arg: r for r in weight_layout_report(module, func_name).hoistable}
    if not args <= fresh.keys():
        raise RewriteRefused("selected arguments lack a verified sole-input-use transpose")
    selected = {i: fresh[i] for i in sorted(args)}
    if relayouts is not None:
        supplied = tuple(relayouts)
        if (
            any(type(r) is not WeightRelayout for r in supplied)
            or len(supplied) != len(selected)
            or {r.arg: r for r in supplied} != selected
        ):
            raise RewriteRefused("weight relayout selection is stale for the current source permutation")
    return selected


def _references_symbol(attribute, name):
    from xdsl.dialects import builtin

    if isinstance(attribute, builtin.SymbolRefAttr):
        return attribute.root_reference.data == name
    if isinstance(attribute, builtin.DictionaryAttr):
        children = attribute.data.values()
    elif isinstance(attribute, builtin.ArrayAttr):
        children = attribute.data
    else:
        children = getattr(attribute, "parameters", ())
    return any(_references_symbol(child, name) for child in children)


def rewrite_transposed_args(
    mlir_text: str, args: set[int], func_name: str = "forward", *, relayouts=None
) -> tuple[str, int]:
    """Rewrite verified rank-N tensor transposes by SSA identity, retaining unrelated IR.

    A supplied selection must match the current parsed permutation, including permutations
    whose shapes happen to be equal. Only an actually dead tensor.empty init is erased.
    Internal callers of the changed function ABI are unsupported and refuse.
    """
    from xdsl.dialects import tensor
    from xdsl.rewriter import Rewriter

    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK
    from ..xdsl_dialects._common import text
    from ..xdsl_dialects.lowering.weight_layout import _func, layout_chain

    with IR_LOCK:
        module = mq.parse(mlir_text)
        selected = _selected_relayouts(module, args, func_name, relayouts)
        fn = _func(module, func_name)
        if any(
            _references_symbol(attribute, func_name)
            for op in module.walk()
            for attribute in (*op.attributes.values(), *op.properties.values())
        ):
            raise RewriteRefused("weight relayout cannot change an internally referenced function ABI")
        dropped = 0
        for index in selected:
            arg = fn.args[index]
            operations, _steps = layout_chain(arg, fn)
            inits = [op.outputs[0].owner for op in operations if op.name == "linalg.transpose"]
            replacement = Rewriter.replace_value_with_new_type(arg, operations[-1].results[0].type)
            Rewriter.replace_op(operations[-1], [], [replacement])
            for op in reversed(operations[:-1]):
                Rewriter.erase_op(op)
            dropped += len(operations)
            for init in inits:
                if isinstance(init, tensor.EmptyOp) and init.parent is not None and not init.results[0].uses:
                    Rewriter.erase_op(init)
                    dropped += 1
        fn.update_function_type()
        module.verify()
        return text(module, generic=True) + "\n", dropped


class RewriteRefused(ValueError):
    """A rewrite whose soundness condition does not hold on this bundle. Raised, never worked around:
    a rewrite that cannot be proven value-preserving must not be applied silently."""


def hoist_safety_problems(
    src: Path | str,
    man: dict[str, Any],
    want: dict[str, int],
    hoisted_args: set[int],
    *,
    relayouts=None,
    func_name: str = "forward",
    immutable_arguments: tuple[int, ...] | None = None,
) -> list[str]:
    """Join fresh typed layouts to bounded, unaliased storage and invocation ownership.

    ``immutable_arguments`` is an ORIGINAL caller premise, never inferred from IR attributes.
    It permits externalized buffers only when the caller has established they stay immutable.
    Without it, legacy manifest-declared parameters are the only eligible stored arguments.
    Session state and streams always refuse, even if the premise names them.
    """
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK

    src = Path(src)
    problems: list[str] = []
    try:
        with IR_LOCK:
            selected = _selected_relayouts(
                mq.parse(src / "model.mlir"), hoisted_args, func_name, relayouts
            )
        header, payload_offset = load_header(src / "weights.safetensors")
        if "__metadata__" in header:
            metadata = header.pop("__metadata__")
            if type(metadata) is not dict or any(type(k) is not str or type(v) is not str for k, v in metadata.items()):
                raise ValueError("safetensors metadata must map strings to strings")
    except (ValueError, OSError) as exc:
        return [str(exc)]
    if type(man) is not dict or any(
        type(k) is not str or not k.isdigit() or type(v) is not dict for k, v in man.items()
    ):
        return ["weight manifest must map numeric argument indices to records"]
    if immutable_arguments is not None and (
        type(immutable_arguments) is not tuple
        or any(type(i) is not int or i < 0 for i in immutable_arguments)
        or len(immutable_arguments) != len(set(immutable_arguments))
    ):
        return ["immutable arguments require distinct explicit nonnegative indices"]
    payload_bytes = (src / "weights.safetensors").stat().st_size - payload_offset
    ranges = {}
    for name, spec in header.items():
        if type(spec) is not dict or type(spec.get("dtype")) is not str:
            problems.append(f"weight {name!r} has an invalid header record")
            continue
        shape, offsets = spec.get("shape"), spec.get("data_offsets")
        if (
            type(shape) is not list
            or any(type(d) is not int or d < 0 for d in shape)
            or type(offsets) is not list
            or len(offsets) != 2
            or any(type(i) is not int for i in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= payload_bytes
        ):
            problems.append(f"weight {name!r} has invalid shape or payload bounds")
            continue
        ranges[name] = tuple(offsets)
        dtype = _NP.get(spec.get("dtype"))
        if dtype is not None and prod(shape) * np.dtype(dtype).itemsize != offsets[1] - offsets[0]:
            problems.append(f"weight {name!r} header shape/dtype does not match its payload byte extent")
    session = src / "session_contract.yaml"
    updated = set()
    if session.is_file():
        from ..common.yaml import load_yaml

        try:
            contract = load_yaml(session)
            if type(contract) is not dict or contract.get("version") != 1:
                raise ValueError("only version 1 single-program session ownership is supported")
            for role in ("states", "streams"):
                entries = contract.get(role, [])
                if entries is None:
                    entries = []
                if type(entries) is not list:
                    raise ValueError(f"session {role} must be a list")
                for entry in entries:
                    if type(entry) is not dict or type(entry.get("input_arg")) is not int or entry["input_arg"] < 0:
                        raise ValueError(f"session {role} requires nonnegative input_arg indices")
                    updated.add(entry["input_arg"])
        except Exception as exc:
            problems.append(f"cannot establish session argument immutability: {exc}")

    for name, arg in sorted(want.items(), key=lambda kv: kv[1]):
        entry = man.get(str(arg), {})
        if arg not in selected or entry.get("weight") != name:
            problems.append(f"arg {arg}: weight selection differs from the source or manifest")
            continue
        if (
            entry.get("stub")
            or entry.get("mutable")
            or arg in updated
            or entry.get("kind") not in ("param", "buffer")
            or (immutable_arguments is None and entry.get("kind") != "param")
            or (immutable_arguments is not None and arg not in immutable_arguments)
        ):
            problems.append(
                f"arg {arg}: stored argument is stubbed, mutable/session-carried or lacks original immutable ownership"
            )
        spec = header.get(name)
        if spec is None:
            stubbed = bool(man.get(str(arg), {}).get("stub"))
            problems.append(
                f"arg {arg}: weight {name!r} has no bytes in weights.safetensors "
                f"({'manifest marks it stub=true' if stubbed else 'dangling manifest entry'}); "
                "its layout cannot be pre-applied, and flipping only the manifest shape would "
                "describe a transpose that never happened"
            )
            continue
        layout = selected[arg]
        if type(spec) is not dict or type(spec.get("dtype")) is not str:
            continue
        if spec.get("shape") != layout.shape or ("shape" in entry and entry["shape"] != layout.shape):
            problems.append(f"arg {arg}: source, manifest and header shapes disagree")
        dtype = spec.get("dtype")
        if _MLIR_DTYPE.get(dtype) != layout.dtype or dtype not in _NP:
            problems.append(f"arg {arg}: source and safetensors dtype disagree or are unsupported")
        if "dtype" in entry and str(entry["dtype"]).removeprefix("torch.") not in _MANIFEST_DTYPE.get(dtype, ()):
            problems.append(f"arg {arg}: manifest and safetensors dtype disagree")

    for key, entry in man.items():
        w = entry.get("weight")
        if w not in want or not key.isdigit():
            continue
        if int(key) != want[w]:
            problems.append(
                f"weight {w!r} is also read by arg {key}; pre-transposing it would change what that argument sees"
            )

    if set(want.values()) != hoisted_args:
        problems.append("selected argument identities alias one stored weight name")
    for name in sorted(want):
        mine = ranges.get(name)
        if mine is None:
            continue
        for other, theirs in ranges.items():
            if other != name and theirs[0] < mine[1] and mine[0] < theirs[1]:
                problems.append(
                    f"weight {name!r} shares bytes [{mine[0]}, {mine[1]}) with {other!r}; "
                    "transposing it would rewrite the other tensor's data underneath it"
                )
    return problems


def retarget_weights_file(mlir_text: str, weights_path: Path | str) -> tuple[str, bool]:
    """Point the module's ``prov.weights_file`` at `weights_path`. Returns (text, changed).

    A rewritten bundle whose provenance still names the SOURCE blob claims to have been built from
    bytes it was not built from. That is the live defect this module's docstring describes:
    `gemma2_2b_int8_full_seq8_pretransposed` and every other `_pretransposed` bundle on disk still
    name the ORIGINAL `weights.safetensors`, so anyone reading one concludes it is stock.

    Parsed attributes retain the same behavior for custom and generic module syntax.
    """
    from xdsl.dialects.builtin import StringAttr

    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK
    from ..xdsl_dialects._common import text

    with IR_LOCK:
        module = mq.parse(mlir_text)
        if "prov.weights_file" not in module.attributes:
            return mlir_text, False
        if not isinstance(module.attributes["prov.weights_file"], StringAttr):
            raise RewriteRefused("prov.weights_file must be a string attribute")
        module.attributes["prov.weights_file"] = StringAttr(str(weights_path))
        module.verify()
        return text(module, generic=True) + "\n", True


#: written by the hoist itself -- never carried over from the source bundle
_HOIST_WRITES = {"model.mlir", "weights.safetensors", "weights.safetensors.manifest.json", REWRITES_FILE}
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


def _weight_selection(src, func_name, immutable_arguments):
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK
    from ..xdsl_dialects.lowering.weight_layout import weight_layout_report

    src = Path(src)
    mlir_text = (src / "model.mlir").read_text()
    with IR_LOCK:
        report = weight_layout_report(mq.parse(src / "model.mlir", source_text=mlir_text), func_name)

    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    if type(man) is not dict or any(
        type(key) is not str
        or not key.isdigit()
        or type(entry) is not dict
        or ("weight" in entry and type(entry["weight"]) is not str)
        for key, entry in man.items()
    ):
        raise RewriteRefused("weight manifest must bind numeric arguments to stored tensor names")
    if immutable_arguments is not None and (
        type(immutable_arguments) is not tuple
        or any(type(i) is not int or i < 0 for i in immutable_arguments)
        or len(immutable_arguments) != len(set(immutable_arguments))
    ):
        raise RewriteRefused("immutable arguments require distinct explicit nonnegative indices")
    weight_args = {int(k) for k, entry in man.items() if k.isdigit() and "weight" in entry}
    # The IR-only analysis also sees sole-use transposes of runtime inputs.  Those values have no
    # stored payload to rewrite and must remain graph operations; they do not invalidate independent
    # manifest-declared weight candidates.  Storage eligibility is decided here, where the manifest
    # is authoritative, rather than guessed from provenance or argument position.
    by_arg = {
        r.arg: r
        for r in report.hoistable
        if r.arg in weight_args and (immutable_arguments is None or r.arg in immutable_arguments)
    }
    unpriceable_weights = [
        problem for problem in report.unpriceable if any(problem.startswith(f"arg {arg}:") for arg in by_arg)
    ]
    if unpriceable_weights:
        raise ValueError(  # fail closed: an unpriced weight is not a free one
            f"cannot hoist safely, {len(unpriceable_weights)} weight re-layout(s) could not be "
            f"priced: {unpriceable_weights}"
        )
    if not by_arg:
        raise ValueError(f"no hoistable weight transposes in {src}")

    want: dict[str, int] = {}  # safetensors tensor name -> arg index
    for arg in by_arg:
        entry = man.get(str(arg))
        assert entry is not None and "weight" in entry
        want[entry["weight"]] = arg

    # The IR analysis proves the transpose is the argument's SOLE CONSUMER. That is a fact about
    # the graph and says nothing about how the bytes are STORED -- a stubbed weight, a weight two
    # arguments name, an aliased byte range. Each of those produces a bundle that builds and grades
    # while reading the wrong bytes, so they are checked here and refused, never worked around.
    problems = hoist_safety_problems(
        src,
        man,
        want,
        set(by_arg),
        relayouts=tuple(by_arg.values()),
        func_name=func_name,
        immutable_arguments=immutable_arguments,
    )
    if problems:
        raise RewriteRefused(f"cannot pre-apply the weight layout of {src.name}: " + "; ".join(problems))

    return mlir_text, report, man, weight_args, by_arg, want


def _hoist_record(src, report, weight_args, by_arg, dropped, retargeted, skipped):
    """Describe the source-derived rewrite after its actual IR/payload checks."""
    return RewriteRecord(
        name="hoist_weight_transposes",
        source_bundle=src.name,
        soundness=(
            "each hoisted argument's transpose is its SOLE consumer, so pre-applying the "
            "layout cannot change what any other reader sees; verified per argument by "
            "weight_layout_report, and each weight's raw element bytes follow the checked layout chain; "
            "immutable invocation ownership is a separate original caller/manifest premise"
        ),
        effect={
            "weights_pre_transposed": len(by_arg),
            "transposes_removed": sum(r.transposes for r in by_arg.values()),
            "ops_removed": dropped,
            "layouts": [asdict(r) for r in by_arg.values()],
            "bytes_moved_per_inference_before": sum(r.bytes_moved for r in by_arg.values()),
            "mib_moved_per_inference_before": round(sum(r.bytes_moved for r in by_arg.values()) / 2**20, 1),
            "bytes_moved_per_inference_after": 0,
            "blocked_not_hoisted": len(report.blocked),
            "outside_immutable_selection": len(
                [r for r in report.hoistable if r.arg in weight_args and r.arg not in by_arg]
            ),
            "non_weight_transposes_not_hoisted": sum(r.arg not in weight_args for r in report.hoistable),
            "weights_file_retargeted": retargeted,
            "sidecars_not_carried": skipped,
            "analysis": "merlin.xdsl_dialects.lowering.weight_layout.weight_layout_report",
        },
        caveats=(
            [f"{len(report.blocked)} re-layout(s) were NOT hoisted (argument has other readers)"]
            if report.blocked
            else []
        )
        + (
            [
                f"{sum(r.arg not in weight_args for r in report.hoistable)} sole-use layout chain(s) "
                "were NOT hoisted because "
                "their arguments are runtime inputs, not manifest-declared weights"
            ]
            if any(r.arg not in weight_args for r in report.hoistable)
            else []
        )
        + (
            ["stored layout chains outside the original immutable argument selection remain unchanged"]
            if any(r.arg in weight_args and r.arg not in by_arg for r in report.hoistable)
            else []
        )
        + ([f"stale, NOT carried over from the source bundle: {skipped}"] if skipped else [])
        + (
            []
            if retargeted
            else ["prov.weights_file was absent, so it could not be retargeted at this bundle's own weights"]
        ),
    )


def hoist_weight_transposes(
    src: Path | str,
    dst: Path | str,
    func_name: str = "forward",
    *,
    immutable_arguments: tuple[int, ...] | None = None,
) -> RewriteRecord:
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
    that the bytes written follow the source's exact layout chain, so an error surfaces here rather than as wrong
    logits hours later on a simulator.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve() or dst.exists():
        raise RewriteRefused("weight relayout requires a fresh destination distinct from its source")
    mlir_text, report, man, weight_args, by_arg, want = _weight_selection(src, func_name, immutable_arguments)

    new_text, dropped = rewrite_transposed_args(mlir_text, set(by_arg), func_name, relayouts=tuple(by_arg.values()))
    if (src / "model.mlir").read_text() != mlir_text:
        raise RewriteRefused("weight layout source changed during selection")
    dst.mkdir(parents=True)
    done = _rewrite_safetensors(src, dst, {name: by_arg[arg] for name, arg in want.items()})
    if done != len(want):  # belt-and-braces: the checks above should make this
        raise RewriteRefused(  # unreachable, and a silent undercount is the failure
            f"{src.name}: transposed {done} of {len(want)} weights; refusing to write a bundle whose "
            "manifest claims a layout its bytes do not have"
        )

    for entry in man.values():
        if entry.get("weight") in want and "shape" in entry:
            entry["shape"] = list(by_arg[want[entry["weight"]]].result_shape)
    (dst / "weights.safetensors.manifest.json").write_text(json.dumps(man, indent=2))

    # Point the provenance at the blob this bundle actually has (see `retarget_weights_file`).
    new_text, retargeted = retarget_weights_file(new_text, (dst / "weights.safetensors").resolve())
    (dst / "model.mlir").write_text(new_text)

    skipped = _carry_sidecars(src, dst, _HOIST_WRITES)
    if (src / REWRITES_FILE).is_file():  # carry the chain forward, do not start a new one
        shutil.copy2(src / REWRITES_FILE, dst / REWRITES_FILE)

    rec = _hoist_record(src, report, weight_args, by_arg, dropped, retargeted, skipped)
    record_rewrite(dst, rec)
    return rec


def _rewrite_safetensors(src: Path, dst: Path, want: dict) -> int:
    """Apply verified layouts to raw element-byte records, keeping tensor order and metadata."""
    from ..xdsl_dialects.lowering.weight_layout import apply_layout_steps

    with open(src / "weights.safetensors", "rb") as f:
        header, data_start = read_header(f)
        meta = header.pop("__metadata__", None)

        new_header: dict[str, Any] = {}
        order: list[tuple[str, int, int, dict]] = []
        off = 0
        for name, spec in header.items():
            s, e = spec["data_offsets"]
            shape = list(spec["shape"])
            if name in want:
                shape = list(want[name].result_shape)
            new_header[name] = {"dtype": spec["dtype"], "shape": shape, "data_offsets": [off, off + (e - s)]}
            order.append((name, s, e, spec))
            off += e - s
        if meta is not None:
            new_header["__metadata__"] = meta
        blob = json.dumps(new_header, separators=(",", ":")).encode()
        blob += b" " * ((-len(blob)) % 8)  # safetensors wants 8-byte aligned data

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
                        if not chunk:
                            raise RewriteRefused(f"{name}: source payload truncated while copying")
                        out.write(chunk)
                        left -= len(chunk)
                    continue
                dt = _NP.get(spec["dtype"])
                if dt is None:
                    raise ValueError(f"{name}: unknown safetensors dtype {spec['dtype']!r}")
                raw = f.read(e - s)
                if len(raw) != e - s:
                    raise RewriteRefused(f"{name}: source payload truncated while relaying out")
                arr = np.frombuffer(raw, dtype=np.dtype(f"V{np.dtype(dt).itemsize}")).reshape(spec["shape"])
                t = apply_layout_steps(arr, want[name].steps)
                if t.shape != tuple(new_header[name]["shape"]) or t.nbytes != e - s:
                    raise RewriteRefused(f"{name}: layout shape/byte extent changed")
                out.write(t.tobytes())
                done += 1
    return done


def verify_hoisted_bundle(
    src: Path | str,
    dst: Path | str,
    func_name: str = "forward",
    *,
    immutable_arguments: tuple[int, ...] | None = None,
) -> dict:
    """Reconstruct a produced bundle from current source semantics and original bytes.

    A receipt or matching output digest does not establish this relation. The actual
    output IR must equal the typed source rewrite, and inverse layout steps must recover
    every original element byte. Untouched tensors are compared in bounded chunks.
    This establishes a conditional storage rewrite, not implementation qualification.
    """
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK
    from ..common.jsonio import canonical_json, strict_json_equal
    from ..xdsl_dialects._common import text

    src, dst = Path(src), Path(dst)
    original, report, manifest, weight_args, by_arg, want = _weight_selection(src, func_name, immutable_arguments)
    expected, removed = rewrite_transposed_args(original, set(by_arg), func_name, relayouts=tuple(by_arg.values()))
    expected, retargeted = retarget_weights_file(expected, (dst / "weights.safetensors").resolve())
    with IR_LOCK:
        actual_module, expected_module = mq.parse(dst / "model.mlir"), mq.parse(expected)
        actual_module.verify()
        if text(actual_module, generic=True) != text(expected_module, generic=True):
            raise RewriteRefused("hoisted model IR differs from the checked source layout rewrite")
    for entry in manifest.values():
        if entry.get("weight") in want and "shape" in entry:
            entry["shape"] = list(by_arg[want[entry["weight"]]].result_shape)
    actual_manifest = json.loads((dst / "weights.safetensors.manifest.json").read_text())
    if not strict_json_equal(actual_manifest, manifest):
        raise RewriteRefused("hoisted manifest differs from source layout geometry or ownership")
    source_header, source_offset = load_header(src / "weights.safetensors")
    actual_header, actual_offset = load_header(dst / "weights.safetensors")
    expected_header, cursor = {}, 0
    for name, spec in source_header.items():
        if name == "__metadata__":
            continue
        begin, end = spec["data_offsets"]
        shape = by_arg[want[name]].result_shape if name in want else spec["shape"]
        expected_header[name] = dict(
            dtype=spec["dtype"], shape=list(shape), data_offsets=[cursor, cursor + end - begin]
        )
        cursor += end - begin
    if "__metadata__" in source_header:
        expected_header["__metadata__"] = source_header["__metadata__"]
    if (
        not strict_json_equal(actual_header, expected_header)
        or (dst / "weights.safetensors").stat().st_size != actual_offset + cursor
    ):
        raise RewriteRefused("hoisted safetensors header, bounds or final payload extent disagree with source")
    with (
        (src / "weights.safetensors").open("rb") as source_file,
        (dst / "weights.safetensors").open("rb") as actual_file,
    ):
        for name, spec in source_header.items():
            if name == "__metadata__":
                continue
            begin, end = spec["data_offsets"]
            actual_begin, _actual_end = actual_header[name]["data_offsets"]
            source_file.seek(source_offset + begin)
            actual_file.seek(actual_offset + actual_begin)
            count = end - begin
            if name in want:
                layout = by_arg[want[name]]
                width = np.dtype(_NP[spec["dtype"]]).itemsize
                raw = actual_file.read(count)
                if len(raw) != count:
                    raise RewriteRefused(f"{name}: hoisted payload truncated")
                inverse = np.frombuffer(raw, dtype=np.dtype(f"V{width}")).reshape(layout.result_shape)
                for step in reversed(layout.steps):
                    if step.operation == "linalg.transpose":
                        permutation = tuple(step.permutation.index(axis) for axis in range(len(step.permutation)))
                        inverse = np.transpose(inverse, permutation)
                    else:
                        inverse = inverse.reshape(step.shape)
                if np.ascontiguousarray(inverse).tobytes() != source_file.read(count):
                    raise RewriteRefused(f"{name}: inverse layout does not recover original stored element bytes")
            else:
                while count:
                    size = min(count, 1 << 20)
                    left, right = source_file.read(size), actual_file.read(size)
                    if len(left) != size or left != right:
                        raise RewriteRefused(f"{name}: untouched stored element bytes changed")
                    count -= size
    # A cache-owned hash cannot establish that carried invocation/state metadata
    # is still the original metadata. Compare that relation independently, too.
    from ..common.digest import sha256_file

    source_files = {p.name: p for p in src.iterdir() if p.is_file()}
    skipped = sorted(name for name in source_files if name.startswith(_STALE_PREFIX))
    expected_names = (source_files.keys() - set(skipped)) | _HOIST_WRITES
    actual_names = {p.name for p in dst.iterdir() if p.is_file()}
    # This one cache receipt is not part of the capture-bundle format.
    actual_names.discard("weight-prepack-identity.json")
    if actual_names != expected_names:
        raise RewriteRefused("hoisted sidecar roster differs from the original bundle")
    for name in expected_names - _HOIST_WRITES:
        if sha256_file(source_files[name]) != sha256_file(dst / name):
            raise RewriteRefused(f"hoisted carried sidecar differs from original source: {name}")
    rec = _hoist_record(src, report, weight_args, by_arg, removed, retargeted, skipped)
    original_history = (
        json.loads((src / REWRITES_FILE).read_text()) if (src / REWRITES_FILE).is_file() else {"rewrites": []}
    )
    expected_history = {**original_history, "rewrites": [*original_history["rewrites"], asdict(rec)]}
    actual_history = json.loads((dst / REWRITES_FILE).read_text())
    # JSON normalization only changes tuples into lists; no supplied field is
    # ignored, including prior history, layouts, caveats and effect counters.
    if canonical_json(actual_history) != canonical_json(expected_history):
        raise RewriteRefused("hoisted rewrite history or reported effects differ from reconstructed source")
    return rec.effect


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
    from ..common import mlir_query as mq
    from ..common.ir_lock import IR_LOCK
    from ..xdsl_dialects.lowering.gather_specialization import find_gather_specializations, kept_rows

    src, dst = Path(src), Path(dst)
    mlir_text = (src / "model.mlir").read_text()
    with IR_LOCK:
        specs, rejections = find_gather_specializations(mq.parse(mlir_text), func_name)
    if not specs:
        raise ValueError(
            f"no specializable gather in {src}"
            + (f" ({len(rejections)} rejected: {[r.reason for r in rejections]})" if rejections else "")
        )
    if len(specs) > 1:
        raise ValueError(f"{len(specs)} specializable gathers; this applier handles exactly one")
    spec = specs[0]

    man = json.loads((src / "weights.safetensors.manifest.json").read_text())
    entry = man.get(str(spec.table_arg))
    if entry is None or "weight" not in entry:
        raise ValueError(f"table arg {spec.table_arg} names no weight in the manifest")

    order = json.loads((src / "input_order.json").read_text()) if (src / "input_order.json").is_file() else None
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

    inputs[idx_key] = np.asarray(renumbered, dtype=inputs[idx_key].dtype).reshape(inputs[idx_key].shape)
    np.savez(dst / "inputs.npz", **inputs)

    (dst / "model.mlir").write_text(
        _retype_arg(
            mlir_text, spec.table_arg, spec.table_shape, [len(kept)] + list(spec.table_shape[1:]), spec.table_dtype
        )
    )

    for name in ("extra.npz", "input_order.json", "golden.npy", "golden_w8a8.npy", "region_goldens.npz"):
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
        soundness=(
            f"the index tensor (arg {spec.index_arg}) has exactly one consumer, so "
            "renumbering the stored ids and keeping only the rows they select is "
            "value-preserving; checked by find_gather_specializations"
        ),
        effect={
            "table_arg": spec.table_arg,
            "rows_before": rows_before,
            "rows_after": len(kept),
            "table_mib_before": round(rows_before * trailing * width / 2**20, 3),
            "table_mib_after": round(len(kept) * trailing * width / 2**20, 3),
            "index_arg": spec.index_arg,
            "index_arg_consumers": 1,
            "analysis": "merlin.xdsl_dialects.lowering.gather_specialization",
        },
        caveats=[
            f"VALID FOR THESE ROW INDICES ONLY: {kept}. This is input specialization, not dead "
            "code elimination -- any other input selects rows that are no longer present."
        ],
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
    if len(inputs) == 1:  # unambiguous: one input, one index tensor
        return next(iter(inputs))
    raise ValueError(
        f"cannot tell which inputs.npz array feeds arg {arg} (manifest name {name!r}, "
        f"arrays {sorted(inputs)}); refusing to guess"
    )


def _slice_table(src: Path, dst: Path, weight: str, kept: list[int]) -> None:
    """Rewrite the blob keeping only `kept` rows of `weight`, in order, everything else verbatim."""
    with open(src / "weights.safetensors", "rb") as f:
        header, data_start = read_header(f)
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
            new_header[name] = {"dtype": spec["dtype"], "shape": shape, "data_offsets": [off, off + nbytes]}
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
                for r in kept:  # row-at-a-time: the table can be gigabytes
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


def _retype_arg(mlir_text: str, arg: int, old_shape: list[int], new_shape: list[int], dtype: str) -> str:
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
        raise ValueError(
            f"argument {name} never appears with type {old_t}; refusing to write IR whose table type was not updated"
        )
    return "\n".join(out) + "\n"
