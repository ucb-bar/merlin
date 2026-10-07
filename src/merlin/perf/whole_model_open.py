"""An OPEN model -- host regions between its groups -- as one program: the model's own host code, and
one dispatch call per device group answered by the package, the target's library, or the host.

    record = build(package_dir, model_capsule, target=t, machine=m, header=h, extra=npz, out=o)

:mod:`merlin.perf.whole_model_build` builds a CLOSED model: every compute group is a device call and
the only host work is the input's quantization, so the program is a list of library/kernel calls. A
transformer is not closed. Its norms, softmax, rotary embedding and per-token dynamic quantization
compute between the contractions, and on an int8 unit they are host work by the target's own
declaration (a float datapath the unit does not have). Refusing such a model -- ``model_closure`` --
is right for a program that only strings group calls together; it is wrong for a program that can
run the host part too.

THE HOST PART IS THE MODEL'S OWN IR. Nothing here writes a second spelling of a layer norm. The
capture's module is lowered for the target's host ISA by the stock whole-model lowering
(:mod:`merlin.llvmlower`), with each device group cut out and replaced by a call to
``merlin_dispatch_g<index>`` -- an external function with the C interface the lowering already emits.
The cut is EXACTLY the group's committed value (the accumulator an int8 contraction commits, before
the widening cast the capture follows it with), so the host code on either side is unchanged.

EACH DISPATCH HAS THREE ANSWERS, attributed per group, never summed:

* ``package`` -- the package's own kernel for that group, asked through the same capsule route the
  closed build asks (:func:`merlin.llvmlower.whole_program.whole_program_buffer` with
  ``open_model=True``) and bound by the same ABI resolution;
* ``vendor`` -- the target's library call, on the loop-free path the target driver names when an
  instruction role is prohibited;
* ``host`` -- the group's own IR, lowered as a function of its own (``merlin_host_g<index>``), which is
  also the answer when the machine cannot read the group's result out (a full-width accumulator on a
  narrow-readout machine) -- with that cause.

Which one a group takes is recorded with the reason. The C that makes the call -- the library call,
the kernel's argument order, the timing brackets -- is the TARGET's (its whole-model driver's
``render_dispatch``); this module writes none of it and names no target.

THE ORACLE IS NUMPY. :mod:`merlin.runtime.linalg_numpy` evaluates the capture's module with no
compiler in it. Every device group is then graded LOCALLY on the device: its exact integer result is
checked on the core against the inputs it actually read (a projection check, see the driver). The
host code is graded where it hands data to the device (a sample of each dispatch's input against the
oracle, within a declared bound) and at the model's output against the capsule's golden under the
capsule's own numeric policy.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.perf.whole_model_chunks import (  # noqa: F401
    _CHEAP_PRODUCERS,
    _chunk_bounds,
    chunk_forward,
    forward_body_size,
    resolve_chunk_ops,
)

__all__ = [
    "DISPATCH_PREFIX",
    "HOST_PREFIX",
    "Dispatch",
    "OpenModelError",
    "externalize_dispatches",
]

SCHEMA = "whole_model_open_build_v1"
#: The lowering features every open model's host code is built with. Exactly specified math ops (floor,
#: ceil, trunc, round, round-even, fabs, and a power of two of an integer) become inline code instead of
#: per-element libm calls: the same bits, and a loop the compiler can vectorize for a host hart that has
#: vectors. Measured on an integer-nonlinear SigLIP layer: 114 ``floor``, 16 ``roundevenf`` and the
#: softmax's per-element ``pow(2, z)`` calls left in the host loops; and on int8-full SmolVLA, 53,426
#: ``__truncsfbf2`` call sites, one per bf16 op per element, which the same feature now spells as f32
#: arithmetic and an integer rounding (:mod:`merlin.llvmlower.roundeven_intrinsic`).
HOST_LOWERING_FEATURES = ("lower_exact_math_inline",)

#: Optimizer options for the host code's compile, beside the cross flags. Loop-versioning LICM runs
#: a loop twice-compiled behind a runtime no-alias check, so an invariant row value (a per-row scale's
#: ``log``, which linalg fusion places in the per-element loop) is computed once per row and the loop
#: vectorizes. Measured: it is the GELU's remaining per-element ``logf``. Semantics-preserving.
HOST_CODE_OPTIONS = ("-mllvm", "-enable-loop-versioning-licm")

#: The bare-metal harness files an open-model program is compiled and linked from (and identified by).
HARNESS_SOURCES = ("crt.S", "htif.c", "libc_min.c", "printf_min.c", "merlin_malloc.c", "model_link.ld")

#: Why a model cannot be built for a machine at all: its device groups commit an accumulator the
#: machine has no full-width readout for (see ``build(require_device_readout=...)``).
MACHINE_CANNOT_READ_OUT = "machine_cannot_read_out_the_models_accumulators"

from .whole_model_dispatches import (
    BUFFER_ACCESS as BUFFER_ACCESS,
)
from .whole_model_dispatches import (
    DISPATCH_PREFIX,
    HOST_PREFIX,
    Dispatch,
    OpenModelError,
    externalize_dispatches,
)
from .whole_model_dispatches import (
    _committed_members as _committed_members,
)
from .whole_model_dispatches import (
    _is_zero as _is_zero,
)
from .whole_model_dispatches import (
    _shape_dtype as _shape_dtype,
)


def two_harts(machine: str, host_hart: int) -> dict[str, Any]:
    """The hart roles of a two-hart program for ``machine``, from its registry entry's declared harts:
    ``{host, unit, host_isa, count}``. Refused when the machine declares no harts, when ``host_hart`` is
    not one of them or is the unit's own, or when not exactly one hart reaches the unit."""
    from .whole_model_headers import machine_harts

    declared = machine_harts(machine)
    if not declared:
        raise OpenModelError(f"{machine!r} declares no harts in the hardware registry; a two-hart program is UNKNOWN")
    units = [row["hart"] for row in declared if row["unit"]]
    if len(units) != 1:
        raise OpenModelError(
            f"{machine!r} declares {len(units)} harts that reach the unit; a two-hart program needs one"
        )
    host = next((row for row in declared if row["hart"] == int(host_hart)), None)
    if host is None or host["unit"]:
        raise OpenModelError(
            f"hart {host_hart} of {machine!r} is {'the unit hart' if host else 'not declared'}; the host code runs "
            f"on a hart other than the unit's ({[r['hart'] for r in declared]})"
        )
    unit = next(row for row in declared if row["hart"] == units[0])
    return {
        "host": host["hart"],
        "unit": unit["hart"],
        "host_isa": host["isa"],
        "unit_isa": unit["isa"],
        "count": max(r["hart"] for r in declared) + 1,
    }


def _vector_extensions(tokens: Sequence[str]) -> list[str]:
    """The vector extensions among an ISA's extension tokens (``v``, ``zve*``, ``zv*``)."""
    return [t for t in tokens[1:] if t == "v" or t.startswith("zv")]


def check_unit_hart_code(objects: Sequence[Path], *, host_objects: Sequence[Path], unit_isa: str) -> list[str]:
    """Every linked object but the host code's must run on the unit's hart, so none may be compiled for a
    vector extension that hart lacks; returns the objects checked, or raises naming each offender. Read
    from what each object records it was compiled for, never from the flags meant for it."""
    from merlin.runtime.backends.spike_model import arch_extensions

    tokens = unit_isa.split("_")
    allowed = set(_vector_extensions([tokens[0], *tokens[0][4:], *tokens[1:]]))
    hosts = {Path(o).resolve() for o in host_objects}
    checked, offenders = [], []
    for obj in objects:
        if Path(obj).resolve() in hosts:
            continue
        found = set(_vector_extensions(arch_extensions(obj)))
        # A hart with the vector extension runs any of its sub-extensions; one without runs none.
        extra = sorted(found if "v" not in allowed else set())
        checked.append(Path(obj).name)
        if extra:
            offenders.append(f"{Path(obj).name} ({', '.join(extra)})")
    if offenders:
        raise OpenModelError(
            f"object(s) the unit's hart ({unit_isa}) may execute are compiled for a vector extension it lacks: "
            f"{offenders[:8]}; only the host code runs on the vector hart"
        )
    return checked


def vector_host_hart(machine: str) -> int | None:
    """The hart an open model's host code runs fastest on, from ``machine``'s declared harts: the first
    one that does not reach the unit and has a vector extension. ``None`` when the machine declares no
    such hart (every declared-less machine), so its programs keep the one-hart layout."""
    from .whole_model_headers import machine_harts

    for row in machine_harts(machine):
        if not row["unit"] and "v" in row["isa"].split("_")[0][4:]:
            return int(row["hart"])
    return None


def module_text(module) -> str:
    """``module`` as MLIR text with every external declaration's argument attributes kept.

    xDSL's custom printer drops a declaration's ``arg_attrs`` (it prints them only beside a body's block
    arguments), and the dispatches' ``bufferization.access`` is what spares every operand a copy. The
    declarations are printed in the generic form instead, which states them, and which MLIR parses."""
    import io

    from xdsl.printer import Printer

    declarations = [op for op in module.body.block.ops if op.name == "func.func" and not op.body.blocks]
    if not any(op.arg_attrs for op in declarations):
        return str(module)
    for op in declarations:
        op.detach()
    try:
        text = str(module)
    finally:
        for op in declarations:
            module.body.block.add_op(op)
    generic = []
    for op in declarations:
        stream = io.StringIO()
        Printer(stream=stream, print_generic_format=True).print_op(op)
        generic.append("  " + stream.getvalue().strip())
    head, brace, tail = text.rpartition("}")
    if not brace:
        raise OpenModelError("the printed module has no closing brace to place its declarations before")
    return head.rstrip() + "\n" + "\n".join(generic) + "\n}" + tail


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------ arguments


def _lifted(meta: Mapping[str, Any]) -> bool:
    """A leaf a capture leaves as an ARGUMENT without storing it in the weights file: a registered
    buffer, or a constant the export lifted out of the graph."""
    return meta.get("kind") == "buffer" or "lifted_tensor" in str(meta.get("name") or "")


def forward_arguments(capsule, *, extra: str | Path | None = None) -> tuple[list[Any], dict[str, Any]]:
    """Every argument of the capsule's ``@forward`` in storage form, and where each class came from.

    Weights come from the capsule's weights file by manifest name; the model inputs from the capsule's
    golden, in the capsule's declared order; a lifted constant or registered buffer from ``extra`` (an
    ``.npz`` in the capture layout: ``buf::<dotted name>`` for a buffer, the manifest name for a lifted
    constant). A leaf nobody supplies is REFUSED by name -- a zero in its place computes another model.
    """
    import numpy as np

    from merlin.llvmlower.model_runner import parse_forward_signature
    from merlin.llvmlower.weights_pack import load_safetensors_header
    from merlin.runtime import linalg_numpy as LN
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import capsule_inputs as CI

    declared = CC.load_capsule(capsule.directory)
    if extra is None:
        # A capsule that carries its own leaves names the file: `operation.attributes.lifted_leaves`, or
        # `operation.attributes.extra` (the spelling the int8-attention capsule uses). An explicit
        # `extra` argument still wins, so a caller can always say which file it means.
        attributes = (declared.get("operation") or {}).get("attributes") or {}
        named = attributes.get("lifted_leaves") or attributes.get("extra")
        extra = Path(capsule.directory) / str(named) if named else None
    inputs = CI.canonical_input_values(declared, capsule.directory)
    order = [str(spec["name"]) for spec in declared.get("inputs") or ()]
    signature = parse_forward_signature(capsule.interface)
    manifest = json.loads(Path(capsule.weights_manifest).read_text(encoding="utf-8"))
    header, payload = load_safetensors_header(capsule.weights)
    blob = np.memmap(capsule.weights, dtype=np.uint8, mode="r")
    lifted = np.load(extra) if extra is not None else None
    arguments: list[Any] = []
    missing: list[str] = []
    used: list[str] = []
    next_input = 0
    for index, (shape, dtype) in enumerate(signature):
        meta = manifest.get(str(index)) or {}
        storage = LN.storage_dtype(dtype)
        if meta.get("kind") == "param" and meta.get("weight") in header:
            begin, end = header[meta["weight"]]["data_offsets"]
            arguments.append(np.frombuffer(blob[payload + begin : payload + end], dtype=storage).reshape(shape))
            continue
        if _lifted(meta):
            name = str(meta.get("name") or "")
            found = None
            for key in getattr(lifted, "files", None) or ():
                if key == name or (key.startswith("buf::") and "b_" + key[5:].replace(".", "_") == name):
                    found = np.asarray(lifted[key])
                    used.append(key)
                    break
            if found is None:
                missing.append(name or f"arg{index}")
                continue
            arguments.append(np.ascontiguousarray(found.astype(storage).reshape(shape)))
            continue
        if next_input >= len(order):
            raise OpenModelError(f"argument {index} ({meta}) is neither stored, lifted nor a declared input")
        spec = inputs[order[next_input]]
        next_input += 1
        arguments.append(np.ascontiguousarray(np.asarray(spec["values"]).astype(storage).reshape(shape)))
    if missing:
        raise OpenModelError(
            f"the capsule's @forward reads {len(missing)} leaf argument(s) its files do not supply "
            f"({missing[:6]}); name an extra .npz that holds them"
        )
    sources = {
        "weights": {"path": str(capsule.weights), "sha256": _sha256(capsule.weights)},
        "inputs": {"path": str(Path(capsule.directory) / "golden.yaml"), "order": order},
        "extra": {"path": str(extra), "sha256": _sha256(extra), "keys": used} if extra is not None else None,
    }
    return arguments, sources


# --------------------------------------------------------------------------------------- oracle

#: Positions sampled from each dispatch's activation for the host-code check, per dispatch.
HOSTIN_SAMPLES = 64


def sample_positions(group: int, elements: int, count: int = HOSTIN_SAMPLES) -> list[int]:
    """``count`` distinct flat positions of an ``elements``-element tensor, a function of ``group``."""
    import numpy as np

    rng = np.random.default_rng(1_000_003 * (int(group) + 1))
    count = min(int(count), int(elements))
    return sorted(int(i) for i in rng.choice(int(elements), size=count, replace=False))


def oracle(
    module, arguments: Sequence[Any], dispatches: Sequence[Dispatch], groups: Sequence[Any], *, group_digest=None
) -> dict[str, Any]:
    """The model evaluated in numpy, with what each device dispatch reads and returns recorded.

    ``module`` is the capture's ORIGINAL module (not the cut one) and ``groups`` its compute groups,
    formed over it. Returns ``{"outputs": [arrays], "dispatch": {group: {...}}}``: per dispatch, the
    activation's sample positions and values and the digests of the activation, the weight operand
    and the exact result -- so a build can check that a package kernel's laid-out weight is the
    operand's own bytes, and a run's result against the same numbers.
    """
    import numpy as np

    from merlin.runtime import linalg_numpy as LN

    roots = {id(g.root): g for g in groups if g.root is not None}
    wanted = {d.group for d in dispatches}
    rows: dict[int, dict[str, Any]] = {}

    def digest(value) -> str:
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

    def observe(op, _index, results, operands):
        group = roots.get(id(op))
        if group is None or int(group.index) not in wanted:
            return
        lhs, rhs = (np.asarray(v) for v in operands[:2])
        positions = sample_positions(group.index, lhs.size)
        rows[int(group.index)] = {
            "lhs_sha256": digest(lhs),
            "rhs_sha256": digest(rhs),
            "result_sha256": digest(results[0]),
            # The chained oracle digests of the result, in the program's own digest (information: a
            # dispatch is graded locally), so the verdict can name every group's oracle basis.
            "sum": int(np.asarray(results[0]).astype(np.int64).sum()),
            "fnv1a": int(group_digest(np.asarray(results[0]).reshape(-1))) if group_digest else None,
            "sample_indices": positions,
            "sample_values": [int(v) for v in lhs.reshape(-1)[positions]],
        }

    outputs = LN.evaluate(module, arguments, observe=observe)
    missing = sorted(wanted - set(rows))
    if missing:
        raise OpenModelError(f"the oracle never evaluated the root of group(s) {missing[:8]}")
    return {"outputs": [np.asarray(o) for o in outputs], "dispatch": rows}


# ---------------------------------------------------------------------------------------- grade


def _floats(bits: Sequence[int]):
    import struct

    import numpy as np

    return np.array([struct.unpack("<f", struct.pack("<I", int(b) & 0xFFFFFFFF))[0] for b in bits], np.float32)


def _words(uart: str) -> dict[str, tuple[str, str]]:
    found: dict[str, tuple[str, str]] = {}
    for line in uart.splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[0] == "GM_WORDS":
            fields = dict(p.split("=", 1) for p in parts if "=" in p)
            found[parts[1]] = (fields.get("bytes", ""), fields.get("digest", ""))
    return found


def words_bridge(graded_uart: str, timed_uart: str) -> dict[str, Any]:
    """Whether a timing run wrote, dispatch by dispatch, the bytes a locally graded run wrote.

    Both builds print a digest of every dispatch's result after the window. Equal digests for every
    dispatch are what let a cycle count taken without checks stand on the grade of the build that
    had them; a dispatch missing from either side is a disagreement, never a pass."""
    graded, timed = _words(graded_uart), _words(timed_uart)
    groups = sorted(set(graded) | set(timed), key=lambda g: (len(g), g))
    differ = [g for g in groups if graded.get(g) is None or graded.get(g) != timed.get(g)]
    return {
        "dispatches": len(groups),
        "agree": len(groups) - len(differ),
        "differ": differ,
        "bridged": not differ and bool(groups),
    }


def grade(
    uart: str,
    expectations: Mapping[str, Any],
    *,
    reference: Mapping[str, Any] | None = None,
    declaration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A run's console against the build's expectations. Read structurally (whitespace and ``=``).

    TWO gates, each named in the verdict, and a dispatch with no line is ABSENT, never agreed:

    * ``local`` -- every device dispatch is exact on the inputs it actually read
      (``GM_LOCAL <g> mismatches=0``, the target's projection check);
    * ``end_result`` -- the output, and every dispatch's result digest, against the model's REFERENCE
      ARM (``reference``, a :mod:`.whole_model_reference` entry) under the model's declared criterion,
      bit-identical by default. With no reference the run is not quotable.

    Reported, never gated: ``hostin`` (the host code's activation at sampled positions against the
    oracle's -- a chained check that dynamic quantization's rounding makes drift by design) and
    ``output`` (the output against the oracle under the capsule's policy, and the oracle's own standing
    against the golden; no two faithful computations of such a model agree within that policy).
    """
    import numpy as np

    local: dict[str, dict[str, str]] = {}
    hostin: dict[str, dict[str, str]] = {}
    out_bits: list[int] | None = None
    stated: tuple[int, int] | None = None
    for line in uart.splitlines():
        parts = line.split()
        if not parts:
            continue
        fields = dict(p.split("=", 1) for p in parts if "=" in p)
        if parts[0] == "GM_LOCAL" and len(parts) > 1:
            local[parts[1]] = fields
        elif parts[0] == "GM_HOSTIN" and len(parts) > 1:
            hostin[parts[1]] = fields
        elif parts[0] == "OUT" and len(parts) > 1 and parts[1].isdigit():
            out_bits = [int(v) for v in parts[2 : 2 + int(parts[1])]]
        elif parts[0] == "GM_OUTPUT" and {"within", "of"} <= set(fields):
            stated = (int(fields["within"]), int(fields["of"]))
    groups = [str(g) for g in expectations.get("groups") or ()]
    local_bad = [g for g in groups if (local.get(g) or {}).get("mismatches") != "0"]
    host_bad = [g for g in groups if (hostin.get(g) or {}).get("over") != "0"]
    verdict: dict[str, Any] = {
        "local": {"groups": len(groups), "agree": len(groups) - len(local_bad), "disagree_or_absent": local_bad},
        "hostin": {
            "groups": len(groups),
            "agree": len(groups) - len(host_bad),
            "disagree_or_absent": host_bad,
            "bound": expectations.get("hostin_bound"),
            "worst": max((int((hostin.get(g) or {}).get("max_abs") or 0) for g in groups), default=0),
            "gates": False,
        },
    }
    golden = np.asarray(expectations["golden"], np.float32).reshape(-1)
    oracle_output = np.asarray(expectations["oracle_output"], np.float32).reshape(-1)
    reference_size = oracle_output.size
    policy = expectations.get("numeric_policy") or {}
    atol, rtol = float(policy.get("atol", 0.0)), float(policy.get("rtol", 0.0))

    def within(values, against) -> int:
        return int((np.abs(values - against) <= atol + rtol * np.abs(against)).sum())

    output = _floats(out_bits) if out_bits is not None else None
    if output is None or stated is None:
        verdict["output"] = {"status": "absent"}
    else:
        # REPORTED, NOT GATED: the program's own count over the whole tensor (GM_OUTPUT); the printed prefix is
        # checked against the same oracle here too, and reported against the golden beside it.
        output = output[: oracle_output.size]
        prefix = oracle_output[: output.size]
        verdict["output"] = {
            "status": "graded",
            "policy": {"atol": atol, "rtol": rtol},
            "of": int(oracle_output.size),
            "within_policy_of_oracle": int(stated[0]) if stated[1] == oracle_output.size else -1,
            "printed_prefix": int(output.size),
            "prefix_within_policy_of_oracle": within(output, prefix),
            "max_abs_vs_oracle": float(np.abs(output - prefix).max()),
            "prefix_within_policy_of_golden": within(output, golden[: output.size]),
            "max_abs_vs_golden": float(np.abs(output - golden[: output.size]).max()),
            "cosine_vs_golden": float(
                output
                @ golden[: output.size]
                / (np.linalg.norm(output) * np.linalg.norm(golden[: output.size]) + 1e-30)
            ),
            "oracle_within_policy_of_golden": within(oracle_output, golden),
        }
    from . import whole_model_reference as REF

    if reference is None:
        verdict["end_result"] = {"passed": False, "note": "no reference arm was given, so the end result is not judged"}
    else:
        verdict["end_result"] = REF.judge(uart, reference, declaration=declaration, elements=int(reference_size))
    verdict["quotable"] = not local_bad and bool(verdict["end_result"].get("passed"))
    return verdict


# ---------------------------------------------------------------------------------------- build

#: The passes the dispatch runtime applies before it executes a capture, so the compiled host code
#: computes what the capture's framework computed (a bool cast to 1.0 rather than the IR's signed -1,
#: a half-precision contraction accumulated in f32). The numpy oracle already evaluates with those
#: semantics; the host code has to be given them.
NORMALIZATION = (
    "lower_torchao_affine_quant",
    "collapse_overrank_matmul",
    "lower_quant_ext",
    "lower_bf16_matmul_f32acc",
    "fix_bool_sitofp",
    "fix_bool_fptosi",
)


def normalize(module) -> dict[str, int]:
    """Apply :data:`NORMALIZATION` in order; returns what each pass rewrote."""
    from merlin.llvmlower import passes_xdsl as PX
    from merlin.llvmlower.torchao_affine import lower_torchao_affine_quant

    done: dict[str, int] = {}
    for name in NORMALIZATION:
        fn = lower_torchao_affine_quant if name == "lower_torchao_affine_quant" else getattr(PX, name)
        result = fn(module)
        done[name] = int(result) if isinstance(result, (int, bool)) else int(bool(result))
    return done


def _region_of(group) -> str | None:
    from xdsl.dialects.builtin import StringAttr

    attr = group.root.attributes.get("prov.region_id") if group.root is not None else None
    return attr.data if isinstance(attr, StringAttr) else None


def _cross_flags(recipe_flags: Sequence[str]) -> list[str]:
    """The ISA flags of the target's own harness recipe, as the clang cross flags the lowered model
    object is compiled with: the same ``-march``/``-mabi``/``-mcmodel``, so both halves agree."""
    keep = [f for f in recipe_flags if f.startswith(("-march=", "-mabi=", "-mcmodel="))]
    if not any(f.startswith("-march=") for f in keep):
        raise OpenModelError("the target's harness recipe states no -march; the model object's ISA is unknown")
    return keep


def _check_host_interface(ll_path: Path, dispatches: Sequence[Dispatch]) -> None:
    """Each group body's C interface takes its arguments and then ONE caller-allocated result.

    The lowering turns a public function's tensor result into an out-parameter appended last; the
    target's dispatch calls it that way. Read off the lowered module's own definitions rather than
    assumed, because a lowering that returned the result instead would be called with a stray pointer.
    """
    want = {d.host_symbol: len(d.arguments) + 1 for d in dispatches}
    seen: dict[str, int] = {}
    prefix = "define void @_mlir_ciface_"
    for line in Path(ll_path).read_text(encoding="utf-8").splitlines():
        if not line.startswith(prefix):
            continue
        name, _sep, rest = line[len(prefix) :].partition("(")
        if name in want:
            seen[name] = len([p for p in rest.split(")")[0].split(",") if p.strip()])
    wrong = {name: seen.get(name) for name, arity in want.items() if seen.get(name) != arity}
    if wrong:
        raise OpenModelError(
            f"the lowered group bodies do not take (arguments..., result): {dict(list(wrong.items())[:4])}"
        )


def _c_include_dirs(compiler: str | Path) -> list[str]:
    """The system include directories ``compiler`` searches, read off its own ``-v`` listing -- so a
    second compiler building for the same bare-metal environment sees the same C library headers."""
    import subprocess

    listing = subprocess.run(
        [str(compiler), "-xc", "-E", "-v", "-"], input="", capture_output=True, text=True
    ).stderr.splitlines()
    dirs, inside = [], False
    for line in listing:
        if line.startswith("#include <...> search starts here:"):
            inside = True
        elif line.startswith("End of search list."):
            break
        elif inside and line.strip():
            dirs.append(str(Path(line.strip()).resolve()))
    if not dirs:
        raise OpenModelError(f"{compiler} lists no system include directories")
    # Only the C library's; the compiler's own builtin headers are clang's to supply.
    return [d for d in dirs if "/lib/gcc/" not in d]


def _run(argv: Sequence[Any], *, cwd: Path | None = None, timeout: int | None = None) -> None:
    import subprocess

    done = subprocess.run([str(a) for a in argv], capture_output=True, text=True, cwd=cwd, timeout=timeout)
    if done.returncode != 0:
        raise OpenModelError(f"{Path(str(argv[0])).name} failed: {(done.stderr or done.stdout)[-1500:]}")


def run_concurrently(commands: Sequence[Sequence[Any]], *, jobs: int, timeout: int | None = None) -> None:
    """Run independent compile commands ``jobs`` at a time; the first failure is raised after all end."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=max(1, min(int(jobs), len(commands) or 1))) as pool:
        for future in [pool.submit(_run, argv, timeout=timeout) for argv in commands]:
            future.result()


def _device_part(
    capsule, *, target: str, package_dir, out: Path, timeout: int, jobs: int, decline, binder=None
) -> dict[str, Any]:
    """The model's device part: the package's statement and bindings, and the cut of the normalized module
    into host code plus one dispatch per device group (groups joined to the statement by their region)."""
    from merlin.common import mlir_query as mq
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    from . import whole_model_build as WMB

    # 1. THE DEVICE PART, stated and put to the package through the closed build's own route.
    buffer = WMB.state(
        capsule,
        target=target,
        package_dir=package_dir,
        work=out / "lower",
        timeout=timeout,
        jobs=jobs,
        open_model=True,
        binder=binder,
        prewarm_objects=True,
    )
    rows = WMB.decline_ops(WMB.bind_groups(buffer), decline)
    # THE OBJECTS COMPILE BEHIND THE CUT, the oracle and the host code: none of those reads them, and a
    # package's largest kernel (a 23 MB artifact, minutes of clang) would otherwise hold all of them up.
    # The routes join it (``part["objects"].result()``).
    from concurrent.futures import ThreadPoolExecutor

    objects_runner = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kernel-objects")
    objects = objects_runner.submit(WMB._kernel_objects, rows, target=target, out=out / "objects", jobs=jobs)
    objects_runner.shutdown(wait=False)
    stated = {int(r["group"]): r for r in buffer["whole_program"]["per_group"]}

    # 2. THE CAPTURE AS CAPTURED (the oracle's module), and its groups.
    # By path: the statement's own parse of the capture is reused (it is never mutated here).
    original = mq.parse(Path(capsule.interface))
    original_groups = CG.form_groups(original, target)
    region_index = {_region_of(g): int(g.index) for g in original_groups if int(g.index) in stated}

    # 3. THE CUT, over the normalized module; groups joined to the statement by their region.
    # A private COPY of the shared parse to normalize and cut (a clone prints as a fresh parse does, at a
    # fraction of the cost: 4 s against a minute for SmolVLA's capture).
    module = original.clone()
    normalized = normalize(module)
    device = []
    for group in CG.form_groups(module, target):
        if group.placement == CG.HOST:
            continue
        index = region_index.get(_region_of(group))
        if index is None:
            raise OpenModelError(f"device group {group.index} ({_region_of(group)}) is no group of the statement")
        device.append(dataclasses.replace(group, index=index))
    if sorted(g.index for g in device) != sorted(stated):
        raise OpenModelError("normalizing the module changed which groups the device takes")
    main, host, dispatches = externalize_dispatches(module, device)
    return {
        "buffer": buffer,
        "rows": rows,
        "by_group": {int(r["group"]): r for r in rows},
        "stated": stated,
        "original": original,
        "original_groups": original_groups,
        "normalized": normalized,
        "main": main,
        "host": host,
        "dispatches": dispatches,
        "objects": objects,
    }


def host_code_units(program: Path) -> list[tuple[str, str]]:
    """The host code as independently compiled UNITS: ``[(name, module text), ...]``.

    SEAM. Today the units are the two modules the cut writes -- ``main`` (the forward, with a call per
    dispatch) and ``host`` (the groups' own bodies). A transformer's forward is one function of about
    100k lines that the stock lowering compiles single-threaded for over an hour; splitting it into
    chunks that compile in parallel is a change to THIS function (and to the forward's own emission),
    and nothing downstream needs to know how many units there are.
    """
    return [(name, (program / f"{name}.mlir").read_text(encoding="utf-8")) for name in ("main", "host")]


def lowering_identity(root: Path | None = None, environ: Mapping[str, str] | None = None) -> str:
    """A digest of everything the host-code lowering runs: every source file of ``merlin.llvmlower``
    (the pass pipeline, the runner it writes, every rewrite prelude) and the value of every
    ``MERLIN_*`` switch those files name. A host object is reused only under the same identity, so an
    edited lowering or a flipped switch recompiles instead of serving the object the old one made.
    ``root``/``environ`` default to the installed package and the process environment."""
    import ast
    import os

    if root is None:
        from merlin import llvmlower

        root = Path(llvmlower.__file__).parent
    environ = os.environ if environ is None else environ
    digest, switches = hashlib.sha256(), set()
    for path in sorted(root.rglob("*.py")):
        data = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + data + b"\0")
        for node in ast.walk(ast.parse(data)):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith("MERLIN_"):
                switches.add(node.value)
    for name in sorted(switches):
        digest.update(f"{name}={environ.get(name, '')}\0".encode())
    return digest.hexdigest()


def compile_host_units(
    units: Sequence[tuple[str, str]],
    program: Path,
    *,
    cross: Sequence[str],
    dispatches: Sequence[Dispatch],
    compile_timeout: int,
    jobs: int,
) -> list[Path]:
    """Lower and compile every host-code unit to an object, concurrently, in ``units``' order.

    A unit's object is reused only when the module text, the flags it was compiled with, the
    toolchains and the lowering itself (:func:`lowering_identity`) are those it came from (their
    digest is kept beside it), so a rebuild of an unchanged cut costs nothing. The unit named ``host`` is checked against the dispatches it must define.
    """
    from concurrent.futures import ThreadPoolExecutor

    from merlin.llvmlower import target_data_layout, toolchain
    from merlin.llvmlower.lower import lower_model

    # The target's own layout, asked of the compiler with these cross flags: a host loop over int64 or
    # f64 data is vectorized only when its accesses carry the target's natural alignment.
    layout = target_data_layout.of(toolchain.clang(), cross)
    identity = lowering_identity()

    def one(unit: tuple[str, str]) -> Path:
        name, text = unit
        key = hashlib.sha256(
            "\0".join(
                [
                    text,
                    *map(str, cross),
                    *HOST_CODE_OPTIONS,
                    *sorted(HOST_LOWERING_FEATURES),
                    # What the lowering itself is: a changed pass, prelude or switch under the same
                    # flags is a different object, never a reused one.
                    identity,
                    str(toolchain.clang()),
                    str(toolchain.m2m_python()),
                ]
            ).encode()
        ).hexdigest()
        obj, stamp = program / f"model_{name}.o", program / f"model_{name}.o.key"
        if obj.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8") == key:
            return obj
        lowered = lower_model(
            text,
            program / f"lower_{name}",
            targets=(),
            textual=True,
            features=frozenset(HOST_LOWERING_FEATURES),
            data_layout=layout,
        )
        if name == "host":
            _check_host_interface(lowered.ll_path, dispatches)
        _run([toolchain.clang(), *cross, *HOST_CODE_OPTIONS, "-c", lowered.ll_path, "-o", obj], timeout=compile_timeout)
        stamp.write_text(key, encoding="utf-8")
        return obj

    with ThreadPoolExecutor(max_workers=max(1, min(jobs, len(units)))) as pool:
        return list(pool.map(one, units))


#: An explicit activation arena for the functional simulator's own map (``dram_bytes=None``), in bytes;
#: ``None`` (the default) derives it from the program (:func:`functional_arena_bytes`). Read at build
#: time, so a harness can set it for every build in its process -- the candidate's and the reference
#: arm's, which must link the same map.
FUNCTIONAL_ARENA_BYTES: int | None = None

#: What the arena serves besides the host code's own allocations (nothing in the C runtime allocates
#: today; this is headroom, not a measured need), and the alignment the bump allocator rounds every
#: request up to (``merlin_malloc.c``: ``bump(n, 64)``).
_ARENA_MARGIN_BYTES = 64 << 20
_ARENA_ALIGN = 64


def _dispatch_heap_bound(dispatches: Sequence[Dispatch], row_multiple: int | None) -> tuple[int, list[str]]:
    """``(bytes, unbounded)``: what the C dispatch layer may allocate over one run.

    The dispatch layer is target C the build renders, not lowered host code, so the host code's demand
    does not see it. Per dispatch it may copy each argument into one contiguous block and once more into
    the kernel ABI's row-padded layout, and it allocates the result and the result's padded scratch:
    twice each argument's and the result's padded bytes, plus alignment, bounds it. A dtype or extent
    whose size is unknown is named rather than guessed."""
    from merlin.common.mlir_query import _DTYPE_BYTES

    total, unbounded = 0, []
    for d in dispatches:
        for shape, dtype in (*d.arguments, d.result):
            width = _DTYPE_BYTES.get(dtype)
            if width is None or any(e < 0 for e in shape):
                unbounded.append(f"g{d.group} {list(shape)}x{dtype}")
                continue
            cols = shape[-1] if shape else 1
            rows = 1
            for e in shape[:-1]:
                rows *= e
            pitch = -(-cols // row_multiple) * row_multiple if row_multiple else cols
            total += 2 * (rows * pitch * width + _ARENA_ALIGN)
    return total, unbounded


def functional_arena_bytes(
    program: Path,
    units: Sequence[str] = ("main", "host"),
    *,
    dispatches: Sequence[Dispatch] = (),
    row_multiple: int | None = None,
) -> dict[str, Any]:
    """The functional map's arena, from what one run of the program allocates.

    The bare-metal allocator never frees within a forward, so the arena must hold EVERY allocation the
    program makes in one run: the sum of the host code's ``malloc`` sites, each counted once per call of
    its function (:func:`merlin.llvmlower.arena_bind.heap_demand`, read off the lowered units' LLVM IR),
    plus the alignment each request can waste, plus what the C dispatch layer may allocate
    (:func:`_dispatch_heap_bound`), plus :data:`_ARENA_MARGIN_BYTES`, rounded to a MiB. The
    reference arm lowers the same ``main``/``host`` text, so it derives the same arena.

    Returns ``{"bytes": ..., "basis": ..., "demand": ...}``. Raises :class:`OpenModelError` when the
    program allocates in a loop (or from a function called in one) and no explicit
    :data:`FUNCTIONAL_ARENA_BYTES` is set: no headroom bounds a count nobody knows. A run-time-SIZED
    allocation outside any loop gets headroom under a stated assumption (see ``dynamic_headroom``).
    """
    from merlin.llvmlower.arena_bind import heap_demand

    if FUNCTIONAL_ARENA_BYTES is not None:
        return {"bytes": int(FUNCTIONAL_ARENA_BYTES), "basis": "FUNCTIONAL_ARENA_BYTES", "demand": None}
    texts = []
    for name in units:
        ll = program / f"lower_{name}" / "model.ll"
        if not ll.is_file():
            raise OpenModelError(f"cannot size the arena: the lowered {name!r} unit ({ll}) is missing")
        texts.append(ll.read_text(encoding="utf-8"))
    demand = heap_demand(texts)
    dispatch_bytes, dispatch_unbounded = _dispatch_heap_bound(dispatches, row_multiple)
    # A site that runs an unknown number of times has no bound at all; refuse it.
    looped = {k: v for k, v in demand.unbounded.items() if k != "dynamic_size"}
    if looped or dispatch_unbounded:
        raise OpenModelError(
            f"cannot size the functional arena: the program makes allocations no static sum bounds "
            f"(host code {looped}, dispatches {dispatch_unbounded[:4]}); set "
            "whole_model_open.FUNCTIONAL_ARENA_BYTES for this model"
        )
    # A site that runs once but whose size is computed at run time (a data-dependent extent: SmolVLA's
    # host code sizes one buffer by a sum over a 1024-element vector) is bounded by an ASSUMPTION, stated
    # in the record: no such buffer exceeds the program's largest statically sized one. If it is wrong
    # the bump allocator stops at that allocation and names it, so it cannot pass silently.
    dynamic = int(demand.unbounded.get("dynamic_size", 0))
    headroom = dynamic * (demand.largest + _ARENA_ALIGN)
    if dynamic:
        print(
            f"[arena] {dynamic} run-time-sized allocation(s): {headroom:#x} bytes of headroom, assuming none "
            "exceeds the largest static allocation",
            flush=True,
        )
    megabyte = 1 << 20
    need = demand.bytes + _ARENA_ALIGN * demand.allocations + dispatch_bytes + headroom + _ARENA_MARGIN_BYTES
    return {
        "bytes": (need + megabyte - 1) & ~(megabyte - 1),
        "basis": "heap_demand",
        "demand": demand.to_dict(),
        "dispatch_bound": dispatch_bytes,
        "dynamic_headroom": (
            {
                "sites": dynamic,
                "bytes": headroom,
                "assumption": "no run-time-sized buffer exceeds the program's largest static allocation",
            }
            if dynamic
            else None
        ),
    }


def build(
    package_dir: str | Path | None,
    model_capsule: str | Path,
    *,
    target: str,
    machine: str,
    header: str | Path,
    header_sha256: str | None = None,
    extra: str | Path | None = None,
    out: str | Path,
    verify: str = "local",
    timeout: int = 600,
    jobs: int | None = None,
    decline: Sequence[Any] = (),
    prohibited_roles: Sequence[str] = (),
    dram_base: int = 0x80000000,  # derived-ok: the RISC-V platform DRAM base the bare-metal harness links at
    dram_bytes: int | None = None,
    hostin_bound: int = 1,
    compile_timeout: int = 7200,
    profile: bool = False,
    require_device_readout: bool = True,
    prune: bool = True,
    host_hart: int | None = None,
    phase0_recipe: str | Path | None = None,
    descriptor: str | Path | None = None,
    chunk_ops: int | str | None = None,
) -> dict[str, Any]:
    """Build ``model_capsule`` as one program: its host code and one dispatch per device group.

    ``package_dir`` answers the dispatches it lowers (``None``: the target's library, on the loop-free
    path when ``prohibited_roles`` names a role). ``extra`` supplies the leaf arguments the capsule's
    files do not (see :func:`forward_arguments`). ``verify='local'`` adds each dispatch's projection
    check and the sampled activation check (outside every bracket, inside the whole window -- a timing
    build passes ``'none'``). ``dram_bytes`` packs the image into a board's DRAM; ``None`` uses the
    functional simulator's map. ``hostin_bound`` is the DECLARED bound, in the activation's integer
    steps, of the chained activation check. Returns the build record, also written to
    ``<out>/whole_model_open_build.json``; the expectations a run is graded against are in
    ``<out>/expectations.json``. ``host_hart`` builds a TWO-HART program for a machine whose registry
    entry declares its harts: the host code, compiled for that hart's own ISA (its vector unit, when it
    has one), runs there, and every device dispatch is handed to the unit's hart (see :func:`two_harts`).
    ``phase0_recipe`` / ``descriptor`` name the corpus binding every dispatch put to the package is
    stated under (:func:`merlin.perf.whole_model_build.corpus_binder`); required with a package.

    ``chunk_ops`` splits ``forward``'s one flat body into that many-op-bounded, sequentially-called
    functions before it is lowered (see :func:`chunk_forward`): LLVM's compile cost on the single giant
    function a large open model emits is superlinear in its size. ``None`` (the default) keeps the
    unchunked program byte for byte -- an opt-in path until a real build's gate validates it.
    ``"auto"`` derives the size from the forward itself (:func:`resolve_chunk_ops`); the record's
    ``forward_chunks`` carries the size actually used (``None`` when the forward fit in one chunk).
    """
    import os
    import shutil

    import numpy as np

    from merlin.common import provenance as PROV
    from merlin.llvmlower import c_runtime, toolchain
    from merlin.runtime.backends import base as backends
    from merlin.runtime.backends import spike_model as SM
    from merlin.targetgen import capsule_common as CC

    from . import whole_model_build as WMB

    if verify not in ("local", "none"):
        raise OpenModelError(f"verify is 'local' or 'none', not {verify!r}")
    try:
        resolve_chunk_ops(chunk_ops)  # a misspelled size is refused before anything is built
    except ValueError as exc:
        raise OpenModelError(str(exc)) from exc
    stages = WMB._StageClock()
    capsule = WMB.load_model_capsule(model_capsule)
    abi_header = WMB.machine_header(machine, header, header_sha256)
    harts = two_harts(machine, host_hart) if host_hart is not None else None
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = jobs or min(16, os.cpu_count() or 1)
    driver = backends.whole_model_driver(target)

    # THE MACHINE FIRST. Every dispatch of an open model commits its accumulator, so on a machine whose
    # header states no full-width readout none of them can run on the unit, and the build below would
    # refuse after lowering every group (about an hour for a transformer). Refused here instead, before
    # the package is asked anything, unless the caller asked for exactly that all-host program.
    recipe = WMB._with_header(backends.harness_build_recipe(target), Path(header), out / "harness")
    includes = [f"-I{root}" for root in recipe.include_roots]
    parameter_header = driver.program._parameter_header(None, includes)
    full_width = driver.program.full_width_readout(parameter_header)
    if full_width is not True and require_device_readout:
        raise OpenModelError(
            f"{MACHINE_CANNOT_READ_OUT}: every device group of an open model commits the accumulator, and "
            f"{machine!r}'s parameter header states no full-width accumulator readout, so none of them can "
            f"run on its unit; build for a machine whose header states one"
        )

    # THE ORACLE depends on the model and the target, never on the package: it starts now, in a process
    # of its own (or is read back from its cache), and is joined where the routes first need it.
    from .whole_model_open_oracle import OpenOracleJob

    oracle_job = OpenOracleJob(capsule, target=target, extra=extra)
    try:
        arguments, argument_sources = forward_arguments(capsule, extra=extra)

        # 1-3. THE DEVICE PART, stated and put to the package; the cut over the normalized module.
        # Every dispatch put to the package is stated as the capsule the corpus writes for it.
        binding = WMB.corpus_binder(target, phase0_recipe=phase0_recipe, descriptor=descriptor) if package_dir else None
        with stages("device_part"):
            part = _device_part(
                capsule,
                target=target,
                package_dir=package_dir,
                out=out,
                timeout=timeout,
                jobs=jobs,
                decline=decline,
                binder=binding.binder if binding is not None else None,
            )
        buffer, rows, by_group, stated = part["buffer"], part["rows"], part["by_group"], part["stated"]
        main, host, dispatches, normalized = part["main"], part["host"], part["dispatches"], part["normalized"]
        # 5. THE HOST CODE AND THE GROUPS' OWN BODIES, lowered by the stock whole-model lowering.
        program = out / "program"
        program.mkdir(parents=True, exist_ok=True)
        chunks_made = 0
        chunk_size = resolve_chunk_ops(
            chunk_ops, forward_ops=forward_body_size(main) if chunk_ops is not None else None
        )
        if chunk_size is not None:
            with stages("chunk_forward"):
                chunks_made = chunk_forward(main, chunk_ops=chunk_size)
        main_text = module_text(main)
        if profile:
            # WHERE THE HOST CODE'S CYCLES GO: a mark before every top-level op of the host code (the
            # profiler the whole-model bare-metal path already has), with the id -> op table beside it.
            # It changes the emitted code, so a profiled image is for the split, never for the count.
            from merlin.llvmlower import op_profile
            from merlin.perf.whole_model_chunks import chunk_symbols

            # A chunked forward's own top-level ops are its chunk calls: mark inside them too.
            main_text, table = op_profile.instrument(main_text, ["forward", *chunk_symbols(chunks_made)])
            op_profile.write_table(table, out / "op_profile_table.json")
        (program / "main.mlir").write_text(main_text, encoding="utf-8")
        (program / "host.mlir").write_text(str(host), encoding="utf-8")
        cross = [SM.CLANG_TARGET, *_cross_flags(recipe.cflags), "-O2", "-ffreestanding", "-fno-builtin"]
        if harts is not None:
            # The host code, and only the host code, is compiled for the host hart's own ISA: everything the
            # unit's hart executes (the dispatches, the harness, the kernels) keeps the recipe's.
            cross = [f"-march={harts['host_isa']}" if f.startswith("-march=") else f for f in cross]
        with stages("host_code"):
            objects = compile_host_units(
                host_code_units(program),
                program,
                cross=cross,
                dispatches=dispatches,
                compile_timeout=compile_timeout,
                jobs=jobs,
            )

        # THE JOINS, where the routes first need what ran behind the cut and the host code.
        with stages("objects_join"):
            part["object_dedup"] = part["objects"].result()
        with stages("oracle_join"):
            expected = oracle_job.result([d.group for d in dispatches])
    except BaseException:
        oracle_job.cancel()
        raise

    # 4. ROUTES. The package's kernel where it answered and its operands are the dispatch's own bytes;
    # else the library, on the loop-free path when an instruction role is prohibited; else the host.
    loop_free = None
    library = None
    if prohibited_roles:
        loop_free = driver.program._trap_prohibited_in_library(
            includes, None, sorted(WMB._prohibited(target, prohibited_roles)), out
        )
        library = driver.program.library_path_without_loops(parameter_header.read_text(encoding="utf-8"))
    host_library = getattr(driver.program, "LIBRARY_HOST", None)
    routes: dict[int, dict[str, Any]] = {}
    attribution: list[dict[str, Any]] = []
    for dispatch in dispatches:
        row, statement = by_group[dispatch.group], stated[dispatch.group]
        seen = expected["dispatch"][dispatch.group]
        op = str(statement.get("op"))
        entry = {"group": dispatch.group, "op": op, "region": dispatch.region, "zero_init": dispatch.zero_init}
        route: dict[str, Any]
        if row["on"] == WMB.ON_PACKAGE:
            operands = {str(v): str(k) for k, v in (statement.get("operands") or {}).items()}
            roles = [operands.get(str(a.get("program"))) for a in row["args"]]
            weight = next((a for a in row["args"] if operands.get(str(a.get("program"))) == "rhs"), None)
            if None in roles or any(a.get("gather") for a in row["args"]):
                route = {"on": WMB.ON_HOST, "cause": "kernel_argument_not_a_dispatch_operand", "why": f"roles {roles}"}
            elif weight is not None and weight.get("sha256") not in (None, seen["rhs_sha256"]):
                route = {
                    "on": WMB.ON_HOST,
                    "cause": "device_weight_is_not_the_operand",
                    "why": "the package reads its weight in a layout whose bytes differ from the operand the "
                    "host code hands the dispatch; linking it would need the laid-out copy embedded",
                }
            else:
                route = {"on": WMB.ON_PACKAGE, "symbol": row["symbol"], "object": row["object"], "args": roles}
        else:
            path = (library or {}).get(op, {}) if library else {}
            route = {
                "on": WMB.ON_VENDOR,
                "path": path.get("path") or getattr(driver.program, "LIBRARY_DEFAULT", "WS"),
                "declined_as": row.get("cause"),
                "why": row.get("why"),
            }
            if path.get("host"):
                route.update({"library_path_on_host": True, "library_path_why": path.get("why")})
        accelerated = route["on"] == WMB.ON_PACKAGE or (
            route["on"] == WMB.ON_VENDOR and route.get("path") != host_library
        )
        if accelerated and full_width is not True:
            route = {
                "on": WMB.ON_HOST,
                "cause": driver.program.ACCUMULATOR_READOUT_UNAVAILABLE,
                "why": "the machine's parameter header states no full-width accumulator readout, and this "
                "dispatch commits the accumulator; its own IR computes it on the core",
                "declined": route["on"],
            }
        route["zero_init"] = dispatch.zero_init
        routes[dispatch.group] = route
        entry.update(route)
        if route["on"] == WMB.ON_VENDOR and route.get("path") == host_library:
            # THE LIBRARY'S OWN HOST CODE: the call is the library's, the work is the core's.
            entry.update({"on": WMB.ON_HOST, "cause": "library_loop_free_path", "library_call": True})
        attribution.append(entry)
    stages.mark("routes")

    # 6. THE RUNTIME: the capture's arguments as the generic C runtime binds them.
    bundle = out / "bundle"
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    (bundle / "model.mlir").write_text(str(main), encoding="utf-8")
    os.link(capsule.weights, bundle / "weights.safetensors")
    shutil.copyfile(capsule.weights_manifest, bundle / "weights.safetensors.manifest.json")
    manifest = json.loads(capsule.weights_manifest.read_text(encoding="utf-8"))
    signature_inputs = [
        (str((manifest.get(str(i)) or {}).get("name") or ""), i)
        for i in range(len(arguments))
        if (manifest.get(str(i)) or {}).get("kind") == "input" and not _lifted(manifest.get(str(i)) or {})
    ]
    np.savez(bundle / "inputs.npz", **{f"in{n}": arguments[i] for n, (_name, i) in enumerate(signature_inputs)})
    (bundle / "input_order.json").write_text(json.dumps({name: n for n, (name, _i) in enumerate(signature_inputs)}))
    if argument_sources["extra"] is not None:
        # ONLY THE LEAVES THE MODEL READS, under the capture's own keys: the runtime generator matches
        # lifted constants by sorted position, so an unrelated key in the file would shift them.
        source = np.load(argument_sources["extra"]["path"])
        np.savez(bundle / "extra.npz", **{key: source[key] for key in argument_sources["extra"]["keys"]})
    cgen = program / "cgen"
    info = c_runtime.generate(bundle, cgen, bundle / "inputs.npz", prepared_dir=program)
    # THE CODE RESERVE GROWS WITH WHAT IS LINKED. A package's kernels are its own code, and a
    # fully-scheduled kernel per group is large (measured: 0.4-1.7 MB each, ~150 MB for 302 groups),
    # so the region ahead of the weights is sized by the objects' own bytes -- an upper bound on what
    # they load -- rather than a constant that the first large package overruns at link time.
    linked_code = sum(Path(o).stat().st_size for o in objects) + sum(
        Path(str(r["object"])).stat().st_size for r in routes.values() if r["on"] == WMB.ON_PACKAGE
    )
    code_reserve = SM._CODE_RESERVE_FIXED + int(info.get("static_io_bytes", 0)) + linked_code
    arena = (
        {
            # Everything the board has past the code reserve and the weights, less the alignment the
            # map places each region on (a megabyte per region, two regions).
            "bytes": (int(dram_bytes) - code_reserve - int(info["weights_bytes"])) - (4 << 20) & ~((1 << 20) - 1),
            "basis": "dram_bytes",
        }
        if dram_bytes
        else functional_arena_bytes(
            program,
            [name for name, _text in host_code_units(program)],
            dispatches=dispatches,
            row_multiple=WMB.pointee_row_padding(target)["multiple"],
        )
    )
    arena_bytes = int(arena["bytes"])
    layout = SM._layout(
        arena_bytes,
        int(info["weights_bytes"]),
        dram_base=int(dram_base),
        dram_bytes=int(dram_bytes) if dram_bytes else None,
        code_reserve=code_reserve,
    )

    stages.mark("runtime")

    # 7. THE TARGET'S C: every dispatch's call, and the program's main with its protocol lines.
    samples = {
        d.group: {
            "indices": expected["dispatch"][d.group]["sample_indices"],
            "values": expected["dispatch"][d.group]["sample_values"],
            "bound": int(hostin_bound),
        }
        for d in dispatches
    }
    rendered = driver.dispatch.render_dispatch(
        [{**d.to_dict(), "op": stated[d.group].get("op")} for d in dispatches],
        routes,
        uart=driver.program.UART,
        verify=verify,
        samples=samples,
        # A digest of every dispatch's result bytes, taken after the window in EVERY build: it is what
        # shows a timing build (no checks) wrote the same bytes as the locally graded build.
        words_helper=getattr(driver.program, "_WORDS_HELPER", ""),
        # The pitch the kernel ABI pads every pointee row to: a package kernel reads and writes its
        # operands at it, and a dense row that is not a whole number of tile edges is not that layout.
        row_padding=WMB.pointee_row_padding(target)["multiple"],
        **({"unit_rpc": True} if harts is not None else {}),
    )
    # THE TARGET'S WORD IS THE LAST: a route its driver cannot write (a library path with no
    # full-width result) is the host's, with the driver's cause, and the attribution says so.
    final = {int(row["group"]): row for row in rendered["census"]}
    for entry in attribution:
        written = final[int(entry["group"])]
        if written["on"] != routes[int(entry["group"])]["on"]:
            entry.update(
                {
                    "on": written["on"],
                    "cause": written.get("cause"),
                    "why": written.get("why"),
                    "declined": written.get("declined"),
                }
            )
            entry.pop("library_call", None)
    (program / "dispatch.c").write_text(rendered["source"], encoding="utf-8")
    policy = CC.load_capsule(capsule.directory).get("numeric_policy") or {}
    (program / "main.c").write_text(
        driver.dispatch.render_main(
            driver.program.UART,
            rendered["census"],
            profile=profile,
            # THE END RESULT IS THE ORACLE'S OUTPUT UNDER THE CAPSULE'S OWN POLICY, stated by the program.
            reference=np.asarray(expected["outputs"][0], np.float32).reshape(-1).tolist(),
            atol=float(policy.get("atol", 0.0)),
            rtol=float(policy.get("rtol", 0.0)),
            words_helper=getattr(driver.program, "_WORDS_HELPER", ""),
            **({"host_hart": harts["host"], "unit_hart": harts["unit"]} if harts is not None else {}),
        ),
        encoding="utf-8",
    )

    # 8. COMPILE AND LINK, with the target's own compiler for C and the harness's layout.
    harness = SM._harness_dir()
    runtime = SM._c_runtime_dir()
    from merlin.common.paths import runtime_dir

    gcc_flags = [*_cross_flags(recipe.cflags), "-O2", "-ffreestanding", "-fno-builtin"]
    address = [
        f"-DMERLIN_ARENA_BASE_ADDR={hex(layout['arena_base'])}ULL",
        f"-DMERLIN_ARENA_SIZE_BYTES={hex(arena_bytes)}ULL",
        f"-DMERLIN_WEIGHTS_BASE_ADDR={hex(layout['weights_base'])}ULL",
    ]
    base_includes = ["-I", runtime, "-I", cgen, "-I", harness]
    units = {
        "model_call.o": (cgen / "model_call.c", base_includes),
        "merlin_model.o": (runtime / "merlin_model.c", base_includes),
        "main.o": (program / "main.c", [*base_includes, *address]),
        "crt.o": (harness / "crt.S", []),
        "console.o": (harness / "htif.c", []),
        # Word-wide memcpy/memset: the host code's bufferized copies are its own, and byte loops made
        # them ~4 instructions a byte (measured on a one-group SmolVLA regime).
        "libc_min.o": (harness / "libc_min.c", ["-DMERLIN_WORD_MEMOPS", "-fno-tree-loop-distribute-patterns"]),
        "printf_min.o": (harness / "printf_min.c", ["-I", harness]),
        "malloc.o": (harness / "merlin_malloc.c", [*address, "-I", harness]),
        "dispatch.o": (program / "dispatch.c", [*includes, "-DBAREMETAL=1", "-Wno-incompatible-pointer-types"]),
    }
    if profile:
        table_size = len(json.loads((out / "op_profile_table.json").read_text(encoding="utf-8"))) + 2
        units["op_prof.o"] = (
            runtime / "merlin_op_prof.c",
            ["-DMERLIN_PROF_BAREMETAL", f"-DMERLIN_PROF_MAX_OPS={table_size}", "-I", harness],
        )
    # THE MLIR RUNTIME IS BUILT BY THE COMPILER THAT BUILT THE MODEL. Its bf16 conversion helpers are
    # called by the lowered model, and the register a bf16 travels in is the CALLER's convention: the
    # clang-lowered model passes and returns it in a float register, a GCC build of the helper returns
    # it in an integer one (merlin/runtime/abi/mlir_runtime.c says so), and every bf16 value the host
    # code narrows then comes back as garbage. Measured: the text model's activations off by ~100 steps.
    runtime_object = program / "mlir_rt.o"
    system_includes = [flag for root in _c_include_dirs(recipe.compiler) for flag in ("-isystem", root)]
    # Every unit is its own compile and none reads another's object: they run at once (a transformer's
    # dispatch.c and the MLIR runtime each take most of a minute), and link in the order listed.
    compiles = [
        [recipe.compiler, *gcc_flags, *extra_flags, "-c", source, "-o", program / name]
        for name, (source, extra_flags) in units.items()
    ]
    compiles.append(
        [toolchain.clang(), *cross, *system_includes, "-c", runtime_dir() / "abi/mlir_runtime.c", "-o", runtime_object]
    )
    stages.mark("render")
    run_concurrently(compiles, jobs=jobs, timeout=compile_timeout)
    stages.mark("c_units")
    objects.extend(program / name for name in units)
    objects.append(runtime_object)
    compiler = Path(recipe.compiler)
    linker = compiler.with_name(compiler.name[: -len("gcc")] + "ld") if compiler.name.endswith("gcc") else None
    if linker is None or not linker.is_file():
        raise OpenModelError(f"no linker beside the target's compiler {compiler} to package the weights blob")
    _run([linker, "-r", "-b", "binary", "-o", program / "weights_blob.o", "weights.bin"], cwd=cgen)
    objects.append(program / "weights_blob.o")
    kernels = [Path(o) for o in rendered["objects"]]
    elf = program / "model.elf"
    _run(
        [
            recipe.compiler,
            *gcc_flags,
            "-nostdlib",
            "-nostartfiles",
            f"-Wl,--defsym,MERLIN_WEIGHTS_BASE={hex(layout['weights_base'])}",
            f"-Wl,--defsym,MERLIN_STACK_BYTES={hex(1 << 22)}",
            # The DRAM span the image was laid out for, stated IN the image: a simulator that runs it
            # reads this rather than defaulting to a span the arena lies past (spike_model.declared_memory).
            f"-Wl,--defsym,{SM.DRAM_BASE_SYMBOL}={hex(int(dram_base))}",
            f"-Wl,--defsym,{SM.DRAM_SPAN_SYMBOL}={hex(int(layout['mem_bytes']))}",
            # How many harts the image runs on, stated IN it, as its memory span is.
            *([f"-Wl,--defsym,{SM.HART_COUNT_SYMBOL}={harts['count']}"] if harts is not None else []),
            "-T",
            harness / "model_link.ld",
            *objects,
            *kernels,
            "-lm",
            "-lgcc",
            "-o",
            elf,
        ],
        timeout=compile_timeout,
    )
    stages.mark("link")

    if harts is not None:
        harts["unit_code_checked"] = len(
            check_unit_hart_code(
                [*objects, *kernels],
                host_objects=[program / "model_main.o", program / "model_host.o", runtime_object],
                unit_isa=harts["unit_isa"],
            )
        )
    isa = None
    if prohibited_roles:
        from .isa_prohibition import check_program

        isa = check_program(
            elf,
            target=target,
            roles=prohibited_roles,
            compiler=recipe.compiler,
            group_objects={str(r["group"]): Path(r["object"]) for r in rows if r.get("object")},
            library_groups=[str(a["group"]) for a in attribution if a["on"] != WMB.ON_PACKAGE],
        )
        if not isa["clean"]:
            raise OpenModelError(f"the linked program issues a prohibited instruction: {isa['summary']}")
    stages.mark("isa_scan")

    golden = np.asarray(next(iter(capsule.outputs.values())), np.float32).reshape(-1)
    declared = CC.load_capsule(capsule.directory)
    expectations = {
        "schema": "whole_model_open_expectations_v1",
        "groups": [d.group for d in dispatches],
        "golden": golden.tolist(),
        "oracle_output": np.asarray(expected["outputs"][0], np.float32).reshape(-1).tolist(),
        "numeric_policy": declared.get("numeric_policy") or {},
        "hostin_bound": int(hostin_bound),
        "dispatch": {str(k): v for k, v in expected["dispatch"].items()},
    }
    (out / "expectations.json").write_text(json.dumps(expectations) + "\n", encoding="utf-8")
    # WHAT THIS PROGRAM'S OUTPUT DEPENDS ON APART FROM WHO ANSWERED ITS GROUPS -- the key its reference
    # arm is cached under (whole_model_reference). The layout's addresses are not in it: they move with
    # the size of the linked kernels and change no number the host code computes.
    from . import whole_model_reference as REF

    host_files = {
        **{f"program/{n}": program / n for n in ("main.mlir", "host.mlir", "main.c", "mlir_rt.o")},
        **{f"program/{p.name}": p for p in (program / "model_main.o", program / "model_host.o")},
        **{f"cgen/{p.name}": p for p in sorted(cgen.iterdir()) if p.is_file() and p.suffix in (".c", ".h")},
        **{f"harness/{n}": harness / n for n in HARNESS_SOURCES},
        "runtime/merlin_model.c": runtime / "merlin_model.c",
        "runtime/abi/mlir_runtime.c": runtime_dir() / "abi/mlir_runtime.c",
    }
    leaves = (argument_sources.get("extra") or {}).get("path")
    reference_identity = {
        "capsule": REF.files_identity(
            {"interface": capsule.interface, "weights": capsule.weights, **({"leaves": leaves} if leaves else {})}
        ),
        "host_code": REF.files_identity(
            host_files,
            [
                *cross,
                *gcc_flags,
                *(["profile"] if profile else []),
                # How the host code was lowered changes its objects as surely as its flags do.
                *(f"lowering:{feature}" for feature in HOST_LOWERING_FEATURES),
                *(f"host_code:{option}" for option in HOST_CODE_OPTIONS),
                # A unit's own flags change its object as surely as its source does.
                *(f"{name}:{flag}" for name, (_source, flags) in sorted(units.items()) for flag in flags),
            ],
            root=out,
        ),
        "toolchain": REF.toolchain_identity(recipe.compiler, toolchain.clang(), gcc_flags),
        "machine": {"machine": machine, "header_sha256": (abi_header or {}).get("sha256") or REF.UNKNOWN},
    }
    counts: dict[str, int] = {}
    for row in attribution:
        counts[row["on"]] = counts.get(row["on"], 0) + 1
    stages.mark("expectations")
    record = {
        "schema": SCHEMA,
        "target": target,
        "machine": machine,
        # Wall seconds per build stage: a slow build is fixed at the stage that is slow.
        "stage_seconds": stages.record(),
        "corpus_binding": binding.record if binding is not None else None,
        "object_dedup": part["object_dedup"],
        "forward_chunks": {
            "chunk_ops": chunk_size,
            "chunks": chunks_made,
            **({"requested": chunk_ops} if chunk_ops != chunk_size else {}),
        },
        "oracle_cache": oracle_job.state,
        "capsule": {
            "name": capsule.name,
            "directory": str(capsule.directory),
            "interface_sha256": _sha256(capsule.interface),
        },
        "package": None
        if package_dir is None
        else {
            "directory": str(Path(package_dir).resolve()),
            "replies": buffer["whole_program"].get("package_replies"),
        },
        "elf": str(elf),
        "elf_sha256": _sha256(elf),
        "verify": verify,
        "profile": profile,
        "arguments": argument_sources,
        "normalization": normalized,
        "layout": {k: (hex(v) if isinstance(v, int) else v) for k, v in layout.items()}
        | {"arena_bytes": hex(arena_bytes), "dram_base": hex(int(dram_base))},
        # Where the arena's size came from: the board's DRAM, or the program's own heap demand.
        "arena": arena,
        "program": {
            "abi_header": abi_header,
            "compiler": str(recipe.compiler),
            "flags": gcc_flags,
            "cross_flags": cross,
            "includes": includes,
            "full_width_readout": full_width,
            "harts": harts,
            "library_paths": library,
            "loop_free_header": loop_free,
            "linked_kernels": [{"path": str(k), "sha256": _sha256(k)} for k in kernels],
        },
        # The kernel objects linked, by content, in the closed build's spelling (isa_prohibition.check_build).
        "linked_objects": [{"path": str(k), "sha256": _sha256(k)} for k in kernels],
        "dispatches": [d.to_dict() for d in dispatches],
        "host_regions": buffer["whole_program"].get("host_regions"),
        "attribution": {"counts": counts, "per_group": attribution},
        "isa_prohibition": isa,
        "expectations": str(out / "expectations.json"),
        "reference_identity": reference_identity,
        "provenance": PROV.record(
            pins=WMB._pins_for(target),
            sources=[capsule.interface],
            artifacts={"elf": elf, "weights": capsule.weights},
        ),
    }
    (out / "whole_model_open_build.json").write_text(json.dumps(record, indent=1, default=str) + "\n", encoding="utf-8")
    if prune:
        record["pruned_bytes"] = prune_intermediates(out)
    return record


def prune_intermediates(out: str | Path) -> int:
    """Remove what a finished open build kept only to link; returns the bytes freed.

    A package build asks 302 groups and compiles each, and its work trees reached ~14 GB per build.
    What stays is what a reader or a later step needs: the ELF and its records, the expectations, and
    each linked kernel object (the prohibition check reads their symbols)."""
    import shutil

    out = Path(out)
    targets = [
        out / "lower",
        out / "bundle",
        out / "harness",
        out / "program" / "lower_main",
        out / "program" / "lower_host",
    ]
    objects = out / "objects"
    if objects.is_dir():
        targets += [d for d in objects.iterdir() if d.is_dir()]
    freed = 0
    for target in targets:
        if target.is_dir():
            freed += sum(f.lstat().st_size for f in target.rglob("*") if f.is_file() and not f.is_symlink())
            shutil.rmtree(target, ignore_errors=True)
    return freed


# The kernel bench and the service half live beside this module; their names stay importable here.
from .whole_model_open_bench import BENCH_SCHEMA, build_kernel_bench, grade_kernel_bench  # noqa: E402,F401
from .whole_model_open_service import (  # noqa: E402,F401
    PRUNABLE,
    is_open_model,
    main,
    run_functional,
    service_build,
)

if __name__ == "__main__":
    raise SystemExit(main())
