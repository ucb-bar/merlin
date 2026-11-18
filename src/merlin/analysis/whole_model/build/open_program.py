"""Externalize an accelerator compute group as a host/device dispatch."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.analysis.whole_model.build.builder import (
    _CHEAP_PRODUCERS,
    _chunk_bounds,
    chunk_forward,
    forward_body_size,
    resolve_chunk_ops,
)
from merlin.analysis.whole_model.build.open_execution import (
    BENCH_SCHEMA,
    PRUNABLE,
    build_kernel_bench,
    grade_kernel_bench,
    is_open_model,
    main,
    run_functional,
    service_build,
)
from merlin.compiler.ir.boundary import free_values

# ---- whole_model_dispatches --------------------------------------------------------------------

#: The symbol stems of the cut: the call the host code makes, and the group's own lowered body.
DISPATCH_PREFIX = "merlin_dispatch_g"
#: The argument attribute one-shot bufferization reads an external function's access to an operand from.
BUFFER_ACCESS = "bufferization.access"
HOST_PREFIX = "merlin_host_g"


class OpenModelError(RuntimeError):
    """The open model cannot be built as one program, and the message says which part stopped it."""


@dataclasses.dataclass(frozen=True)
class Dispatch:
    """One device group, cut out of the host code: its symbols and the values that cross the cut."""

    group: int
    dispatch_symbol: str
    host_symbol: str
    #: ``(shape, dtype)`` of each argument, in call order, and of the committed result.
    arguments: tuple[tuple[tuple[int, ...], str], ...]
    result: tuple[tuple[int, ...], str]
    #: Which argument is the group's root operand ``i`` (``None`` when a root operand is produced
    #: inside the cut), so a caller can say which argument is the activation and which the weight.
    root_operands: tuple[int | None, ...]
    region: str | None = None
    #: The argument the contraction accumulates INTO, and whether the IR states it zero (a splat or
    #: fill of a constant zero): only then is ``lhs @ rhs`` the dispatch's whole answer.
    init_argument: int | None = None
    zero_init: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "dispatch_symbol": self.dispatch_symbol,
            "host_symbol": self.host_symbol,
            "arguments": [{"shape": list(s), "dtype": d} for s, d in self.arguments],
            "result": {"shape": list(self.result[0]), "dtype": self.result[1]},
            "root_operands": list(self.root_operands),
            "region": self.region,
            "init_argument": self.init_argument,
            "zero_init": self.zero_init,
        }


def _shape_dtype(value) -> tuple[tuple[int, ...], str]:
    from merlin.compiler.ir import mlir_query as mq

    shape, dtype = mq.type_shape_dtype(value.type)
    return tuple(int(e) for e in shape), str(dtype)


def _committed_members(group) -> list[Any]:
    """The members up to the value the device COMMITS: a trailing widening cast (an integer result put
    in a float box) stays on the host side of the cut, because the device commits the integers."""
    from merlin.compiler.lower.device.whole_program import _pure_retype

    members = list(group.members)
    retype = _pure_retype(group)
    if retype is not None and members and members[-1] is retype:
        members = members[:-1]
    return members


def externalize_dispatches(
    module, groups: Sequence[Any], *, function: str = "forward"
) -> tuple[Any, Any, list[Dispatch]]:
    """Cut each device group of ``groups`` out of ``module``'s ``function``. Mutates ``module``.

    Returns ``(module, host_module, dispatches)``: ``module`` now calls ``merlin_dispatch_g<i>`` (an
    external declaration with the C interface) where each group's committed value was computed, and
    ``host_module`` defines ``merlin_host_g<i>`` -- the group's own members, with the cheap producers
    they read cloned in -- so any dispatch can be answered by the group's own IR. A group whose
    members are read outside the cut (other than through its committed value) is refused by name:
    cutting it would drop a value the host code still reads.
    """
    from xdsl.dialects import func
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, ModuleOp, StringAttr, UnitAttr
    from xdsl.ir import Block, Region

    from merlin.compiler.partition import outline as OL

    functions = [op for op in module.walk() if op.name == "func.func" and op.body.blocks]
    functions = [f for f in functions if f.sym_name.data == function] or functions[:1]
    if not functions:
        raise OpenModelError(f"the module has no function @{function} with a body")
    block = functions[0].body.blocks[0]
    order = {id(op): index for index, op in enumerate(block.ops)}
    host_functions: list[Any] = []
    dispatches: list[Dispatch] = []
    declarations: list[Any] = []
    for group in groups:
        members = sorted(_committed_members(group), key=lambda m: order.get(id(m), -1))
        if not members or any(id(m) not in order for m in members):
            raise OpenModelError(f"group {group.index} names an operation outside @{function}")
        committed = members[-1].results[0]
        inside = {id(r) for m in members for r in m.results}
        for m in members[:-1]:
            for result in m.results:
                for use in result.uses:
                    if id(use.operation) not in {id(x) for x in members} and use.operation.parent_block() is block:
                        raise OpenModelError(
                            f"group {group.index}: {m.name} is read outside the group, so cutting the group "
                            f"out would drop a value the host code still reads"
                        )
        closure = OL._group_closure(members)
        params = [p for p in free_values(closure) if id(p) not in inside]
        arg_types = [p.type for p in params]
        result_type = committed.type
        index = int(group.index)
        dispatch_symbol, host_symbol = f"{DISPATCH_PREFIX}{index}", f"{HOST_PREFIX}{index}"

        # The group's own body, as a function of the values that cross the cut.
        kblock = Block(arg_types=arg_types)
        mapping = dict(zip(params, kblock.args, strict=True))
        for op in closure:
            clone = op.clone(value_mapper=mapping)
            kblock.add_op(clone)
            for old, new in zip(op.results, clone.results, strict=True):
                mapping[old] = new
        kblock.add_op(func.ReturnOp(mapping[committed]))
        body = func.FuncOp(host_symbol, (arg_types, [result_type]), Region([kblock]))
        body.attributes["llvm.emit_c_interface"] = UnitAttr()
        host_functions.append(body)

        # EVERY ARGUMENT IS ONLY READ: a dispatch writes its result into a fresh buffer it returns, never
        # into an operand (and the group's own body, the host route, copies before any write). Stated, so
        # the bufferization does not copy each operand -- a weight, the zero accumulator -- ahead of every
        # call to protect it from a callee it would otherwise have to assume writes it.
        read = DictionaryAttr({BUFFER_ACCESS: StringAttr("read")})
        declaration = func.FuncOp(
            dispatch_symbol,
            (arg_types, [result_type]),
            Region(),
            visibility="private",
            arg_attrs=ArrayAttr([read for _ in arg_types]),
        )
        declaration.attributes["llvm.emit_c_interface"] = UnitAttr()
        declarations.append(declaration)

        call = func.CallOp(dispatch_symbol, list(params), [result_type])
        block.insert_op_before(call, members[-1])
        committed.replace_all_uses_with(call.results[0])
        for op in reversed(members):
            op.detach()
            op.erase()

        root_operands = []
        for operand in list(group.root.operands)[:2] if group.root is not None else ():
            root_operands.append(next((i for i, p in enumerate(params) if p is operand), None))
        inits = list(getattr(group.root, "outputs", ()) or ()) if group.root is not None else []
        init_argument = next((i for i, p in enumerate(params) if inits and p is inits[0]), None)
        region = group.root.attributes.get("prov.region_id") if group.root is not None else None
        dispatches.append(
            Dispatch(
                group=index,
                dispatch_symbol=dispatch_symbol,
                host_symbol=host_symbol,
                arguments=tuple(_shape_dtype(p) for p in params),
                result=_shape_dtype(committed),
                root_operands=tuple(root_operands),
                region=region.data if isinstance(region, StringAttr) else None,
                init_argument=init_argument,
                zero_init=bool(inits) and _is_zero(inits[0]),
            )
        )
    # AFTER the function: the whole-model lowering attaches its C interface to the first function
    # WITH A BODY, and a declaration placed first is only skipped by that step, never needed first.
    for declaration in declarations:
        module.body.block.add_op(declaration)
    host_module = ModuleOp(host_functions)
    return module, host_module, dispatches


def _is_zero(value) -> bool:
    """Whether ``value`` is a tensor the IR states is all zeros: a splat or a fill of a constant 0,
    through views. Anything else -- including a zero computed at run time -- is not known zero."""
    from merlin.compiler.partition.groups import compute_groups as CG

    for _ in range(8):
        owner = getattr(value, "owner", None)
        if owner is None:
            return False
        if owner.name in ("tensor.splat", "linalg.fill"):
            scalar = getattr(owner.operands[0], "owner", None)
            if scalar is None or scalar.name != "arith.constant":
                return False
            attr = scalar.properties.get("value")
            raw = getattr(getattr(attr, "value", None), "data", None)
            return raw is not None and float(raw) == 0.0
        if owner.name == "arith.constant":
            attr = owner.properties.get("value")
            values = list(attr.get_values()) if hasattr(attr, "get_values") else []
            return bool(values) and all(float(v) == 0.0 for v in values)
        stage = CG.classify(owner)
        if stage is None or stage.kind != CG.VIEW or not owner.operands:
            return False
        value = owner.operands[0]
    return False


# ---- whole_model_open --------------------------------------------------------------------------
# An OPEN model -- host regions between its groups -- as one program: the model's own host code, and
# one dispatch call per device group answered by the package, the target's library, or the host.
#
#     record = build(package_dir, model_capsule, target=t, machine=m, header=h, extra=npz, out=o)
#
# :mod:`merlin.analysis.whole_model.build.stages` builds a CLOSED model: every compute group is a device call and
# the only host work is the input's quantization, so the program is a list of library/kernel calls. A
# transformer is not closed. Its norms, softmax, rotary embedding and per-token dynamic quantization
# compute between the contractions, and on an int8 unit they are host work by the target's own
# declaration (a float datapath the unit does not have). Refusing such a model -- ``model_closure`` --
# is right for a program that only strings group calls together; it is wrong for a program that can
# run the host part too.
#
# THE HOST PART IS THE MODEL'S OWN IR. Nothing here writes a second spelling of a layer norm. The
# capture's module is lowered for the target's host ISA by the stock whole-model lowering
# (:mod:`merlin.compiler.lower.model`), with each device group cut out and replaced by a call to
# ``merlin_dispatch_g<index>`` -- an external function with the C interface the lowering already emits.
# The cut is EXACTLY the group's committed value (the accumulator an int8 contraction commits, before
# the widening cast the capture follows it with), so the host code on either side is unchanged.
#
# EACH DISPATCH HAS THREE ANSWERS, attributed per group, never summed:
#
# * ``package`` -- the package's own kernel for that group, asked through the same capsule route the
#   closed build asks (:func:`merlin.compiler.lower.device.whole_program.whole_program_buffer` with
#   ``open_model=True``) and bound by the same ABI resolution;
# * ``vendor`` -- the target's library call, on the loop-free path the target driver names when an
#   instruction role is prohibited;
# * ``host`` -- the group's own IR, lowered as a function of its own (``merlin_host_g<index>``), which is
#   also the answer when the machine cannot read the group's result out (a full-width accumulator on a
#   narrow-readout machine) -- with that cause.
#
# Which one a group takes is recorded with the reason. The C that makes the call -- the library call,
# the kernel's argument order, the timing brackets -- is the TARGET's (its whole-model driver's
# ``render_dispatch``); this module writes none of it and names no target.
#
# THE ORACLE IS NUMPY. :mod:`merlin.evaluation.model.linalg_semantics` evaluates the capture's module with no
# compiler in it. Every device group is then graded LOCALLY on the device: its exact integer result is
# checked on the core against the inputs it actually read (a projection check, see the driver). The
# host code is graded where it hands data to the device (a sample of each dispatch's input against the
# oracle, within a declared bound) and at the model's output against the capsule's golden under the
# capsule's own numeric policy.

SCHEMA = "whole_model_open_build_v1"


HOST_LOWERING_FEATURES = ("lower_exact_math_inline",)


HOST_CODE_OPTIONS = ("-mllvm", "-enable-loop-versioning-licm")


HARNESS_SOURCES = ("crt.S", "htif.c", "libc_min.c", "printf_min.c", "merlin_malloc.c", "model_link.ld")


MACHINE_CANNOT_READ_OUT = "machine_cannot_read_out_the_models_accumulators"


def two_harts(machine: str, host_hart: int) -> dict[str, Any]:
    """The hart roles of a two-hart program for ``machine``, from its registry entry's declared harts:
    ``{host, unit, host_isa, count}``. Refused when the machine declares no harts, when ``host_hart`` is
    not one of them or is the unit's own, or when not exactly one hart reaches the unit."""
    from merlin.analysis.whole_model.build.products import machine_harts

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
    from merlin.runtime.backends.baremetal.model_build import arch_extensions

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
    from merlin.analysis.whole_model.build.products import machine_harts

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

    from merlin.compiler.lower.c.weights import load_safetensors_header
    from merlin.compiler.lower.model.oracle import parse_forward_signature
    from merlin.development.targetgen.corpus import capsule_io as CC
    from merlin.development.targetgen.grading import golden_inputs as CI
    from merlin.evaluation.model import linalg_semantics as LN

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

    from merlin.evaluation.model import linalg_evaluator as LN

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
    from merlin.analysis.whole_model.build import oracle as REF

    if reference is None:
        verdict["end_result"] = {"passed": False, "note": "no reference arm was given, so the end result is not judged"}
    else:
        verdict["end_result"] = REF.judge(uart, reference, declaration=declaration, elements=int(reference_size))
    verdict["quotable"] = not local_bad and bool(verdict["end_result"].get("passed"))
    return verdict


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
    from merlin.compiler.frontend import preprocess as PX
    from merlin.compiler.quantization.torchao.affine import lower_torchao_affine_quant

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
    from merlin.analysis.whole_model.build import stages as WMB
    from merlin.compiler.ir import mlir_query as mq
    from merlin.compiler.partition.groups import compute_groups as CG
    from merlin.compiler.partition.groups import (
        formation_compute_groups as _merlin_formation_compute_groups_1,
    )

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
    original_groups = _merlin_formation_compute_groups_1.form_groups(original, target)
    region_index = {_region_of(g): int(g.index) for g in original_groups if int(g.index) in stated}

    # 3. THE CUT, over the normalized module; groups joined to the statement by their region.
    # A private COPY of the shared parse to normalize and cut (a clone prints as a fresh parse does, at a
    # fraction of the cost: 4 s against a minute for SmolVLA's capture).
    module = original.clone()
    normalized = normalize(module)
    device = []
    for group in _merlin_formation_compute_groups_1.form_groups(module, target):
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
    """A digest of everything the host-code lowering runs: every source file of ``merlin.compiler.lower.model``
    (the pass pipeline, the runner it writes, every rewrite prelude) and the value of every
    ``MERLIN_*`` switch those files name. A host object is reused only under the same identity, so an
    edited lowering or a flipped switch recompiles instead of serving the object the old one made.
    ``root``/``environ`` default to the installed package and the process environment."""
    import ast
    import os

    if root is None:
        from merlin.compiler.lower import model as llvmlower

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

    from merlin.compiler.lower.model import target_data_layout, toolchain
    from merlin.compiler.lower.model.lower import lower_model

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


FUNCTIONAL_ARENA_BYTES: int | None = None


_ARENA_MARGIN_BYTES = 64 << 20


_ARENA_ALIGN = 64


def _dispatch_heap_bound(dispatches: Sequence[Dispatch], row_multiple: int | None) -> tuple[int, list[str]]:
    """``(bytes, unbounded)``: what the C dispatch layer may allocate over one run.

    The dispatch layer is target C the build renders, not lowered host code, so the host code's demand
    does not see it. Per dispatch it may copy each argument into one contiguous block and once more into
    the kernel ABI's row-padded layout, and it allocates the result and the result's padded scratch:
    twice each argument's and the result's padded bytes, plus alignment, bounds it. A dtype or extent
    whose size is unknown is named rather than guessed."""
    from merlin.compiler.ir.mlir_query import _DTYPE_BYTES

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
    its function (:func:`merlin.compiler.lower.host.arena.heap_demand`, read off the lowered units' LLVM IR),
    plus the alignment each request can waste, plus what the C dispatch layer may allocate
    (:func:`_dispatch_heap_bound`), plus :data:`_ARENA_MARGIN_BYTES`, rounded to a MiB. The
    reference arm lowers the same ``main``/``host`` text, so it derives the same arena.

    Returns ``{"bytes": ..., "basis": ..., "demand": ...}``. Raises :class:`OpenModelError` when the
    program allocates in a loop (or from a function called in one) and no explicit
    :data:`FUNCTIONAL_ARENA_BYTES` is set: no headroom bounds a count nobody knows. A run-time-SIZED
    allocation outside any loop gets headroom under a stated assumption (see ``dynamic_headroom``).
    """
    from merlin.compiler.lower.host.arena import heap_demand

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


__all__ = [
    "DISPATCH_PREFIX",
    "Dispatch",
    "HOST_PREFIX",
    "OpenModelError",
    "externalize_dispatches",
]
