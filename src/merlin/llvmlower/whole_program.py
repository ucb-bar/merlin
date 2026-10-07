"""A captured model as ONE command buffer: the program its compute groups are.

The device route (:mod:`.device_offload`, :mod:`.group_offload`) turns each closed compute group
into one call and states it as the entry a backend builds from. That is a program per LAYER. This is
the step after: the same groups, in the model's own order, as a single command buffer over named
buffers -- the model's inputs and weights in, one command per group, the intermediates between them,
the model's result out.

**Why this exists at all.** Everything that asks "can this compiler emit THIS MODEL" reads a command
buffer, not a module: the phase-2 analysis hands a whole ``linalg-on-tensors`` capsule to a package
and reads back one buffer. A per-layer answer cannot be handed to it, and a buffer built from only
the layers that happened to route would be an INCOMPLETE program presented as a whole one -- which is
worse than a refusal, because a consumer cannot tell.

**So closure is a precondition, not an outcome.** A model is CLOSED when its only host region is the
quantization of a model ARGUMENT onto the integer grid: the one conversion a device program is
allowed to leave outside itself, because the caller hands the model a float image and something has
to put it on the grid. Any other host region computes BETWEEN the groups, and then "the groups" is
not the model. That rule is not restated here -- it is
:func:`merlin.xdsl_dialects.lowering.model_closure.require_closed`, the same predicate the
group-model harness refuses on, because two spellings of "closed" would drift and the second one
would silently accept a different set of models.

Nothing here knows a target: the device is a parameter, the dtypes are read off the module's own
types, and the accumulator format comes from the device's derived datapath facts.
"""

from __future__ import annotations

import functools
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = [
    "SCHEMA",
    "attribution",
    "input_permutation",
    "WholeProgramError",
    "whole_program_buffer",
]

#: ABI version of the command buffer this emits. Read from the shared contract rather than spelled,
#: so a buffer cannot claim a version the grammar module does not implement.
SCHEMA = "whole_program_v1"

#: entry ``op`` -> the command-buffer opcode that states it. Both vocabularies are declared: the left
#: is what :mod:`~merlin.xdsl_dialects.lowering.group_command` states a group as, the right is the
#: enum ``merlin/contract/schemas/command_buffer.schema.json`` admits. An op in neither is refused by
#: name rather than mapped to the nearest thing, which would emit a command computing something else.
_OPCODE_OF_OP = {
    "matmul": "MATMUL",
    "conv2d": "CONV2D",
    "residual_add": "RESIDUAL_ADD",
}


#: How this program's one command for a group was obtained. Named because they are DIFFERENT CLAIMS:
#: one is the submission's own lowering of that layer, the other is this repo's ABI-level statement of
#: it. A buffer that mixed them while reporting only "71 commands" would say nothing about which of a
#: model's layers the compiler under test actually lowered -- the same reason
#: `group_model_program.py`'s schedule census records `on: vendor` against `on: sched` per group.
FROM_SUBMISSION = "submission"
FROM_REFERENCE = "reference"


from .whole_program_splice import (  # noqa: E402  (re-exported: every cause token is part of this API)
    CALLER_DECLINED,
    CAPSULE_FAILED,
    MALFORMED,
    NOT_ASKED,
    PACKAGE_DECLINED,
    PACKAGE_FAILED,
    _is_declined,
    _refuse,
    _same_bytes,
    _spliced,
)
from .whole_program_splice import (
    DTYPE_MISMATCH as DTYPE_MISMATCH,
)
from .whole_program_splice import (
    NO_COMMANDS as NO_COMMANDS,
)
from .whole_program_splice import (
    NO_SHAPE as NO_SHAPE,
)
from .whole_program_splice import (
    ROLE_AMBIGUOUS as ROLE_AMBIGUOUS,
)
from .whole_program_splice import (
    ROLE_UNBOUND as ROLE_UNBOUND,
)
from .whole_program_splice import (
    _decline_key as _decline_key,
)


def input_permutation(quantize, rank: int) -> list[int]:
    """The axis order carrying the model argument ``quantize`` writes into the layout a command reads.

    A rank-4 activation is declared ``[batch, spatial..., features]`` because that is the only layout
    CONV2D defines; where the capture keeps its features is read off the capture itself
    (:func:`group_demand.feature_axis`), never assumed. Any other rank is declared as captured. ONE
    function for it, used by this buffer's declaration and by every caller that lays the argument out,
    so the two cannot disagree about which axis moved.
    """
    from merlin.xdsl_dialects.lowering import group_demand as GD

    if rank != 4:
        return list(range(rank))
    axis = GD.feature_axis(quantize, 4)
    return [0, *(i for i in range(4) if i not in (0, axis)), axis]


class WholeProgramError(RuntimeError):
    """This model cannot be stated as one command buffer, and the message says which part."""


def _dtype_of(value) -> str:
    from merlin.common import mlir_query as mq

    _shape, dtype = mq.type_shape_dtype(value.type)
    return str(dtype)


def _shape_of(value) -> list[int]:
    from merlin.common import mlir_query as mq

    shape, _dtype = mq.type_shape_dtype(value.type)
    return [int(extent) for extent in shape]


def _accumulator_of(device: str, operand: str) -> str:
    """What ``device`` accumulates ``operand`` x ``operand`` into, from its own datapath facts.

    Fails closed in both directions, exactly as :func:`.device_build._accum_from_facts` does: no
    matching triple means the device has not said what it accumulates this pair into, and more than
    one means it has said several -- and a group that leaves as the accumulator would then be
    labelled with a precision chosen by dictionary order.
    """
    from merlin.system.offload import device_dtype_triples
    from merlin.targetgen.routing import _fmt_ok  # noqa: PLC2701 -- one format-equality predicate

    found = {
        acc for lhs, rhs, acc in device_dtype_triples(device) if _fmt_ok(operand, (lhs,)) and _fmt_ok(operand, (rhs,))
    }
    if len(found) != 1:
        raise WholeProgramError(
            f"{device!r} declares {len(found)} accumulate format(s) for {operand} x {operand} "
            f"({sorted(found) or 'none'}); a group that leaves as the accumulator cannot be typed"
        )
    return str(found.pop())


def _integer_source(operand) -> Any | None:
    """The integer value behind a root operand: past its dequantize, or the operand itself.

    A capture that fake-quantizes reaches its contraction through a dequantize, and the integers the
    device reads are that dequantize's input. A capture that is ALREADY integer (M2's matmuls take
    ``i8`` directly) has no dequantize to look through, and the operand is the integer. Requiring the
    dequantize refuses the second kind for the shape of its provenance rather than its arithmetic.
    ``None`` when the operand reaches the root as a float with nothing behind it, which an integer
    datapath genuinely cannot read.
    """
    from xdsl.dialects.builtin import IntegerType

    from merlin.xdsl_dialects.lowering import compute_groups as CG

    _adapters, dequantize, _dtype = CG._input_chain(operand)
    value = dequantize.operands[0] if dequantize is not None else operand
    element = getattr(value.type, "element_type", None)
    return value if isinstance(element, IntegerType) else None


def _stored_argument(value) -> Any:
    """The model argument ``value`` is a relayout of, looking back through views and movement only;
    ``value`` itself when there is none (the caller then refuses it)."""
    from xdsl.ir import BlockArgument

    from merlin.xdsl_dialects.lowering import compute_groups as CG

    current = value
    for _ in range(16):
        if isinstance(current, BlockArgument):
            return current
        owner = getattr(current, "owner", None)
        stage = CG.classify(owner) if owner is not None else None
        if owner is None or stage is None or stage.kind not in (CG.VIEW, CG.MOVEMENT) or not owner.operands:
            return value
        current = owner.operands[0]
    return value


def _dequantizes(group) -> bool:
    """Whether any member of ``group`` dequantizes (carries a scale into the float domain)."""
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    return any((CG.classify(m) is not None and CG.classify(m).kind == CG.DEQUANTIZE) for m in group.members)


def _pure_retype(group) -> Any | None:
    """The group's closing member iff it only WIDENS what the group commits, else ``None``.

    Structural, and every clause is load-bearing. The member must be the group's last; its body must
    be exactly one conversion and a yield, with no constant in it (a constant would be a scale, and a
    scale changes the numbers); and the conversion must go from an INTEGER element to a FLOAT one,
    read off the types rather than off an operation's spelling. What is left is a signed integer put
    in a wider box: the values are unchanged, so a readout that commits those integers computes the
    same function, and a program that says so is not dropping work.

    ``None`` for anything else, including a cast that carries a constant -- which is a scaled
    dequantize wearing a cast's name, and committing the integers for THAT would be wrong by the
    scale with nothing to point at.
    """
    from xdsl.dialects.builtin import IntegerType

    from merlin.xdsl_dialects.lowering import compute_groups as CG

    member = group.members[-1] if group.members else None
    stage = CG.classify(member) if member is not None else None
    if member is None or stage is None or stage.kind != CG.CAST:
        return None
    body = CG._body_ops(member)
    if len(body) != 2 or CG._body_constants(member):
        return None
    if not member.operands or not member.results:
        return None
    source, sink = member.operands[0], member.results[0]
    source_element = getattr(source.type, "element_type", None)
    sink_element = getattr(sink.type, "element_type", None)
    if not isinstance(source_element, IntegerType) or isinstance(sink_element, IntegerType):
        return None
    return member


#: A group whose permuted device weight the prepack does not supply cannot be spliced.
NO_PREPACK = "prepack_supplies_no_device_weight"


def _prepack_rows(groups, capture) -> tuple[dict[int, Mapping[str, Any]], str]:
    """``{group index: its prepack row}``, or the reason there are none.

    Computed before the walk because the declaration depends on it: a unit holds a contraction's
    weight as ``[K, N]`` and a convolution's as ``[tap_h, tap_w, channel]`` by output, which is a
    PERMUTATION of the capture's ``[cout, cin, kh, kw]`` -- not a reshape, and not something this
    buffer may perform by declaring a different shape over the same bytes.
    """
    if capture is None:
        return {}, "no capture was given, so no weights manifest could be read and no weight laid out"
    beside = Path(capture)
    beside = beside if beside.is_dir() else beside.parent
    manifest = next(iter(sorted(beside.glob("*.manifest.json"))), None)
    weights = next(iter(sorted(beside.glob("*.safetensors"))), None)
    if manifest is None or weights is None:
        return {}, f"no weights manifest and safetensors pair beside {beside}"
    from merlin.xdsl_dialects.lowering import group_prepack as GP

    try:
        record = GP.prepack(list(groups), manifest, weights, device_layout=True)["record"]
    except Exception as error:  # noqa: BLE001 -- named, never a silently absent prepack
        return {}, f"prepack: {type(error).__name__}: {error}"
    return {int(row["group"]): row for row in (record.get("groups") or []) if "group" in row}, ""


def _device_weight(row, name: str) -> tuple[dict[str, Any] | None, str]:
    """What supplies this group's weight in device layout, or why nothing does.

    The layout comes from `group_prepack.device_weight`, which states its own ``column_order`` and
    digests the bytes it produced. Both travel into the buffer: a reader can see WHICH permutation
    was applied and to which bytes, rather than being told that a differently-shaped tensor is
    somehow the same one. A group the prepack refused is refused here, by name and with its reason.
    """
    if not isinstance(row, Mapping):
        return None, "the prepack holds no row for this group, so nothing supplies its device weight"
    weight = row.get("weight")
    if not isinstance(weight, Mapping) or "array" not in weight:
        refused = weight.get("refused") if isinstance(weight, Mapping) else None
        return None, f"the prepack supplies no device weight{f': {refused}' if refused else ''}"
    return {
        "tensor": name,
        "from": "prepack",
        "array": weight.get("array"),
        "shape": [int(v) for v in (weight.get("shape") or [])],
        "column_order": weight.get("column_order"),
        "transposed": weight.get("transposed"),
        "sha256": weight.get("sha256"),
        "stored_tensor": weight.get("stored_tensor"),
    }, ""


# ------------------------------------------------------------------------ the one group capsule path
#
# SEAM (port W1/W2). A group put to a package is stated as the capsule the Phase 0 corpus writes for
# it: ONE grouping (``compute_groups`` + ``group_command``) and ONE binding (``corpus_spec.derive_binding``
# with the accumulator taken from the target's own facts), reached through
# ``merlin.targetgen.group_capsule_entries.group_binding`` / ``interface_capsule``. The statement never
# builds a binding of its own -- a per-call ``_mesh_tile_binding(device, dtype, "int32")`` was a second
# binding free to disagree with the corpus, and a literal accumulator. Until that API is present in
# this checkout, a group is refused by name (``interface_capsule_failed``) rather than asked under a
# substitute binding.


class UnboundGroupCapsule(RuntimeError):
    """No binding was supplied, so no group can be stated as the capsule the corpus would write."""


def group_binder(te, datapath: Mapping[str, Any] | None, *, contract=None, facts=None):
    """``operand_dtype -> CorpusBinding``, the corpus's own binding for a group of that operand format.

    Built ONCE per program (each operand format is derived once and reused), from the explicitly
    loaded recipe ``datapath`` block and the target experiment ``te``; never discovered here.
    """
    cache: dict[str, Any] = {}

    def binder(operand_dtype: str):
        key = str(operand_dtype or "")
        if key not in cache:
            from merlin.targetgen.group_capsule_entries import group_binding

            cache[key] = group_binding(te, datapath, operand_dtype=key or None, contract=contract, facts=facts)
        return cache[key]

    return binder


def group_interface(entry: Mapping[str, Any], binder, *, source_reference: str) -> str:
    """The interface MLIR a package is asked to lower for one stated group, through the one path."""
    if binder is None:
        raise UnboundGroupCapsule(
            "no group binding was supplied (whole_program_buffer(binder=...)); a group is stated only "
            "under the corpus binding, never under one assumed here"
        )
    from merlin.targetgen.group_capsule_entries import interface_capsule

    stated = {**dict(entry), "source_reference": str(entry.get("source_reference") or source_reference)}
    _capsule, iface = interface_capsule(stated, binder(str(entry.get("operand_dtype") or "")))
    return iface


def _asked_orientation(entry: Mapping[str, Any], group: Any, operands: Mapping[str, str], tensors: Mapping[str, Any]):
    """The entry in the orientation this program HOLDS the group's operands -- the one a package has to
    be asked for if its kernel is to bind (:func:`~merlin.xdsl_dialects.lowering.group_command.device_orientation`,
    the corpus's own rule, never a second derivation of it).

    CHECKED, NEVER ASSUMED: the result is used only when its operand extents are exactly the shapes this
    program declares for the command's two operands; otherwise the stated entry is returned and the
    splice refuses by shape, as it would have. What binding ACCEPTS does not change: a kernel declaring
    the transposed tensors is still refused.
    """
    if getattr(group, "window_mean", None) is None or "lhs" not in operands or "rhs" not in operands:
        return dict(entry)
    from merlin.xdsl_dialects.lowering import group_command as GC

    oriented = GC.device_orientation(entry, {"stored_operand": None})
    try:
        want = ([int(oriented["M"]), int(oriented["K"])], [int(oriented["K"]), int(oriented["N"])])
    except (KeyError, TypeError, ValueError):
        return dict(entry)
    held = tuple(list((tensors.get(operands[key]) or {}).get("shape") or []) for key in ("lhs", "rhs"))
    return oriented if held == want else dict(entry)


def _ask_package(
    package,
    entry,
    operands,
    device,
    work,
    tag,
    *,
    timeout,
    shapes=None,
    run=None,
    views=None,
    record=None,
    pinned=None,
    binder=None,
    memo=None,
):
    """Put ONE group to the submission as a merlin_iface v0.1 capsule and splice what it returns.

    The capsule is the one the corpus would write for this group (:func:`group_interface`, under
    ``binder``'s binding): a group asked of a package and a group certified by a capsule are the
    same bytes, so a model build cannot ask for something no capsule ever tested.

    Every failure is a per-group fallback with a reason, never an exception: a group the package
    cannot lower is a fact about that layer, and losing the other seventy over it would tell nobody
    anything. The roles the capsule declares are what bind its tensors to this program's buffers, so
    a package that names its operands differently still splices -- what has to agree is the ROLE, not
    the spelling -- except for a NEGOTIATED group, whose tensors are ``pinned`` by the names the
    offer declared (see :func:`_spliced`), because two of its data operands share a role.
    """
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import oot_runner as OR
    from merlin.xdsl_dialects.lowering import group_command as GC

    bound: dict[str, tuple[str, ...]] = {}
    if pinned is None:
        bound["output"] = (operands["dst"],)
        # AN ACTIVATION IN THE STATIONARY SLOT binds where a weight would: the capsule states the
        # contraction's right operand in the weight role (the operand the unit holds), and the kernel's
        # argument for it is whatever buffer the program hands it at run time. This is checked FIRST and
        # independently of any declared ``shapes`` role, because a stationary activation's own role is
        # not (and must not be) "weight" -- it is a runtime tensor, not a constant -- yet the kernel still
        # expects it positionally where a weight would sit.
        stationary = GC.stationary_is_activation(entry)
        if "rhs" in operands and stationary:
            bound["input"], bound["weight"] = (operands["lhs"],), (operands["rhs"],)
        else:
            # A CONSTANT OPERAND BINDS BY ITS OWN DECLARED ROLE, never by whether this group also has a
            # bias. The role every constant (a stored conv/matmul weight, or a synthesized reduction
            # constant like a window-mean's ones) is declared under is already recorded in ``shapes`` at
            # the point it was produced -- gating the split on bias presence instead was a coincidence
            # that held only because every bias-bearing group in this model happens to have a stored
            # weight too: a constant operand with no bias (a window-mean's ones, or any future reduction
            # against a synthesized scale) fell into the same "input" role as the real activation, and a
            # package that declared both distinctly by shape still lost the tie to `bound["input"]`'s
            # positional pairing when the counts happened to agree, or was refused outright when they did
            # not -- `group 70 is a window mean whose constant operand nothing supplies` was this, not a
            # missing package kernel: the package's own kernel was never told which of its two operands
            # the program considers a weight.
            weight_keys = [
                key
                for key in ("lhs", "rhs")
                if key in operands and str(((shapes or {}).get(operands[key]) or {}).get("role")) == "weight"
            ]
            input_keys = [key for key in ("lhs", "rhs") if key in operands and key not in weight_keys]
            if weight_keys:
                bound["weight"] = tuple(operands[key] for key in weight_keys)
            if input_keys:
                bound["input"] = tuple(operands[key] for key in input_keys)
        if "bias" in operands:
            bound["bias"] = (operands["bias"],)
    try:
        iface = group_interface(entry, binder, source_reference=f"compute group {tag} of a captured model")
    except Exception as error:  # noqa: BLE001 -- recorded per group, never swallowed
        return _refuse(CAPSULE_FAILED, f"interface capsule: {type(error).__name__}: {error}")
    source = Path(work) / f"{tag}.iface.mlir"
    source.write_text(iface, encoding="utf-8")
    if record is not None:
        # The interface IS what the package was told: its scales, epilogue and operand order. Kept by
        # path so a check of "what did we pass" reads the bytes that were passed.
        record["interface"] = str(source)
    # THE SAME CALL THE CAPSULE ROUTE MAKES. A capsule, a kernel and a model group are three
    # spellings of one job, and this used to be the second implementation of it: one entrypoint
    # chosen by an inline feature-detection, no `parse`, no `lower_interface_to_target`, and no
    # contract validation of the buffer that came back -- so a group could splice a buffer the
    # grader would have refused, and a package declaring no analysis bundle produced no target
    # artifact at all for the whole-model arm that reads one. `capsule_common.lower_interface` is
    # now the one walk, and `capsule_common.run_entrypoints` (the grader's own front half) is the
    # capsule route's call into it.
    #
    # THE CALLER STILL DECIDES HOW THE PACKAGE IS INVOKED. An analysis worker runs a submission
    # inside a host-created sandbox and must keep doing so here; a plain compile has no sandbox to
    # keep. Only the invocation is injected -- the sequence is the shared one.
    try:
        emitted, _artifact = CC.lower_interface(
            package,
            source,
            Path(work) / f"{tag}.generated",
            contract=None,
            timeout=timeout,
            invoke=run or OR.run_entrypoint,
            artifact_name=f"{tag}.artifact.txt",
            memo=memo,
        )
    except OR.BackendDeclined as declined:
        # A ZERO EXIT IS NOT AN ANSWER, and reading it as one measured 71 of 71: the entrypoint
        # reports success and puts the refusal in the buffer, under the ABI's own `declined` key.
        return _refuse(PACKAGE_DECLINED, f"the package declined this group: {str(declined)[:200]}")
    except OR.CertFailure as failure:
        # The contract's own plane names WHICH kind of failure this was. A buffer that parsed and
        # violated the schema is a malformed answer; everything else is the invocation failing.
        cause = MALFORMED if str(getattr(failure, "plane", "")) == "command_buffer_schema" else PACKAGE_FAILED
        return _refuse(cause, f"the package did not lower this group: {str(failure)[:300]}")
    except Exception as error:  # noqa: BLE001
        return _refuse(PACKAGE_FAILED, f"the package did not run: {type(error).__name__}: {error}")
    # The whole-model arm reads one artifact per group beside the work dir, in the model's order.
    (Path(work) / f"{tag}.artifact.txt").write_text(_artifact, encoding="utf-8")
    if record is not None:
        record["command_buffer"] = str(Path(work) / f"{tag}.generated" / "command_buffer.json")
        record["artifact"] = str(Path(work) / f"{tag}.artifact.txt")
    return _spliced(emitted, bound, tag, shapes, views, record, pinned)


def _abi_opcodes() -> Mapping[str, Any]:
    """The ``opcodes`` block of the command-buffer ABI, parsed once per distinct file content.

    Read on every call it was the single largest cost of stating a model: two YAML parses per group
    and per opcode, 456 of them for one ResNet-50, which was 12 of the 21 seconds the statement took.
    Keyed on the file's bytes rather than cached forever, so an edited contract is read again.
    """
    from merlin.common.paths import contract_dir

    text = (contract_dir() / "command_buffer_abi.yaml").read_text(encoding="utf-8")
    return _parse_abi_opcodes(text)


@functools.lru_cache(maxsize=4)
def _parse_abi_opcodes(text: str) -> Mapping[str, Any]:
    import yaml

    return (yaml.safe_load(text) or {}).get("opcodes") or {}


def _declared_operand_names(opcode: str) -> tuple[str, ...]:
    """The operand keys the command-buffer ABI declares for ``opcode``, in its own order.

    READ FROM THE CONTRACT, never spelled here. A buffer whose CONV2D carried ``lhs``/``rhs``
    instead of the declared ``ifm``/``weight`` validated against the JSON schema -- which
    constrains structure, not per-opcode operand names -- and was then unreadable by every
    consumer that follows the ABI, the independent reference recomputation included. The schema
    said yes and the contract said no, which is the shape of a check that cannot fail.
    """
    block = _abi_opcodes().get(str(opcode)) or {}
    return tuple(block.get("operands") or {})


def _as_declared(opcode: str, operands: Mapping[str, str]) -> dict[str, str]:
    """``operands`` under the names the ABI declares for ``opcode``.

    The statement is built with generic roles -- a contraction has a left and a right operand
    whatever the opcode calls them -- and renamed once, here, against the contract. A name the
    contract does not declare and this cannot place is left alone rather than dropped: an operand
    nobody can name is a refusal downstream, never a silently absent pointer.
    """
    names = _declared_operand_names(opcode)
    if not names:
        return dict(operands)
    # `lhs`/`rhs` are this module's spelling for "the activation" and "the stationary operand".
    # Where the contract names them otherwise (CONV2D's `ifm`/`weight`), take the contract's.
    renamed: dict[str, str] = {}
    positional = [name for name in names if name != "dst" and name != "bias"]
    generic = [key for key in ("lhs", "rhs") if key in operands]
    for key, value in operands.items():
        if key in names:
            renamed[key] = value
        elif key in generic and len(generic) <= len(positional):
            renamed[positional[generic.index(key)]] = value
        else:
            renamed[key] = value
    return renamed


#: Entry keys that describe the GEOMETRY a command states through declared attributes instead. They
#: are consumed by `_as_declared_attributes` and must not survive into the command: an engine that
#: refuses an attribute it does not implement (as the reference recomputation does, on purpose) is
#: the only reason this was ever noticed.
_GEOMETRY_KEYS = ("N", "ci", "Himg", "Wimg", "kh", "kw", "op", "operand_dtype", "scale_granularity")


def _as_declared_attributes(opcode: str, entry: Mapping[str, Any], attributes: dict[str, Any]) -> dict[str, Any]:
    """``attributes`` in the vocabulary the command-buffer ABI declares for ``opcode``.

    A contraction's entry names its extents the way a generator asks for them (``kh``, ``ci``,
    ``Himg``); a CONV2D command declares one ``kernel`` of ``[kh, kw, ci, co]`` and takes the image
    extents from the tensor it reads. Emitting the entry verbatim produced a buffer the JSON schema
    accepted -- it constrains structure, not per-opcode attribute names -- and that no consumer
    following the ABI could execute. Only CONV2D differs today; every other opcode's entry already
    speaks the declared vocabulary and is passed through untouched.
    """
    if opcode != "CONV2D":
        return attributes
    stated = dict(attributes)
    for key in _GEOMETRY_KEYS:
        stated.pop(key, None)
    stated["kernel"] = [int(entry["kh"]), int(entry["kw"]), int(entry["ci"]), int(entry["N"])]
    stated["layout"] = "nhwc"  # the only layout the contract defines; anything else it rejects by name
    stated.setdefault("stride", [1, 1])
    stated.setdefault("padding", [0, 0, 0, 0])
    stated.setdefault("dilation", [1, 1])
    # `pool_in_dims` is the conv's OWN output extent, which the contract cross-checks against the
    # geometry it derives from kernel/stride/padding -- a disagreement is rejected, not reconciled.
    if "maxpool" in (stated.get("epilogue") or []):
        out_h = (
            int(entry["Himg"])
            + int(stated["padding"][0])
            + int(stated["padding"][2])
            - (int(stated["dilation"][0]) * (int(entry["kh"]) - 1) + 1)
        ) // int(stated["stride"][0]) + 1
        out_w = (
            int(entry["Wimg"])
            + int(stated["padding"][1])
            + int(stated["padding"][3])
            - (int(stated["dilation"][1]) * (int(entry["kw"]) - 1) + 1)
        ) // int(stated["stride"][1]) + 1
        stated.setdefault("pool_in_dims", [out_h, out_w])
    return stated


def _declared_attribute_names(opcode: str) -> frozenset[str]:
    """The attribute keys the command-buffer ABI declares for ``opcode``."""
    block = _abi_opcodes().get(str(opcode)) or {}
    return frozenset(block.get("attributes") or {})


def _readout_pair(
    opcode: str,
    operands: Mapping[str, str],
    attributes: Mapping[str, Any],
    accumulator: str,
    shape: list[int],
    acc_dtype: str,
) -> tuple[list[dict], dict] | None:
    """A contraction and the COMMIT that reads it out, or ``None`` when the opcode needs no pair.

    MATMUL declares NO attributes: the contraction leaves its product in the accumulator and a
    separate COMMIT applies the readout. Folding the epilogue onto the contraction produced a
    command the JSON schema accepted -- it does not constrain attributes per opcode -- and that no
    engine following the ABI would apply, so the readout silently did not happen and the
    accumulator's integers were read as if they were the committed output.

    CONV2D is the stated exception: its contract says in as many words that there is no separate
    COMMIT and the epilogue happens on the command. RESIDUAL_ADD declares its own scales and
    epilogue too. Both are left alone.
    """
    if opcode not in ("MATMUL", "MATMUL_RESIDENT"):
        return None
    # EVERY OPERAND THE OPCODE DECLARES, OR A REFUSAL BY NAME. Selecting the ones that happen to be
    # present lets a contraction missing its stationary operand through as a command with one input,
    # and the engine that reads it fails somewhere else entirely -- on a KeyError naming the operand
    # rather than the group that lacked it. A window mean is exactly this case: its stationary
    # operand is a constant one, not a model argument, so a positional filter eats it silently.
    missing = [name for name in _declared_operand_names(opcode) if name != "dst" and name not in operands]
    if missing:
        raise WholeProgramError(
            f"a {opcode} states no {', '.join(missing)} operand, which its opcode declares; "
            "a contraction missing an operand is refused rather than emitted with the rest"
        )
    contraction = {
        "opcode": opcode,
        "operands": {key: operands[key] for key in _declared_operand_names(opcode) if key != "dst"}
        | {"dst": accumulator},
        "attributes": {},
    }
    allowed = _declared_attribute_names("COMMIT")
    commit_operands = {"src": accumulator, "dst": operands["dst"]}
    if "bias" in operands:
        commit_operands["bias"] = operands["bias"]
    commit = {
        "opcode": "COMMIT",
        "operands": commit_operands,
        "attributes": {key: value for key, value in attributes.items() if key in allowed},
    }
    # The accumulator the contraction writes and the COMMIT reads. Declared because a tensor a
    # command names and nothing declares is a buffer nobody allocated. No explanatory key rides
    # along: the tensor sub-schema admits no extra properties, deliberately, so that a reader is
    # never handed a field it does not know.
    staged = {"shape": list(shape), "dtype": acc_dtype, "role": "intermediate"}
    return [contraction, commit], staged


def whole_program_buffer(
    module,
    device: str,
    *,
    weight_args=None,
    manifest: dict | None = None,
    oracle=None,
    model: str = "",
    package_dir: str | Path | None = None,
    package: Any = None,
    capture: str | Path | None = None,
    workdir: str | Path | None = None,
    timeout: int = 300,
    run=None,
    allow_regions: bool = False,
    decline: Sequence[Any] = (),
    open_model: bool = False,
    binder=None,
    lowering_memo: dict | None = None,
    region_internal_ops: Sequence[str] = (),
) -> dict[str, Any]:
    """``module`` as one command buffer, or raise :class:`WholeProgramError` naming what stopped it.

    ``lowering_memo`` (see :func:`merlin.targetgen.capsule_common.lower_interface`) shares accepted
    lowerings of identical group interfaces across the groups -- and the passes -- of one build.

    ``binder`` (:func:`group_binder`) states every group put to ``package`` as the capsule the corpus
    writes for it; with a package named and no binder, each group is refused by name.

    ``weight_args`` are the model-argument indices a weights manifest calls stored; ``manifest`` is
    that manifest, used only to NAME the weight tensors, so a buffer's operands carry the capture's
    own tensor names instead of positions nobody can check.

    ``decline`` (op names / group indices) routes a group to the reference/library statement WITHOUT
    asking the package at all -- a group the CALLER declined is stated exactly as a group the package
    itself refused would be: it gets real ``FROM_REFERENCE`` commands, its own committed buffer and its
    weights, so it is still checked by ``memory_map`` and the host-side grade downstream. Relabeling a
    row's attribution AFTER this buffer is built (as a post-hoc ``on: vendor`` rewrite would) leaves the
    buffer itself carrying the package's spliced commands and scratch for that group, and the group
    then has neither a package kernel nor a library one -- its output buffer never exists at all.

    ``package_dir`` is what makes this a lowering OF A SUBMISSION rather than of this repo. Each
    group is put to that package as one ``merlin_iface`` v0.1 capsule -- the grammar every backend
    already lowers -- and the commands it returns are spliced into this program in place of the
    reference statement. The grading boundary is then exactly where the group-model harness puts it:
    the harness owns the plumbing (forming the groups, naming the buffers, wiring the dataflow) and
    the submission owns the per-group KERNEL, which is the part phase 1 grades. With no package the
    reference statement stands, which is a different claim and is recorded as one.

    A group the package refuses is recorded BY NAME and falls back to the reference statement rather
    than failing the program: a mixed buffer that says which groups were the submission's is more
    useful than an all-or-nothing refusal, and it is how that harness already reports.

    ``region_internal_ops`` is the TARGET's own statement of which ops it can link as a fused region's
    internal member (its whole-model driver restates them on the core to grade the region's boundary);
    empty -- a target whose driver links no fused region -- and no region is ever offered. A region is
    offered across a boundary only when the producing member's op is one of them and it commits a
    requantized value, never the accumulator.

    ``allow_regions`` OFF BY DEFAULT, and even when on, a region is only ever OFFERED to a package
    that itself declares ``whole_model_regions: true`` in its manifest -- the builder still decides
    WHICH windows are legal (that is never the package's to state), but WHETHER this program's own
    walk ever tries the offer is the package's own opt-in. A package that omits it gets a BYTE-FOR-
    BYTE identical statement to ``allow_regions=False``: ``window_of`` stays empty and every group is
    put to the package exactly as before this existed, because ``allow_regions`` alone is not enough
    to change one line of a submission that never asked for the capability -- a generic package
    whose compiler happens to succeed at lowering a merged, unrequested capsule is not "asking".
    ``open_model`` states a model whose HOST regions compute between its groups (a transformer's
    norms, softmax and dynamic quantization). Such a model is not closed, so its groups are not the
    whole program -- and the buffer does not pretend they are: it states the DEVICE PART only. A
    value a device group reads that no group produced is a HOST-PRODUCED tensor (role ``input``,
    listed under ``whole_program.host_produced`` with the group that reads it), every device group's
    destination is an output the host reads back, and each host region is listed by index with its
    own refusal under ``whole_program.host_regions``. The host part is the model's own IR, compiled
    by the program that embeds this buffer; nothing here computes it.
    """
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC
    from merlin.xdsl_dialects.lowering import group_numerics as GN
    from merlin.xdsl_dialects.lowering import model_closure as MC

    groups = CG.form_groups(module, device, oracle=oracle)
    if not open_model:
        MC.require_closed(groups, error=WholeProgramError)
    host_produced: dict[str, dict[str, Any]] = {}
    host_regions: list[dict[str, Any]] = []

    work = Path(workdir) if workdir is not None else None
    if package is None and package_dir is not None:
        from merlin.targetgen import oot_runner as OR

        package = OR.load_package(str(package_dir))

    # WHICH RUNS OF GROUPS A PACKAGE MAY BE OFFERED TOGETHER, from the statement's own dataflow alone
    # -- computed once, before anyone is asked anything, so the offer never depends on how any one
    # package answers it. Gated on the PACKAGE'S OWN declared opt-in (never inferred from whether its
    # compiler happens to succeed): without it, this is empty and nothing below ever differs from
    # ``allow_regions=False``.
    wants_regions = bool(
        allow_regions and region_internal_ops and package is not None and package.manifest.get("whole_model_regions")
    )
    window_of: dict[int, tuple[int, int]] = {}
    if wants_regions:
        from merlin.llvmlower import region_legality as RL

        for start, end in RL.legal_regions(groups):
            for index in range(start, end + 1):
                window_of[index] = (start, end)

    if package is not None:
        work = work or Path(tempfile.mkdtemp(prefix="whole_program_", dir=os.environ.get("TMPDIR") or None))
        work.mkdir(parents=True, exist_ok=True)
    provenance: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    prepacked: dict[str, dict[str, Any]] = {}
    # THE DEVICE LAYOUT IS A PROPERTY OF THE TARGET, NOT OF WHO ANSWERS THE GROUP. This was skipped
    # whenever no package was named, on the reasoning that the prepack exists to serve a splice. It
    # does not: a weight the capture holds as [cout, cin, kh, kw] has to be declared as the
    # [kh*kw*ci, cout] the command reads whether the group is answered by a submission or stated by
    # the reference. Skipping it produced a reference buffer whose every convolution declared a
    # weight in a layout its own opcode does not define, and the independent recomputation refused
    # it -- correctly. The REFUSAL stays conditional (a missing prepack only blocks a splice); the
    # declaration does not.
    prepack_rows, why_no_prepack = (
        ({}, "") if capture is None else _prepack_rows([g for g in groups if g.placement != CG.HOST], capture)
    )
    # The module's contraction extents, read once for every group stated below.
    from merlin.xdsl_dialects.lowering.group_prepack import module_extents

    extents = module_extents([g for g in groups if g.placement != CG.HOST])
    result: str | None = None
    reshaped: dict[str, list[int]] = {}  # tensor -> the one shape a consumer re-declared it in

    tensors: dict[str, dict[str, Any]] = {}
    commands: list[dict[str, Any]] = []
    buffer_of: dict[int, str] = {}  # id(SSA value) -> the tensor name holding it
    args: list[dict[str, str]] = []
    readout: dict[str, Any] | None = None
    entry_domain: dict[str, Any] | None = None
    names = {
        int(key): str(entry.get("weight") or entry.get("name") or f"arg{key}")
        for key, entry in (manifest or {}).items()
        if str(key).isdigit() and isinstance(entry, dict)
    }

    def declare(name: str, value, role: str, dtype: str | None = None, shape: list[int] | None = None) -> str:
        tensors[name] = {
            "shape": list(shape) if shape else _shape_of(value),
            "dtype": dtype or _dtype_of(value),
            "role": role,
        }
        return name

    def produce(value, name: str, role: str, dtype: str | None = None, shape: list[int] | None = None) -> str:
        buffer_of[id(value)] = declare(name, value, role, dtype, shape)
        return name

    def consume(value, *, why: str) -> str:
        found = buffer_of.get(id(value))
        if found is None and open_model:
            # A VALUE THE HOST PRODUCED. Declared once, in the capture's own shape and type, and
            # named for the order it was first read in -- position is all the host code and this
            # buffer share, and the reader is recorded beside it.
            name = f"H_{len(host_produced)}"
            buffer_of[id(value)] = declare(name, value, "input")
            args.append({"tensor": name, "access": "read"})
            host_produced[name] = {"first_reader": why, "shape": _shape_of(value), "dtype": _dtype_of(value)}
            return name
        if found is None:
            raise WholeProgramError(f"{why} reads a value no group of this model produced")
        return found

    def argument(value, *, role: str) -> str:
        """A model argument as a named tensor, declared the first time it is read."""
        from xdsl.ir import BlockArgument

        if not isinstance(value, BlockArgument) and open_model:
            # THROUGH VIEWS AND MOVEMENT ONLY, to the stored argument the operand is a relayout of --
            # the relayout the prepack states and the device weight carries. Anything else is not a
            # stored operand and stays a refusal.
            value = _stored_argument(value)
        if not isinstance(value, BlockArgument):
            raise WholeProgramError("a stored operand of this model is not a model argument")
        index = int(value.index)
        name = names.get(index, f"arg{index}")
        if id(value) not in buffer_of:
            buffer_of[id(value)] = declare(name, value, role)
            args.append({"tensor": name, "access": "read"})
        return buffer_of[id(value)]

    def emit_single(ctx: dict[str, Any]) -> None:
        """One group, put to the package alone (or stated by the reference) -- today's whole path,
        factored out so a region that a package declines falls back to exactly this, member by
        member, rather than a second copy of it."""
        tag, entry, operands = ctx["tag"], ctx["entry"], ctx["operands"]
        opcode, output_dtype = ctx["opcode"], ctx["output_dtype"]
        spliced, scratch, why, cause = _refuse(
            NOT_ASKED, "no package was named, so nothing was asked to lower this group"
        )
        asked: dict[str, Any] = {}
        # DECLINE IS CHECKED FIRST, AHEAD OF EVEN A WEIGHT-PREPACK REFUSAL: a caller-declined group is
        # never asked at all, so whether the package COULD have been given a correctly device-laid-out
        # weight is moot for it -- that concern exists only to gate an ask that is not going to happen.
        # THE PACKAGE IS NEVER ASKED. Asking it and then discarding a real reply (a post-hoc
        # attribution rewrite) is what left this group's buffer holding the package's spliced commands
        # and scratch with none of the reference's -- neither a package kernel nor a library one, so
        # its output buffer never existed. Declining HERE, before the ask, is what makes this group's
        # provenance identical in shape to one the package itself refused.
        if package is not None and _is_declined(int(ctx["group"].index), str(entry.get("op")), decline):
            spliced, scratch, why, cause = _refuse(
                CALLER_DECLINED,
                f"the caller routed group {ctx['group'].index} ({entry.get('op')}) to the "
                "target's library (decline=); the package's kernel for it was not linked",
            )
        elif ctx["weight_refusal"]:
            spliced, scratch, why, cause = _refuse(NO_PREPACK, ctx["weight_refusal"])
        elif package is not None:
            asked_entry = dict(entry) if "output_dtype" in entry else {**entry, "output_dtype": output_dtype}
            asked_entry = _asked_orientation(asked_entry, ctx["group"], operands, tensors)
            spliced, scratch, why, cause = _ask_package(
                package, asked_entry, operands, device, work, tag,
                timeout=timeout, shapes=tensors, run=run, views=views, record=asked, binder=binder,
                memo=lowering_memo,
            )  # fmt: skip
        stated_row = {"operands": dict(operands), "entry": dict(entry), **({"asked": asked} if asked else {})}
        if spliced:
            tensors.update(scratch)
            commands.extend(spliced)
            provenance.append(
                {
                    "group": ctx["group"].index,
                    "op": str(entry.get("op")),
                    "on": FROM_SUBMISSION,
                    "commands": len(spliced),
                    **stated_row,
                }
            )
        else:
            paired = _readout_pair(
                opcode,
                ctx["reference"]["operands"],
                ctx["reference"]["attributes"],
                f"ACC_{tag}",
                list((tensors.get(operands["dst"]) or {}).get("shape") or []),
                _accumulator_of(device, str(entry.get("operand_dtype") or "")),
            )
            if paired is None:
                emitted = [ctx["reference"]]
            else:
                emitted, staged = paired
                tensors[f"ACC_{tag}"] = staged
            commands.extend(emitted)
            provenance.append(
                {
                    "group": ctx["group"].index,
                    "op": str(entry.get("op")),
                    "on": FROM_REFERENCE,
                    "commands": len(emitted),
                    "why": why,
                    "cause": cause,
                    **stated_row,
                }
            )

    def can_hold(ctx: dict[str, Any]) -> bool:
        """Whether ``ctx`` may be a region's INTERNAL member: an op the target restates, committing a
        requantized value -- the facts that decide where a fused region may cross a boundary."""
        return str(ctx["entry"].get("op")) in set(region_internal_ops) and not ctx["accumulates"]

    def ask_region(window: list[dict[str, Any]]) -> dict[str, Any] | None:
        """``window`` put to the package as ONE fused region; the accepted reply, or None. Nothing of
        this program's own state changes here -- an offer a longer window supersedes leaves no trace."""
        if package is None or any(ctx["weight_refusal"] for ctx in window):
            return None
        if any(_is_declined(int(ctx["group"].index), str(ctx["entry"].get("op")), decline) for ctx in window):
            return None
        if not all(can_hold(ctx) for ctx in window[:-1]):
            return None
        from merlin.llvmlower import region_capsule as RC

        record: dict[str, Any] = {}
        offered_views: list[dict[str, Any]] = []
        members = [
            {
                "tag": ctx["tag"],
                "entry": ctx["entry"],
                "operands": ctx["operands"],
                "opcode": ctx["opcode"],
                "output_dtype": ctx["output_dtype"],
            }
            for ctx in window
        ]
        spliced, scratch, _why, _cause, region_info = RC.ask_package_region(
            package,
            members,
            device,
            work,
            timeout,
            run,
            shapes=tensors,
            views=offered_views,
            record=record,
            binder=binder,
        )
        if not spliced or region_info is None:
            return None
        return {"commands": spliced, "scratch": scratch, "views": offered_views, "record": record, "info": region_info}

    def emit_region(window: list[dict[str, Any]], answer: dict[str, Any]) -> None:
        """An accepted region, into this program: ONE kernel's commands, every member a row of its own.

        The INTERNAL members' rows carry no command (the kernel is the boundary's) and are answered by
        the package as members of the region; the BOUNDARY's row carries all of them and is the one
        this program grades -- its output is the only one anything outside the window reads."""
        tensors.update(answer["scratch"])
        views.extend(answer["views"])
        commands.extend(answer["commands"])
        info = dict(answer["info"])
        info["member_groups"] = [int(ctx["group"].index) for ctx in window]
        for position, ctx in enumerate(window):
            boundary = position == len(window) - 1
            provenance.append(
                {
                    "group": ctx["group"].index,
                    "op": str(ctx["entry"].get("op")),
                    "on": FROM_SUBMISSION,
                    "commands": len(answer["commands"]) if boundary else 0,
                    "operands": dict(ctx["operands"]),
                    "entry": dict(ctx["entry"]),
                    "region": {**info, "role": "boundary" if boundary else "internal"},
                    "asked": answer["record"],
                }
            )

    def emit_window(window: list[dict[str, Any]]) -> None:
        """A maximal legal window, as the longest fused regions the package accepts, left to right.

        From each member, the region is grown one member at a time while the package keeps accepting
        it; the longest accepted region is kept and the walk resumes after it. A member no accepted
        region starts at is asked ALONE, exactly as if its window had never been legal -- a region is
        only ever an ADDITIONAL affordance and never narrows what an honest group-by-group answer does."""
        start = 0
        while start < len(window):
            kept: tuple[int, dict[str, Any]] | None = None
            end = start + 1
            while end < len(window):
                answer = ask_region(window[start : end + 1])
                if answer is None:
                    break
                kept = (end, answer)
                end += 1
            if kept is None:
                emit_single(window[start])
                start += 1
                continue
            emit_region(window[start : kept[0] + 1], kept[1])
            start = kept[0] + 1

    pending: list[dict[str, Any]] = []

    for group in groups:
        tag = f"g{group.index}"
        if group.placement == CG.HOST and open_model and not MC.quantizes_an_input(group):
            host_regions.append(
                {
                    "group": int(group.index),
                    "stages": list(group.stages),
                    "refusal": group.refusal,
                    "why": group.reason,
                    "gap": group.gap or CG.gap_class(group.refusal),
                }
            )
            continue
        if group.placement == CG.HOST:
            # THE ONE PERMITTED HOST REGION: the model's float argument put on the integer grid. It
            # is not a command -- the caller hands the program an argument already on the grid -- so
            # what it contributes is the ENTRY TENSOR and the scale that defines it. Recorded, never
            # dropped: a reader has to be able to see what the program's input domain is.
            quantize = group.members[-1]
            scale = GN._scale_source(quantize)
            source, result = list(quantize.operands)[0], quantize.results[0]
            name = names.get(int(getattr(source, "index", -1)), "input")
            # DECLARED IN THE LAYOUT THE COMMAND DEFINES, NOT THE CAPTURE'S. CONV2D is defined for
            # `nhwc` and for nothing else -- the contract rejects any other layout by name -- while
            # the capture holds the image as [N, C, H, W]. Declaring the capture's shape produced a
            # buffer whose very first command could not read its own input: the channel extent
            # landed where the image extent belongs, and only an engine that checks caught it.
            # The permutation is the CALLER's to perform and it already does (the group-model
            # driver transposes the quantized image before it writes it), so what is fixed here is
            # the statement, which was describing a tensor the program is never handed.
            entry_shape = _shape_of(result)
            device_entry = list(entry_shape)
            permutation = input_permutation(quantize, len(entry_shape))
            device_entry = [int(entry_shape[i]) for i in permutation]
            produce(result, name, "input", None, device_entry)
            args.append({"tensor": name, "access": "read"})
            # The tensor sub-schema admits no extra keys -- deliberately, so a reader cannot be
            # handed a field it does not know -- so the conversion is stated beside the program
            # rather than on the tensor. It is the program's INPUT DOMAIN: what the caller's float
            # argument has to be divided by before the first command reads it.
            entry_domain = {
                "tensor": name,
                "from_dtype": _dtype_of(source),
                "scale": None if scale.value is None else float(scale.value),
                "zero_point": scale.zero_point,
                # WHICH REGION this is and HOW the caller's argument becomes the declared tensor: the
                # capture's extents and the axis order that carries them into the declared layout.
                # Stated so a caller performs the permutation this buffer declares instead of one it
                # assumes (a transpose written as a literal axis tuple is the same fact spelled twice).
                "group": int(group.index),
                "capture_shape": [int(e) for e in entry_shape],
                "permutation": permutation,
            }
            continue

        try:
            stated = GC.program(
                group, weight_args=weight_args, name=f"{model}_{tag}" if model else tag, extents=extents
            )
        except CG.NoCapsuleForm as refusal:
            raise WholeProgramError(f"group {group.index} cannot be stated as a device program: {refusal}") from refusal
        entry = stated.entry
        opcode = _OPCODE_OF_OP.get(str(entry.get("op")))
        if opcode is None:
            raise WholeProgramError(
                f"group {group.index} is stated as {entry.get('op')!r}, which no command-buffer opcode states"
            )

        sink = group.members[-1].results[0]
        operands: dict[str, str] = {}
        bias_divisor: float | None = None
        weight_refusal = ""
        if group.operand_sum is not None:
            # Two activations and no stored side. The operands are the integer tensors behind each
            # summand's own dequantize, in the order the entry states its two scales in.
            for key, operand in zip(("lhs", "rhs"), list(group.root.operands)[:2], strict=False):
                value = _integer_source(operand)
                if value is None:
                    raise WholeProgramError(f"group {group.index} sums a value that reaches it as a float")
                operands[key] = consume(value, why=f"group {group.index}")
        elif group.window_mean is not None:
            # A MEAN OVER A TRAILING WINDOW IS STILL A CONTRACTION, AND ITS STATIONARY OPERAND IS A
            # REAL TENSOR. It is a constant one for every element of the window rather than a model
            # argument, which is why it was left as prose -- but MATMUL declares an `rhs`, and a
            # command missing an operand its opcode declares is not a command. The prepack supplies
            # the array (it states the same constant), so the tensor is declared from the shape the
            # prepack produced rather than from an extent invented here; with no prepack row there
            # is nothing to declare it from and the group is refused by name.
            dequantize = next(m for m in group.members if CG.classify(m).kind == CG.DEQUANTIZE)
            operands["lhs"] = consume(dequantize.operands[0], why=f"group {group.index}")
            ones, why_ones = _device_weight(prepack_rows.get(int(group.index)), f"ONES_{tag}")
            if ones is None or not ones["shape"]:
                raise WholeProgramError(
                    f"group {group.index} is a window mean whose constant operand nothing supplies: {why_ones}"
                )
            # THE CONSTANT GOES ON THE LEFT, AND THAT IS WHAT MAKES IT STATABLE. Summing a window
            # for every channel of a [window, features] activation is `ones[1, window] @ x`, which
            # is a plain contraction over the operand's own bytes. Putting the activation on the
            # left would demand [features, window] -- a TRANSPOSE of what is stored, which the
            # vendor's library takes as a flag and which MATMUL, declaring no attributes at all,
            # cannot express. The two statements compute the same sum; only one of them is sayable
            # here, and choosing the other would have needed a permutation nobody performs.
            window, features = int(group.window_mean["window"]), int(group.window_mean["rows"])
            # THE SHAPE THE BUFFER DECLARES, NOT THE ONE THE CAPTURE'S SSA CARRIES. The command
            # reads a named tensor, and what that name is declared as is what a consumer binds by:
            # the producing group already committed this activation in the [positions, features]
            # plane the device writes, while the capture still calls it [1, C, H, W]. Checking the
            # capture's spelling asks a question about a tensor no command references.
            held = list((tensors.get(operands["lhs"]) or {}).get("shape") or [])
            if held != [window, features]:
                raise WholeProgramError(
                    f"group {group.index} means a window of {window} over {features} features and its "
                    f"operand is {held}; the contraction would have to read it transposed"
                )
            operands["lhs"], operands["rhs"] = f"ONES_{tag}", operands["lhs"]
            tensors[f"ONES_{tag}"] = {
                "shape": [1, window],
                "dtype": str(entry.get("operand_dtype") or "i8").replace("int", "i"),
                "role": "weight",
            }
            prepacked[f"ONES_{tag}"] = dict(ones) | {"shape": [1, window]}
        elif GC.stationary_is_activation(entry):
            # TWO ACTIVATIONS, BOTH HANDED OVER AT RUN TIME. The statement read them as lhs[M, K] @
            # rhs[K, N] in the capture's own orientation, so each is the integer value the root reads
            # -- a group's destination or a host-produced tensor -- and nothing is prepacked: there is
            # no stored tensor to lay out, and the stationary operand's bytes exist only per inference.
            for key, operand in zip(("lhs", "rhs"), list(group.root.operands)[:2], strict=True):
                value = _integer_source(operand)
                if value is None:
                    raise WholeProgramError(f"group {group.index} reads its {key} activation as a float")
                operands[key] = consume(value, why=f"group {group.index}")
        else:
            stored = int(stated.stored_operand)
            activation = list(group.root.operands)[1 - stored]
            value = _integer_source(activation)
            if value is None:
                raise WholeProgramError(f"group {group.index} reads its activation as a float")
            operands["lhs"] = consume(value, why=f"group {group.index}")
            weight = list(group.root.operands)[stored]
            operands["rhs"] = argument(_integer_source(weight) or weight, role="weight")
            # THE WEIGHT IS DECLARED IN THE LAYOUT THE UNIT HOLDS IT, and the prepack is what puts it
            # there. [cout, cin, kh, kw] in the capture against [ci*kh*kw, cout] on the device is a
            # PERMUTATION, so re-declaring the capture's own bytes under the device shape would hand
            # the kernel the right size and the wrong order. The layout, the column order that
            # produced it and a digest of the bytes all travel into the buffer, so a reader sees
            # WHICH permutation was applied rather than being told two shapes are the same tensor.
            supplied, why_weight = _device_weight(prepack_rows.get(int(group.index)), operands["rhs"])
            if supplied is not None and supplied["shape"]:
                tensors[operands["rhs"]]["shape"] = list(supplied["shape"])
                prepacked[operands["rhs"]] = supplied
            elif package is not None:
                weight_refusal = f"{why_weight}{f' ({why_no_prepack})' if why_no_prepack else ''}"
            if stated.bias_arg is not None:
                block_args = list(getattr(group.root.parent, "args", ()))
                if int(stated.bias_arg) >= len(block_args):
                    raise WholeProgramError(
                        f"group {group.index} names bias argument {stated.bias_arg}, which the entry does not take"
                    )
                operands["bias"] = argument(block_args[int(stated.bias_arg)], role="bias")

        # WHAT THE GROUP COMMITS. A group that closes on a requantize commits the narrow integer its
        # own quantize produces; one that does not (a model's final classifier) leaves as the
        # ACCUMULATOR, and the dequantize that follows it is the program's readout rather than one of
        # its commands -- the same division the group-model harness makes.
        closes = CG.QUANTIZE in group.stages or "acc_scale" in (entry.get("epilogue") or ())
        retype = _pure_retype(group)
        if retype is not None:
            # A GROUP THAT ONLY WIDENS WHAT IT COMMITS. It commits the integer its contraction
            # produced; the float the capture materializes is the same numbers in a wider box, so the
            # conversion is recorded as a readout of divisor one rather than issued as a command.
            # Saying nothing here would leave the buffer's integers to be read as float bit patterns.
            output_dtype = _dtype_of(retype.operands[0])
            readout = {"tensor": f"B_{tag}", "dequantize": 1.0, "dtype": _dtype_of(sink)}
        elif closes:
            output_dtype = _dtype_of(sink)
            if output_dtype.startswith("f"):
                raise WholeProgramError(f"group {group.index} closes on a requantize but escapes as {output_dtype}")
        elif not _dequantizes(group) and _dtype_of(sink) == _accumulator_of(
            device, str(entry.get("operand_dtype") or "")
        ):
            # THE CAPTURE ITSELF HOLDS THE ACCUMULATOR. An integer contraction whose result the model
            # reads as integers (`torch._int_mm`: attention's scores before their dequantize) has no
            # scale in the group and needs none: the committed integers ARE the capture's tensor, so
            # there is no readout to state and nothing for a bias divisor to divide.
            output_dtype = _dtype_of(sink)
        else:
            output_dtype = _accumulator_of(device, str(entry.get("operand_dtype") or ""))
            sources = [GN._scale_source(m) for m in group.members if CG.classify(m).kind == CG.DEQUANTIZE]
            if len(sources) != 2 or any(s.value is None or s.zero_point for s in sources):
                raise WholeProgramError(
                    f"group {group.index} leaves as the accumulator and its scales are not static, so "
                    "nothing can say what its integers mean"
                )
            divisor = float(sources[0].value) * float(sources[1].value)
            readout = {"tensor": f"B_{tag}", "dequantize": divisor, "dtype": _dtype_of(sink)}
            # THE BIAS OF AN ACCUMULATOR-LEAVING GROUP IS PRE-DIVIDED BY THE SAME DIVISOR, by the
            # same rule a closed group's is folded by its own scales. The buffer names tensors and
            # carries no arrays, so the number goes on the command: whoever materializes the bias has
            # to divide by it, and a bias staged in float units into an integer accumulator is wrong
            # by the whole dynamic range with nothing to point at.
            bias_divisor = divisor

        # AN INTERMEDIATE IS DECLARED IN THE SHAPE THE DEVICE WRITES, not the capture's tensor. The
        # capture holds a convolution's result as [1, 64, 56, 56]; the device commits [3136, 64], and
        # a buffer declared in the first is not the one the next group's kernel reads. A group whose
        # entry states no device output keeps the capture's shape and is refused downstream by name
        # rather than silently declared in a shape nobody computed.
        try:
            device_shape = GC.device_output_shape(entry)
        except GC.NoDeviceShape:
            device_shape = None
        operands["dst"] = produce(sink, f"B_{tag}", "intermediate", output_dtype, device_shape)
        result = operands["dst"]
        attributes = {key: value for key, value in entry.items() if key not in ("name", "kind", "cat", "label")}
        attributes["output_dtype"] = output_dtype
        if bias_divisor is not None and "bias" in operands:
            attributes["bias_divisor"] = bias_divisor
        reference = {
            "opcode": opcode,
            "operands": _as_declared(opcode, operands),
            "attributes": _as_declared_attributes(opcode, entry, attributes),
        }
        if opcode == "CONV2D":
            # A CONVOLUTION'S ACTIVATION IS DECLARED RANK-4, BECAUSE ITS OPCODE DEFINES NO OTHER
            # FORM. The producing group commits the flat [positions, features] the device writes,
            # and those are the same bytes in the same order as [1, H, W, C] -- but the shape a
            # tensor is DECLARED in is what a consumer reads it by, and CONV2D rejects anything
            # that is not rank-4 nhwc. The spatial split the flat form dropped is not guessed: it
            # is stated by this very convolution, which declares the image extents it expects.
            ifm = reference["operands"]["ifm"]
            want = [1, int(entry["Himg"]), int(entry["Wimg"]), int(entry["ci"])]
            held = list((tensors.get(ifm) or {}).get("shape") or [])
            if held and held != want:
                if not _same_bytes(held, want):
                    raise WholeProgramError(
                        f"group {group.index} reads {ifm!r} as {want}, which is not a reshape of the "
                        f"{held} it was committed in: that is a permutation and no view performs it"
                    )
                previous = reshaped.get(ifm)
                if previous is not None and previous != want:
                    raise WholeProgramError(
                        f"{ifm!r} is read as {previous} and as {want}; a tensor has one declared shape"
                    )
                reshaped[ifm] = want
                tensors[ifm]["shape"] = want
                views.append(
                    {
                        "tensor": ifm,
                        "committed": held,
                        "read_as": want,
                        "why": "a convolution reads its activation rank-4 nhwc; the bytes are the same "
                        "and the spatial extents come from the convolution's own declared image",
                    }
                )

        # THE INTERFACE STATES THE WIDTH THIS PROGRAM COMMITS. Left undeclared, the capsule builder
        # derives a width from the epilogue alone -- the narrow readout for any group carrying a
        # bias -- and a group that leaves as the accumulator was then asked for an i8 commit into
        # a buffer this program declares at the accumulator's width. Declared, a width the target
        # cannot read out with those stages is refused by the builder, by name.
        ctx = {
            "tag": tag,
            "group": group,
            "entry": dict(entry),
            "operands": dict(operands),
            "opcode": opcode,
            "output_dtype": output_dtype,
            "weight_refusal": weight_refusal,
            "reference": reference,
            # Whether this group leaves as the ACCUMULATOR rather than a requantized value.
            "accumulates": output_dtype == _accumulator_of(device, str(entry.get("operand_dtype") or "")),
        }
        window = window_of.get(group.index)
        if window is None or window[0] == window[1]:
            emit_single(ctx)
        else:
            pending.append(ctx)
            if group.index == window[1]:
                window_pending = list(pending)
                pending.clear()
                # THE PACKAGE MAY ANSWER ANY RUN OF THE WINDOW AS ONE FUSED REGION, OR NONE: whatever
                # it declines (or cannot be offered -- no package, a member's device weight missing, a
                # caller-declined member, a boundary the target cannot restate) is put to it ONE GROUP
                # AT A TIME, exactly as if the window had never been legal.
                emit_window(window_pending)

    if not commands:
        raise WholeProgramError("this model has no device group, so there is no program to state")

    # THE MODEL'S RESULT IS THE LAST GROUP'S DESTINATION, not the last COMMAND's. A spliced group
    # ends with whatever its package emitted last, which for a residency backend is an `EVICT` of a
    # weight handle and carries no destination at all -- so reading the result off the final command
    # names a scratch buffer, or nothing.
    if result is None:
        raise WholeProgramError("no group produced this model's result")
    outputs = [result]
    if open_model:
        # Every device result is read back by host code, so every one is an output of this program;
        # the model's own result is host code's, and no single group's destination stands for it.
        outputs = [str(row["operands"]["dst"]) for row in provenance]
        readout = None
    for name in outputs:
        tensors[name]["role"] = "output"
        args.append({"tensor": name, "access": "write"})
    buffer: dict[str, Any] = {
        "abi_version": "0.1",
        "target": device,
        "tensors": tensors,
        "commands": commands,
        "outputs": outputs,
        "kernel_abi": {"kind": "whole_program", "args": args, "outputs": outputs},
        "whole_program": {
            "schema": SCHEMA,
            "groups": len(groups),
            "commands": len(commands),
            # A tensor a spliced kernel READS UNDER ANOTHER SHAPE, with the basis it was admitted on.
            # Stated so a reader is never left to infer that two shapes hold the same bytes.
            "views": views,
            # A tensor whose bytes the PREPACK supplies rather than the capture: the weight in the
            # layout the unit holds it, with the column order that produced it and a digest of it.
            "prepacked": prepacked,
            # The conversion the program's caller applies BEFORE the first command: the one host
            # region a closed model keeps. Stated, never implied -- the buffer's first operand is
            # integers, and nothing else says what they are integers OF.
            "input_domain": entry_domain,
            # The float conversion the program leaves to its caller, stated rather than implied: the
            # buffer's integers are not the model's logits until this is applied.
            "readout": readout,
            # WHICH GROUPS THE SUBMISSION LOWERED, and which fell back to this repo's own statement of
            # them. Never summed: they are different claims about different compilers.
            "per_group": provenance,
            "on_submission": sum(1 for row in provenance if row["on"] == FROM_SUBMISSION),
            "on_reference": sum(1 for row in provenance if row["on"] == FROM_REFERENCE),
            **(
                {"open_model": True, "host_produced": host_produced, "host_regions": host_regions} if open_model else {}
            ),
        },
    }
    return buffer


from .whole_program_attribution import SPLIT_UNMEASURED_BEFORE, attribution  # noqa: E402,F401  (re-exported)
