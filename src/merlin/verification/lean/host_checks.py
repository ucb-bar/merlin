"""Fresh host checks of emitted assignments and authored sequential rewrites.

The caller selects original, reviewed semantics. Author requests cannot provide
these arguments through the common broker. Each acceptance reconstructs the
projection and requires the live Lean completion from the same host process.
This is modeled preservation, not a device or asynchronous execution proof.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.verification.lean.checker import _check_generated, accepted_receipt
from merlin.verification.lean.contract import AUTHORED_DOMAINS, SCHEMA, LeanTransformReceipt, digest, document_digest

# ---- lean_checked ------------------------------------------------------------------------------


def _identity(model: str, projection: str) -> str:
    return document_digest(
        {
            "model": model,
            "projection": projection,
            "checker": digest(Path(__file__).read_bytes()),
        }
    )


def _unsupported(before: str, after: str, domain: str, reason: str) -> LeanTransformReceipt:
    from merlin.common.access import scrub_host_paths

    return LeanTransformReceipt(
        SCHEMA,
        "lean_kernel",
        domain,
        digest(before),
        digest(after),
        "",
        "unsupported",
        (),
        (),
        reason=scrub_host_paths(reason),
    )


def _resource_projection(before: str, after: str, model: dict):
    from merlin.compiler.transforms.certificates import parse_ir
    from merlin.verification.lean.resource_geometry import project_resource_geometry

    return project_resource_geometry(parse_ir(before), parse_ir(after), **model)


def verify_resource_assignment(
    before: str, after: str, *, toolchain, deadline: float | None = None, **original_model
) -> LeanTransformReceipt:
    """Check actual emitted addresses against the original host's resource model.

    Geometry arguments are internal host authority, not an author message or an
    annotation's self-declared capacities. The model's physical truth and final
    ISA consumption remain independent checks.
    """
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.verification.lean.resource_geometry import lean_resource_query

    domain = AUTHORED_DOMAINS[6]
    try:
        projection = _resource_projection(before, after, original_model)
        receipt = LeanTransformReceipt(
            SCHEMA,
            "lean_kernel",
            domain,
            digest(before),
            digest(after),
            _identity(projection.semantics_sha256, projection.sha256),
            "unsupported",
            projection.scope,
            (
                "original host representation, capacities, backing stores, units and instruction constraints",
                "actual SSA lifetime and annotation projection is trusted",
                "physical effects, overwrite meanings, ISA consumption and hardware correspondence are separate",
            ),
        )
        query = lean_resource_query(projection)
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ParseError, VerifyException) as exc:
        return _unsupported(before, after, domain, str(exc))
    return _check_generated(query, receipt, toolchain, deadline=deadline)


def accepted_resource_receipt(receipt, before: str, after: str, *, toolchain, **original_model) -> bool:
    """Reproject the original model and current artifacts before accepting evidence."""
    from xdsl.utils.exceptions import ParseError, VerifyException

    if not isinstance(receipt, LeanTransformReceipt) or not accepted_receipt(receipt, before, after):
        return False
    try:
        projection = _resource_projection(before, after, original_model)
        return (
            receipt.domain == AUTHORED_DOMAINS[6]
            and receipt.scope == projection.scope
            and receipt.semantics_sha256 == _identity(projection.semantics_sha256, projection.sha256)
            and receipt.toolchain_sha256 == toolchain.identity()
        )
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ParseError, VerifyException):
        return False


def _effect_projection(before: str, after: str, entry: str, contract):
    from merlin.verification.lean.state_effects import derive_effect_schedule, effect_semantic_identity

    projection = derive_effect_schedule(before, after, entry=entry, contract=contract)
    identity = _identity(effect_semantic_identity(), projection.identity())
    return projection, identity


def _query(statements: list[str], proofs: list[str]) -> str:
    """Balanced obligations avoid depth proportional to the program's operation count."""
    if not statements or len(statements) != len(proofs):
        raise ValueError("state projection has no complete modeled obligations")

    def join(first, limit):
        if limit - first == 1:
            return statements[first], f"effect_{first}"
        middle = (first + limit) // 2
        left, a = join(first, middle)
        right, b = join(middle, limit)
        return f"({left}) ∧ ({right})", f"And.intro ({a}) ({b})"

    declarations = [
        f"theorem effect_{index} : {statement} := {proof}"
        for index, (statement, proof) in enumerate(zip(statements, proofs, strict=True))
    ]
    statement, proof = join(0, len(statements))
    return (
        "import Merlin.StateEffects\nset_option warningAsError true\n"
        + "\n".join(declarations)
        + f"\ntheorem merlin_transform : {statement} := {proof}\n#print axioms merlin_transform\n"
    )


def verify_effect_schedule(
    before: str, after: str, *, entry: str, contract, toolchain, deadline: float | None = None
) -> LeanTransformReceipt:
    """Verify every actual adjacent exchange under immutable sequential semantics."""
    from xdsl.utils.exceptions import ParseError, VerifyException

    domain = AUTHORED_DOMAINS[7]
    try:
        projection, identity = _effect_projection(before, after, entry, contract)
        receipt = LeanTransformReceipt(
            SCHEMA,
            "lean_kernel",
            domain,
            digest(before),
            digest(after),
            identity,
            "unsupported",
            projection.scope,
            projection.assumptions,
        )
        query = _query(*projection.lean_obligations())
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ParseError, VerifyException) as exc:
        return _unsupported(before, after, domain, str(exc))
    return _check_generated(query, receipt, toolchain, deadline=deadline)


def accepted_effect_receipt(receipt, before: str, after: str, *, entry: str, contract, toolchain) -> bool:
    from xdsl.utils.exceptions import ParseError, VerifyException

    if not isinstance(receipt, LeanTransformReceipt) or not accepted_receipt(receipt, before, after):
        return False
    try:
        projection, identity = _effect_projection(before, after, entry, contract)
        return (
            receipt.domain == AUTHORED_DOMAINS[7]
            and receipt.scope == projection.scope
            and receipt.assumptions == projection.assumptions
            and receipt.semantics_sha256 == identity
            and receipt.toolchain_sha256 == toolchain.identity()
        )
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ParseError, VerifyException):
        return False


def verify_authored_transform(
    before: str, after: str, *, entry: str, registry, contract, toolchain, deadline: float | None = None
):
    """Try the fixed scalar domain, then exact sequential reorder correspondence.

    A failed covered proof remains a failure. Unsupported scalar IR may enter
    the separately modeled state domain. This is no general tensor equivalence
    claim, and a refusal does not become evidence through a saved report.
    """
    from merlin.verification.lean.checker import semantic_identity, verify_scalar_transform

    if deadline is None and toolchain is not None:
        deadline = time.monotonic() + toolchain.timeout_s
    scalar = verify_scalar_transform(
        before,
        after,
        entry=entry,
        registry=registry,
        toolchain=toolchain,
        deadline=deadline,
    )
    if scalar.status == "unknown" and toolchain is not None and time.monotonic() < deadline:
        # Host-owned finite tactics may prove a nontrivial exact bit-vector
        # rewrite after simplification fails. They share one original budget;
        # no author-supplied theorem, tactic or tolerance is admitted.
        scalar = verify_scalar_transform(
            before,
            after,
            entry=entry,
            registry=registry,
            toolchain=toolchain,
            tactic="bv_decide",
            deadline=deadline,
        )
    if scalar.status != "unsupported":
        accepted = (
            (
                accepted_receipt(scalar, before, after)
                and scalar.semantics_sha256 == semantic_identity(registry)
                and scalar.toolchain_sha256 == toolchain.identity()
            )
            if scalar.verified
            else False
        )
        return (
            scalar if not scalar.verified or accepted else replace(scalar, status="error", reason="invalid completion")
        )
    return verify_effect_schedule(
        before,
        after,
        entry=entry,
        contract=contract,
        toolchain=toolchain,
        deadline=deadline,
    )


def accepted_authored_receipt(receipt, before: str, after: str, *, entry: str, registry, contract, toolchain) -> bool:
    """Fresh modeled correspondence, fixed semantics and live original completion."""
    from xdsl.utils.exceptions import ParseError, VerifyException

    from merlin.verification.lean.checker import semantic_identity

    if not isinstance(receipt, LeanTransformReceipt) or not accepted_receipt(receipt, before, after):
        return False
    if receipt.domain == AUTHORED_DOMAINS[7]:
        return accepted_effect_receipt(
            receipt,
            before,
            after,
            entry=entry,
            contract=contract,
            toolchain=toolchain,
        )
    if receipt.domain != AUTHORED_DOMAINS[0]:
        return False
    try:
        source, target = export_scalar(before, entry, registry), export_scalar(after, entry, registry)
        return (
            source.inputs == target.inputs
            and source.outputs == target.outputs
            and receipt.semantics_sha256 == semantic_identity(registry)
            and receipt.toolchain_sha256 == toolchain.identity()
            and receipt.scope == ("actual scalar integer SSA input/output equality for every argument bit pattern",)
        )
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ParseError, VerifyException):
        return False

        # Preserve historical object lookup; implementation and completion state live here.


# ---- lean_export -------------------------------------------------------------------------------
# Structural export of a deliberately small, exact integer xDSL domain.
#
# The source and target use identical argument bindings and return identities.
# Unknown operations, poison flags, memory effects and floats abstain. Primitive
# declarations are frozen caller inputs, never inferred from instruction names.


class UnsupportedLeanIR(ValueError):
    pass


@dataclass(frozen=True)
class ScalarFunction:
    name: str
    inputs: tuple[int, ...]
    outputs: tuple[int, ...]
    bindings: tuple[str, ...]
    values: tuple[str, ...]

    def term(self) -> str:
        body = "\n".join(self.bindings)
        returned = "(" + ", ".join(self.values) + ")" if len(self.values) > 1 else self.values[0]
        return (body + "\n" if body else "") + returned


def parse_ir(text: str):
    from xdsl.context import Context
    from xdsl.dialects import arith, builtin, func, memref, scf
    from xdsl.parser import Parser

    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, arith.Arith, func.Func, memref.MemRef, scf.Scf):
        context.load_dialect(dialect)
    module = Parser(context, text).parse_module()
    module.verify()
    return module


def _width(typ) -> int:
    from xdsl.dialects.builtin import IntegerType, Signedness

    if not isinstance(typ, IntegerType) or typ.signedness.data != Signedness.SIGNLESS or not 1 <= typ.width.data <= 256:
        raise UnsupportedLeanIR("formal scalar domain requires integer widths 1..256")
    return typ.width.data


def _no_flags(op) -> None:
    from xdsl.dialects.arith import IntegerOverflowAttr

    flags = op.properties.get("overflowFlags")
    if flags is not None and flags != IntegerOverflowAttr("none"):
        raise UnsupportedLeanIR("poison-generating arithmetic flags are outside the wrapping model")


def export_scalar(text: str, entry: str, registry: Mapping[str, dict] | None = None) -> ScalarFunction:
    from xdsl.dialects import arith, func
    from xdsl.dialects.builtin import IntegerAttr

    module = parse_ir(text)
    functions = {op.sym_name.data: op for op in module.body.block.ops if isinstance(op, func.FuncOp)}
    if len(functions) != 1 or len(tuple(module.body.block.ops)) != 1 or entry not in functions:
        raise UnsupportedLeanIR("formal scalar input requires exactly one defined entry function")
    selected = functions[entry]
    if (
        module.attributes
        or module.properties
        or selected.attributes
        or set(selected.properties) - {"sym_name", "function_type", "sym_visibility"}
    ):
        raise UnsupportedLeanIR("module or function metadata is outside the modeled domain")
    if len(selected.body.blocks) != 1:
        raise UnsupportedLeanIR("formal scalar input requires a single-block function")
    block = selected.body.block
    inputs = tuple(_width(arg.type) for arg in block.args)
    outputs = tuple(_width(t) for t in selected.function_type.outputs)
    if not outputs:
        raise UnsupportedLeanIR("formal scalar input requires complete integer outputs")
    env = {arg: f"v{i}" for i, arg in enumerate(block.args)}
    bindings: list[str] = []
    returned: tuple[str, ...] | None = None
    binary = {arith.AddiOp: "add", arith.SubiOp: "sub", arith.MuliOp: "mul"}
    for op in block.ops:
        if returned is not None:
            raise UnsupportedLeanIR("operations after return are outside the formal domain")
        if isinstance(op, func.ReturnOp):
            if op.attributes or op.properties:
                raise UnsupportedLeanIR("return metadata is outside the modeled domain")
            returned = tuple(env[value] for value in op.arguments)
            if tuple(_width(v.type) for v in op.arguments) != outputs:
                raise UnsupportedLeanIR("return types do not match complete function outputs")
            continue
        if len(op.results) != 1 or op.regions or op.successors:
            raise UnsupportedLeanIR("formal scalar operations require one result and no regions or successors")
        width = _width(op.results[0].type)
        args = [env[v] for v in op.operands]
        from xdsl.dialects.builtin import UnregisteredOp

        name = op.op_name.data if isinstance(op, UnregisteredOp) else op.name
        allowed_attributes = {"op_name__"} if isinstance(op, UnregisteredOp) else set()
        allowed_properties = (
            {"value"} if isinstance(op, arith.ConstantOp) else {"overflowFlags"} if type(op) in binary else set()
        )
        if set(op.attributes) - allowed_attributes or set(op.properties) - allowed_properties:
            raise UnsupportedLeanIR("unmodeled operation flags, immediates or metadata")
        if isinstance(op, arith.ConstantOp):
            value = op.value
            if not isinstance(value, IntegerAttr):
                raise UnsupportedLeanIR("formal scalar constant must be an integer")
            expr = f"BitVec.ofInt {width} ({value.value.data})"
        elif type(op) in binary:
            _no_flags(op)
            if len(args) != 2 or any(_width(v.type) != width for v in op.operands):
                raise UnsupportedLeanIR("formal binary operation requires two equal-width operands")
            expr = f"Merlin.bv{binary[type(op)].capitalize()} {width} {args[0]} {args[1]}"
        elif isinstance(op, arith.ExtSIOp):
            expr = f"BitVec.signExtend {width} {args[0]}"
        elif isinstance(op, arith.ExtUIOp):
            expr = f"BitVec.setWidth {width} {args[0]}"
        elif isinstance(op, arith.TruncIOp):
            expr = f"BitVec.setWidth {width} {args[0]}"
        else:
            declaration = (registry or {}).get(name)
            if declaration is None or declaration.get("effects") != {"reads": [], "writes": []}:
                raise UnsupportedLeanIR("operation has no admitted pure primitive semantics")
            semantic = declaration.get("semantics") or {}
            primitive = semantic.get("primitive")
            if primitive not in {"integer.add", "integer.sub", "integer.mul"}:
                raise UnsupportedLeanIR("declared primitive is outside the formal scalar model")
            if len(args) != 2 or any(_width(v.type) != width for v in op.operands):
                raise UnsupportedLeanIR("declared integer primitive has an unsupported signature")
            expected = declaration.get("operand_types"), declaration.get("result_types")
            if declaration.get("immediates") != [] or not all(isinstance(types, list) for types in expected):
                raise UnsupportedLeanIR("primitive requires its complete frozen typed signature")
            for declared, actual in zip(expected, (tuple(op.operands), tuple(op.results)), strict=True):
                if len(declared) != len(actual) or any(
                    row.get("kind") != "integer" or row.get("bits") != _width(v.type)
                    for row, v in zip(declared, actual, strict=True)
                ):
                    raise UnsupportedLeanIR("actual operation differs from its frozen instruction signature")
                    # Binding order is semantic: an instruction may encode its rhs first.
            operand_names = declaration.get("operand_names")
            mapping = semantic.get("bindings")
            if (
                not isinstance(operand_names, list)
                or len(operand_names) != 2
                or len(set(operand_names)) != 2
                or not isinstance(mapping, dict)
            ):
                raise UnsupportedLeanIR("primitive requires explicit operand names and lhs/rhs bindings")
            if set(mapping) != {"lhs", "rhs"} or set(mapping.values()) != set(operand_names):
                raise UnsupportedLeanIR("primitive operand bindings are incomplete")
            parameters = semantic.get("parameters") or {}
            if parameters != {"width": width, "overflow": "wrap"}:
                raise UnsupportedLeanIR("primitive requires its exact width and wrapping policy")
            lhs, rhs = (args[operand_names.index(mapping[key])] for key in ("lhs", "rhs"))
            expr = f"Merlin.bv{primitive.partition('.')[2].capitalize()} {width} {lhs} {rhs}"
        result_name = f"t{len(bindings)}"
        bindings.append(f"let {result_name} : BitVec {width} := {expr}")
        env[op.results[0]] = result_name
    if returned is None or len(returned) != len(outputs):
        raise UnsupportedLeanIR("formal scalar program has incomplete outputs")
    return ScalarFunction(entry, inputs, outputs, tuple(bindings), returned)


def equivalence_query(before: ScalarFunction, after: ScalarFunction, *, tactic: str = "simp") -> str:
    if before.inputs != after.inputs or before.outputs != after.outputs:
        raise UnsupportedLeanIR("source and target have different complete typed signatures")
    if tactic not in {"simp", "rfl", "bv_decide"}:
        raise ValueError("proof strategy is outside the admitted tactic vocabulary")
    args = " ".join(f"(v{i} : BitVec {w})" for i, w in enumerate(before.inputs))
    proof = (
        "simp"
        if tactic == "simp"
        else "simp only [Merlin.bvAdd_eq, Merlin.bvSub_eq, Merlin.bvMul_eq]; bv_decide"
        if tactic == "bv_decide"
        else tactic
    )
    return (
        "import Merlin.Foundation\nset_option warningAsError true\nset_option linter.unusedSimpArgs false\n"
        f"theorem merlin_transform {args} :\n"
        f"  ({before.term()}) = ({after.term()}) := by\n  {proof}\n"
        "#print axioms merlin_transform\n"
    )

    # Preserve historical object lookup; implementation and completion state live here.
