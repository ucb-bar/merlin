"""Independent actual-source effect admission before word-stage early returns."""

from xdsl.dialects import builtin
from xdsl.parser import Parser
from xdsl.utils.exceptions import ParseError


def checked_word_source_effects(source, stage, memory, records):
    """Reconstruct selected primitives without importing producer/evaluator code.

    Original inventories may retain unreachable declarations. Every actual
    effect still needs a compatible selection and matching primitive facts.
    This supplies no source equivalence, address, dependency or target proof.
    """
    from merlin.compiler.frontend.intake.kernel_atomic_evidence import (
        EFFECT_PROFILE,
        TYPED_EFFECT_PROFILE,
        checked_source_atomic,
    )
    from merlin.compiler.frontend.intake.kernel_barrier_evidence import _ids, _name
    from merlin.compiler.frontend.intake.kernel_volatile_evidence import PROFILE as VOLATILE_PROFILE
    from merlin.compiler.frontend.intake.kernel_volatile_evidence import checked_source_volatile
    from merlin.compiler.frontend.linalg_mlir import make_context

    if (
        not isinstance(source, (str, bytes))
        or not isinstance(stage, dict)
        or not isinstance(memory, dict)
        or not isinstance(records, list)
        or any(not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] for row in records)
    ):
        raise ValueError("source-word effects require typed source, stage, policy and inventory objects")
    try:
        module = Parser(make_context(), source.decode() if isinstance(source, bytes) else source).parse_module()
    except ParseError as error:
        raise ValueError("source-word effects require actual parseable typed IR") from error
    declarations = {row.get("id"): row for row in records}
    if None in declarations or len(declarations) != len(records):
        raise ValueError("word source effects require distinct retained source identities")
    atomics = {sid: row for sid, row in declarations.items() if row.get("opcode") in ("tt.atomic_rmw", "tt.atomic_cas")}
    volatile = {sid: row for sid, row in declarations.items() if "volatile_read" in row}
    atomic_ids, volatile_ids = set(atomics), set(volatile)
    profile = memory.get("effect_profile")
    observation_profile = memory.get("volatile_profile", profile)
    protected = stage.get("schema") in ("t2m.source-word-stages.v3", "t2m.source-word-stages.v4")
    legacy = stage.get("schema") in ("t2m.triton-shared-stage.v1", "t2m.source-word-stages.v2")
    for op in module.walk():
        name = _name(op)
        if name.startswith(("ttg.", "ttng.", "gpu.", "nvvm.", "rocdl.", "llvm.")) or name in (
            "tt.inline_asm",
            "tt.elementwise_inline_asm",
        ):
            raise ValueError("word source retains an unexpanded physical/effect operation: " + name)
        if name == "tt.load":
            flag = op.properties.get("isVolatile", op.attributes.get("isVolatile"))
            if flag is not None and (not isinstance(flag, builtin.IntegerAttr) or flag.type != builtin.i1):
                raise ValueError("word source volatile flag requires the actual pinned boolean")
            selected = set(_ids(op.location)) & volatile_ids
            if flag is None or not flag.value.data:
                if selected:
                    raise ValueError("word source lost an original volatile observation")
                continue
            if observation_profile != VOLATILE_PROFILE or not protected:
                raise ValueError(
                    "actual volatile source requires its selected nondestructive word/effect profile at "
                    + str(op.location)
                )
            if not selected:
                raise ValueError("actual volatile source lacks its primitive declaration")
            for sid in selected:
                if volatile[sid].get("opcode") != name:
                    raise ValueError("word source volatile primitive differs from its original declaration")
                checked_source_volatile(op, volatile[sid]["volatile_read"], sid)
        elif name in ("tt.atomic_rmw", "tt.atomic_cas"):
            if profile not in (EFFECT_PROFILE, TYPED_EFFECT_PROFILE):
                raise ValueError(
                    "actual atomic source requires its selected private numeric/effect profile at " + str(op.location)
                )
            selected = set(_ids(op.location)) & atomic_ids
            if not selected:
                raise ValueError("actual atomic source lacks its primitive declaration")
            for sid in selected:
                declared = atomics[sid].get("atomic")
                if atomics[sid].get("opcode") != name or not isinstance(declared, dict):
                    raise ValueError("word source atomic lacks its original primitive fields")
                primitive = {key: value for key, value in declared.items() if key not in ("word_view", "byte_view")}
                # Preserve the exact historical label in evidence and compare
                # its existing f64 IEEE source meaning only. This grants no
                # new quiet-NaN behavior or executable-payload permission.
                if (
                    legacy
                    and primitive.get("operation") == "fadd"
                    and primitive.get("value_dtype") == "f64"
                    and primitive.get("numeric_contract") == "ieee-f64-rne-add"
                ):
                    primitive["numeric_contract"] = "ieee-f64-rne-add-quiet-nans"
                selection = {"profile": profile, "f32_atomic_add": memory.get("f32_atomic_add")}
                checked_source_atomic(op, primitive, selection)
