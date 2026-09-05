"""The ``contract`` dialect (xDSL): facts, obligations, capabilities, legality.

``contract`` says what is true and what must be proven. It never contains target
instructions. Ops: region, assume, fact, require, prove, capability, check.
See docs/core_dialects.md and docs/dialects.md.
"""
from __future__ import annotations

from ._common import HAS_XDSL

DIALECT_NAME = "contract"
OPS = ["region", "assume", "fact", "require", "prove", "capability", "check"]
TYPES = ["feature", "proof", "capability"]

# Assumption kinds `contract.assume` accepts (verifier-checked).
ASSUMPTION_KINDS = {
    "immutable", "stable_over_lifetime", "shape_static", "dtype_known",
    "layout_known", "capacity_fit", "reuse_known", "lifetime",
}

# Requirement predicates `contract.require` may demand (verifier-checked).
KNOWN_PREDICATES = {
    "rhs_immutable", "lhs_immutable", "capacity_fit", "layout_preserved",
    "runtime_persistent_handle", "accumulator_not_user_visible",
    "epilogue_consumes_accumulator", "event_token_support",
}

if HAS_XDSL:
    from xdsl.ir import (Attribute, Dialect, EnumAttribute, ParametrizedAttribute,
                         SpacedOpaqueSyntaxAttribute, TypeAttribute)
    from xdsl.irdl import (IRDLOperation, irdl_attr_definition, irdl_op_definition,
                           operand_def, opt_operand_def, opt_prop_def, prop_def,
                           region_def, result_def, traits_def, var_operand_def)
    from xdsl.traits import NoTerminator
    from xdsl.utils.exceptions import VerifyException
    from xdsl.utils.str_enum import StrEnum
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, StringAttr

    # -- enums ---------------------------------------------------------------

    class Lifetime(StrEnum):
        SINGLE_USE = "single_use"
        WITHIN_OP = "within_op"
        WITHIN_REGION = "within_region"
        ACROSS_DISPATCH = "across_dispatch"
        ACROSS_INVOCATION = "across_invocation"
        PERSISTENT_UNTIL_EVICT = "persistent_until_evict"

    class MemoryRole(StrEnum):
        STREAMING_ACTIVATION = "streaming_activation"
        REUSABLE_WEIGHT = "reusable_weight"
        RESIDENT_PACKED_TENSOR = "resident_packed_tensor"
        ACCUMULATOR = "accumulator"
        COMMITTED_OUTPUT = "committed_output"
        TEMPORARY = "temporary"
        FIFO_PAYLOAD = "fifo_payload"

    class LayoutRole(StrEnum):
        CANONICAL = "canonical"
        PACKED_LHS = "packed_lhs"
        PACKED_RHS = "packed_rhs"
        TARGET_NATIVE = "target_native"
        OPAQUE_TARGET = "opaque_target"
        LAYOUT_PRESERVED = "layout_preserved"

    @irdl_attr_definition
    class LifetimeAttr(EnumAttribute[Lifetime], SpacedOpaqueSyntaxAttribute):
        """#contract.lifetime — how long a fact/object remains valid."""
        name = "contract.lifetime"

    @irdl_attr_definition
    class MemoryRoleAttr(EnumAttribute[MemoryRole], SpacedOpaqueSyntaxAttribute):
        """#contract.memory_role — the role a tensor plays in the workload."""
        name = "contract.memory_role"

    @irdl_attr_definition
    class LayoutRoleAttr(EnumAttribute[LayoutRole], SpacedOpaqueSyntaxAttribute):
        """#contract.layout_role — the layout class of a tensor."""
        name = "contract.layout_role"

    # -- types ---------------------------------------------------------------

    @irdl_attr_definition
    class FeatureType(ParametrizedAttribute, TypeAttribute):
        """!contract.feature<"name"> — symbolic feature handle (use sparingly)."""
        name = "contract.feature"
        feature: StringAttr

    @irdl_attr_definition
    class ProofType(ParametrizedAttribute, TypeAttribute):
        """!contract.proof<"requirement"> — a proof token for a discharged fact."""
        name = "contract.proof"
        requirement: StringAttr

    @irdl_attr_definition
    class CapabilityType(ParametrizedAttribute, TypeAttribute):
        """!contract.capability<"target"> — a declared target/runtime capability set."""
        name = "contract.capability"
        target: StringAttr

    # -- ops -----------------------------------------------------------------

    @irdl_op_definition
    class RegionOp(IRDLOperation):
        """contract.region — wraps an analyzed region with facts and requirements."""
        name = "contract.region"
        sym_name = prop_def(StringAttr)
        workload = opt_prop_def(StringAttr)
        candidate_features = opt_prop_def(ArrayAttr)
        body = region_def()
        traits = traits_def(NoTerminator())

        def verify_(self) -> None:
            if not self.sym_name.data:
                raise VerifyException("contract.region requires a non-empty symbol name")
            if self.candidate_features is not None:
                for entry in self.candidate_features:
                    if not isinstance(entry, StringAttr) or not entry.data:
                        raise VerifyException(
                            "contract.region candidate_features must be non-empty strings")

    @irdl_op_definition
    class AssumeOp(IRDLOperation):
        """contract.assume — declares a fact assumed true (not proven) at this stage."""
        name = "contract.assume"
        value = operand_def()
        kind = prop_def(StringAttr)
        lifetime = opt_prop_def(LifetimeAttr)

        def verify_(self) -> None:
            if self.kind.data not in ASSUMPTION_KINDS:
                raise VerifyException(
                    "contract.assume kind %r not registered (known: %s)"
                    % (self.kind.data, sorted(ASSUMPTION_KINDS)))

    @irdl_op_definition
    class FactOp(IRDLOperation):
        """contract.fact — records an inferred (not assumed) fact about a value."""
        name = "contract.fact"
        value = operand_def()
        role = opt_prop_def(MemoryRoleAttr)
        reuse_count = opt_prop_def(IntegerAttr)
        layout = opt_prop_def(LayoutRoleAttr)

        def verify_(self) -> None:
            if self.reuse_count is not None and self.reuse_count.value.data < 0:
                raise VerifyException("contract.fact reuse_count must be non-negative")

    @irdl_op_definition
    class RequireOp(IRDLOperation):
        """contract.require — a requirement to discharge before interface lowering."""
        name = "contract.require"
        feature = prop_def(StringAttr)
        requires = opt_prop_def(ArrayAttr)

        def verify_(self) -> None:
            if not self.feature.data:
                raise VerifyException("contract.require feature name must be non-empty")
            if self.requires is not None:
                for entry in self.requires:
                    pred = entry.data if isinstance(entry, StringAttr) else None
                    if pred not in KNOWN_PREDICATES:
                        raise VerifyException(
                            "contract.require predicate %r not registered (known: %s)"
                            % (pred, sorted(KNOWN_PREDICATES)))

    @irdl_op_definition
    class ProveOp(IRDLOperation):
        """contract.prove — produces a proof token for a requirement on a value.

        **The verifier below checks a NAME MATCH, not a proof.** It confirms the result token's
        requirement string equals this op's requirement string; nothing here establishes that the
        property actually holds. That is a real limit of what a verifier can do, not an oversight —
        but it means a `contract.prove` in the IR must not be read as evidence.

        Which tokens are backed by evidence is therefore a MEASUREMENT, not a syntactic property:
        :func:`merlin.verify.proofs.audit_proofs` classifies each token as ``verified`` (a
        verification layer discharged this requirement for the producing pass), ``asserted`` (it
        exists but nothing discharged it) or ``unattributed`` (it names no producer, so nothing
        could). Measured on the reference workload, the baseline is 2 asserted, 0 verified.
        """
        name = "contract.prove"
        value = opt_operand_def()
        requirement = prop_def(StringAttr)
        producer_pass = opt_prop_def(StringAttr)
        proof = result_def(ProofType)

        def verify_(self) -> None:
            if not self.requirement.data:
                raise VerifyException("contract.prove requirement must be non-empty")
            if self.proof.type.requirement.data != self.requirement.data:
                raise VerifyException(
                    "contract.prove result proof token %r does not match requirement %r"
                    % (self.proof.type.requirement.data, self.requirement.data))

    @irdl_op_definition
    class CapabilityOp(IRDLOperation):
        """contract.capability — declares a target/runtime capability set."""
        name = "contract.capability"
        sym_name = prop_def(StringAttr)
        features = prop_def(ArrayAttr)
        runtime = opt_prop_def(ArrayAttr)
        cap = result_def(CapabilityType)

        def verify_(self) -> None:
            if not self.sym_name.data:
                raise VerifyException("contract.capability requires a symbol name")
            for entry in self.features:
                f = entry.data if isinstance(entry, StringAttr) else None
                if not f or not f.replace("_", "").isalnum():
                    raise VerifyException(
                        "contract.capability feature %r is not a valid identifier" % (f,))
            if self.cap.type.target.data != self.sym_name.data:
                raise VerifyException(
                    "contract.capability result type target %r does not match @%s"
                    % (self.cap.type.target.data, self.sym_name.data))

    @irdl_op_definition
    class CheckOp(IRDLOperation):
        """contract.check — asserts a requirement is satisfied for a value.

        Must be matched by a proof or a target capability before interface lowering;
        that matching is a cross-op analysis (lowering/analyses.py), not a local check.
        """
        name = "contract.check"
        value = operand_def()
        proofs = var_operand_def(ProofType)
        requirement = prop_def(StringAttr)

        def verify_(self) -> None:
            if not self.requirement.data:
                raise VerifyException("contract.check requirement must be non-empty")
            for p in self.proofs:
                if p.type.requirement.data != self.requirement.data:
                    raise VerifyException(
                        "contract.check proof token %r does not discharge %r"
                        % (p.type.requirement.data, self.requirement.data))

    _OP_CLASSES = [RegionOp, AssumeOp, FactOp, RequireOp, ProveOp, CapabilityOp, CheckOp]
    _ATTR_CLASSES = [FeatureType, ProofType, CapabilityType,
                     LifetimeAttr, MemoryRoleAttr, LayoutRoleAttr]
    CONTRACT_DIALECT = Dialect(DIALECT_NAME, _OP_CLASSES, _ATTR_CLASSES)

    def get_dialect() -> Dialect:
        return CONTRACT_DIALECT

    def build_example():
        """A small, verifiable module exercising every contract op."""
        from xdsl.ir import Block, Region
        from xdsl.dialects.builtin import (FunctionType, ModuleOp, TensorType, i8)
        from xdsl.dialects.func import FuncOp, ReturnOp

        Wt = TensorType(i8, [128, 64])
        blk = Block(arg_types=[Wt])
        (w,) = blk.args
        cap = CapabilityOp(
            result_types=[CapabilityType(StringAttr("toy_npu"))],
            properties={
                "sym_name": StringAttr("toy_npu"),
                "features": ArrayAttr([StringAttr("resident_packed_tensor"),
                                       StringAttr("accumulator_commit")]),
                "runtime": ArrayAttr([StringAttr("command_buffer"),
                                      StringAttr("metrics")]),
            })
        assume = AssumeOp(operands=[w], properties={
            "kind": StringAttr("immutable"),
            "lifetime": LifetimeAttr(Lifetime.WITHIN_REGION)})
        fact = FactOp(operands=[w], properties={
            "role": MemoryRoleAttr(MemoryRole.REUSABLE_WEIGHT),
            "reuse_count": IntegerAttr(8, 64),
            "layout": LayoutRoleAttr(LayoutRole.CANONICAL)})
        req = RequireOp(properties={
            "feature": StringAttr("resident_packed_tensor"),
            "requires": ArrayAttr([StringAttr("rhs_immutable"),
                                   StringAttr("capacity_fit")])})
        prove = ProveOp(
            operands=[w],
            result_types=[ProofType(StringAttr("rhs_immutable"))],
            properties={"requirement": StringAttr("rhs_immutable"),
                        "producer_pass": StringAttr("merlin-infer-contract-facts")})
        check = CheckOp(operands=[w, [prove.proof]],
                        properties={"requirement": StringAttr("rhs_immutable")})
        inner = Block()
        inner.add_ops([])
        region = RegionOp(regions=[Region([inner])], properties={
            "sym_name": StringAttr("rrhs_matmul"),
            "workload": StringAttr("repeated_rhs_matmul"),
            "candidate_features": ArrayAttr([StringAttr("resident_packed_tensor")])})
        ret = ReturnOp()
        blk.add_ops([cap, assume, fact, req, prove, check, region, ret])
        fn = FuncOp("facts", FunctionType.from_lists([Wt], []), Region([blk]))
        return ModuleOp([fn])

else:  # pragma: no cover - exercised only when xDSL is absent

    def get_dialect():
        return None

    def build_example():
        return None
