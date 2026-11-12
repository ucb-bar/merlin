"""One-shot current-source scalar-carrier binding for ordinary model lowering."""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

from merlin.compiler.lower.model.source_expression_interval import (
    IntervalEffectContract,
    find_closed_scalar_i8_observers,
    validate_closed_scalar_observer,
)
from merlin.compiler.lower.model.source_scalar_carrier import (
    ScalarCarrierHelperBinding,
    prepare_source_scalar_carrier,
    reify_source_scalar_carrier_helper_family,
)
from merlin.compiler.lower.model.source_scalar_carrier_policy import ApproximateScalarCarrierPolicy
from merlin.compiler.lower.model.source_stage_transport import ScalarLeafPatch, operation_paths, value_locator


def _identity(value):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        or value == "0" * 64
    ):
        raise ValueError("explicit implementation SHA256 identity required")


@dataclass(frozen=True)
class CompilerHostNumericAdmission:
    """Trusted caller's scoped host capability; core implements no native fenv.

    The factory must restore the complete incoming environment on every exit.
    verify_rne must inspect the actual current host environment, not return an
    unconditional promise. Its implementation is separately qualified/pinned.
    """

    context_factory: Callable
    verify_rne: Callable
    implementation_sha256: str

    def validate(self):
        _identity(self.implementation_sha256)
        if not callable(self.context_factory) or not callable(self.verify_rne):
            raise ValueError("scoped actual compiler-host RNE capability required")

    @contextmanager
    def enter(self):
        self.validate()
        with self.context_factory():
            if self.verify_rne() is not True:
                raise ValueError("actual compiler-host RNE admission refused")
            yield
            if self.verify_rne() is not True:
                raise ValueError("compiler-host RNE changed during source preparation")


@dataclass(frozen=True)
class IncomingRNECapability:
    """Provider-owned runtime ()→i1 predicate, read once per scalar point.

    The provider proves the actual linked implementation and ordinary ABI.
    This declaration grants no all-region or all-model FRM stability theorem.
    """

    symbol: str
    implementation_sha256: str
    reads_actual_incoming_rounding: bool = False
    preserves_rounding_mode: bool = False
    preserves_flags: bool = False
    nontrapping: bool = False
    no_memory_writes: bool = False
    placement: str = "per_point"

    def validate(self):
        _identity(self.implementation_sha256)
        if not isinstance(self.symbol, str) or not self.symbol.isascii() or not self.symbol.isidentifier():
            raise ValueError("explicit runtime predicate ABI symbol required")
        if self.placement != "per_point":
            raise ValueError("cross-FRM predicate hoisting has no complete region theorem")
        if any(
            value is not True
            for value in (
                self.reads_actual_incoming_rounding,
                self.preserves_rounding_mode,
                self.preserves_flags,
                self.nontrapping,
                self.no_memory_writes,
            )
        ):
            raise ValueError("actual incoming-RNE predicate and complete runtime effects required")


@dataclass(frozen=True)
class SourceScalarCarrierSelection:
    policy: ApproximateScalarCarrierPolicy
    effects: IntervalEffectContract
    coefficient_proposal: Callable
    incoming_rne: IncomingRNECapability
    compiler_host: CompilerHostNumericAdmission
    max_total_table_bytes: int
    namespace: str = "merlin_scalar_carrier"
    max_source_bytes: int = 16 * 1024 * 1024
    max_response_bytes: int = 16 * 1024 * 1024

    def validate(self):
        if type(self.policy) is not ApproximateScalarCarrierPolicy or type(self.effects) is not IntervalEffectContract:
            raise ValueError("distinct typed scalar policy and explicit source effects required")
        self.policy.validate()
        self.effects.validate()
        if (
            type(self.incoming_rne) is not IncomingRNECapability
            or type(self.compiler_host) is not CompilerHostNumericAdmission
        ):
            raise ValueError("typed runtime and separate compiler-host capabilities required")
        self.incoming_rne.validate()
        self.compiler_host.validate()
        if not callable(self.coefficient_proposal):
            raise ValueError("invocation-local current-expression coefficient proposal required")
        if not isinstance(self.namespace, str) or not self.namespace.isascii() or not self.namespace.isidentifier():
            raise ValueError("explicit private helper namespace required")
        if any(type(value) is not int or value <= 0 for value in (self.max_source_bytes, self.max_response_bytes)):
            raise ValueError("explicit positive private source/response quotas required")
        if (
            type(self.max_total_table_bytes) is not int
            or self.max_total_table_bytes < (1 << self.policy.leading_bits) * 12
        ):
            raise ValueError("explicit aggregate immutable table budget required")

    def callback(self, payload, directory):
        """Discover and prove current IR; no retained helper or call-site proof input."""
        from xdsl.dialects import func
        from xdsl.dialects.builtin import ModuleOp, StringAttr, f32, i1, i8
        from xdsl.ir import Block, Region

        from merlin.compiler.dialects._common import text
        from merlin.compiler.frontend.linalg_mlir import parse_mlir_text

        self.validate()
        with self.compiler_host.enter():
            module = parse_mlir_text(payload.decode("utf-8"))
            proofs, refusals = find_closed_scalar_i8_observers(module, effects=self.effects)
            if not proofs:
                raise ValueError("current source has no admitted closed scalar observers")
            # Authenticate ALL source witnesses before the coefficient callback
            # and again before constructing any edit. Source IR is never mutated.
            for proof in proofs:
                validate_closed_scalar_observer(proof)
                # Whole current IR is verified in its native session before
                # insertion. xDSL's unrelated operation constraints can lag
                # native resource support; verify every selected container,
                # never re-create or relax an original external resource.
                container = proof.integer_result.owner.parent_op()
                while container is not None and container.name not in ("linalg.generic", "func.func"):
                    container = container.parent_op()
                if container is None:
                    raise ValueError("current closed scalar observer has no typed source container")
                container.verify()
            paths, blocks = operation_paths(module)
            groups = {}
            for proof in proofs:
                groups.setdefault(proof.expression.canonical_sha256, []).append(proof)
            if len(groups) * (1 << self.policy.leading_bits) * 12 > self.max_total_table_bytes:
                raise ValueError("current expression families exceed aggregate table budget")
            fragments, replacements, source_records, private_paths = [], [], [], set()
            predicate = func.FuncOp.external(self.incoming_rne.symbol, [], [i1])
            fragments.append(predicate)
            physical_table_bytes = 0
            for family_index, members in enumerate(groups.values()):
                members = tuple(members)
                proposal = self.coefficient_proposal(members[0].expression, self.effects, self.policy)
                for proof in proofs:
                    validate_closed_scalar_observer(proof)
                carrier = prepare_source_scalar_carrier(
                    members,
                    effects=self.effects,
                    policy=self.policy,
                    coefficient_words=proposal,
                )
                prefix = f"{self.namespace}_{family_index}"
                bindings = tuple(
                    ScalarCarrierHelperBinding(
                        proof,
                        f"{prefix}_expression_{index}",
                        f"{prefix}_observer_{index}",
                        f"{prefix}_carrier_{index}",
                    )
                    for index, proof in enumerate(members)
                )
                table_name = f"{prefix}_coefficients"
                family = reify_source_scalar_carrier_helper_family(carrier, bindings, table_symbol=table_name)
                for operation in tuple(family.body.block.ops):
                    operation.detach()
                    if isinstance(operation, func.FuncOp):
                        operation.properties["sym_visibility"] = StringAttr("private")
                    fragments.append(operation)
                physical_table_bytes += (1 << self.policy.leading_bits) * 12
                for index, binding in enumerate(bindings):
                    proof = binding.proof
                    block = Block(arg_types=[f32, f32])
                    predicate_call = func.CallOp(self.incoming_rne.symbol, [], [i1])
                    call = func.CallOp(binding.carrier_symbol, [*block.args, predicate_call.results[0]], [i8])
                    block.add_ops([predicate_call, call, func.ReturnOp(call.results[0])])
                    adapter_name = f"{prefix}_point_{index}"
                    adapter = func.FuncOp(adapter_name, ([f32, f32], [i8]), Region(block), visibility="private")
                    fragments.append(adapter)
                    anchor = proof.integer_result.uses
                    uses = tuple(anchor)
                    if (
                        len(uses) != 1
                        or uses[0].index != 0
                        or uses[0].operation.name not in ("linalg.yield", "func.return", "tensor.insert")
                    ):
                        raise ValueError("current integer observation escaped its original terminator")
                    terminator = uses[0].operation
                    private = set((*proof.expression_operations, *proof.observer_operations))
                    private = {operation for operation in private if operation.name != "arith.constant"}
                    if proof.cut.owner in private or proof.up.owner in private:
                        raise ValueError("original cut/up producer overlaps the replaced private scalar DAG")
                    order = [operation for operation in terminator.parent.ops if operation in private]
                    if len(order) != len(private):
                        raise ValueError("current scalar closure is not private to its original block")
                    remove = []
                    for operation in order:
                        path = paths[operation]
                        if path in private_paths:
                            raise ValueError("selected scalar closures overlap")
                        private_paths.add(path)
                        remove.append({"path": path, "name": operation.name})
                    replacements.append(
                        {
                            "anchor": paths[terminator],
                            "cut": value_locator(proof.cut, paths, blocks),
                            "up": value_locator(proof.up, paths, blocks),
                            "adapter": adapter_name,
                            "remove": remove,
                        }
                    )
                    source_records.append(
                        {
                            "expression_sha256": proof.expression.canonical_sha256,
                            "quant_factor_bits": proof.quant_factor_bits,
                            "anchor": paths[terminator],
                            "carrier": binding.carrier_symbol,
                            "diagnostic_paths_not_policy": True,
                        }
                    )
            for proof in proofs:
                validate_closed_scalar_observer(proof)
            helper_module = ModuleOp(fragments)
            helper_module.verify()
            fragment = text(helper_module, generic=True)
            # Native fragment namespace verification repeats this before mutation.
            existing = {op.sym_name.data for op in module.body.block.ops if hasattr(op, "sym_name")}
            fresh = [op.sym_name.data for op in helper_module.body.block.ops if hasattr(op, "sym_name")]
            if len(set(fresh)) != len(fresh) or existing.intersection(fresh):
                raise ValueError("current source helper/runtime namespace conflict")
            permission = {
                "source_effects": asdict(self.effects),
                "runtime_capability": asdict(self.incoming_rne),
                "compiler_host_capability_sha256": self.compiler_host.implementation_sha256,
                "policy_sha256": self.policy.canonical_sha256,
                "original_source_fallback": True,
                "private_publication": True,
            }
            digest = hashlib.sha256(payload).hexdigest()
            record = {
                "schema": "merlin.current_scalar_carrier_binding.v1",
                "status": "CURRENT_TYPED_EDITS_PREPARED",
                "source_sha256": digest,
                "fragment_sha256": hashlib.sha256(fragment.encode()).hexdigest(),
                "members": source_records,
                "refusals": [{"operation": op.name, "reason": reason} for op, reason in refusals],
                "family_count": len(groups),
                "physical_table_bytes": physical_table_bytes,
                "permissions": permission,
                "default_policy": False,
                "profitability": "UNKNOWN",
                "whole_output_validation": "PENDING_ORIGINAL_GATE_REQUIRED",
                "predicate_placement": "per_point_no_cross_FRM_hoist",
                "native_original_module_or_producer_reparse": False,
                "whole_current_module_verifier": "original native context before any mutation",
                "parent_verification": "every current closed selected container; original external resources untouched",
                "lifetime": "one-shot packet bound to exact current stage bytes",
            }
            Path(directory, "binding.json").write_text(json.dumps(record, indent=2) + "\n")
            return ScalarLeafPatch(digest, fragment, tuple(replacements), permission)


def host_scope(selection):
    if selection is None:
        return nullcontext()
    if type(selection) is not SourceScalarCarrierSelection:
        raise ValueError("typed invocation-local scalar-carrier selection required")
    selection.validate()
    return selection.compiler_host.enter()


def host_admitted(function):
    """Cover direct ordinary APIs before their first xDSL operation."""
    signature = inspect.signature(function)

    @functools.wraps(function)
    def invoke(*args, **kwargs):
        arguments = signature.bind(*args, **kwargs).arguments
        if arguments.get("source_scalar_carrier") is not None:
            from merlin.compiler.lower.model.prov_cse import FEATURE as provenance_stripping_feature

            if provenance_stripping_feature in (arguments.get("features") or ()):
                raise ValueError("current typed scalar rewrite cannot follow provenance-stripping CSE")
        with host_scope(arguments.get("source_scalar_carrier")):
            return function(*args, **kwargs)

    return invoke
