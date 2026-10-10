"""Derive limited source applicability from actual registered tensor IR.

This analysis says what the mathematical source can observe. It never proves
physical buffer ownership, placement, links, synchronization or repeated execution.
The selected target verifier must still evaluate those complete runtime obligations.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from merlin_experiments.phase2 import contracts as C

from .component_witness import REQUIRED_EXECUTION_EFFECTS

_SUBJECTS = (
    "static_input_domain",
    "observable_input_mutation",
    "observable_input_address_alias",
    "cross_invocation_source_state",
    "source_synchronization",
)


def _plain(path: Path) -> None:
    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not path.is_file()
        or path.resolve() != path
        or any(part.is_symlink() for part in (path, *path.parents))
    ):
        raise C.StageGateError("component source analysis requires an exact direct source file")


def _analysis_sources() -> tuple[tuple[Path, str], ...]:
    """Pin the complete parser membership and the concrete analysis owners."""
    paths = {Path(__file__)}
    for name in ("merlin.xdsl_dialects._common", "merlin.xdsl_dialects.fp8", "xdsl"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue  # Missing parsing authority is reported UNKNOWN below.
        owner = Path(module.__file__)
        paths.add(owner)
        if name == "xdsl":
            paths.update(owner.parent.rglob("*.py"))
    for path in paths:
        _plain(path)
    return tuple((path, C.sha256_file(path)) for path in sorted(paths))


@dataclass(frozen=True)
class ComponentSourceApplicability:
    source: Path
    source_program_sha256: str
    frontend: str
    analysis_sources: tuple[tuple[Path, str], ...]
    analysis_json: str

    def verify(self) -> None:
        """Rederive source facts; a typed object or caller status is insufficient."""
        observed = evaluate_component_source_applicability(
            source=self.source,
            source_program_sha256=self.source_program_sha256,
            frontend=self.frontend,
        )
        if observed.analysis_sources != self.analysis_sources or observed.analysis_json != self.analysis_json:
            raise C.StageGateError("component source applicability or parser authority changed")

    def record(self) -> dict:
        return {
            "schema": "merlin.component_source_applicability.v1",
            "source": str(self.source),
            "source_program_sha256": self.source_program_sha256,
            "frontend": self.frontend,
            "analysis_sources": [[str(path), digest] for path, digest in self.analysis_sources],
            **json.loads(self.analysis_json),
        }


def _inspect_source(text: str) -> dict:
    from xdsl.dialects.arith import Arith
    from xdsl.dialects.builtin import AnyFloat, IndexType, IntegerType, ModuleOp, NoneAttr, TensorType
    from xdsl.dialects.func import FuncOp, ReturnOp
    from xdsl.dialects.linalg import Linalg
    from xdsl.dialects.linalg.abstract_ops import LinalgStructuredOperation
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.dialects.math import Math
    from xdsl.dialects.tensor import Tensor
    from xdsl.parser import Parser
    from xdsl.traits import get_effects

    from merlin.xdsl_dialects._common import make_context

    module = Parser(make_context(Arith, Tensor, Linalg, Math), text).parse_module()
    module.verify()
    functions = tuple(module.body.block.ops)
    if len(functions) != 1 or type(functions[0]) is not FuncOp:
        raise ValueError("source must contain exactly one defined mathematical function")
    function = functions[0]
    if len(function.body.blocks) != 1 or not function.body.block.args:
        raise ValueError("source requires one concrete function block with explicit tensor inputs")

    def static_tensor(value_type):
        # A rank-zero tensor has one logical element and no dimensions. Its
        # empty shape is static; it remains a tensor, not a by-value scalar ABI.
        return (
            isinstance(value_type, TensorType)
            and isinstance(value_type.encoding, NoneAttr)
            and all(dim >= 0 for dim in value_type.get_shape())
        )

    if (
        not all(static_tensor(arg.type) for arg in function.body.block.args)
        or not function.function_type.outputs.data
        or not all(static_tensor(value_type) for value_type in function.function_type.outputs.data)
    ):
        raise ValueError("source function inputs/outputs are not explicit static unencoded tensors")

    def mathematical_type(value_type):
        if isinstance(value_type, TensorType):
            return static_tensor(value_type) and isinstance(
                value_type.get_element_type(), (IntegerType, AnyFloat, IndexType)
            )
        return isinstance(value_type, (IntegerType, AnyFloat, IndexType))

    operations, unresolved = [], []
    for ordinal, op in enumerate(module.walk()):
        types = [value.type for value in (*op.operands, *op.results)]
        types += [arg.type for region in op.regions for block in region.blocks for arg in block.args]
        if not all(mathematical_type(value_type) for value_type in types):
            unresolved.append(f"operation {ordinal} {op.name} has non-mathematical or dynamic storage types")
        if type(op) in {ModuleOp, FuncOp, ReturnOp, YieldOp}:
            effect = "structural_container_or_value_publication"
        elif isinstance(op, LinalgStructuredOperation):
            # The registered structured operation contract states that memref
            # outs mutate buffers, while tensor outs return updated SSA results.
            if (
                len(op.outputs) != len(op.res)
                or not op.outputs
                or any(not static_tensor(value.type) for value in op.outputs)
                or any(lhs.type != rhs.type for lhs, rhs in zip(op.outputs, op.res, strict=True))
            ):
                unresolved.append(f"operation {ordinal} {op.name} lacks tensor-only destination semantics")
            effect = "structured_tensor_value_update"
        else:
            effects = get_effects(op)
            effect = "unknown" if effects is None else "present" if effects else "none"
            if effects is None or effects:
                unresolved.append(f"operation {ordinal} {op.name} has unknown or observable memory effects")
        operations.append(
            {
                "ordinal": ordinal,
                "operation": op.name,
                "types": [str(value_type) for value_type in types],
                "source_effect": effect,
            }
        )
    return {
        "operations": operations,
        "unresolved": unresolved,
        "inputs": [str(arg.type) for arg in function.body.block.args],
        "outputs": [str(value_type) for value_type in function.function_type.outputs.data],
    }


def evaluate_component_source_applicability(
    *,
    source: Path,
    source_program_sha256: str,
    frontend: str,
) -> ComponentSourceApplicability:
    """Observe exact source semantics; unavailable/unknown authority stays UNKNOWN."""
    _plain(source)
    if C.sha256_file(source) != source_program_sha256 or frontend not in {"mlir", "pytorch"}:
        raise C.StageGateError("component source applicability differs from the admitted source/frontend")
    before = _analysis_sources()
    observation = {"operations": [], "unresolved": [], "inputs": [], "outputs": []}
    if frontend != "mlir":
        observation["unresolved"].append("Python source requires independent actual FX capture/import correspondence")
    else:
        try:
            observation = _inspect_source(source.read_text())
        except Exception as exc:  # noqa: BLE001 - parsing/verification never substitutes for missing source facts
            observation["unresolved"].append(type(exc).__name__ + ": " + str(exc))
    pure = not observation["unresolved"]
    facts = {}
    for subject in _SUBJECTS:
        if pure:
            status = "PASS" if subject == "static_input_domain" else "N_A"
            reason = (
                "registered verified function has static unencoded tensor inputs/outputs"
                if subject == "static_input_domain"
                else "closed tensor SSA has no observable addresses, input writes, persistent state "
                "or external synchronization"
            )
        else:
            status, reason = "UNKNOWN", "source applicability authority is incomplete"
        facts[subject] = {"status": status, "reason": reason, "scope": "mathematical source only"}
    observation.update(
        {
            "facts": facts,
            "runtime_effects": {effect: "UNKNOWN" for effect in REQUIRED_EXECUTION_EFFECTS},
            "numerical_finiteness": "UNKNOWN",
            "scope": (
                "source semantics only; no physical alias/lifetime/epoch, partition, execution or numerical credit"
            ),
        }
    )
    after = _analysis_sources()
    if before != after or C.sha256_file(source) != source_program_sha256:
        raise C.StageGateError("component source or parsing authority changed during applicability analysis")
    return ComponentSourceApplicability(
        source, source_program_sha256, frontend, after, json.dumps(observation, sort_keys=True, allow_nan=False)
    )
