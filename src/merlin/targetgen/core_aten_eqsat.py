"""Sound e-class quotienting for bounded Core ATen test candidates.

Equality saturation is used only after two candidates have byte-identical executable semantics: exact
overload/schema, encoded inputs, eager output, post-call state, mutation/alias observations, comparison
policy, and RNG seed.  Such candidates differ only in the coverage labels by which the generator reached
the same concrete program.  They are placed in one xDSL ``equivalence.class``; ``eqsat-extract`` chooses
a deterministic representative, and that representative inherits the union of the class obligations.

No algebraic PyTorch rewrite is assumed.  In particular, floating reassociation, NaN identities,
integer overflow, mutation, aliasing, and RNG programs are never equated by this pass.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

_LABEL_FIELDS = {
    "case_id",
    "covered_obligations",
    "partition_assignment",
    "equivalent_partition_assignments",
    "eqsat_eclass",
}


def observational_fingerprint(case: Mapping[str, Any]) -> str:
    """Digest the complete executable/oracle semantics while excluding coverage-only labels."""

    semantic = {key: value for key, value in case.items() if key not in _LABEL_FIELDS}
    payload = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()


def _extract_representative(identifiers: list[str]) -> str:
    """Use the repository's real xDSL equality-saturation passes for deterministic extraction."""

    from xdsl.context import Context
    from xdsl.dialects import builtin, equivalence, test
    from xdsl.dialects.builtin import IndexType, IntAttr, ModuleOp, StringAttr
    from xdsl.ir import Block, Region
    from xdsl.transforms import eqsat_add_costs, eqsat_extract

    names = sorted(set(identifiers))
    if not names:
        raise ValueError("cannot extract an empty observational equivalence class")
    if len(names) == 1:
        return names[0]
    index = IndexType()
    operations = []
    for rank, name in enumerate(names, 1):
        operations.append(
            test.TestOp(
                result_types=[index],
                attributes={
                    "case_id": StringAttr(name),
                    equivalence.EQSAT_COST_LABEL: IntAttr(rank),
                },
            )
        )
    equivalence_class = equivalence.ClassOp(*[operation.results[0] for operation in operations])
    block = Block([*operations, equivalence_class, equivalence.YieldOp(equivalence_class.results[0])])
    module = ModuleOp([equivalence.GraphOp([index], Region([block]))])
    context = Context(allow_unregistered=True)
    for dialect in (builtin.Builtin, equivalence.Equivalence, test.Test):
        context.load_dialect(dialect)
    eqsat_add_costs.EqsatAddCostsPass(default=None).apply(context, module)
    eqsat_extract.EqsatExtractPass().apply(context, module)
    remaining = [
        str(operation.attributes["case_id"].data) for operation in module.walk() if "case_id" in operation.attributes
    ]
    if len(remaining) != 1 or remaining[0] not in names:
        raise RuntimeError(f"eqsat extraction did not retain exactly one known candidate: {remaining}")
    return remaining[0]


def quotient_observational_equivalents(
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Merge exact observational equivalents and return ``(reduced, audit)``.

    Each retained case carries every partition assignment whose concrete execution document is exactly
    the same.  This is a semantic no-op and may strictly improve the later global set-cover optimum.
    """

    groups: dict[str, list[str]] = defaultdict(list)
    for identifier, case in candidates.items():
        if identifier != case.get("case_id"):
            raise ValueError(f"candidate map key differs from case_id: {identifier}")
        groups[observational_fingerprint(case)].append(identifier)
    reduced: dict[str, dict[str, Any]] = {}
    classes = []
    for fingerprint in sorted(groups):
        members = sorted(groups[fingerprint])
        representative_id = _extract_representative(members)
        representative = dict(candidates[representative_id])
        assignments_by_json: dict[str, dict[str, str]] = {}
        obligations: set[str] = set()
        for identifier in members:
            member = candidates[identifier]
            assignment = dict(sorted(member["partition_assignment"].items()))
            assignments_by_json[json.dumps(assignment, sort_keys=True, separators=(",", ":"))] = assignment
            obligations.update(str(item) for item in member["covered_obligations"])
        assignments = [assignments_by_json[key] for key in sorted(assignments_by_json)]
        primary = dict(sorted(representative["partition_assignment"].items()))
        representative["partition_assignment"] = primary
        representative["equivalent_partition_assignments"] = assignments
        representative["covered_obligations"] = sorted(obligations)
        representative["eqsat_eclass"] = {
            "observational_fingerprint": fingerprint,
            "members": members,
            "representative": representative_id,
        }
        reduced[representative_id] = representative
        if len(members) > 1:
            classes.append(representative["eqsat_eclass"])
    audit = {
        "engine": "xdsl-eqsat",
        "xdsl_version": importlib.metadata.version("xdsl"),
        "equivalence": "byte-identical complete executable/oracle documents",
        "rewrite_rules": [],
        "raw_candidate_count": len(candidates),
        "quotient_candidate_count": len(reduced),
        "eliminated_candidate_count": len(candidates) - len(reduced),
        "nontrivial_eclass_count": len(classes),
        "nontrivial_eclasses": classes,
    }
    return dict(sorted(reduced.items())), audit
