"""Fresh host checks of emitted assignments and authored sequential rewrites.

The caller selects original, reviewed semantics. Author requests cannot provide
these arguments through the common broker. Each acceptance reconstructs the
projection and requires the live Lean completion from the same host process.
This is modeled preservation, not a device or asynchronous execution proof.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

from .lean_checker import _check_generated, accepted_receipt
from .lean_contract import AUTHORED_DOMAINS, SCHEMA, LeanTransformReceipt, digest, document_digest


def _identity(model: str, projection: str) -> str:
    return document_digest(
        {
            "model": model,
            "projection": projection,
            "checker": digest(Path(__file__).read_bytes()),
        }
    )


def _unsupported(before: str, after: str, domain: str, reason: str) -> LeanTransformReceipt:
    from merlin.common.path_scrub import scrub_host_paths

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
    from merlin.transforms.certificates import parse_ir

    from .lean_resource_geometry import project_resource_geometry

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

    from .lean_resource_geometry import lean_resource_query

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
    from .lean_state_effects import derive_effect_schedule, effect_semantic_identity

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
    from .lean_checker import semantic_identity, verify_scalar_transform

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

    from .lean_checker import semantic_identity
    from .lean_export import export_scalar

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
