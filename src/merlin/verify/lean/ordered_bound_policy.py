"""A separately versioned W8A8 wrapper binding live originals to checked contract snapshots.

The sealed ordered checker and its Lean numerical theorem remain unchanged. This wrapper
adds explicit HOST content/lifetime checks; it does not establish artifact authenticity,
native fidelity, independent policy admission or whole-pass correctness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from merlin.verify.lean import ordered_evidence, ordered_policy, rotary_evidence, rotary_policy
from merlin.verify.lean.check import _file_ref, _function_attempt, _work, _write
from merlin.verify.lean.snapshot_binding import FileReference, SnapshotBinding, contract_bindings

_BOUNDARY = "w8a8-ordered-contract-bound-candidate"
_SCHEMA = "merlin.w8a8.ordered.contract-bound.v1"
_COPIED = ("evidence", "proof_axioms", "runtime", "runtime_sha256", "requested_prefix_checks_passed")
_FIELDS = {"schema", "base_receipt", "base_report", "contract_bindings", "candidate_sources", *_COPIED}


def _sources() -> list[dict[str, Any]]:
    paths = {Path(item["path"]) for item in ordered_policy._sources()}
    paths.update(Path(__file__).resolve().parent / name for name in ("ordered_bound_policy.py", "snapshot_binding.py"))
    return [_file_ref(path) for path in sorted(paths)]


def _fresh(records, label: str) -> None:
    for raw in records:
        FileReference.decode(raw, label).assert_current(label)


def _base_dependencies(report: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *report["candidate_sources"],
        *report["runtime"]["files"],
        *report["program_exports"],
        report["compiler_python"],
        report["structural_preflight"],
        report["structural_receipt"],
        *[
            report[kind][part]
            for kind in ("input_contracts", "exponent_contracts")
            for part in ("original", "snapshot")
        ],
    ]


def _bindings_equal(raw: Any, expected: dict[str, SnapshotBinding]) -> None:
    encoded = {kind: binding.encode() for kind, binding in expected.items()}
    if not ordered_evidence.same(raw, encoded):
        raise ValueError("bound contract records differ from the actual base report")


def _receipt(base: dict[str, Any], certificate: dict[str, Any]) -> dict[str, Any]:
    return {**base, "boundary": _BOUNDARY, "cert": certificate}


def _load(receipt: dict[str, Any]):
    if (
        receipt.get("boundary") != _BOUNDARY
        or receipt.get("theorem") != ordered_evidence.PREFIX
        or receipt.get("verdict") != "abstracted"
    ):
        raise ValueError("receipt is not a contract-bound ordered candidate")
    certificate = FileReference.decode(receipt["cert"], "bound report")
    certificate.assert_current("bound report")
    report = rotary_policy._decode(Path(certificate.path))
    rotary_evidence._object(report, _FIELDS, "contract-bound report")
    if type(report["schema"]) is not str or report["schema"] != _SCHEMA:
        raise ValueError("unsupported contract-bound report schema")
    base_receipt_ref = FileReference.decode(report["base_receipt"], "base receipt")
    base_report_ref = FileReference.decode(report["base_report"], "base report")
    base_receipt_ref.assert_current("base receipt")
    base_report_ref.assert_current("base report")
    base_receipt = rotary_policy._decode(Path(base_receipt_ref.path))
    if (
        base_receipt.get("boundary") != "w8a8-ordered-contraction-prefix-candidate"
        or base_receipt.get("theorem") != ordered_evidence.PREFIX
        or base_receipt.get("verdict") != "abstracted"
    ):
        raise ValueError("bound report names an unsupported base receipt")
    if FileReference.decode(base_receipt["cert"], "base certificate") != base_report_ref:
        raise ValueError("base receipt and report are not associated")
    if not ordered_evidence.same(receipt, _receipt(base_receipt, certificate.encode())):
        raise ValueError("contract-bound receipt differs from its actual base receipt")
    base_report = rotary_policy._decode(Path(base_report_ref.path))
    rotary_evidence._object(base_report, ordered_policy._REPORT_FIELDS, "base ordered report")
    bindings = contract_bindings(base_report)
    _bindings_equal(report["contract_bindings"], bindings)
    for key in _COPIED:
        if not ordered_evidence.same(report[key], base_report[key]):
            raise ValueError(f"bound report changed base {key}")
    if not ordered_evidence.same(report["candidate_sources"], _sources()):
        raise ValueError("contract-bound wrapper sources changed")
    _fresh(
        [*report["candidate_sources"], *_base_dependencies(base_report), receipt["before"], receipt["after"]],
        "bound receipt dependency",
    )
    return report, base_report, base_receipt, bindings


def _new_destination(destination: Path, contracts: tuple[Path, ...] = ()) -> None:
    receipt = (destination / "receipt.json").resolve()
    if receipt.exists() or receipt in {path.resolve() for path in contracts}:
        raise ValueError("contract-bound qualification requires a new destination distinct from its contracts")


def check_w8a8_ordered_bound_draft(
    stage: Path,
    destination: Path,
    *,
    input_contracts: Path,
    exponent_contracts: Path,
    m2m_python: Path,
    scope: str = "captured",
    index_bits: int = 64,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    """Check the unchanged numerical route and both live original/snapshot byte associations."""
    _new_destination(destination, (input_contracts, exponent_contracts))
    return _function_attempt(
        destination,
        index_bits,
        lambda: _check(
            stage, destination, input_contracts, exponent_contracts, m2m_python, scope, index_bits, timeout_seconds
        ),
    )


def _check(stage, destination, input_contracts, exponent_contracts, m2m_python, scope, index_bits, timeout_seconds):
    if stage.resolve() == destination.resolve():
        raise ValueError("contract-bound destination must differ from the source stage")
    sources = _sources()
    work = _work(destination)
    base = ordered_policy.check_w8a8_ordered_draft(
        stage,
        work / "base",
        input_contracts=input_contracts,
        exponent_contracts=exponent_contracts,
        m2m_python=m2m_python,
        scope=scope,
        index_bits=index_bits,
        timeout_seconds=timeout_seconds,
    )
    if "cert" not in base:
        raise ValueError(f"contract-bound base check failed: {base.get('reason')}")
    base_report_ref = FileReference.decode(base["cert"], "base report")
    base_report_ref.assert_current("base report")
    report = rotary_policy._decode(Path(base_report_ref.path))
    bindings = contract_bindings(report)
    wrapper = work / "contract-bound-report.json"
    wrapper.write_text(
        json.dumps(
            {
                "schema": _SCHEMA,
                "base_receipt": _file_ref(work / "base/receipt.json"),
                "base_report": base_report_ref.encode(),
                "contract_bindings": {kind: binding.encode() for kind, binding in bindings.items()},
                "candidate_sources": sources,
                **{key: report[key] for key in _COPIED},
            },
            indent=2,
        )
        + "\n"
    )
    result = _receipt(base, _file_ref(wrapper))
    # Includes both association checks and all source/report/runtime dependencies after the
    # numerical checker, immediately before exposing the outer receipt.
    _load(result)
    return _write(destination, result)


def replay(receipt_path: Path, destination: Path, *, m2m_python: Path) -> dict[str, Any]:
    """Recheck both submitted contracts throughout replay; publish only after final freshness."""
    _new_destination(destination)
    original = FileReference.decode(_file_ref(receipt_path), "original receipt")
    receipt = rotary_policy._decode(receipt_path)
    report, base_report, _base, bindings = _load(receipt)
    if FileReference.decode(base_report["compiler_python"], "compiler") != FileReference.decode(
        _file_ref(m2m_python), "compiler"
    ):
        raise ValueError("contract-bound replay selected a different compiler")
    candidate = _work(destination) / "candidate"
    fresh_receipt = check_w8a8_ordered_bound_draft(
        Path(base_report["stage"]),
        candidate,
        input_contracts=Path(bindings["input_contracts"].original.path),
        exponent_contracts=Path(bindings["exponent_contracts"].original.path),
        m2m_python=m2m_python,
        scope=report["evidence"]["policy_scope"],
        index_bits=report["evidence"]["index_bits"],
        timeout_seconds=base_report["checker_timeout_seconds"],
    )
    if "cert" not in fresh_receipt:
        raise ValueError(f"contract-bound replay failed: {fresh_receipt.get('reason')}")
    fresh, fresh_base, _receipt_base, fresh_bindings = _load(fresh_receipt)
    for key in (*_COPIED, "candidate_sources"):
        if not ordered_evidence.same(report[key], fresh[key]):
            raise ValueError(f"contract-bound replay differs in {key}")
    for key in ("before", "after", "checker", "theorem", "domain", "axioms", "functions", "lean", "verdict", "reason"):
        if not ordered_evidence.same(receipt[key], fresh_receipt[key]):
            raise ValueError(f"contract-bound replay receipt differs in {key}")
    def identities(value):
        return [(item["sha256"], item["bytes"]) for item in value["program_exports"]]

    if not ordered_evidence.same(identities(base_report), identities(fresh_base)):
        raise ValueError("contract-bound replay exports changed")
    for kind, binding in bindings.items():
        fresh_binding = fresh_bindings[kind]
        if binding.original != fresh_binding.original or binding.snapshot.content != fresh_binding.snapshot.content:
            raise ValueError(f"contract-bound replay changed the submitted {kind}")
    original.assert_current("original receipt")
    _load(receipt)
    _load(fresh_receipt)
    return _write(destination, fresh_receipt)
