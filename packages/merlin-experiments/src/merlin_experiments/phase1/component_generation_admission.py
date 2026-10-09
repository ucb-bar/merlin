"""Fresh authoring admission for actual bounded Phase 0 reference generation.

Legacy coverage remains inspectable through its original report reader. A fresh
origin requires the complete budgeted ledger and ordinary source-cost replay.
These bounds do not qualify process heap, compiler execution or hardware timing.
"""

import hashlib
from pathlib import Path

from merlin_experiments.phase0.component_coverage import BUDGETED_REPORT_SCHEMA, verify_report
from merlin_experiments.phase2.contracts import StageGateError


def verify_bounded_generation(corpus_root: Path) -> dict:
    """Reopen actual source/member costs before fresh compiler-origin admission."""
    try:
        root = Path(corpus_root).absolute()
        if any(path.is_symlink() for path in (root, *root.parents)) or root.resolve() != root:
            raise StageGateError("fresh Phase 1 bounded generation refuses indirect paths")
        report = verify_report(root)
    except (OSError, TypeError, ValueError) as error:
        raise StageGateError("fresh Phase 1 bounded generation is unavailable: " + str(error)) from error
    if report["schema"] != BUDGETED_REPORT_SCHEMA:
        raise StageGateError("fresh Phase 1 requires verified budgeted Phase 0 coverage v2")
    return report


def verify_generation_inputs(corpus_root: Path, *, preparation, hardware, software) -> dict:
    """Select original source preparation without reinterpreting legacy coverage.

    This is only the source-input leg. FreshPhase1Inputs separately requires the
    actual independent runtime, library, compile roster and isolated tool route.
    Completion of a source owner is never a candidate execution verdict.
    """
    if preparation is None:
        return verify_bounded_generation(corpus_root)
    from merlin_experiments.phase0.source_preparation_release import SourcePreparation

    try:
        if type(preparation) is not SourcePreparation:
            raise ValueError("fresh Phase1 needs its actual live versioned source preparation")
        root = Path(corpus_root).absolute()
        if preparation.root != root or preparation.hardware is not hardware or preparation.software is not software:
            raise ValueError("fresh Phase1 source preparation differs from its exact original corpus/hardware/software")
        selected = preparation.require_complete()
        from merlin.common.strict_json import loads

        with preparation.coverage.open("rb") as stream:
            raw = stream.read(preparation.budget.max_report_bytes + 1)
        if hashlib.sha256(raw).hexdigest() != selected["coverage_source"]["sha256"]:
            raise ValueError("fresh Phase1 source coverage changed after original completion replay")
        report = loads(raw, max_bytes=preparation.budget.max_report_bytes)
        for pin in preparation.source_pins:
            pin.verify()
        if report["schema"] != BUDGETED_REPORT_SCHEMA:
            raise ValueError("fresh Phase1 source preparation requires actual budgeted source coverage v2")
        return report
    except (OSError, TypeError, ValueError) as error:
        raise StageGateError("fresh Phase1 original source preparation is unavailable: " + str(error)) from error
