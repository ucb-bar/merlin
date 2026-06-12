"""Check that the generated target does not add code to Merlin core.

This is the per-file complement to the R4 architecture rule (which does a git diff).
This validator checks for structural integration markers in generated files.
"""

from __future__ import annotations

from pathlib import Path


_MERLIN_CORE_PATHS = [
    "merlin/compiler",
    "merlin/python/merlin",
    "merlin/runtime",
    "merlin/schemas",
]


def run(run_dir: Path, manifest: dict) -> dict:
    target = manifest["target"]
    generated_dir = run_dir / "generated" / f"{target}-mlir"

    metrics: dict = {
        "validator": "merlin_integration",
        "merlin_core_files_modified": 0,
        "integration_markers_found": [],
        "errors": [],
    }

    if not generated_dir.exists():
        metrics["errors"].append(
            f"generated/{target}-mlir/ does not exist; "
            "skipping Merlin integration check"
        )
        return metrics

    # Look for any file that references merlin core paths (would indicate cross-contamination)
    markers = []
    for py_file in generated_dir.rglob("*.py"):
        text = py_file.read_text(errors="replace")
        for core_path in _MERLIN_CORE_PATHS:
            if core_path in text:
                markers.append(f"{py_file.relative_to(run_dir)}: references {core_path}")

    metrics["integration_markers_found"] = markers
    if markers:
        metrics["errors"].append(
            "Generated files contain references to Merlin core paths; "
            "target-specific code must not depend on Merlin internals"
        )

    return metrics
