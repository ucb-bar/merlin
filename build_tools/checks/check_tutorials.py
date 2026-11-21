"""Execute the five public tutorials in this interpreter; retain outputs outside source.

This checks the authored walkthroughs once. It does not discover extra test cases,
run hardware, or establish the examples' illustrative target assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = ("01_contracts", "02_test_generation", "03_scheduling", "04_transformations", "05_cca")


def _code_cells(notebook, source: Path):
    """Require executable work before a tutorial can be reported as executed."""
    code = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    if not code:
        raise ValueError(f"tutorial has no executable cells: {source}")
    return code


def execute(name: str, output: Path, kernels: Path) -> dict:
    import nbformat
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    from nbclient import NotebookClient

    source = ROOT / "examples" / "tutorials" / (name + ".ipynb")
    destination = output / source.name
    if destination.resolve() == source.resolve():
        raise ValueError("choose an output directory outside the source tutorials")
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    code = _code_cells(notebook, source)
    for cell in code:
        cell.outputs = []
        cell.execution_count = None
    manager = KernelManager(
        kernel_name="merlin-tutorials",
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernels)]),
    )
    client = NotebookClient(
        notebook,
        km=manager,
        timeout=120,
        allow_errors=False,
        record_timing=False,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        # A supplied kernel manager is otherwise left alive by nbclient.
        client.execute(cleanup_kc=True)
        # Output widgets capture callback exceptions instead of raising a cell
        # error. A rendered traceback must still fail an executable tutorial.
        state = (
            notebook.metadata.get("widgets", {}).get("application/vnd.jupyter.widget-state+json", {}).get("state", {})
        )
        for model in state.values():
            for item in model.get("state", {}).get("outputs", []):
                if item.get("output_type") == "error":
                    raise RuntimeError(f"{name} widget callback: {item.get('ename')}: {item.get('evalue')}")
    finally:
        nbformat.write(notebook, destination)
    return {
        "name": name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "code_cells": len(code),
        "executed": str(destination),
        "status": "executed",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    kernels = output / "kernels"
    kernel = kernels / "merlin-tutorials"
    kernel.mkdir(parents=True, exist_ok=True)
    (kernel / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Merlin tutorial checks",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )
    report = {"python": sys.executable, "tutorials": []}
    status = 0
    try:
        for name in TUTORIALS:
            row = execute(name, output, kernels)
            report["tutorials"].append(row)
            print(f"executed {name}: {row['code_cells']} code cells", flush=True)
    except Exception as exc:
        status = 1
        report["failure"] = {"type": type(exc).__name__, "detail": str(exc)}
        print(f"tutorial execution failed: {exc}", file=sys.stderr)
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
