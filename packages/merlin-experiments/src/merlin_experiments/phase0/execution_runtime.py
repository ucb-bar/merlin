"""Byte-checked host execution selections for selected-only Phase 0 children.

These remain host paths, not copied sources or a sandbox claim. External RTL
custody covers declared probe/evidence files; MLC custody covers its package,
schemas, selected target outputs and fallback tools, not an entire checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

from merlin_experiments.phase1.source_inputs import fingerprint

FILES = (
    "MERLIN_RISCV_GCC",
    "MERLIN_CLANG",
    "MERLIN_MLIR_TRANSLATE",
    "MERLIN_OBJDUMP",
    "MERLIN_SPIKE",
    "MERLIN_COMPILER_PYTHON",
    "MERLIN_CIRCT_OPT",
    "MERLIN_FIRTOOL",
)
ROOTS = ("MERLIN_MLIR_INSTALL", "MERLIN_CLANG_INSTALL", "MERLIN_MLC_DIR", "MERLIN_EXT_CHIPYARD", "MERLIN_CHIPYARD")


def _pin(path: Path) -> dict:
    resolved = path.resolve()
    if path.is_dir():
        from merlin_experiments.capture_execution.sealed_m2m import _source_tree

        digest = _source_tree(resolved, skip_python_cache=True)
    else:
        digest = fingerprint(path) if path.exists() else None
    return {"path": str(path), "resolved": str(resolved), "digest": digest}


def _probe_paths(document, pins, artifacts):
    """Include inline locators and absent sources as well as named sources."""
    if isinstance(document, dict):
        if "ext_root" in document:
            value = os.environ.get("MERLIN_EXT_" + str(document["ext_root"]).upper())
            if value:
                yield Path(value).expanduser().absolute() / document["path"]
        elif "pin" in document and "path" in document:
            pin = pins.get(str(document["pin"]))
            root = pin.checkout() if pin else None
            if root is not None:
                yield (root / document["path"]).absolute()
        elif "artifact" in document:
            artifact = artifacts.get(str(document["artifact"]))
            path = artifact.resolve() if artifact else None
            if path is not None:
                yield path.absolute()
        elif "file" in document:
            yield Path(document["file"]).absolute()
        for value in document.values():
            yield from _probe_paths(value, pins, artifacts)
    elif isinstance(document, list):
        for value in document:
            yield from _probe_paths(value, pins, artifacts)


def select(target: str, evidence) -> dict:
    """Observe only the allowlisted exports while freezing the live selection."""
    from merlin.common import provenance
    from merlin.targetgen.rtl.semantic_facts import load_probes

    sources = {
        *evidence.source_paths,
        *_probe_paths(load_probes(target) or {}, provenance.load_pins(), provenance.load_artifacts()),
    }
    selected = {}
    for name in (*FILES, *ROOTS):
        value = os.environ.get(name)
        if not value:
            selected[name] = None
            continue
        path = Path(value).expanduser().absolute()
        if name in FILES:
            if not path.is_file() or not os.access(path, os.X_OK):
                raise ValueError(f"selected Phase 0 executable is unavailable: {name}")
            members = [path]
            if name == "MERLIN_COMPILER_PYTHON":
                # Preserve bin/python's venv spelling, and bind its startup and
                # dependency bytes rather than just the shared base executable.
                venv = path.parent.parent
                members += [venv / "pyvenv.cfg", venv / "lib"]
        else:
            if not path.is_dir():
                raise ValueError(f"selected Phase 0 execution root is unavailable: {name}")
            if name == "MERLIN_MLC_DIR":
                from merlin.targetgen.rtl.mlc_bridge import _arc_target

                members = [path / "mlc", path / "schemas"]
                members += [path / "runs/circt-arc" / key for key in {target, _arc_target(target)}]
                members += [path / "third_party/circt/build/bin" / tool for tool in ("circt-opt", "firtool")]
            elif name in {"MERLIN_MLIR_INSTALL", "MERLIN_CLANG_INSTALL"}:
                members = [path / "bin" / tool for tool in ("clang-23", "mlir-opt", "mlir-translate", "llvm-objdump")]
            else:
                members = [source for source in sources if source.is_relative_to(path)]
        selected[name] = {
            "path": str(path),
            "resolved": str(path.resolve()),
            "members": [_pin(member) for member in sorted(set(members))],
        }
    return selected


def environment(selection: dict) -> dict[str, str]:
    if not isinstance(selection, dict) or set(selection) != set((*FILES, *ROOTS)):
        raise ValueError("frozen Phase 0 execution selection is absent; freeze a new run")
    return {name: row["path"] for name, row in selection.items() if row is not None}


def verify(selection: dict, env: dict, *, live: bool = True) -> None:
    expected = environment(selection)
    if {name: env[name] for name in (*FILES, *ROOTS) if name in env} != expected:
        raise ValueError("frozen Phase 0 execution environment routing changed")
    if live:
        for name, row in selection.items():
            if row is None:
                continue
            path = Path(row["path"])
            if (
                str(path.resolve()) != row["resolved"]
                or (name in ROOTS and not path.is_dir())
                or any(_pin(Path(pin["path"])) != pin for pin in row["members"])
            ):
                raise ValueError(f"frozen Phase 0 execution selection changed: {name}")
            if name in FILES and not os.access(path, os.X_OK):
                raise ValueError(f"frozen Phase 0 executable is no longer executable: {name}")
