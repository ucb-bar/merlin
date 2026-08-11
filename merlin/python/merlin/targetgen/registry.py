"""Plug-and-play loader for ISOLATED, per-run target-dialect packages.

Generated target dialects do NOT live in the core merlin tree and are NOT hardcoded into the
shared lowering tables. Each is a self-contained directory (one per generation round), loaded
dynamically and registered for a run:

    artifacts/targets/<target>/<run_id>/
        manifest.yaml                         # target, run_id, provenance, status
        dialect.py                            # self-contained xDSL dialect; exposes SPEC_OPS + DIALECT_NAME
        lowering.yaml                         # interface->target + target->opcode tables (data)
        contracts/{target_contract,dialect_plan}.yaml

This gives: (1) isolation — rounds never clobber the core or each other; (2) import/export —
the directory is the portable unit; (3) run isolation — multiple candidate dialects coexist.
The core ships only the reference targets (toy_npu, saturn); everything generated is loaded
from its package via :func:`load_target`.
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TargetPackage:
    """A loaded, isolated target-dialect package."""

    name: str
    run_id: str
    directory: Path
    dialect_module: Any
    spec: Any                       # a target_lowering.TargetSpec (op/type classes)
    lowering_table: dict[str, str]  # interface op -> target op
    opcode_table: dict[str, str]    # target op -> command-buffer opcode
    contract: dict[str, Any]

    def dialect_plan(self) -> dict[str, Any]:
        """A dialect_plan dict (lowering rules) the core pipeline can consume."""
        return {"target": self.name, "dialect_name": self.name,
                "lowering": [{"from": k, "to": v} for k, v in self.lowering_table.items()]}


def _import_module(path: Path, name: str):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so xDSL's annotation/type-hint resolution (get_type_hints) can find
    # the module globals (e.g. `element_type: Attribute`) — IRDL definitions need this.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_target(package_dir: str | Path) -> TargetPackage:
    """Load an isolated target package directory into a :class:`TargetPackage`.

    The package's ``dialect.py`` must expose ``DIALECT_NAME`` and ``SPEC_OPS`` (a dict mapping
    ``pack/matmul/commit/evict/resident_type/accumulator_type`` to its op/type classes). The
    ``lowering.yaml`` carries ``interface_to_target`` and ``target_to_opcode`` maps.
    """
    from ..xdsl_dialects.lowering.target_lowering import TargetSpec

    d = Path(package_dir)
    manifest = yaml.safe_load((d / "manifest.yaml").read_text())
    run_id = manifest.get("run_id", d.name)
    mod = _import_module(d / "dialect.py", f"gen_target_{manifest['target']}_{run_id}")
    ops = mod.SPEC_OPS
    low = yaml.safe_load((d / "lowering.yaml").read_text())
    contract = {}
    cpath = d / "contracts" / "target_contract.yaml"
    if cpath.is_file():
        contract = yaml.safe_load(cpath.read_text())

    # A package MAY require properties on its target ops that only its own contract can supply —
    # a SIMT target's warp width, for instance. It derives them itself (and fails closed if the
    # contract does not carry them); the core rebuild loop merges them without interpreting them.
    op_properties = None
    derive = getattr(mod, "op_properties", None)
    if callable(derive):
        op_properties = derive(contract)
    spec = TargetSpec(mod.DIALECT_NAME, mod, ops["pack"], ops["matmul"], ops["commit"],
                      ops["evict"], ops["resident_type"], ops["accumulator_type"],
                      op_properties=op_properties)
    return TargetPackage(
        name=mod.DIALECT_NAME, run_id=run_id, directory=d, dialect_module=mod, spec=spec,
        lowering_table=dict(low["interface_to_target"]),
        opcode_table=dict(low["target_to_opcode"]), contract=contract)


def default_run(generated_root: str | Path, target: str) -> Path:
    """The most-recent run_id directory for a target under the generated-targets root."""
    base = Path(generated_root) / target
    runs = sorted([p for p in base.iterdir() if p.is_dir()]) if base.is_dir() else []
    if not runs:
        raise FileNotFoundError(f"no generated runs for {target!r} under {base}")
    return runs[-1]
