"""Record the effective upstream lowering invocation without changing its IR.

This receipt observes commands, selected features, source bytes and the redacted
project environment. It is not a hermetic toolchain lock or a cache identity:
imported pass implementations, non-project environment and later host/device
compilation still require their own provenance.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from merlin.common.digest import sha256_bytes, sha256_file
from merlin.common.jsonio import write_pretty_json

from . import codegen_env

FILENAME = "lowering_recipe.json"


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


class LoweringRecipe:
    """One invocation's ordinary lowering receipt, reset before its first command."""

    def __init__(self, workdir: Path, *, features: Iterable[str], sources: dict[str, Path]):
        self.path = workdir / FILENAME
        self.record = {
            "schema": "merlin.lowering_recipe.v1",
            "scope": "upstream lowering through returned LLVM IR; not final codegen or link",
            "status": "prepared",
            "features": sorted(features),
            "sources": {name: _identity(path) for name, path in sorted(sources.items())},
            "commands": [],
        }
        write_pretty_json(self.path, self.record)

    def command(self, argv: list[str]) -> None:
        """Retain the actual pipeline and positional gates in their launch order."""
        environment = codegen_env.snapshot()
        self.record["commands"].append(
            {
                "argv": list(argv),
                "executable": _identity(Path(argv[0])),
                "environment": environment,
                "environment_digest": codegen_env.digest(environment),
            }
        )
        self.record["status"] = "invoked"
        write_pretty_json(self.path, self.record)

    def returned(self, llvm_ir: str) -> None:
        """Bind the successfully returned text, including normalization/loop outlining."""
        payload = llvm_ir.encode("utf-8")
        self.record.update(
            status="returned",
            returned_llvm_ir={"sha256": sha256_bytes(payload), "bytes": len(payload)},
        )
        write_pretty_json(self.path, self.record)

    def bind_source(self, name: str, path: Path) -> None:
        """Bind an exact native-stage source that later lowering consumed."""
        self.record["sources"][name] = _identity(Path(path))
        write_pretty_json(self.path, self.record)


def bind_host_transform_chain(recipe: Path, receipt: Path, *, source: Path, selected: Path) -> None:
    """Bind an actual complete late transform chain to its ordinary upstream receipt.

    The original returned LLVM witness is retained. This extends observation
    through host LLVM emission only, never claiming codegen/link qualification.
    """
    from .host_transform_chain import recheck_host_transform_chain

    record = json.loads(recipe.read_text())
    chain = recheck_host_transform_chain(receipt)
    original, output = _identity(source), _identity(selected)
    if (
        record.get("schema") != "merlin.lowering_recipe.v1"
        or record.get("status") != "returned"
        or record.get("returned_llvm_ir") != {"sha256": original["sha256"], "bytes": original["bytes"]}
        or chain["source"] != original
        or chain["selected"] != output
    ):
        raise ValueError("host chain is not bound to the actual returned upstream LLVM and selected emission")
    record["host_transform_chain"] = _identity(receipt)
    record["selected_host_llvm_ir"] = output
    write_pretty_json(recipe, record)


def bind_ciface_annotation(recipe: Path, before: str, after: str, report: dict) -> None:
    """Extend the returned witness through the ordinary ABI annotation only.

    Retain the upstream witness and independently reproduce the exact annotation.
    This does not authorize arbitrary text changes or alter any lifetime premise.
    """
    from . import ciface_inline

    def identity(text: str) -> dict:
        payload = text.encode("utf-8")
        return {"sha256": sha256_bytes(payload), "bytes": len(payload)}

    record = json.loads(recipe.read_text())
    selected, expected = ciface_inline.inline_wide_ciface_implementations(before)
    if (
        record.get("schema") != "merlin.lowering_recipe.v1"
        or record.get("status") != "returned"
        or record.get("returned_llvm_ir") != identity(before)
        or selected != after
        or expected != report
        or "ciface_boundary_inline" in record
    ):
        raise ValueError("ABI annotation differs from the actual returned LLVM and ordinary repair")
    record["upstream_returned_llvm_ir"] = identity(before)
    record["returned_llvm_ir"] = identity(after)
    record["ciface_boundary_inline"] = {
        "input": identity(before),
        "selected": identity(after),
        "report": report,
        "implementation": _identity(Path(ciface_inline.__file__)),
    }
    write_pretty_json(recipe, record)
