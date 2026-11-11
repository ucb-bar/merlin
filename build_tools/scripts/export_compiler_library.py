#!/usr/bin/env python3
"""Materialize an explicitly selected compiler library, without dependency discovery.

Run in the trusted host setup, never as a compiler candidate. The selection is
an authored source roster; hash/import-closure checks do not themselves review
or qualify it. Namespace-only initializers intentionally expose no parent API.
"""

import argparse
import json
from pathlib import Path
import shutil
import sys

import _source_layout

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_source_layout.core_package(ROOT).parent))
from merlin.targetgen.compiler_library import freeze_compiler_library


def export(selection, source_root, destination):
    if set(selection) != {"schema", "review_id", "public_modules", "sources", "empty_namespaces"} or (
        selection["schema"] != "merlin.compiler_library_selection.v1"
    ):
        raise ValueError("explicit compiler library selection required")
    source_root, destination = Path(source_root).resolve(strict=True), Path(destination).resolve()
    # Validate the entire roster before creating anything.
    members, payloads = [], {}
    for row in selection["sources"]:
        if set(row) != {"path", "module"}:
            raise ValueError("library source needs an exact path and module")
        module = row["module"]
        if not isinstance(module, str) or any(not part.isidentifier() for part in module.split(".")):
            raise ValueError("invalid source module")
        relative = Path(row["path"])
        source = source_root / relative
        if (relative.is_absolute() or ".." in relative.parts or source.is_symlink() or
                not source.resolve(strict=True).is_relative_to(source_root) or not source.is_file()):
            raise ValueError("library source escapes selected root")
        target = Path(*module.split("."))
        target = target / "__init__.py" if source.name == "__init__.py" else target.with_suffix(".py")
        payloads[str(target)] = source.read_bytes()
        members.append((str(target), module))
    for module in selection["empty_namespaces"]:
        if any(not part.isidentifier() for part in module.split(".")):
            raise ValueError("invalid namespace module")
        target = str(Path(*module.split(".")) / "__init__.py")
        if target in payloads:
            raise ValueError("duplicate namespace ownership")
        payloads[target] = b'"""Isolated compiler-library namespace; public APIs are explicit leaf modules."""\n'
        members.append((target, module))
    destination.mkdir(parents=True, exist_ok=False)
    try:
        for relative, payload in payloads.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        contract = freeze_compiler_library(destination, review_id=selection["review_id"],
                                           public_modules=tuple(selection["public_modules"]), sources=tuple(members))
        (destination / "compiler-library.json").write_text(json.dumps(contract.record(), indent=2) + "\n")
        return contract
    except Exception:
        shutil.rmtree(destination)  # Only the fresh directory created above.
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    contract = export(json.loads(args.selection.read_text()), args.source_root, args.destination)
    print(json.dumps(dict(contract_sha256=contract.sha256, members=len(contract.members),
                         public_modules=contract.public_modules), indent=2))


if __name__ == "__main__":
    main()
