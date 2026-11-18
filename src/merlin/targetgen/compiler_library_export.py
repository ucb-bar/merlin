"""Export a host-selected compiler source closure for independent backends and authoring tools."""

import hashlib
import json
import os
import shutil
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from zipfile import ZipFile, ZipInfo

from .compiler_library import freeze_compiler_library


def export(selection, source_root, destination, *, python_source_root=None):
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
        if python_source_root is None:
            base = source_root
        else:
            # A reviewed release reads the Python installed for this run, even
            # when the example and the installed package have different owners.
            if not relative.is_relative_to("src/merlin"):
                raise ValueError("installed compiler source must be under src/merlin")
            base = Path(python_source_root).resolve(strict=True)
            relative = relative.relative_to("src")
            module_path = Path(*module.split("."))
            if relative not in (module_path.with_suffix(".py"), module_path / "__init__.py"):
                raise ValueError("authoring source path must match its declared import identity")
        source = base / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or source.is_symlink()
            or not source.resolve(strict=True).is_relative_to(base)
            or not source.is_file()
        ):
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
        contract = freeze_compiler_library(
            destination,
            review_id=selection["review_id"],
            public_modules=tuple(selection["public_modules"]),
            sources=tuple(members),
        )
        (destination / "compiler-library.json").write_text(json.dumps(contract.record(), indent=2) + "\n")
        return contract
    except Exception:
        shutil.rmtree(destination)  # Only the fresh directory created above.
        raise


def authoring_archive(selection, source_root, destination, *, python_source_root=None):
    """Package only the verified source roster, with deterministic member order and timestamps."""
    with TemporaryDirectory() as tmp:
        root = Path(tmp) / "library"
        contract = export(
            selection, source_root, root, python_source_root=python_source_root or Path(source_root) / "src"
        )
        payload = _archive_bytes(contract, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent) as staged:
        staged.write(payload)
        staged.flush()
        try:
            os.link(staged.name, destination)
        except FileExistsError:
            if destination.is_symlink() or not destination.is_file() or destination.read_bytes() != payload:
                raise ValueError("different shared pass archive already exists, select a fresh bundle destination")
    return destination


def _archive_bytes(contract, root):
    buf = BytesIO()
    with ZipFile(buf, "w") as archive:
        for member in sorted(contract.members, key=lambda m: m.path):
            payload = (root / member.path).read_bytes()
            if hashlib.sha256(payload).hexdigest() != member.sha256:
                raise ValueError(f"library source changed during archive creation: {member.path}")
            archive.writestr(ZipInfo(member.path), payload)
        archive.writestr(ZipInfo("compiler-library.json"), json.dumps(contract.record(), sort_keys=True))
    return buf.getvalue()
