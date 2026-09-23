"""Native torch-MLIR printer inspection, never an executable transformation.

The function's tracked source is embedded in the existing isolated compiler worker;
the compiler environment need not import Merlin or reparse another dialect's text.
"""

from __future__ import annotations

import inspect


def _inspection_pass_manager(pipeline, context):
    """Preserve native pass execution; enable its own per-pass printer."""
    from pathlib import Path

    from torch_mlir.passmanager import PassManager

    manager = PassManager.parse(pipeline, context)
    root = Path(_MERLIN_INSPECTION_DIRECTORY) / "passes"
    root.mkdir(exist_ok=True)
    # The worker is serial and the audit root is fresh. Number invocations so
    # feature-driven pipeline segments keep both their order and native filenames.
    segment = root / f"segment-{len(list(root.iterdir())):04d}"
    segment.mkdir()
    (segment / "pipeline.txt").write_text(pipeline, encoding="utf-8")
    manager.enable_ir_printing(
        print_before_all=True,
        print_after_all=True,
        large_elements_limit=64,
        large_resource_limit=64,
        tree_printing_dir_path=str(segment),
    )
    return manager


def _write_view(operation, directory, name, keep_exact):
    import hashlib
    import json
    from pathlib import Path

    root = Path(directory) / "native"
    root.mkdir(exist_ok=True)
    exact = operation.get_asm().encode("utf-8")
    parent = hashlib.sha256(exact).hexdigest()
    header = "// INSPECTION ONLY: elided tensor/resource contents; NOT EXECUTABLE IR.\n"
    header += "// Exact native get_asm() parent SHA-256: " + parent + "\n"
    compact = operation.get_asm(large_elements_limit=64, large_resource_limit=64)
    payload = (header + compact).encode("utf-8")
    filename = name + ".inspection.mlir.txt"
    (root / filename).write_bytes(payload)
    exact_name = name + ".exact.mlir" if keep_exact else None
    if exact_name:
        (root / exact_name).write_bytes(exact)
    record = {
        "name": name,
        "representation": "inspection-only",
        "executable": False,
        "file": "native/" + filename,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "parent": {
            "serialization": "torch_mlir.Operation.get_asm()",
            "sha256": parent,
            "bytes": len(exact),
            "file": "native/" + exact_name if exact_name else None,
        },
        "limits": {"large_elements_limit": 64, "large_resource_limit": 64},
        "sidecars": "index.json#sidecars",
    }
    (root / (name + ".json")).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def bind_inspection(source: str, directory: str | None, *, keep_exact: bool = False) -> str:
    """Bind declared runner template hooks; absent hooks refuse rather than skip views."""
    hooks = {"# __MERLIN_INSPECT_PARSED__": "00-parsed", "# __MERLIN_INSPECT_LOWERED__": "01-lowered"}
    for token, name in hooks.items():
        if source.count(token) != 1:
            raise ValueError(f"compiler runner must declare one {token} hook")
        replacement = f"_write_view(module.operation, {directory!r}, {name!r}, {keep_exact!r})" if directory else ""
        source = source.replace(token, replacement)
    if not directory:
        return source
    if "PassManager.parse(" not in source:
        raise ValueError("compiler runner must declare a native pass-manager invocation")
    source = source.replace("PassManager.parse(", "_inspection_pass_manager(")
    return (
        f"_MERLIN_INSPECTION_DIRECTORY = {directory!r}\n"
        + inspect.getsource(_inspection_pass_manager)
        + "\n"
        + inspect.getsource(_write_view)
        + "\n"
        + source
    )
