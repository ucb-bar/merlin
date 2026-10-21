"""Keep selected application entry points public when linking generated stages.

Independent C exports can reuse names for region helpers and bundled runtime
functions. Localize those definitions in each object, preserving its selected
public API and unresolved library dependencies. No instructions are generated.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from merlin.common.digest import sha256_file
from merlin.common.proc import run_checked


def _symbols(tool, path, kind):
    result = run_checked(
        [tool, "--extern-only", kind, "--no-demangle", "--format=posix", path], timeout=60,
    )
    symbols = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 4 or len(fields[1]) != 1 or fields[0] in symbols:
            raise ValueError("unsupported or repeated nm symbol record")
        name, binding = fields[:2]
        if not name.isascii() or any(not (c.isalnum() or c in "_.$") for c in name):
            raise ValueError("unsupported object symbol spelling")
        symbols[name] = binding
    return symbols


def localize_object_symbols(
    source: Path, destination: Path, public_symbols: tuple[str, ...], *, nm: Path, objcopy: Path,
) -> dict:
    """Publish a fresh object with only explicitly selected definitions global.

    Use LLVM nm and objcopy selected by the host toolchain. Required external
    library symbols remain unresolved. Missing exports, changed inputs, tool
    failures and unexpected output bindings leave no successful destination.
    This checks symbol visibility, not whole-program arithmetic or linking.
    """
    source, destination = Path(source), Path(destination)
    if (
        source.is_symlink() or not source.is_file() or destination.exists() or destination.is_symlink()
        or type(public_symbols) is not tuple
        or any(not isinstance(name, str) or not name for name in public_symbols)
        or len(set(public_symbols)) != len(public_symbols)
    ):
        raise ValueError("object exports need a regular source, distinct symbols and a fresh destination")
    tools = {name: Path(tool).resolve(strict=True) for name, tool in (("nm", nm), ("objcopy", objcopy))}
    tool_hashes = {name: sha256_file(tool) for name, tool in tools.items()}
    before_sha256 = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="object-exports-", dir=destination.parent) as temporary:
        root = Path(temporary)
        copied, result = root / "input.o", root / "output.o"
        shutil.copyfile(source, copied)
        if sha256_file(copied) != before_sha256:
            raise ValueError("input object changed while copying")
        definitions = _symbols(tools["nm"], copied, "--defined-only")
        dependencies = _symbols(tools["nm"], copied, "--undefined-only")
        if not set(public_symbols) <= set(definitions):
            raise ValueError("selected public symbol is absent from the input definitions")
        private = sorted(set(definitions) - set(public_symbols))
        roster = root / "private-symbols.txt"
        roster.write_text("".join(name + "\n" for name in private))
        run_checked([tools["objcopy"], "--localize-symbols=" + str(roster), copied, result], timeout=60)
        if (
            _symbols(tools["nm"], result, "--defined-only") != {key: definitions[key] for key in public_symbols}
            or _symbols(tools["nm"], result, "--undefined-only") != dependencies
            or sha256_file(source) != before_sha256
            or any(sha256_file(tools[name]) != digest for name, digest in tool_hashes.items())
        ):
            raise ValueError("object inputs, exports or external dependencies changed unexpectedly")
        result_sha256 = sha256_file(result)
        # Publish atomically and reject a destination created during the tool calls.
        os.link(result, destination)
    return {
        "input_sha256": before_sha256,
        "output_sha256": result_sha256,
        "public_symbols": sorted(public_symbols),
        "localized_symbols": private,
        "unresolved_symbols": sorted(dependencies),
        "tools": {name: {"path": str(tool), "sha256": tool_hashes[name]} for name, tool in tools.items()},
    }
