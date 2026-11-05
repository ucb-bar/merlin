"""Validate the exported call edge between a kernel object and its host shim."""

from __future__ import annotations

from pathlib import Path


def verify_object_symbol_binding(
    kernel_object: Path,
    shim_object: Path,
    *,
    entry_symbol: str,
    kernel_symbol: str,
    original_kernel_symbol: str,
    timeout: int,
    nm,
    run,
) -> dict[str, str]:
    """Check the exact staged objects' exported call edge before claiming build evidence.

    This is an object-level check, not a link or execution verdict. In particular,
    every external reference must resolve between these two objects; otherwise a
    later link could silently pick up an unrelated runtime definition.
    """
    if nm is None:
        raise ValueError("staged object symbol binding needs a readable nm tool")

    def symbols(path: Path) -> tuple[list[tuple[str, str]], list[str]]:
        if not path.is_file() or path.is_symlink():
            raise ValueError("staged object symbol binding needs regular object files")
        result = run([nm, "--format=posix", "--extern-only", str(path)], timeout=timeout)
        if result.returncode != 0:
            raise ValueError(f"could not inspect staged object symbols: {(result.stderr or '')[-200:]}")
        defined, undefined = [], []
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2 or len(parts[1]) != 1:
                raise ValueError("nm returned an unrecognized staged object symbol record")
            if parts[1].upper() == "U":
                undefined.append(parts[0])
            else:
                defined.append((parts[0], parts[1]))
        return defined, undefined

    kernel_defined, kernel_undefined = symbols(kernel_object)
    shim_defined, shim_undefined = symbols(shim_object)
    if (
        kernel_defined.count((kernel_symbol, "T")) != 1
        or shim_defined.count((entry_symbol, "T")) != 1
        or any(name == kernel_symbol for name, _kind in shim_defined)
        or any(name == entry_symbol for name, _kind in kernel_defined)
        or shim_undefined.count(kernel_symbol) != 1
        or kernel_undefined
        or sorted(shim_undefined) != [kernel_symbol]
        or original_kernel_symbol
        in [
            *(name for name, _kind in kernel_defined),
            *kernel_undefined,
            *(name for name, _kind in shim_defined),
            *shim_undefined,
        ]
    ):
        raise ValueError("staged kernel and shim object symbols disagree or have unresolved references")
    return {"status": "object_symbol_binding_verified", "entry_symbol": entry_symbol, "kernel_symbol": kernel_symbol}
