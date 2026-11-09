"""Explicit LLVM loop extraction for large scalar host functions.

LLVM's CodeExtractor preserves buffer accesses and operation order after
bufferization. Extracted helpers normally carry ``alwaysinline``; this policy
replaces it with ``noinline`` only for newly extracted functions. Original
functions retain their inlining policies. Helper call overhead is a real cost:
this is a measured, default-off compiler choice, not a default.
"""

from __future__ import annotations

from pathlib import Path

from merlin.common.proc import run_checked

FEATURE = "outline_llvm_loops"
MERGE_FEATURE = "outline_llvm_loops_merge_identical"


def ensure_registered() -> str:
    from .shared_feature_registry import ImprFeature, known, register

    if FEATURE not in known():
        register(
            ImprFeature(
                name=FEATURE,
                action_class="PASS",
                description="Extract LLVM loops and keep new helpers out of line.",
            )
        )
    if MERGE_FEATURE not in known():
        register(
            ImprFeature(
                name=MERGE_FEATURE,
                action_class="PASS",
                description="Extract LLVM loops, keep helpers out of line and merge identical functions with LLVM's exact function comparator.",
            )
        )
    return FEATURE


def _defined_functions(symbols: str) -> set[str]:
    """Read strict llvm-nm POSIX records, refusing ambiguous symbol spellings.

    Attribute selectors and response files use LLVM's simple identifier subset.
    Unusual quoted names must not be guessed from whitespace-delimited output.
    """
    result = set()
    allowed = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$-")
    for line in symbols.splitlines():
        fields = line.split()
        if len(fields) != 4 or not fields[0] or not set(fields[0]) <= allowed:
            raise ValueError("LLVM loop outlining requires unambiguous simple symbol names")
        name, kind, address, size = fields
        if len(kind) != 1:
            raise ValueError("invalid LLVM symbol kind")
        if set(address) != {"-"}:
            int(address, 16)
        int(size, 16)
        if kind in {"t", "T", "w", "W"}:
            result.add(name)
    return result


def outline_loops(
    llvm_ir: str,
    workdir: Path,
    *,
    timeout: float = 7200,
    audit=None,
    merge_identical: bool = False,
) -> str:
    """Run actual LLVM transformations; unavailable or rejected passes fail closed.

    Optional function merging is the LLVM ``mergefunc`` pass, with its ordinary
    signature, attribute, body and function-address rules. It adds no numerical
    equivalence assumption. Original exported symbols must remain defined.
    There is no floating-point reassociation permission, tensor ownership change,
    target selection or model-specific selection. The caller records the selected
    feature in the ordinary compiler feature identity.
    """
    from .toolchain import llvm_nm, llvm_opt

    if type(merge_identical) is not bool:
        raise ValueError("merge_identical requires an explicit boolean")
    workdir.mkdir(parents=True, exist_ok=True)
    source = workdir / "loops-input.ll"
    output = workdir / "loops-outlined.ll"
    original = workdir / "loops-input.bc"
    extracted = workdir / "loops-extracted.bc"
    source.write_text(llvm_ir, encoding="utf-8")

    def run(command, sources=(__file__,)):
        if audit is not None:
            audit.command([str(x) for x in command], sources=sources)
        return run_checked(command, timeout=timeout)

    run([llvm_opt(), "-passes=verify", source, "-o", original])
    before = _defined_functions(run([llvm_nm(), "--format=posix", "--defined-only", original]).stdout)
    run([llvm_opt(), "-passes=loop-extract,verify", original, "-o", extracted])
    after = _defined_functions(run([llvm_nm(), "--format=posix", "--defined-only", extracted]).stdout)
    if not before <= after:
        raise ValueError("LLVM loop extraction unexpectedly removed original function symbols")
    # A response file avoids the operating system argument-count limit when
    # thousands of helpers are extracted. No shell parses this file.
    remove_selectors = workdir / "helper-remove-attributes.rsp"
    remove_selectors.write_text(
        "".join(f"-force-remove-attribute={name}:alwaysinline\n" for name in sorted(after - before)),
        encoding="utf-8",
    )
    removed = workdir / "loops-without-inline.bc"
    run(
        [llvm_opt(), "-passes=forceattrs,verify", f"@{remove_selectors}", extracted, "-o", removed],
        sources=(__file__, str(remove_selectors)),
    )
    # Forceattrs will not add noinline while alwaysinline is present. Removing
    # alwaysinline and adding noinline in the same pass leaves neither on those
    # functions, so addition must run on the completed removal result.
    selectors = workdir / "helper-attributes.rsp"
    selectors.write_text(
        "".join(f"-force-attribute={name}:noinline\n" for name in sorted(after - before)),
        encoding="utf-8",
    )
    run(
        [llvm_opt(), "-S", "-passes=forceattrs,verify", f"@{selectors}", removed, "-o", output],
        sources=(__file__, str(selectors)),
    )
    if merge_identical:
        merged = workdir / "loops-merged.bc"
        run([llvm_opt(), "-passes=mergefunc,verify", output, "-o", merged])
        retained = _defined_functions(run([llvm_nm(), "--format=posix", "--defined-only", merged]).stdout)
        if not before <= retained:
            raise ValueError("LLVM function merging removed an original function symbol")
        run([llvm_opt(), "-S", "-passes=verify", merged, "-o", output])
    result = output.read_text(encoding="utf-8")
    if audit is not None:
        audit.stage(
            "llvm-loops-outlined-merged" if merge_identical else "llvm-loops-outlined",
            result,
            format="llvm-ir",
        )
    return result
