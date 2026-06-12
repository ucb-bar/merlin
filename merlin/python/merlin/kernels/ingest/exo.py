"""Ingest Exo schedules by compiling them to C, then mining the generated C.

Exo ships *scheduling specifications* (``.py``), not generated C. Per the chosen design we
compile each spec to C with ``exo.compile_procs_to_strings`` and mine that C with the same
ISA-family marker table used for XNNPACK/Autocomp — so an Exo→Gemmini kernel and an
Autocomp→Gemmini kernel share one marker set.

Robustness is the priority: Exo may be absent from the environment, a spec may fail to
import, or a proc may fail to compile. Every such case is **skipped and logged** (never
fatal); the corpus is still satisfied by XNNPACK + Autocomp alone. Skip counts are surfaced
via the returned diagnostics so the report can state them honestly.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path
from typing import Iterator

from merlin.kernels.types import NormalizedKernel

log = logging.getLogger("merlin.kernels.ingest.exo")

# Map an Exo platform import to an ISA family used as the kernel target.
_PLATFORM_TARGET = (
    ("platforms.gemmini", "gemmini"),
    ("platforms.x86", "avx"),
    ("platforms.avx", "avx"),
    ("platforms.aarch64", "neon"),
    ("platforms.neon", "neon"),
    ("platforms.rvm", "rvv"),
    ("rvv", "rvv"),
)


def _detect_target(source_text: str, default: str | None) -> str:
    for needle, fam in _PLATFORM_TARGET:
        if needle in source_text:
            return fam
    return default or "unknown"


def _guess_op(name: str) -> str:
    n = name.lower()
    for kw, op in (("matmul", "matmul"), ("sgemm", "gemm"), ("gemm", "gemm"),
                   ("conv", "conv"), ("filter", "conv")):
        if kw in n:
            return op
    return "unknown"


def _sniff_dtype(c_code: str) -> str:
    if "int8_t" in c_code:
        return "i8"
    if "_Float16" in c_code or "__fp16" in c_code:
        return "f16"
    if "float" in c_code:
        return "f32"
    return "unknown"


def _spec_files(root: Path) -> list[Path]:
    pats = ["apps/*/src/exo/*.py", "apps/*/*.py", "apps/*/*/*.py", "examples/*/*.py", "examples/*/exo/*.py"]
    seen: dict[Path, None] = {}
    for pat in pats:
        for p in sorted(root.glob(pat)):
            if p.name.startswith("_") or p.name in {"conftest.py"} or p.name.startswith("test_"):
                continue
            seen[p] = None
    return list(seen)


def _load_module(path: Path):
    mod_name = f"_merlin_exo_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    parent = str(path.parent)
    added = parent not in sys.path
    if added:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    finally:
        if added:
            try:
                sys.path.remove(parent)
            except ValueError:
                pass
    return module


def ingest_exo_schedules(repo: str, limit: int | None = None) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels from Exo *schedule* ``.py`` files (no compilation).

    The schedule directives (``set_memory(GEMM_ACCUM)``, ``stage_mem``, ``divide_loop``,
    ``replace_gemmini_calls``, ``acc_scale``/``clamp``/``relu``) are the explicit decision
    record — the strongest evidence form — which compiling to C discards. Mined with the
    ``exo_schedule`` marker family. Pure text; needs no Exo install.
    """
    root = Path(repo).resolve()
    count = 0
    for spec_path in _spec_files(root):
        text = spec_path.read_text(encoding="utf-8", errors="replace")
        # Only schedule files: must invoke at least one scheduling directive.
        if not any(d in text for d in ("set_memory", "stage_mem", "divide_loop",
                                       "replace_all", "replace_gemmini_calls", "tile_outer_loops")):
            continue
        try:
            rel = str(spec_path.relative_to(root))
        except ValueError:
            rel = str(spec_path)
        op = _guess_op(spec_path.stem) if _guess_op(spec_path.stem) != "unknown" else _guess_op(text[:2000])
        yield NormalizedKernel(
            source="exo", target="exo_schedule", path=rel, op=op, dtype="unknown",
            raw_text=text, meta={"kind": "schedule"},
        )
        count += 1
        if limit is not None and count >= limit:
            return


def ingest_exo(
    repo: str,
    target: str | None = None,
    out_dir: str | None = None,
    limit: int | None = None,
    diagnostics: dict | None = None,
) -> Iterator[NormalizedKernel]:
    """Yield NormalizedKernels by compiling Exo specs under ``repo`` to C.

    ``diagnostics`` (if provided) is populated with ``{"specs": n, "compiled": n,
    "skipped": n, "skips": [(path, reason), ...]}`` for honest reporting.
    """
    diag = diagnostics if diagnostics is not None else {}
    diag.setdefault("specs", 0)
    diag.setdefault("compiled", 0)
    diag.setdefault("skipped", 0)
    diag.setdefault("skips", [])

    try:
        import exo  # noqa: F401
        from exo import Procedure, compile_procs_to_strings
    except Exception as e:  # exo not installed / import error
        log.warning("exo unavailable (%s); skipping Exo ingest. `pip install -e .[kernels-exo]`", e)
        diag["skips"].append(("<import exo>", repr(e)))
        return

    root = Path(repo).resolve()
    out = Path(out_dir).resolve() if out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    count = 0

    # Some Exo example modules write generated headers to the *current* directory at import
    # time. Run inside the gitignored output dir so re-runs never litter the repo root.
    prev_cwd = os.getcwd()
    if out:
        os.chdir(out)

    try:
        for spec_path in _spec_files(root):
            diag["specs"] += 1
            try:
                source_text = spec_path.read_text(encoding="utf-8", errors="replace")
                module = _load_module(spec_path)
            except Exception as e:
                diag["skipped"] += 1
                diag["skips"].append((str(spec_path), f"import: {e!r}"))
                log.warning("skip Exo spec %s (import: %s)", spec_path, e)
                continue

            fam = _detect_target(source_text, target)
            try:
                rel = str(spec_path.relative_to(root))
            except ValueError:
                rel = str(spec_path)

            procs = [v for v in vars(module).values() if isinstance(v, Procedure)]
            for proc in procs:
                try:
                    name = getattr(proc, "name", lambda: "proc")()
                except Exception:
                    name = "proc"
                try:
                    c_code, _h = compile_procs_to_strings([proc], "mined.h")
                except Exception as e:
                    diag["skipped"] += 1
                    diag["skips"].append((f"{rel}::{name}", f"compile: {e!r}"))
                    log.warning("skip Exo proc %s::%s (compile: %s)", rel, name, e)
                    continue
                if out:
                    (out / f"{spec_path.stem}__{name}.c").write_text(c_code, encoding="utf-8")
                diag["compiled"] += 1
                yield NormalizedKernel(
                    source="exo", target=fam, path=f"{rel}::{name}",
                    op=_guess_op(name), dtype=_sniff_dtype(c_code), raw_text=c_code,
                    meta={"proc": name},
                )
                count += 1
                if limit is not None and count >= limit:
                    return
    finally:
        os.chdir(prev_cwd)
