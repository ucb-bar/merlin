"""Bind the complete declared capture request and direct source implementations to a cache slot."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.capture.artifacts import _MODEL_CAPTURE_ABI_VERSION
from merlin.common.paths import module_source_path
from merlin.development.targetgen import capture_cache


def capture_slot(
    m2m_dir: Path,
    default_python: Path,
    op: str,
    dtype: str,
    src: str,
    scheme: str | None,
    env: dict | None = None,
    python: Path | None = None,
    recipe_sha256: str = "",
    already_quantized: bool = False,
    agreement_tolerance: tuple[float, float] | None = None,
    static_pt2e: bool = False,
    capabilities: tuple[str, ...] = (),
) -> Path | None:
    """Where a capture of exactly this input already lives, or ``None`` if caching is unavailable.

    Keyed on the declared request -- the loader's source, format, capture scheme and op role.
    Direct Merlin worker/normalization owners are byte-bound too. Imported loader/framework
    dependencies and external data still need independent attribution. Not keyed
    on the target: a capture is a property of the model and the format, and two targets that choose
    the same format are entitled to the same bytes.

    The declared loader ENVIRONMENT and INTERPRETER are part of the key for the same reason the
    source is: they change the program and the operands (which input stream a loader reads, which
    checkpoint it loads, which dependency stack it imports). Keyed on the source alone, a capture
    taken under one input declaration would be served for a request that declared another -- the
    silent kind of substitution this corpus exists to prevent.

    This exists because the roster axis made whole-model captures routine. Four networks per target
    re-captured on every corpus regeneration is tens of minutes of work whose result is identical
    each time, and the alternative that suggests itself -- generating the entries but not building
    them -- is the silent kind of skip this corpus is built to avoid.
    """
    try:
        from merlin.common.artifacts import cache_dir
        from merlin.common.digest import sha256_file

        # A path alone is not a version: active development can replace
        # imported implementation bytes in place. Keep the required direct
        # owners (including the optional static integerizer) as an
        # availability check, and bind *all* Python modules in the selected
        # m2m package. Helpers imported transitively must invalidate the
        # slot too. This still does not close Torch, loader imports or data.
        upstream_files = [
            "m2m/api.py",
            "m2m/ir/import_fx.py",
            "m2m/capture/torchao_pipeline.py",
            "m2m/capture/torchao_schemes.py",
            "m2m/capture/torch_export.py",
            "m2m/capture/torch_mlir_bridge.py",
        ]
        if static_pt2e:
            upstream_files.append("m2m/capture/pt2e_integerize.py")
        upstream_identity = {relative: sha256_file(m2m_dir / relative) for relative in upstream_files}
        # Absence is explicit for older diagnostic producers; adding the
        # trace implementation invalidates their cache identity.
        trace_owner = m2m_dir / "m2m/capture/trace.py"
        upstream_identity["m2m/capture/trace.py"] = sha256_file(trace_owner) if trace_owner.is_file() else None
        package_root = m2m_dir / "m2m"
        if package_root.is_symlink() or not package_root.is_dir():
            return None
        package_sources = {}
        for path in sorted(package_root.rglob("*")):
            # A symlinked package subtree is not an inventory of the bytes
            # Python may import from it; fail closed instead of omitting it.
            if path.is_symlink():
                return None
            if path.suffix != ".py":
                continue
            if not path.is_file():
                return None
            package_sources[path.relative_to(m2m_dir).as_posix()] = sha256_file(path)
        if not package_sources:
            return None

        # Structured encoding preserves field boundaries even when loader source or
        # environment values contain delimiter characters. Historical ambiguous-key
        # slots remain untouched, but are never admitted under this key version.
        request = {
            "key_version": 4,
            "op": op,
            "dtype": str(dtype),
            "scheme": str(scheme or ""),
            "model2mlir_root": str(m2m_dir),
            "loader_source": src,
            "environment": env or {},
            "interpreter": str(python or default_python),
            "capture_abi": _MODEL_CAPTURE_ABI_VERSION,
            "recipe_sha256": recipe_sha256,
            "agreement_tolerance": list(agreement_tolerance) if agreement_tolerance else None,
            "merlin_implementation": capture_cache.implementation_identity(),
            "model2mlir_direct_owners": upstream_identity,
            "model2mlir_python_tree": package_sources,
        }
        if already_quantized:
            request["capture_quantization"] = "already_materialized"
        if capabilities:
            # A capability is part of the program, and so is the module that realises it: its source
            # is keyed by content. Added only when asked for, so every existing slot keeps its key.
            request["capture_capabilities"] = {
                name: sha256_file(module_source_path("merlin.development.targetgen._" + name))
                for name in sorted(capabilities)
            }
        key = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return cache_dir("model_capture") / f"{op}_{dtype}_{key}"
    except Exception:  # noqa: BLE001 -- an unavailable cache is not a failed capture
        return None
