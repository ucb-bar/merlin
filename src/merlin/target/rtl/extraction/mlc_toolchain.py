"""Bridge to the sibling ``mlc`` model-ladder compiler — the CIRCT+xDSL RTL frontend we reuse for
RTL *needle* extraction (finding specific ISA facts in the HW-dialect op graph).

``mlc`` is an EXTERNAL dependency resolved via ``.env MERLIN_MLC_DIR`` (it will be upstreamed to its own
open-source repo). Rather than re-implement the op-graph parser + decoder analysis, we reuse mlc's
``discover.irgraph`` (``circt-opt --mlir-print-op-generic`` -> xDSL ``HwGraph``) and ``discover.decode``
(observed field values from the decoder's ``comb.icmp eq`` fan-out), plus mlc's own prebuilt CIRCT
binaries. ``mlc`` is pip-installed editable into merlin's venv from ``MERLIN_MLC_DIR`` (see the dev
setup docs), so its Python is a normal import. Imports are still function-local behind an availability
guard (``chia_bridge`` style) so importing this module never hard-requires mlc, and a machine without
the mlc package installed degrades honestly rather than crashing. Its NON-python assets (the prebuilt
``circt-opt`` binary, the cached ``runs/circt-arc/<target>`` arc outputs, schemas) still resolve by
directory path via ``MERLIN_MLC_DIR``.

Why this matters: a header parse and an RTL comparison set can disagree. Recording
the selected decoder field exposes that discrepancy; neither alone proves an
exhaustive executable ISA or authorizes a code-generation endpoint.

TARGET-AGNOSTIC: every entry point here takes a ``target`` argument and holds no target name. mlc is
target-parameterized (``artifact_paths(target)`` / ``discovered_memory_map(target)`` /
``discover_opcode_set(graph)`` / per-target ``runs/circt-arc/<target>``), so the same code plugs any HW
RTL repo mlc knows (systolic arrays, NPUs, SIMT cores, CPUs, ...).
"""

from __future__ import annotations

import copy
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.common import external_tools as _modelir

# ---- mlc_bridge --------------------------------------------------------------------------------


def mlc_dir() -> Path | None:
    """The mlc package root from ``.env MERLIN_MLC_DIR``, or None if unset / not an mlc checkout.

    mlc lives outside this repo and has no in-tree fallback. There used to be a hardcoded default
    pointing at one machine's checkout, which meant every other clone silently resolved a path that
    does not exist and reported "no mlc" for the wrong reason -- indistinguishable from an unset
    variable. Callers already handle None by skipping the mlc-derived facts.
    """
    from merlin.common.paths import env

    d = env("MERLIN_MLC_DIR")
    if not d:
        return None
    p = Path(d).expanduser()
    return p if (p / "mlc").is_dir() else None


def circt_opt_bin() -> Path | None:
    """``circt-opt``, which lowers CIRCT custom syntax to generic MLIR for the xDSL ingest.

    Resolved in three places, most explicit first. The binary and the Python package are not the same
    dependency, and tying them together made a checkout the only way to get the tool: mlc's copy lives
    under a ``third_party/`` whose submodules are tens of gigabytes, so a clone that wants one 4 GB
    CIRCT build had to take all of it. Hence:

    1. ``MERLIN_CIRCT_OPT`` -- an explicit path. If it is set and does not exist this returns None
       rather than falling through, because silently using a different binary than the one a caller
       named is how a result gets attributed to the wrong toolchain.
    2. ``PATH`` -- a system or release-tarball install.
    3. mlc's own prebuilt copy -- the historical location, kept last so an existing checkout keeps
       working with nothing set.
    """
    from merlin.common.paths import env

    declared = env("MERLIN_CIRCT_OPT")
    if declared:
        p = Path(declared).expanduser()
        return p if p.exists() else None
    found = shutil.which("circt-opt")
    if found:
        return Path(found)
    d = mlc_dir()
    if d is None:
        return None
    b = d / "third_party" / "circt" / "build" / "bin" / "circt-opt"
    return b if b.exists() else None


def mlc_available() -> tuple[bool, str]:
    """(ok, reason). ok iff MERLIN_MLC_DIR resolves (assets), circt-opt is built, and mlc imports."""
    d = mlc_dir()
    if d is None:
        return False, "MERLIN_MLC_DIR unset/invalid (no mlc/ package) — set it in .env"
    if circt_opt_bin() is None:
        return False, (
            "circt-opt not found: set MERLIN_CIRCT_OPT to it, put it on PATH, or build it under "
            f"{d}/third_party/circt/build/bin"
        )
    try:
        import mlc.discover.decode  # noqa: F401
        import mlc.discover.irgraph  # noqa: F401
    except Exception as e:  # noqa: BLE001 — surface the import failure honestly
        return False, f"mlc import failed: {type(e).__name__}: {e} — pip install -e $MERLIN_MLC_DIR"
    return True, "ok"


def require_mlc() -> None:
    ok, why = mlc_available()
    if not ok:
        raise RuntimeError(f"mlc unavailable: {why}")


def firtool_bin() -> Path | None:
    """``firtool``, which elaborates FIRRTL into the HW dialect the extraction reads.

    Resolved exactly like :func:`circt_opt_bin` -- ``MERLIN_FIRTOOL`` (explicit, and a path that does
    not exist returns None rather than falling through to a different binary), then ``PATH``, then
    mlc's own prebuilt copy. It is resolved here because it is part of the extraction TOOLCHAIN whose
    identity every facts artifact now records; nothing in this module invokes it.
    """
    from merlin.common.paths import env

    declared = env("MERLIN_FIRTOOL")
    if declared:
        p = Path(declared).expanduser()
        return p if p.exists() else None
    found = shutil.which("firtool")
    if found:
        return Path(found)
    d = mlc_dir()
    if d is None:
        return None
    b = d / "third_party" / "circt" / "build" / "bin" / "firtool"
    return b if b.exists() else None


TOOLCHAIN_UNKNOWN = "UNKNOWN"


_TOOL_IDENTITY_CACHE: dict[tuple, dict] = {}


def _binary_identity(name: str, path: Path | None, *, absent_reason: str) -> dict:
    """Identify one toolchain BINARY: its self-reported version, else a digest of its bytes.

    Order is deliberate. A version string is what the tool says about itself and is comparable across
    machines; a digest is comparable only against the same build, but it is still enough to tell two
    extractions apart, which is the question this field exists to answer. When neither is obtainable
    the entry records :data:`TOOLCHAIN_UNKNOWN` **with the reason** -- never an omitted key, because an
    omitted key is indistinguishable from an artifact written before anyone recorded toolchains.
    """
    if path is None:
        return {"tool": name, "version": TOOLCHAIN_UNKNOWN, "unknown_reason": absent_reason}
    try:
        st = path.stat()
    except OSError as exc:
        return {"tool": name, "path": str(path), "version": TOOLCHAIN_UNKNOWN, "unknown_reason": f"{exc}"}
    key = (name, str(path), st.st_size, st.st_mtime_ns)
    hit = _TOOL_IDENTITY_CACHE.get(key)
    if hit is not None:
        return dict(hit)
    out: dict = {"tool": name, "path": str(path), "size_bytes": st.st_size}
    reported, why = _self_reported_version(path)
    if reported is not None:
        out["version"] = reported
    else:
        digest, digest_why = _file_sha256(path)
        if digest is not None:
            # The tool does not report a version, so its BYTES are its identity. Recorded under its own
            # key rather than as `version`, so a reader is never told a digest is a release number.
            out["sha256"] = digest
            out["version"] = TOOLCHAIN_UNKNOWN
            out["unknown_reason"] = f"{why}; identified by the digest of its bytes instead"
        else:
            out["version"] = TOOLCHAIN_UNKNOWN
            out["unknown_reason"] = f"{why}; and the binary could not be digested either ({digest_why})"
    _TOOL_IDENTITY_CACHE[key] = dict(out)
    return out


_VERSION_TEXT_LIMIT = 500


def _self_reported_version(path: Path) -> tuple[str | None, str]:
    """``(version, why-not)`` from ``<bin> --version``.

    ALL the non-empty lines, joined -- not the first one. Measured on the CIRCT build this repo uses,
    ``circt-opt --version`` opens with ``LLVM (http://llvm.org/):`` and states the thing that actually
    identifies the build, ``CIRCT <sha>``, on its FOURTH line. Keeping the first line would have
    recorded a constant for every CIRCT build ever made, which is worse than recording nothing because
    it looks like an answer. Structural: ``splitlines`` + ``strip``, no pattern matching, so a tool
    that lays its version out differently is still recorded in full rather than dropped.
    """
    import subprocess

    try:
        proc = subprocess.run(  # noqa: S603 - a path this repo resolved, with a fixed argument
            [str(path), "--version"], capture_output=True, text=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{path.name} --version could not be run ({type(exc).__name__}: {exc})"
    if proc.returncode != 0:
        return None, f"{path.name} --version exited {proc.returncode}"
    lines = [ln.strip() for ln in (proc.stdout + "\n" + proc.stderr).splitlines() if ln.strip()]
    if not lines:
        return None, f"{path.name} --version printed nothing"
    return " | ".join(lines)[:_VERSION_TEXT_LIMIT], "ok"


def _file_sha256(path: Path) -> tuple[str | None, str]:
    """``(digest, why-not)`` for a file's bytes, read in chunks (these binaries are large)."""
    import hashlib

    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        return None, f"{exc}"
    return h.hexdigest(), "ok"


def _mlc_identity() -> dict:
    """Identify the mlc CHECKOUT + package -- the Python half of the extraction toolchain.

    mlc is an editable install of a git checkout, so its identity is that checkout's revision. The
    revision is READ, never written: this asks git for HEAD and whether the tree is dirty, and a dirty
    tree is recorded as such because the commit alone then does not describe the bytes that ran.
    """
    d = mlc_dir()
    if d is None:
        return {
            "tool": "mlc",
            "version": TOOLCHAIN_UNKNOWN,
            "unknown_reason": "MERLIN_MLC_DIR is unset or does not resolve to an mlc checkout",
        }
    out: dict = {"tool": "mlc", "path": str(d)}
    declared = None
    try:
        import mlc  # noqa: PLC0415 - optional external package, probed on purpose

        declared = getattr(mlc, "__version__", None)
    except Exception as exc:  # noqa: BLE001 — not importable is itself part of the identity
        out["import_error"] = f"{type(exc).__name__}: {exc}"
    if isinstance(declared, str) and declared:
        out["version"] = declared
    head, why = _git_head(d)
    if head is not None:
        out["commit"] = head
    if "version" not in out:
        if head is not None:
            out["version"] = TOOLCHAIN_UNKNOWN
            out["unknown_reason"] = "the mlc package declares no __version__; identified by its checkout commit"
        else:
            out["version"] = TOOLCHAIN_UNKNOWN
            out["unknown_reason"] = (
                f"the mlc package declares no __version__ and its checkout revision is unreadable ({why})"
            )
    return out


def _git_head(checkout: Path) -> tuple[str | None, str]:
    """``(commit, why-not)`` for a checkout, with ``-dirty`` appended when the tree has changes.

    READ-ONLY, like every other provenance read in this repo: other sessions work in these trees, so
    nothing here writes, stashes or checks anything out.
    """
    import subprocess

    def _run(args: list[str]) -> tuple[int, str]:
        try:
            proc = subprocess.run(  # noqa: S603 - fixed git arguments against a resolved checkout
                ["git", "-C", str(checkout), *args], capture_output=True, text=True, timeout=30, check=False
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return 1, f"{type(exc).__name__}: {exc}"
        return proc.returncode, proc.stdout.strip() or proc.stderr.strip()

    rc, head = _run(["rev-parse", "HEAD"])
    if rc != 0 or not head:
        return None, head or "git rev-parse HEAD failed"
    rc, status = _run(["status", "--porcelain"])
    return (head + ("-dirty" if rc == 0 and status else "")), "ok"


def toolchain_identity() -> dict[str, dict]:
    """WHICH BUILD of the extraction toolchain is about to read the RTL, as recordable data.

    The gap this closes. A facts artifact records the bytes it READ -- ``core_hw_sha256`` for the input
    IR, ``extractor_sha`` for merlin's own reader -- and recorded nothing at all about the CIRCT/firtool
    build that parsed them. Two artifacts extracted from identical RTL by different CIRCT builds were
    therefore byte-comparable and indistinguishable, and a fact could not be bound to the toolchain
    revision that produced it. The repo's hardware-provenance rule is that a result records which
    revision it came from; the extractor toolchain is part of that revision, not outside it.

    Every tool gets an entry whether or not it was found, and an entry whose identity could not be
    determined carries :data:`TOOLCHAIN_UNKNOWN` plus ``unknown_reason``. A missing tool is a visible
    fact about the extraction; an omitted key would be invisible, and indistinguishable from an
    artifact written before this field existed.

    DERIVED, never declared: nothing here names a version, a release or a path literal -- each entry is
    what the resolved binary says about itself, or the digest of its own bytes.
    """
    return {
        "circt_opt": _binary_identity(
            "circt-opt",
            circt_opt_bin(),
            absent_reason="circt-opt was not resolved (MERLIN_CIRCT_OPT unset/nonexistent, not on PATH, "
            "and no prebuilt copy under $MERLIN_MLC_DIR/third_party/circt/build/bin)",
        ),
        "firtool": _binary_identity(
            "firtool",
            firtool_bin(),
            absent_reason="firtool was not resolved (MERLIN_FIRTOOL unset/nonexistent, not on PATH, and "
            "no prebuilt copy under $MERLIN_MLC_DIR/third_party/circt/build/bin)",
        ),
        "mlc": _mlc_identity(),
    }


def matmul_reuse_prediction(
    M: int, N: int, K: int, *, dim: int, capacity_bytes: int, elem_bytes: int = 1
) -> dict | None:
    """M3 STATIC data-movement prediction for an M×K·K×N matmul — no arc, no sim, no cycles.

    Reuses mlc's pure-arithmetic Model-3 functions (``predict_dma_volume.matmul_refetch_factor`` +
    ``spills.matmul_footprint_rows``) so the reuse/spill math is the mlc source of truth, not a
    re-derivation here. Returns the resident operand footprint (rows), the on-chip capacity (rows), a
    fits flag, and the loop-nest refetch factor (=1 when the operands are fully resident; >1 when they
    spill and the outer loop re-streams). Returns None (fail-closed) when mlc is not importable — these
    functions need only the mlc package installed, NOT circt, so we gate on the import alone."""
    try:
        from mlc.passes.predict_dma_volume import ReuseFacts, matmul_refetch_factor
        from mlc.passes.spills import matmul_footprint_rows
    except Exception:  # noqa: BLE001 — mlc absent/unimportable ⇒ honest unavailable, never a guess
        return None
    # beat_bytes is irrelevant to the refetch/footprint math (it only scales byte→beat counts), so a
    # benign 1 keeps us from fabricating a DMA beat width we have not discovered for this target.
    facts = ReuseFacts(beat_bytes=1, dim=int(dim), capacity_bytes=int(capacity_bytes))
    refetch = int(matmul_refetch_factor(M, K, N, facts, in_bytes=elem_bytes))
    footprint_rows = int(matmul_footprint_rows(M, N, K, int(dim)))
    capacity_rows = int(capacity_bytes) // (int(dim) * max(elem_bytes, 1))
    return {
        "footprint_rows": footprint_rows,
        "capacity_rows": capacity_rows,
        "fits": footprint_rows <= capacity_rows,
        "refetch": refetch,
        "footprint_tiles": footprint_rows // int(dim),
    }


_ARC_TARGET_CACHE: dict[str, str] = {}


def _arc_target(target: str) -> str:
    """The mlc arc/fingerprint key for a merlin ``target``. A composite/aliased target — e.g. a SIMT SoC
    whose bit-exact arc model mlc registers under the embedding cluster's name, not the merlin-facing
    name — declares ``arc_target`` in its ``contracts/residual.yaml``; every other target maps to itself.
    DERIVED from the residual (no target-name literal here), fail-closed to the target itself when there
    is no residual / no alias. Applied ONLY on the ORACLE path — ``arc_available`` / ``arc_core`` (the
    bit-exact model) and the program-oracle cosim lookups, plus the SIMT fact delegation in
    :func:`_simt_fact_bundle`. It is deliberately NOT applied to the structural-fact discovery
    (``core_hw_mlir`` / ``discovered_dim`` / ``discovered_memory_map`` / ``discovered_memories``): a SIMT
    SoC must not inherit the embedding cluster's systolic geometry as if it were its own RoCC mesh (that
    would fabricate spatial levers). The merlin-facing identity (corpus, contract, suite, results) stays
    the original ``target`` everywhere. The residual is a plain YAML side-input, so this never triggers
    manifest derivation (which would re-enter the mlc boundary through this resolver)."""
    if target in _ARC_TARGET_CACHE:
        return _ARC_TARGET_CACHE[target]
    alias = target
    try:
        from merlin.target.capabilities.manifest import _load_residual

        declared = (_load_residual(target) or {}).get("arc_target")
        if isinstance(declared, str) and declared:
            alias = declared
    except Exception:  # noqa: BLE001 — no residual / no alias ⇒ identity
        alias = target
    _ARC_TARGET_CACHE[target] = alias
    return alias


def core_hw_mlir(target: str) -> Path | None:
    """The CORE HW dialect (command decoder) for ANY target: the selection's, else the selected facts'
    own digest-checked one (``recorded_hw``), else mlc's arc outputs, preferring ``*_core_hw.mlir``."""
    from merlin.target.rtl.extraction.source_custody import recorded_core_hw
    from merlin.target.rtl.extraction.source_selection import active_selection

    selected = active_selection(target)
    if selected is not None:
        return Path(selected["sources"]["core_hw"]["path"])
    recorded, d = recorded_core_hw(target), mlc_dir()
    if recorded is not None or d is None:
        return recorded
    outs = d / "runs" / "circt-arc" / target / "outputs"
    cands = sorted(outs.glob("*_core_hw.mlir")) or sorted(outs.glob("*_hw.mlir"))
    return next((p for p in cands if p.exists() and ".generic." not in p.name), None)


ARC_MODELS_FILE: tuple[str, ...] = ("contract", "arc_models.yaml")


_ARC_MODELS_SCHEMA_VERSION = 1


_ARC_MODEL_FIELDS = frozenset({"kind", "isa_encoding"})


class ArcModelRegistryError(RuntimeError):
    """The arc-model registry is missing or malformed. Raised, never read as empty: an empty registry
    would make every model look like one with no kind and no encoding fact, which reads as a finding."""


_ARC_MODELS_CACHE: dict[str, dict[str, dict]] = {}


def load_arc_models(path: str | Path | None = None) -> dict[str, dict]:
    """The arc-model registry as ``{arc_key: {field: value}}``.

    Fails closed on a missing file, a wrong ``schema_version``, an unknown field, a ``kind`` no family
    profile declares, or an ``isa_encoding`` that is not a bare file name. ``path`` defaults to
    :data:`ARC_MODELS_FILE` resolved through ``common.paths.data_path`` (the checkout's tree, else the copy
    bundled in the wheel)."""
    from merlin.common.paths import data_path

    p = Path(path) if path is not None else data_path(*ARC_MODELS_FILE)
    key = str(p)
    if key in _ARC_MODELS_CACHE:
        return _ARC_MODELS_CACHE[key]
    if not p.is_file():
        raise ArcModelRegistryError(
            f"no arc-model registry at {p}; mlc arc models are declared in merlin/{'/'.join(ARC_MODELS_FILE)}"
        )
    import yaml

    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("arc_models"), dict):
        raise ArcModelRegistryError(f"{p}: expected a mapping with an `arc_models:` mapping of key -> fields")
    if raw.get("schema_version") != _ARC_MODELS_SCHEMA_VERSION:
        raise ArcModelRegistryError(
            f"{p}: schema_version {raw.get('schema_version')!r}, this loader reads {_ARC_MODELS_SCHEMA_VERSION}"
        )
    from merlin.target.capabilities.families import known_kinds

    out: dict[str, dict] = {}
    for name, entry in raw["arc_models"].items():
        where = f"{p}: arc model {name!r}"
        if not isinstance(name, str) or not isinstance(entry, dict):
            raise ArcModelRegistryError(f"{where}: an entry is `<arc key>: {{field: value, ...}}`")
        unknown = sorted(set(entry) - _ARC_MODEL_FIELDS)
        if unknown:
            raise ArcModelRegistryError(f"{where}: unknown field(s) {unknown}; known: {sorted(_ARC_MODEL_FIELDS)}")
        kind = entry.get("kind")
        if kind is not None and kind not in known_kinds():
            raise ArcModelRegistryError(
                f"{where}: kind {kind!r} is not a known compute-unit kind {sorted(known_kinds())}"
            )
        isa = entry.get("isa_encoding")
        if isa is not None and (not isinstance(isa, str) or not isa or Path(isa).name != isa):
            raise ArcModelRegistryError(
                f"{where}: isa_encoding {isa!r} must be a bare file name under the model's arc outputs directory"
            )
        out[name] = dict(entry)
    _ARC_MODELS_CACHE[key] = out
    return out


def arc_model(arc_key: str) -> dict:
    """The registry entry for one mlc arc key, or ``{}`` when the registry does not list it."""
    return dict(load_arc_models().get(arc_key) or {})


def isa_encoding_artifact(target: str) -> tuple[Path | None, str]:
    """``(path, reason)`` for the encoding-fact file ``target``'s arc model declares.

    The file lives under ``runs/circt-arc/<arc key>/outputs/`` and its name comes from the model's
    ``isa_encoding`` entry in the arc-model registry. The arc key is the ORACLE-path alias
    (:func:`_arc_target`), because the encoding is the aliased CORE's own ISA (a SIMT SoC's decoder is the
    embedded core's), unlike structural geometry. ``path`` is None, with ``reason`` naming what is missing,
    when mlc is unresolvable or the model declares no encoding fact: no file name is ever guessed."""
    d = mlc_dir()
    if d is None:
        return None, "MERLIN_MLC_DIR unset/invalid (no mlc/ package), so no arc outputs are resolvable"
    key = _arc_target(target)
    name = arc_model(key).get("isa_encoding")
    if not name:
        return None, (
            f"arc model {key!r} declares no `isa_encoding` in merlin/{'/'.join(ARC_MODELS_FILE)}, "
            f"so {target!r} has no encoding fact to read"
        )
    return d / "runs" / "circt-arc" / key / "outputs" / str(name), "declared"


def isa_encoding_for(target: str) -> dict | None:
    """The mlc-derived instruction-ENCODING fact (``{inst_width, fields, opcodes, provenance}``) for a
    target: the field bit-ranges + opcode table the mlc ``isa_encoding`` pass recovers from the target's
    RTL decoder, read from the file :func:`isa_encoding_artifact` resolves. Returns the parsed fact, or
    None when mlc / the declaration / the cache is absent — an honest fallback the caller degrades on,
    never a guessed encoding."""
    p, _why = isa_encoding_artifact(target)
    if p is None or not p.is_file():
        return None
    try:
        import json

        fact = json.loads(p.read_text())
        return fact if isinstance(fact, dict) and fact.get("fields") else None
    except Exception:  # noqa: BLE001 — unreadable/garbled cache ⇒ honest None
        return None


def mx_mmio_for(target: str) -> dict | None:
    """The target's MX accelerator MMIO command ABI (``{ctrl_base, reg_offsets, inst_word, funct,
    sf_mem, gpu_dram_offset, dim, group, config_ex, bounds, loop_ws, mxquant_config_mvout, read_smem}``) —
    the memory-mapped RoCC command surface a SIMT kernel pushes to drive the block-scaled matmul PE.
    This is an ABI the RTL decoder cannot ground (like a RoCC target's ``encoding`` residual), so it is declared,
    header-derived + human-reviewed, in the target's ``contracts/residual.yaml`` under ``mx_mmio``. Returns
    the parsed fact, or None when the target ships no MX MMIO contract — an honest fallback the emitter/harness
    degrade on (the mxfp8 op is unsupported), never a guessed base/opcode."""
    try:
        from merlin.target.capabilities.manifest import _load_residual

        fact = (_load_residual(target) or {}).get("mx_mmio")
        return fact if isinstance(fact, dict) and fact.get("ctrl_base") is not None else None
    except Exception:  # noqa: BLE001 — no residual / unreadable ⇒ honest None
        return None


def arc_artifact_paths(target: str) -> dict | None:
    """Resolved ``{hw, man, so}`` artifact paths for an arc-modelled target via mlc's per-target
    fingerprint map — which knows the two-level ``runs/circt-arc/<family>/<config>/outputs`` layout the
    OuterProductUnit uses, UNLIKE the flat one-level layout
    :func:`core_hw_mlir` assumes. Absolute paths under the mlc dir. None when mlc is not resolvable or
    the target is not a known mlc arc target — an honest fallback, never a guessed path.

    Only mlc's lightweight ``discover.fingerprint`` is imported (pure pathlib; no circt binary), so OPU
    fact extraction works even when the full ``mlc_available`` gate (circt-opt built) is not met — and
    mlc is imported by-path for this call only (context-managed) so it grounds even when mlc is a sibling
    checkout rather than a pip-installed package."""
    d = mlc_dir()
    if d is None:
        return None
    try:
        with _modelir.discovery_imports(d):
            from mlc.discover.fingerprint import artifact_paths

            return {k: Path(v) for k, v in artifact_paths(target, base=d).items()}
    except Exception:  # noqa: BLE001 — unknown target / mlc layout skew ⇒ honest unavailable
        return None


def compute_unit_dtypes(target: str) -> dict | None:
    """Per-compute-unit INPUT dtype map ``{unit_name: [dtype, ...]}`` for ANY target, DERIVED structurally
    from the RTL facts by mlc's GENERAL datapath-dtype extractor
    (:func:`mlc.discover.datapath_dtypes.compute_unit_dtypes_detail` — typed MAC-mesh/FPU + spatial OPU
    families, no per-target ``if``). This is the general lift of the OPU-only dtype code that grounded
    only saturn_opu; it reproduces that same value (``saturn_opu -> [int8, fp8_e4m3, fp8_e5m2]``) and also
    grounds gemmini (``[int8]``) / atlas (``[fp8_e4m3]``) / an FPU core (``[fp16, f32, f64]``).

    Returns ``None`` — an honest fallback the caller degrades to the existing behavior on — when mlc is
    unresolvable OR the extractor reports the target's facts carry no derivable compute datapath
    (``supported=False``); never a fabricated dtype. mlc is imported by-path for this call only
    (context-managed; never a global sys.path insert that would flip ``mlc_available`` process-wide)."""
    d = mlc_dir()
    if d is None:
        return None
    try:
        with _modelir.discovery_imports(d):
            from mlc.discover.datapath_dtypes import compute_unit_dtypes_detail

            detail = compute_unit_dtypes_detail(target, base=d)
    except Exception:  # noqa: BLE001 — mlc absent / layout skew ⇒ honest unavailable, never a guess
        return None
    if not detail.get("supported"):
        return None
    return {u["unit"]: u["dtypes"] for u in detail.get("units", [])}


@dataclass(frozen=True)
class _EquivalentDecodeSignal:
    module: str
    width: int
    values: tuple[int, ...]
    fanout: int
    cloned_expressions: int


def _equivalent_decode_signals(graph) -> list[_EquivalentDecodeSignal]:
    """Find equality fan-outs split by cloned, equivalent CIRCT expressions.

    The upstream discovery groups comparisons by SSA value. CIRCT may duplicate
    ``comb.extract`` and ``comb.concat`` for every comparison, although every
    expression reads the same input bits. Canonicalize only those two pure ops;
    everything else remains an opaque SSA leaf. Grouping stays within one HW
    module, so a same-width subfield in another module cannot join this signal.
    This is an integration repair, not an instruction-set completeness proof.
    """
    cache: dict[int, tuple] = {}

    def expression(value) -> tuple:
        key = id(value)
        if key in cache:
            return cache[key]
        owner = graph.defining_op(value)
        name = getattr(getattr(owner, "op_name", None), "data", None)
        if name == "comb.extract" and len(owner.operands) == 1:
            bit_range = graph.extract_range(owner)
            result = (name, bit_range, str(value.type), expression(owner.operands[0])) if bit_range else ("ssa", key)
        elif name == "comb.concat" and owner.operands:
            result = (name, str(value.type), tuple(expression(operand) for operand in owner.operands))
        else:
            result = ("ssa", key)
        cache[key] = result
        return result

    buckets: dict[tuple, dict[str, set[int]]] = {}
    for module in graph.modules.values():
        for compare in graph.ops("comb.icmp", within=module):
            if graph.icmp_predicate(compare) != 0 or len(compare.operands) != 2:
                continue
            a, b = compare.operands
            left, right = graph.defining_op(a), graph.defining_op(b)
            left_const = left is not None and getattr(left.op_name, "data", None) == "hw.constant"
            right_const = right is not None and getattr(right.op_name, "data", None) == "hw.constant"
            if left_const == right_const:
                continue
            const, signal = (left, b) if left_const else (right, a)
            value = graph.const_value(const)
            typ = str(signal.type)
            width = int(typ[1:]) if typ.startswith("i") and typ[1:].isdigit() else 0
            if value is None or width == 0 or width > 16:
                continue
            expr = expression(signal)
            if expr[0] == "ssa":
                continue  # upstream already handles the identical-SSA case
            bucket = buckets.setdefault((module.name, width, expr), {"values": set(), "signals": set()})
            bucket["values"].add(value & ((1 << width) - 1))
            bucket["signals"].add(id(signal))

    out: list[_EquivalentDecodeSignal] = []
    for (module, width, _expr), bucket in buckets.items():
        values = bucket["values"]
        if len(bucket["signals"]) < 2 or len(values) < 3 or len(values) >= 0.5 * (1 << width):
            continue
        out.append(_EquivalentDecodeSignal(module, width, tuple(sorted(values)), len(values), len(bucket["signals"])))
    return sorted(out, key=lambda item: (-item.fanout, -item.width, item.module, item.values))


def discover_legal_opcodes(target: str, *, opcode_width: int | None = None) -> dict:
    """Observe one decoder comparison field in a target's core HW dialect.

    ``legal_opcodes`` is a legacy key for observed field values, not a complete
    instruction-set claim. Scope and completeness are returned explicitly.
    """
    require_mlc()
    hw = core_hw_mlir(target)
    if hw is None:
        return {
            "legal_opcodes": None,
            "width": opcode_width,
            "fanout": 0,
            "module": None,
            "hw_source": None,
            "method": "decoder_icmp_fanout(mlc)",
            "evidence": f"no core HW dialect for target {target!r} under mlc runs/circt-arc",
            "scope": "observed_decode_field",
            "complete_isa": False,
        }
    from mlc.discover import decode

    from merlin.target.rtl.hw.graph import load_hw_graph

    graph = load_hw_graph(hw, circt_opt=circt_opt_bin())
    sig = decode.discover_opcode_set(graph, expected_width=opcode_width)
    recovered = [
        candidate
        for candidate in _equivalent_decode_signals(graph)
        if opcode_width is None or candidate.width == opcode_width
    ]
    if recovered and (sig is None or recovered[0].fanout > sig.fanout):
        top = recovered[0]
        # Equal-strength expressions with different values do not identify a
        # unique decoder. Never union them just because their widths match.
        tied = [candidate for candidate in recovered if candidate.fanout == top.fanout]
        if any(candidate.values != top.values for candidate in tied[1:]):
            return {
                "legal_opcodes": None,
                "width": top.width,
                "fanout": top.fanout,
                "module": None,
                "hw_source": str(hw),
                "method": "decoder_icmp_fanout(equivalent_expressions)",
                "evidence": "ambiguous equal-strength equivalent-expression decode signals",
                "scope": "observed_decode_field",
                "complete_isa": False,
            }
        return {
            "legal_opcodes": list(top.values),
            "width": top.width,
            "fanout": top.fanout,
            "module": top.module,
            "hw_source": str(hw),
            "method": "decoder_icmp_fanout(equivalent_expressions)",
            "evidence": (
                f"comb.icmp-eq {top.width}-bit comparisons over {top.cloned_expressions} "
                f"structurally equivalent comb.extract/concat expressions in module {top.module} "
                f"-> {len(top.values)} field values; this is not a complete instruction-set proof"
            ),
            "scope": "observed_decode_field",
            "complete_isa": False,
        }
    if sig is None:
        return {
            "legal_opcodes": None,
            "width": opcode_width,
            "fanout": 0,
            "module": None,
            "hw_source": str(hw),
            "method": "decoder_icmp_fanout(mlc)",
            "evidence": f"no decode signal in {hw.name}",
            "scope": "observed_decode_field",
            "complete_isa": False,
        }
    legal = sorted(int(v) for v in sig.values)
    return {
        "legal_opcodes": legal,
        "width": sig.width,
        "fanout": sig.fanout,
        "module": sig.module,
        "hw_source": str(hw),
        "method": "decoder_icmp_fanout(mlc)",
        "evidence": f"union of comb.icmp-eq {sig.width}-bit decode signals in module {sig.module} "
        f"({sig.fanout} comparisons) -> {len(legal)} observed field values; not a complete ISA proof",
        "scope": "observed_decode_field",
        "complete_isa": False,
    }
