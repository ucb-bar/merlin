"""Precompile kernel sources to per-target object files.

Inputs: a `Manifest` (see `manifest.py`).
Outputs: one or more object artifacts per kernel — keyed by `(kernel_name,
target)` — written under a SHA-keyed cache so repeated `./merlin compile`
invocations don't recompile unchanged kernels.

Toolchain dispatch table:

    source_lang × target family                  driver
    ─────────────────────────────────────────────────────────
    cl          × *-vulkan                       clspv
    glsl        × *-vulkan                       glslangValidator
    spirv       × *-vulkan                       passthrough
    c / cpp     × llvm-cpu / *-llvm-cpu          host clang (-target ...-none-elf)
    c / cpp     × radiance-muon                  llvm-muon clang ($LLVM_MUON)
    ll          × radiance-muon                  llvm-muon clang -c (LLVM IR)

Each driver produces a single binary blob of the right kind for the
`hal.executable.objects` attribute IREE expects:
    .spv  for vulkan-spirv
    .o    for llvm-cpu / radiance-muon

The cache key is `sha256(source_bytes || tool_version || flags)` so that
toolchain upgrades or flag changes invalidate cached artifacts automatically.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import pathlib
import shutil
import subprocess
from collections.abc import Sequence

from . import manifest as _manifest

_LOG = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ObjectArtifact:
    kernel_name: str
    target: str
    path: pathlib.Path  # absolute, lives under the cache dir
    spirv: bool  # whether the file is SPIR-V (vs ELF/.o or .qnn-ctx)
    qnn_context: bool = False  # True when path is a .qnn-ctx blob


@dataclasses.dataclass(frozen=True)
class Toolchain:
    """Resolved external tools and their versions, computed once per session."""

    clspv: pathlib.Path | None
    clspv_version: str | None
    glslang: pathlib.Path | None
    glslang_version: str | None
    clang: pathlib.Path | None
    clang_version: str | None
    spirv_val: pathlib.Path | None
    # Vendored llvm-muon clang for Muon-family targets (radiance-muon, etc.).
    # Resolved from $LLVM_MUON env or NULL when unset (the manifest pipeline
    # then surfaces a precise error only when a Muon-targeted kernel actually
    # gets compiled — matches the same-day-failure policy of the rest of
    # this module).
    llvm_muon_clang: pathlib.Path | None
    llvm_muon_clang_version: str | None
    radiance_kernels_root: pathlib.Path | None


def _which(name: str) -> pathlib.Path | None:
    p = shutil.which(name)
    return pathlib.Path(p) if p else None


def _run_text(args: Sequence[str], cwd: pathlib.Path | None = None) -> str:
    out = subprocess.run(args, capture_output=True, text=True, cwd=cwd, check=False)
    return (out.stdout + out.stderr).strip()


def detect_toolchain() -> Toolchain:
    """Probe PATH for external tools. Missing tools are tolerated; the
    precompile dispatch table will surface a precise error when a kernel
    actually requires the missing tool.
    """
    clspv = _which("clspv")
    glslang = _which("glslangValidator") or _which("glslang")
    clang = _which("clang") or _which("clang-23") or _which("clang-21")
    spirv_val = _which("spirv-val")

    # llvm-muon: vendored bare-metal RV32 clang for Muon GPU kernels.
    # Resolved from $LLVM_MUON (set by the radiance-kernels build env). Not
    # on PATH by default; we look up via env, then probe $LLVM_MUON/bin/clang.
    llvm_muon_root_env = os.environ.get("LLVM_MUON")
    llvm_muon_clang: pathlib.Path | None = None
    if llvm_muon_root_env:
        candidate = pathlib.Path(llvm_muon_root_env) / "bin" / "clang"
        if candidate.is_file():
            llvm_muon_clang = candidate

    radiance_root_env = os.environ.get("RADIANCE_KERNELS_ROOT")
    radiance_root: pathlib.Path | None = None
    if radiance_root_env and pathlib.Path(radiance_root_env).is_dir():
        radiance_root = pathlib.Path(radiance_root_env)

    return Toolchain(
        clspv=clspv,
        clspv_version=_run_text([str(clspv), "--version"]) if clspv else None,
        glslang=glslang,
        glslang_version=_run_text([str(glslang), "--version"]) if glslang else None,
        clang=clang,
        clang_version=_run_text([str(clang), "--version"]) if clang else None,
        spirv_val=spirv_val,
        llvm_muon_clang=llvm_muon_clang,
        llvm_muon_clang_version=(_run_text([str(llvm_muon_clang), "--version"]) if llvm_muon_clang else None),
        radiance_kernels_root=radiance_root,
    )


def _is_vulkan_target(target: str) -> bool:
    # Accepts both legacy ("vulkan-spirv") and Merlin-style targets
    # ("qualcomm-adreno-vulkan", "vulkan-cpu-lavapipe", etc).
    return "vulkan" in target


def _is_cpu_target(target: str) -> bool:
    return "llvm-cpu" in target or target == "host-cpu" or _is_radiance_muon_target(target)


def _is_radiance_muon_target(target: str) -> bool:
    """Match `radiance-muon` and `llvm-cpu-radiance-muon` (compat alias)."""
    return target == "radiance-muon" or target == "llvm-cpu-radiance-muon"


# QNN-specific helpers + dispatch live in kernels.qnn.precompile_extras
# and are imported lazily at the dispatch site below.


def _cache_key(source_bytes: bytes, tool_version: str, flags: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(source_bytes)
    h.update(b"\0\0")
    h.update(tool_version.encode("utf-8"))
    h.update(b"\0\0")
    for f in flags:
        h.update(f.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def _compile_cl_to_spv(
    source: pathlib.Path,
    out: pathlib.Path,
    tc: Toolchain,
) -> list[str]:
    if not tc.clspv:
        raise RuntimeError(
            f"compiling OpenCL ({source}) requires clspv on PATH "
            "(install via: conda install -n merlin-dev -c conda-forge clspv)"
        )
    flags = [
        "--fp16",
        "--cl-std=CL2.0",
        "--spv-version=1.3",
        "--inline-entry-points",
        "-O3",
    ]
    cmd = [str(tc.clspv), *flags, str(source), "-o", str(out)]
    _LOG.info("clspv: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"clspv failed (exit {res.returncode}) for {source}:\n"
            f"--- stderr ---\n{res.stderr}\n--- stdout ---\n{res.stdout}"
        )
    return flags


def _compile_glsl_to_spv(
    source: pathlib.Path,
    out: pathlib.Path,
    tc: Toolchain,
) -> list[str]:
    if not tc.glslang:
        raise RuntimeError(
            f"compiling GLSL ({source}) requires glslangValidator on PATH "
            "(install via: conda install -n merlin-dev -c conda-forge glslang)"
        )
    flags = ["--target-env", "vulkan1.3", "-V", "-S", "comp"]
    cmd = [str(tc.glslang), *flags, str(source), "-o", str(out)]
    _LOG.info("glslang: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"glslangValidator failed (exit {res.returncode}) for {source}:\n"
            f"--- stderr ---\n{res.stderr}\n--- stdout ---\n{res.stdout}"
        )
    return flags


def _compile_c_to_obj(
    source: pathlib.Path,
    out: pathlib.Path,
    arch_triple: str,
    tc: Toolchain,
    *,
    extra_arch_flags: list[str] | None = None,
) -> list[str]:
    if not tc.clang:
        raise RuntimeError(f"compiling C ({source}) requires clang on PATH")
    # Mirrors samples/custom_dispatch/cpu/embedded/CMakeLists.txt:29-58.
    flags = [
        f"--target={arch_triple}",
        "-ffreestanding",
        "-fvisibility=hidden",
        "-fno-plt",
        "-fno-rtti",
        "-fno-exceptions",
        "-c",
        "-O3",
    ]
    if extra_arch_flags:
        flags.extend(extra_arch_flags)
    cmd = [str(tc.clang), *flags, str(source), "-o", str(out)]
    _LOG.info("clang: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(f"clang failed (exit {res.returncode}) for {source}:\n{res.stderr}")
    return flags


def _radiance_muon_baseline_flags(tc: Toolchain) -> list[str]:
    """Baseline CFLAGS for the radiance-muon target.

    Mirrors radiance-kernels/kernels/common.mk MU_CFLAGS exactly. The
    `--target=` flag is implicit — llvm-muon clang's default target is
    riscv32-unknown-unknown-elf, which is what the upstream Make recipe
    relies on. Include paths come from $RADIANCE_KERNELS_ROOT/lib.

    Drift between this list and common.mk is detectable via the byte-
    equivalence smoke (compile vecadd_body.cpp via this path; check the
    .o is sized like the upstream kernel.mu.o under the same flags).
    """
    if tc.radiance_kernels_root is None:
        raise RuntimeError(
            "radiance-muon: RADIANCE_KERNELS_ROOT env var must be set so "
            "the precompile step can locate libmuonrt headers under "
            "$RADIANCE_KERNELS_ROOT/lib/include"
        )
    if tc.llvm_muon_clang is None:
        raise RuntimeError("radiance-muon: $LLVM_MUON not set or " "$LLVM_MUON/bin/clang missing")

    radiance_lib = tc.radiance_kernels_root / "lib"
    radiance_include = radiance_lib / "include"
    gemmini_sw = radiance_lib / "mxgemmini"
    llvm_muon_root = tc.llvm_muon_clang.parent.parent

    return [
        f"--sysroot={llvm_muon_root}",
        "-Xclang",
        "-target-feature",
        "-Xclang",
        "+vortex",
        "-march=rv32im_zfinx_zhinx",
        "-mabi=ilp32",
        "-O3",
        "-std=c++20",
        "-mcmodel=medany",
        "-fno-rtti",
        "-fno-exceptions",
        "-fdata-sections",
        "-ffunction-sections",
        "-mllvm",
        "-inline-threshold=262144",
        f"-I{radiance_include}",
        f"-I{gemmini_sw}",
        "-DRADIANCE",
        "-DRADIANCE_DEVICE",
        "-DNDEBUG",
        "-DLLVM_VORTEX",
        "-c",
    ]


def _compile_radiance_muon_obj(
    source: pathlib.Path,
    out: pathlib.Path,
    tc: Toolchain,
) -> list[str]:
    """Compile a Muon kernel source (.c / .cpp / .ll) into a .o using
    llvm-muon clang with the canonical Muon flag set.

    Uses the *exact* CFLAGS from radiance-kernels/kernels/common.mk so the
    object file is link-compatible with libmuonrt.a + tohost.S + mu_link.ld.
    """
    if tc.llvm_muon_clang is None:
        raise RuntimeError(f"compiling Muon kernel ({source}) requires $LLVM_MUON to be set")

    flags = _radiance_muon_baseline_flags(tc)
    cmd = [str(tc.llvm_muon_clang), *flags, str(source), "-o", str(out)]
    _LOG.info("llvm-muon clang: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"llvm-muon clang failed (exit {res.returncode}) for {source}:\n"
            f"--- stderr ---\n{res.stderr}\n--- stdout ---\n{res.stdout}"
        )
    return flags


def _validate_spv(path: pathlib.Path, tc: Toolchain) -> None:
    """Optional structural validation. Skipped silently if spirv-val is
    missing — clspv/glslang are the source of truth, spirv-val is a
    secondary check."""
    if not tc.spirv_val:
        return
    res = subprocess.run(
        [str(tc.spirv_val), "--target-env", "vulkan1.3", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"spirv-val rejected {path}:\n{res.stdout}\n{res.stderr}")


# Default arch triples per HAL target key. Extended on demand; missing keys
# raise a precise error at compile time rather than silently miscompiling.
#
# Each entry is `(triple, extra_flags_in_addition_to_the_baseline)`. The
# baseline already includes `-ffreestanding -fvisibility=hidden -fno-plt
# -fno-rtti -fno-exceptions -c -O3` (mirroring
# samples/custom_dispatch/cpu/embedded/CMakeLists.txt).
_CPU_TARGET_FLAGS: dict[str, tuple[str, list[str]]] = {
    "llvm-cpu-x86_64": ("x86_64-none-elf", []),
    "llvm-cpu-aarch64": ("aarch64-none-elf", []),
    "llvm-cpu-riscv64": ("riscv64-none-elf", ["-march=rv64gc", "-mabi=lp64d"]),
    # SpacemiT X60: rv64gcv with zvl256b vector length — matches the per-
    # dispatch benchmark MLIR target key in
    # models/compiled_models/dronet/spacemit_RVV_quant/benchmarks/.
    "llvm-cpu-riscv64-rvv": (
        "riscv64-none-elf",
        ["-march=rv64gcv", "-mabi=lp64d"],
    ),
    "llvm-cpu-spacemit-x60": (
        "riscv64-none-elf",
        [
            "-march=rv64gcv_zfh_zba_zbb_zbc_zbs_zicbom_zicboz_zicbop_zihintpause",
            "-mabi=lp64d",
        ],
    ),
    # radiance-muon: special-cased above _CPU_TARGET_FLAGS dispatch in the
    # `c`/`cpp`/`ll` source-lang branches below. The triple here is purely
    # informational (the real flags come from _radiance_muon_baseline_flags
    # which uses llvm-muon clang's default target).
    "radiance-muon": ("riscv32-unknown-unknown-elf", []),
    "llvm-cpu-radiance-muon": ("riscv32-unknown-unknown-elf", []),
}


def precompile(
    manifest: _manifest.Manifest,
    cache_dir: pathlib.Path,
    *,
    targets_filter: Sequence[str] | None = None,
    force: bool = False,
    toolchain: Toolchain | None = None,
) -> dict[tuple[str, str], ObjectArtifact]:
    """Compile every (kernel, target) pair in `manifest` whose target is in
    `targets_filter` (or all targets if filter is None).

    Returns a map from (kernel_name, target) to ObjectArtifact. Idempotent:
    artifacts hashed by source+toolchain+flags are reused across runs.
    """
    cache_dir = cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tc = toolchain or detect_toolchain()

    out_map: dict[tuple[str, str], ObjectArtifact] = {}
    for kernel in manifest.kernels:
        for target in kernel.targets:
            if targets_filter is not None and target not in targets_filter:
                continue

            source_bytes = kernel.source.read_bytes()
            if kernel.source_lang == "spirv":
                # Pass-through: hash + copy into the cache so downstream
                # consumers see a stable cache_dir layout.
                key = _cache_key(source_bytes, "passthrough", [])
                dst = cache_dir / f"{kernel.name}.{key}.spv"
                if force or not dst.exists():
                    dst.write_bytes(source_bytes)
                _validate_spv(dst, tc)
                out_map[(kernel.name, target)] = ObjectArtifact(
                    kernel.name,
                    target,
                    dst,
                    spirv=True,
                )
                continue

            if kernel.source_lang == "cl":
                if not _is_vulkan_target(target):
                    raise RuntimeError(
                        f"kernel '{kernel.name}': source_lang=cl requires a " f"vulkan target, got '{target}'"
                    )
                key = _cache_key(source_bytes, tc.clspv_version or "missing", ["cl-default"])
                dst = cache_dir / f"{kernel.name}.{key}.spv"
                if force or not dst.exists():
                    _compile_cl_to_spv(kernel.source, dst, tc)
                    _validate_spv(dst, tc)
                out_map[(kernel.name, target)] = ObjectArtifact(
                    kernel.name,
                    target,
                    dst,
                    spirv=True,
                )
                continue

            if kernel.source_lang == "glsl":
                if not _is_vulkan_target(target):
                    raise RuntimeError(
                        f"kernel '{kernel.name}': source_lang=glsl requires a " f"vulkan target, got '{target}'"
                    )
                key = _cache_key(source_bytes, tc.glslang_version or "missing", ["glsl-default"])
                dst = cache_dir / f"{kernel.name}.{key}.spv"
                if force or not dst.exists():
                    _compile_glsl_to_spv(kernel.source, dst, tc)
                    _validate_spv(dst, tc)
                out_map[(kernel.name, target)] = ObjectArtifact(
                    kernel.name,
                    target,
                    dst,
                    spirv=True,
                )
                continue

            if kernel.source_lang in ("c", "cpp", "ll"):
                if not _is_cpu_target(target):
                    raise RuntimeError(
                        f"kernel '{kernel.name}': source_lang={kernel.source_lang} "
                        f"requires a CPU target, got '{target}'"
                    )

                # radiance-muon takes a different code path: llvm-muon clang
                # with the canonical Muon flag set. The shared cache key still
                # hashes (source bytes + tool version + flag list).
                if _is_radiance_muon_target(target):
                    flags = _radiance_muon_baseline_flags(tc)
                    key = _cache_key(
                        source_bytes,
                        tc.llvm_muon_clang_version or "missing",
                        flags,
                    )
                    dst = cache_dir / f"{kernel.name}.{key}.muon.o"
                    if force or not dst.exists():
                        _compile_radiance_muon_obj(kernel.source, dst, tc)
                    out_map[(kernel.name, target)] = ObjectArtifact(
                        kernel.name,
                        target,
                        dst,
                        spirv=False,
                    )
                    continue

                # Host-clang CPU targets (existing path).
                if kernel.source_lang != "c":
                    raise RuntimeError(
                        f"kernel '{kernel.name}': source_lang={kernel.source_lang} "
                        f"is only supported on radiance-muon today; got CPU target '{target}'"
                    )
                target_entry = _CPU_TARGET_FLAGS.get(target)
                if not target_entry:
                    raise RuntimeError(
                        f"kernel '{kernel.name}': no CPU target spec for "
                        f"'{target}' (extend kernels/core/precompile.py "
                        f"_CPU_TARGET_FLAGS — provide target_triple + arch flags)"
                    )
                arch_triple, arch_flags = target_entry
                key = _cache_key(
                    source_bytes,
                    tc.clang_version or "missing",
                    [arch_triple, *arch_flags],
                )
                dst = cache_dir / f"{kernel.name}.{key}.{arch_triple}.o"
                if force or not dst.exists():
                    _compile_c_to_obj(
                        kernel.source,
                        dst,
                        arch_triple,
                        tc,
                        extra_arch_flags=arch_flags,
                    )
                out_map[(kernel.name, target)] = ObjectArtifact(
                    kernel.name,
                    target,
                    dst,
                    spirv=False,
                )
                continue

            if kernel.source_lang == "qnn-context-binary":
                # Delegate to the QNN backend's precompile extras.
                # Late import keeps core backend-agnostic and avoids the
                # qnn SDK Python overhead when no QNN kernels are present.
                from kernels.qnn.precompile_extras import compile_qnn_kernel

                out_map[(kernel.name, target)] = compile_qnn_kernel(
                    kernel,
                    target,
                    source_bytes,
                    cache_dir,
                    force=force,
                    cache_key=_cache_key,
                )
                continue

            raise RuntimeError(f"kernel '{kernel.name}': unhandled source_lang " f"'{kernel.source_lang}'")

    return out_map
