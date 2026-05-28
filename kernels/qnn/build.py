"""Build hook for QNN kernel sources.

Driven by the kernel-embedding pipeline (`kernels/core/spec_gen.py` +
`precompile.py`) when a manifest entry has `source_lang=qnn-context-binary`
and the source is a `.qnn.cpp` file.

Two build modes:

`build_qnn_kernel` — host build with libQnnCpu.so validation backend.
    1. g++ on host (x86_64) compiles <kernel>.qnn.cpp + QnnModel.cpp +
       QnnWrapperUtils.cpp + QnnModelPal.cpp into libqnn_<name>.so.
    2. qnn-context-binary-generator on host runs composeGraphs against
       libQnnCpu.so, captures the context binary via QnnContext_getBinary.
    3. The .qnn-ctx is CPU-validated only — the actual QNN backend at
       runtime (GPU / HTA / DSP) re-validates against its own hardware on
       load. Useful for fast smoke testing of new kernel sources, but the
       resulting blob is NOT loadable on libQnnGpu.so / libQnnDspV66Stub.so
       (different flatbuffer major version).

`build_qnn_kernel_on_board` — board-native build for a target backend.
    1. scp source + Qualcomm wrapper sources + headers to a working dir on
       the board over SSH.
    2. ssh runs g++ on the board (aarch64, matching libstdc++) to produce
       libqnn_<name>.so for the target arch.
    3. ssh runs qnn-context-binary-generator on the board with the target
       backend (libQnnGpu.so / libQnnHta.so / libQnnDspV66Stub.so) so the
       composeGraphs validation runs against the *real* accelerator
       hardware.
    4. scp the resulting binary back to the host as <name>.qnn-ctx.
    The blob produced is loadable on the target backend at runtime —
    this is the production path for QRB5165 GPU/HTA/DSP kernels.

The .qnn-ctx byte format is what the IREE QNN HAL driver loads at runtime
via `QnnContext_createFromBinary`. Both modes cache by source-hash so
re-invoking on unchanged inputs is a no-op.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import os
import pathlib
import shutil
import subprocess

_LOG = logging.getLogger(__name__)

# Default SDK root. Overridable via QAIRT_SDK_ROOT env var.
_DEFAULT_SDK_ROOT = pathlib.Path("/scratch2/dima/misc_sw/qualcomm/qairt/2.45.0.260326")


@dataclasses.dataclass(frozen=True)
class QnnBuildConfig:
    sdk_root: pathlib.Path
    host_backend_so: pathlib.Path  # libQnnCpu/Gpu/Hta.so under lib/x86_64-linux-clang
    cxx: str = "g++"
    cxxflags: tuple[str, ...] = (
        "-std=c++17",
        "-march=x86-64",
        "-O2",
        "-fPIC",
        "-shared",
        "-fvisibility=hidden",
    )

    @classmethod
    def from_env(cls, backend: str = "cpu") -> QnnBuildConfig:
        """Resolve SDK + backend lib from env / defaults.

        `backend` is the validation backend used during context-binary
        generation. Use 'cpu' for shape-/op-validation only (works on host).
        Use 'gpu' / 'hta' when the host has the matching backend libs and
        you want backend-specific graph optimisations baked in. The output
        .qnn-ctx is not tied to the chosen backend at runtime — the HAL
        driver picks the runtime backend when loading.
        """
        sdk_root = pathlib.Path(os.environ.get("QAIRT_SDK_ROOT") or _DEFAULT_SDK_ROOT)
        if not sdk_root.exists():
            raise FileNotFoundError(f"QNN SDK not found at {sdk_root} — set QAIRT_SDK_ROOT")
        backend_map = {
            "cpu": "libQnnCpu.so",
            "gpu": "libQnnGpu.so",
            "hta": "libQnnHta.so",
            "htp": "libQnnHtp.so",
        }
        if backend not in backend_map:
            raise ValueError(f"unknown QNN backend '{backend}' — choose from {sorted(backend_map)}")
        backend_so = sdk_root / "lib" / "x86_64-linux-clang" / backend_map[backend]
        if not backend_so.exists():
            raise FileNotFoundError(f"backend library missing: {backend_so}")
        return cls(sdk_root=sdk_root, host_backend_so=backend_so)


def _wrapper_sources(sdk_root: pathlib.Path) -> list[pathlib.Path]:
    """Qualcomm's QnnModel C++ helper sources, used as-is from the SDK.

    These are the same files `qnn-model-lib-generator` compiles internally;
    we link them directly to keep the build straightforward and avoid
    relying on the lib-generator wrapper (which assumes clang++ in PATH and
    a specific Makefile layout).
    """
    jni = sdk_root / "share" / "QNN" / "converter" / "jni"
    return [
        jni / "QnnModel.cpp",
        jni / "QnnWrapperUtils.cpp",
        jni / "linux" / "QnnModelPal.cpp",
    ]


_HELPERS_DIR = pathlib.Path(__file__).resolve().parent / "qnn"


def _wrapper_includes(sdk_root: pathlib.Path) -> list[str]:
    return [
        f"-I{sdk_root / 'include' / 'QNN'}",
        f"-I{sdk_root / 'share' / 'QNN' / 'converter' / 'jni'}",
        f"-I{_HELPERS_DIR}",
    ]


def _qnn_define_visibility() -> str:
    return '-DQNN_API=__attribute__((visibility("default")))'


def _cache_key(source: pathlib.Path, cfg: QnnBuildConfig) -> str:
    h = hashlib.sha256()
    h.update(source.read_bytes())
    h.update(b"\0\0qnn_sdk\0")
    h.update(str(cfg.sdk_root).encode())
    h.update(b"\0\0backend\0")
    h.update(str(cfg.host_backend_so).encode())
    h.update(b"\0\0cxxflags\0")
    for f in cfg.cxxflags:
        h.update(f.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def _ld_library_path(cfg: QnnBuildConfig) -> str:
    """Construct LD_LIBRARY_PATH so qnn-context-binary-generator finds
    libQnnSystem.so + libc++ (clang14-runtime ships a private libc++)."""
    paths = [
        cfg.sdk_root / "lib" / "x86_64-linux-clang",
        cfg.sdk_root / "lib" / "clang14-runtime",
    ]
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return os.pathsep.join([str(p) for p in paths] + ([existing] if existing else []))


def build_qnn_kernel(
    source: pathlib.Path,
    kernel_name: str,
    cache_dir: pathlib.Path,
    cfg: QnnBuildConfig | None = None,
) -> pathlib.Path:
    """Compile + serialise a QNN kernel source into a .qnn-ctx blob.

    Returns the absolute path to the resulting `<cache>/<key>/<name>.qnn-ctx`
    artifact. The artifact is cached by content hash; subsequent calls with
    unchanged inputs return the same path without rebuilding.
    """
    cfg = cfg or QnnBuildConfig.from_env()
    source = source.resolve()
    cache_dir = cache_dir.resolve()
    key = _cache_key(source, cfg)
    work = cache_dir / key
    work.mkdir(parents=True, exist_ok=True)
    qnn_ctx = work / f"{kernel_name}.qnn-ctx"
    if qnn_ctx.exists():
        _LOG.info("qnn-build cache hit: %s", qnn_ctx)
        return qnn_ctx

    so_path = work / f"libqnn_{kernel_name}.so"
    cmd: list[str] = [
        cfg.cxx,
        *cfg.cxxflags,
        _qnn_define_visibility(),
        *_wrapper_includes(cfg.sdk_root),
        *(str(p) for p in _wrapper_sources(cfg.sdk_root)),
        str(source),
        "-o",
        str(so_path),
    ]
    _LOG.info("qnn-build compile: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if res.returncode != 0:
        raise RuntimeError(
            f"qnn-build: compile failed for {source}\n" f"stdout:\n{res.stdout}\n" f"stderr:\n{res.stderr}"
        )

    # Serialise the graph to a .qnn-ctx blob via Qualcomm's
    # qnn-context-binary-generator. The tool runs on the host, dlopens
    # `so_path`, calls QnnModel_composeGraphs, then QnnContext_getBinary.
    ctxbin_gen = cfg.sdk_root / "bin" / "x86_64-linux-clang" / "qnn-context-binary-generator"
    if not ctxbin_gen.exists():
        raise FileNotFoundError(f"qnn-context-binary-generator missing: {ctxbin_gen}")
    out_basename = f"{kernel_name}_raw"  # avoid clashing with our final .qnn-ctx
    cmd = [
        str(ctxbin_gen),
        "--model",
        str(so_path),
        "--backend",
        str(cfg.host_backend_so),
        "--binary_file",
        out_basename,
        "--output_dir",
        str(work),
    ]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = _ld_library_path(cfg)
    _LOG.info("qnn-build serialise: %s", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if res.returncode != 0:
        raise RuntimeError(
            f"qnn-build: ctx-bin-generator failed for {kernel_name}\n"
            f"stdout:\n{res.stdout}\n"
            f"stderr:\n{res.stderr}"
        )
    raw_bin = work / f"{out_basename}.bin"
    if not raw_bin.exists():
        raise RuntimeError(
            f"qnn-build: ctx-bin-generator produced no .bin for {kernel_name}\n"
            f"expected at: {raw_bin}\n"
            f"stdout:\n{res.stdout}"
        )
    shutil.move(str(raw_bin), str(qnn_ctx))
    _LOG.info("qnn-build emitted: %s (%d bytes)", qnn_ctx, qnn_ctx.stat().st_size)
    return qnn_ctx


# Board-side build target spec — the board's QAIRT toolchain variant. QRB5165
# RB5 is Ubuntu 20.04 / glibc 2.31; aarch64-ubuntu-gcc9.4 is the matching
# QAIRT toolchain. (The aarch64-oe-linux-gcc11.2 variant requires glibc
# 2.32+ and fails on this image.)
_DEFAULT_BOARD_TARGET_TUPLE = "aarch64-ubuntu-gcc9.4"


@dataclasses.dataclass(frozen=True)
class BoardBuildConfig:
    """Configuration for a board-native QNN kernel build over SSH.

    `ssh_host` is the SSH alias / `user@host` string passed to ssh/scp.
    `board_qairt_root` is the absolute path on the board where the QAIRT
    SDK has been staged (typically /tmp/qnn_probe per the Phase 0 setup).
    """

    sdk_root: pathlib.Path
    ssh_host: str
    board_qairt_root: str  # absolute path on the board
    target_backend: str  # gpu | hta | dspv66 | dsp | htp | cpu
    target_tuple: str = _DEFAULT_BOARD_TARGET_TUPLE
    board_workdir: str = "/tmp/qnn_kernel_build"

    @classmethod
    def from_env(
        cls,
        ssh_host: str,
        board_qairt_root: str,
        target_backend: str = "gpu",
    ) -> BoardBuildConfig:
        sdk_root = pathlib.Path(os.environ.get("QAIRT_SDK_ROOT") or _DEFAULT_SDK_ROOT)
        if not sdk_root.exists():
            raise FileNotFoundError(f"QNN SDK not found at {sdk_root} — set QAIRT_SDK_ROOT")
        valid = {"cpu", "gpu", "hta", "dspv66", "dsp", "htp"}
        if target_backend not in valid:
            raise ValueError(f"unknown board target_backend '{target_backend}' — choose from {sorted(valid)}")
        return cls(
            sdk_root=sdk_root,
            ssh_host=ssh_host,
            board_qairt_root=board_qairt_root,
            target_backend=target_backend,
        )


def _board_backend_so(target_backend: str) -> str:
    """Top-level QNN backend library that exports `QnnInterface_getProviders`.

    Note: `libQnnDspV66Stub.so` and similar `*V66*Stub.so` files are
    arch-specific *companion* stubs loaded via libQnnDsp.so's provider
    machinery, not standalone backends. The top-level lib is `libQnnDsp.so`.
    Same for HTP: top-level is `libQnnHtp.so` (which dispatches across
    HtpV68/V69/V73/etc. via internal arch detection).
    """
    return {
        "cpu": "libQnnCpu.so",
        "gpu": "libQnnGpu.so",
        "hta": "libQnnHta.so",
        "dsp": "libQnnDsp.so",  # top-level DSP backend (v66/v65/etc.)
        "dspv66": "libQnnDsp.so",  # alias (kept for callers using the old name)
        "htp": "libQnnHtp.so",  # top-level HTP backend (v68+)
    }[target_backend]


def _board_cache_key(source: pathlib.Path, cfg: BoardBuildConfig) -> str:
    h = hashlib.sha256()
    h.update(source.read_bytes())
    h.update(b"\0\0qnn_sdk\0")
    h.update(str(cfg.sdk_root).encode())
    h.update(b"\0\0board_target\0")
    h.update(cfg.target_backend.encode())
    h.update(b"\0")
    h.update(cfg.target_tuple.encode())
    h.update(b"\0\0board_host\0")
    h.update(cfg.ssh_host.encode())
    return h.hexdigest()[:16]


def _ssh(host: str, cmd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["ssh", host, cmd], capture_output=True, text=True, check=False)


def _scp_to(host: str, src: pathlib.Path | str, dst: str) -> None:
    res = subprocess.run(
        ["scp", "-q", "-r", str(src), f"{host}:{dst}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"scp {src} -> {host}:{dst} failed: {res.stderr}")


def _scp_from(host: str, src: str, dst: pathlib.Path | str) -> None:
    res = subprocess.run(
        ["scp", "-q", f"{host}:{src}", str(dst)],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        raise RuntimeError(f"scp {host}:{src} -> {dst} failed: {res.stderr}")


def build_qnn_kernel_on_board(
    source: pathlib.Path,
    kernel_name: str,
    cache_dir: pathlib.Path,
    cfg: BoardBuildConfig,
) -> pathlib.Path:
    """Compile + serialise a QNN kernel on the board, returning the local
    path to the resulting `<cache>/<key>/<name>.qnn-ctx` artifact.

    Why on-board: QNN context binaries are validated against the target
    backend during compose. `qnn-context-binary-generator --backend
    libQnnGpu.so` on host fails because the host has no Adreno hardware.
    Running on the board lets the validator compose against real GPU/HTA/
    DSP hardware and produce a binary that's loadable by the same backend
    at runtime.

    Caching is by sha256(source bytes, sdk root, target backend, target
    tuple, ssh host). Re-running with unchanged inputs short-circuits.
    """
    source = source.resolve()
    cache_dir = cache_dir.resolve()
    key = _board_cache_key(source, cfg)
    work = cache_dir / f"board_{cfg.target_backend}_{key}"
    work.mkdir(parents=True, exist_ok=True)
    qnn_ctx = work / f"{kernel_name}.qnn-ctx"
    if qnn_ctx.exists():
        _LOG.info("qnn-build cache hit (board): %s", qnn_ctx)
        return qnn_ctx

    helper_src_dir = _HELPERS_DIR.parent / "qnn"  # kernels/qnn/headers/
    qairt_jni = cfg.sdk_root / "share" / "QNN" / "converter" / "jni"
    qairt_include = cfg.sdk_root / "include" / "QNN"

    # Stage source tree on the board: <workdir>/<key>/{src,include,jni}
    board_dir = f"{cfg.board_workdir}/{key}"
    _ssh(
        cfg.ssh_host, f"rm -rf {board_dir} && mkdir -p {board_dir}/src {board_dir}/jni {board_dir}/include"
    ).check_returncode()
    # Wrapper sources (Qualcomm-shipped, arch-agnostic)
    for fp in [
        qairt_jni / "QnnModel.cpp",
        qairt_jni / "QnnWrapperUtils.cpp",
        qairt_jni / "linux" / "QnnModelPal.cpp",
        qairt_jni / "QnnModel.hpp",
        qairt_jni / "QnnModelPal.hpp",
        qairt_jni / "QnnTypeMacros.hpp",
        qairt_jni / "QnnWrapperUtils.hpp",
    ]:
        _scp_to(cfg.ssh_host, fp, f"{board_dir}/jni/")
    # Our helper
    _scp_to(cfg.ssh_host, helper_src_dir / "QnnKernelHelpers.hpp", f"{board_dir}/jni/")
    # QNN headers
    _scp_to(cfg.ssh_host, qairt_include, f"{board_dir}/include/")
    # The kernel source
    _scp_to(cfg.ssh_host, source, f"{board_dir}/src/")

    # Compile on the board (g++; native arch).
    so_name = f"libqnn_{kernel_name}.so"
    compile_cmd = (
        f"cd {board_dir} && g++ -std=c++17 -O2 -fPIC -shared -fvisibility=hidden "
        f"-DQNN_API='__attribute__((visibility(\"default\")))' "
        f"-Iinclude/QNN -Ijni "
        f"jni/QnnModel.cpp jni/QnnWrapperUtils.cpp jni/QnnModelPal.cpp "
        f"src/{source.name} -o {so_name}"
    )
    _LOG.info("qnn-build (board) compile: %s", compile_cmd)
    res = _ssh(cfg.ssh_host, compile_cmd)
    if res.returncode != 0:
        raise RuntimeError(
            f"qnn-build (board): compile failed for {source}\n" f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )

    # Run qnn-context-binary-generator on the board against the target
    # backend lib.
    backend_so = _board_backend_so(cfg.target_backend)
    out_basename = f"{kernel_name}_raw"
    ctxbin_cmd = (
        f"cd {board_dir} && "
        f"export LD_LIBRARY_PATH={cfg.board_qairt_root}/lib:$LD_LIBRARY_PATH && "
        f'export ADSP_LIBRARY_PATH="{cfg.board_qairt_root}/lib;/dsp/cdsp;'
        f'/usr/lib/rfsa/adsp;/vendor/dsp/cdsp" && '
        f"{cfg.board_qairt_root}/bin/qnn-context-binary-generator "
        f"--model {board_dir}/{so_name} "
        f"--backend {cfg.board_qairt_root}/lib/{backend_so} "
        f"--binary_file {out_basename} "
        f"--output_dir {board_dir}"
    )
    _LOG.info("qnn-build (board) ctxbin: %s", ctxbin_cmd)
    res = _ssh(cfg.ssh_host, ctxbin_cmd)
    if res.returncode != 0:
        raise RuntimeError(
            f"qnn-build (board): ctx-bin-generator failed for {kernel_name}\n"
            f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
        )

    # Pull the .bin back as .qnn-ctx.
    _scp_from(cfg.ssh_host, f"{board_dir}/{out_basename}.bin", qnn_ctx)
    if not qnn_ctx.exists() or qnn_ctx.stat().st_size == 0:
        raise RuntimeError(f"qnn-build (board): empty/missing artifact at {qnn_ctx}\n" f"compose stdout:\n{res.stdout}")
    _LOG.info(
        "qnn-build (board) emitted: %s (%d bytes, backend=%s)",
        qnn_ctx,
        qnn_ctx.stat().st_size,
        cfg.target_backend,
    )
    return qnn_ctx


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=pathlib.Path, required=True)
    parser.add_argument("--name", required=True, help="Kernel name (output basename)")
    parser.add_argument(
        "--cache-dir",
        type=pathlib.Path,
        default=pathlib.Path("build/qnn_kernels"),
    )
    parser.add_argument(
        "--backend",
        default="cpu",
        choices=["cpu", "gpu", "hta", "dspv66", "dsp", "htp"],
        help="Validation backend used during context-binary generation. "
        "For host builds, only 'cpu' works (no GPU/DSP on host). "
        "For --on-board builds, any target backend is allowed.",
    )
    parser.add_argument(
        "--on-board",
        metavar="SSH_HOST",
        default=None,
        help="Build on this SSH host instead of locally. The board must " "have QAIRT staged at --board-qairt-root.",
    )
    parser.add_argument(
        "--board-qairt-root",
        default="/tmp/qnn_probe",
        help="Path on the board where QAIRT bin/ + lib/ have been staged.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )
    if args.on_board:
        bcfg = BoardBuildConfig.from_env(
            ssh_host=args.on_board,
            board_qairt_root=args.board_qairt_root,
            target_backend=args.backend,
        )
        out = build_qnn_kernel_on_board(args.source, args.name, args.cache_dir, bcfg)
    else:
        cfg = QnnBuildConfig.from_env(backend=args.backend)
        out = build_qnn_kernel(args.source, args.name, args.cache_dir, cfg)
    print(out)


# ---------------------------------------------------------------------------
# Phase 3b — parallel-build helpers.
#
# When the partitioner emits N islands, each one needs its own ctxbin.
# Building serially is bottlenecked by the qnn-context-binary-generator
# subprocess (~10s per kernel host-side, ~30s on-board). Parallel
# builds via concurrent.futures.ThreadPoolExecutor give ~Nx wall-clock
# speedup up to the number of available cores / SSH MUX slots.
#
# For board builds we recommend pairing this helper with an SSH
# ControlMaster so the N parallel SSH sessions reuse a single TCP
# connection (avoid the ~1s handshake cost per kernel). Sample
# `~/.ssh/config` snippet:
#
#   Host qdev
#     ControlMaster auto
#     ControlPath ~/.ssh/sockets/%C
#     ControlPersist 5m
#
# Stable cache keys (`digest_qnn_graph_desc`) ensure that two
# recognizers producing structurally-identical graphs (same tensors,
# same nodes, same q-params) hit the same cache entry — a
# Phase 5 prerequisite for compile-determinism across runs.
# ---------------------------------------------------------------------------


def digest_qnn_graph_desc(graph: object) -> str:
    """Deterministic SHA-256 digest of a `QnnGraphDesc`.

    Hashes the structural fields (tensor names, shapes, dtypes,
    quant params; node names, op-types, in/out tensors, params) so two
    semantically-identical graphs produce the same digest regardless
    of how they were constructed. Used as a stable cache key for the
    parallel-build helpers below.

    Returns the first 16 hex chars of the SHA-256 digest — short
    enough for filesystem paths, long enough that collisions are
    statistically negligible.
    """
    import hashlib  # noqa: PLC0415  - lazy

    h = hashlib.sha256()
    h.update(b"qnn_graph_desc_v1\0")
    h.update(getattr(graph, "name", "?").encode())
    h.update(b"\0\0tensors\0")
    for t in getattr(graph, "tensors", ()):
        h.update(t.name.encode())
        h.update(b"|")
        h.update(",".join(str(d) for d in t.shape).encode())
        h.update(b"|")
        h.update(t.dtype.encode())
        h.update(b"|")
        h.update(t.role.encode())
        h.update(b"|")
        if t.quant is not None:
            h.update(f"{t.quant.scale:.7g}".encode())
            h.update(b":")
            h.update(str(t.quant.offset).encode())
        if t.static_data is not None:
            h.update(b"|sd:")
            h.update(hashlib.sha256(t.static_data).digest())
        h.update(b"\0")
    h.update(b"\0\0nodes\0")
    for n in getattr(graph, "nodes", ()):
        h.update(n.name.encode())
        h.update(b"|")
        h.update(n.op_type.encode())
        h.update(b"|in:")
        h.update(",".join(n.inputs).encode())
        h.update(b"|out:")
        h.update(",".join(n.outputs).encode())
        for tp in n.tensor_params:
            h.update(b"|tp:")
            h.update(tp.name.encode())
            h.update(b":")
            h.update(",".join(str(v) for v in tp.values).encode())
        for sp in n.scalar_params:
            h.update(b"|sp:")
            h.update(sp.name.encode())
            h.update(b":")
            h.update(str(sp.value).encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


def build_many_qnn_kernels_on_board(
    sources: list[tuple[pathlib.Path, str]],
    cache_dir: pathlib.Path,
    cfg: BoardBuildConfig,
    *,
    max_workers: int = 4,
) -> list[pathlib.Path]:
    """Parallel on-board build of multiple kernels via
    `concurrent.futures.ThreadPoolExecutor`.

    `sources` is a list of `(source_path, kernel_name)` pairs. Each
    pair is built independently; the returned list has the same order
    as the input. `max_workers` caps the number of concurrent SSH
    sessions (default 4 — typical balance between board parallelism
    and host SSH connection overhead).

    For best wall-clock behavior, configure SSH ControlMaster on the
    host so the N parallel sessions multiplex over one TCP connection
    (see module-level docstring for the `~/.ssh/config` snippet).
    """
    import concurrent.futures  # noqa: PLC0415  - lazy

    def _one(item: tuple[pathlib.Path, str]) -> pathlib.Path:
        src, name = item
        return build_qnn_kernel_on_board(src, name, cache_dir, cfg)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Preserve input order in the output list.
        return list(ex.map(_one, sources))


def build_many_qnn_kernels(
    sources: list[tuple[pathlib.Path, str]],
    cache_dir: pathlib.Path,
    cfg: QnnBuildConfig,
    *,
    max_workers: int = 4,
) -> list[pathlib.Path]:
    """Host-side parallel build of multiple kernels. See
    `build_many_qnn_kernels_on_board` for usage. The host build is
    capped at `max_workers` concurrent g++/qnn-context-binary-generator
    invocations; default 4 balances build parallelism and memory."""
    import concurrent.futures  # noqa: PLC0415  - lazy

    def _one(item: tuple[pathlib.Path, str]) -> pathlib.Path:
        src, name = item
        return build_qnn_kernel(src, name, cache_dir, cfg)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, sources))
