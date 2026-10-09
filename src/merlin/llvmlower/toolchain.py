"""Toolchain resolution for the whole-model path (all env-overridable)."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from merlin.common.paths import ext_path, repo_root


def _env(key: str, default: str | None = None) -> str | None:
    """Process env wins, then the gitignored ``<repo>/.env`` (same source ``ext_path`` reads), then
    ``default``. This lets a dev point at their model2MLIR / clang once in ``.env`` and have the
    toolchain resolve automatically — parity with how the ``aet`` sibling checkout is picked up —
    without exporting vars per shell or committing a personal path."""
    from merlin.common.paths import _dotenv

    return os.environ.get(key) or _dotenv().get(key) or default


DEFAULT_M2M_DIR = "/path/to/model2MLIR"  # external model2MLIR checkout; set MERLIN_M2M_DIR (or .env)
# Standalone LLVM-23 install (mlir-opt/mlir-translate) used where the torch-mlir wheel's
# in-process translate bridge is unreliable (its OpenMPIRBuilder segfaults on whole-model
# omp IR, whereas this build's mlir-translate handles it cleanly).
DEFAULT_LLVM_INSTALL = repo_root() / "third_party" / "llvm-install"


def llvm_install() -> Path:
    """The selected stock LLVM/MLIR install, including detached or installed runs."""
    selected = _env("MERLIN_MLIR_INSTALL")
    return Path(selected) if selected else DEFAULT_LLVM_INSTALL


def m2m_dir() -> Path:
    from merlin.integrations.model2mlir import root

    return root(default=DEFAULT_M2M_DIR)


def m2m_python() -> Path:
    """Compatibility spelling for the compiler interpreter, not the capture interpreter."""
    return compiler_python()


def compiler_python() -> Path:
    """Resolve the LLVM Python environment independently of the PyTorch frontend.

    Explicit configuration is authoritative even if unavailable: never silently use a
    different compiler when the requested one is missing. The legacy model2MLIR venv
    remains a staged fallback, preserving existing machine configurations.
    """
    explicit = _env("MERLIN_COMPILER_PYTHON")
    if explicit:
        return Path(explicit)
    compiler_venv = _env("MERLIN_COMPILER_VENV")
    if compiler_venv:
        return Path(compiler_venv) / "bin" / "python"
    env = _env("MERLIN_M2M_VENV")
    base = Path(env) if env else m2m_dir() / ".venv"
    return base / "bin" / "python"


def _iree_bin(lookup: Callable[[str], str | None] | None = None) -> Path | None:
    """bin/ of the IREE-based Merlin build (ships clang-23), if configured. Set MERLIN_IREE_BIN, or
    MERLIN_EXT_MERLIN_IREE pointing at an external Merlin-IREE checkout build.
    Resolved lazily so importing this module never requires the IREE build to be present."""
    if lookup is not None:
        env = lookup("MERLIN_IREE_BIN")
        if env:
            return Path(env)
        external = lookup("MERLIN_EXT_MERLIN_IREE")
        return Path(external) / "build" / "host-merlin-release" / "install" / "bin" if external else None
    env = _env("MERLIN_IREE_BIN")
    if env:
        return Path(env)
    try:
        return Path(ext_path("merlin_iree")) / "build" / "host-merlin-release" / "install" / "bin"
    except Exception:
        return None


def _resolve_clang(install: Path, lookup: Callable[[str], str | None], iree: Callable[[], Path | None]) -> Path:
    env = lookup("MERLIN_CLANG")
    if env:
        return Path(env)
    local = install / "bin" / "clang-23"
    if local.exists():
        return local
    iree_bin = iree()
    if iree_bin and (iree_bin / "clang-23").exists():
        return iree_bin / "clang-23"
    return (iree_bin / "clang-23") if iree_bin else Path("clang-23")


def clang() -> Path:
    """Resolve clang for x86-64 and riscv64.

    ``MERLIN_CLANG`` wins, then ``MERLIN_MLIR_INSTALL`` (or the checkout's own LLVM
    install), then the legacy IREE build, then ``clang-23`` on PATH.
    """
    return _resolve_clang(llvm_install(), _env, _iree_bin)


def host_llc() -> Path | None:
    """An explicit LLVM IR object producer for the host shared-library path.

    No selection keeps the historical clang compilation. A selected missing
    tool is an error when invoked, never permission to discover another tool.
    C-runtime compilation and shared-library linking retain their own tools.
    """
    selected = _env("MERLIN_LLVM_LLC")
    return Path(selected) if selected else None


def clang_for(root: Path, environ: Mapping[str, str]) -> Path:
    """What :func:`clang` answers in a process rooted at checkout ``root`` with ``environ``.

    The same chain, read from that environment and ``root``'s own ``.env`` and install instead of
    this process's, so a launcher can check the compiler a child will use before starting it.
    """
    from merlin.common.paths import read_dotenv

    dotenv = read_dotenv(Path(root) / ".env")

    def lookup(key: str) -> str | None:
        return environ.get(key) or dotenv.get(key)

    selected = lookup("MERLIN_MLIR_INSTALL")
    install = Path(selected) if selected else Path(root) / "third_party" / "llvm-install"
    return _resolve_clang(install, lookup, lambda: _iree_bin(lookup))


def mlir_translate() -> Path:
    """Standalone LLVM-23 ``mlir-translate`` (handles OpenMP -> LLVM-IR; the in-process
    torch-mlir bridge crashes on whole-model omp). Env-overridable."""
    env = _env("MERLIN_MLIR_TRANSLATE")
    return Path(env) if env else llvm_install() / "bin" / "mlir-translate"


def llvm_opt() -> Path:
    """LLVM optimizer from the selected clang installation, or explicit override.

    A missing selected tool is an error when used; do not silently switch LLVM
    versions. A clang resolved through PATH uses its resolved installation.
    """
    env = _env("MERLIN_LLVM_OPT")
    if env:
        return Path(env)
    import shutil

    selected = clang()
    return Path(shutil.which(str(selected)) or selected).with_name("opt")


def llvm_nm() -> Path:
    """Bitcode symbol inspector paired with the selected optimizer."""
    env = _env("MERLIN_LLVM_NM")
    return Path(env) if env else llvm_opt().with_name("llvm-nm")


def llvm_link() -> Path:
    """LLVM IR linker paired with the selected optimizer, or explicit override."""
    env = _env("MERLIN_LLVM_LINK")
    return Path(env) if env else llvm_opt().with_name("llvm-link")


def available() -> bool:
    return m2m_python().is_file() and clang().is_file()


def objdump() -> Path:
    """LLVM ``objdump``, from the same install as :func:`clang`. Env-overridable.

    Used by the post-codegen census to count what was actually EMITTED for a symbol. It has to be
    the LLVM one, and the same one that produced the object: GNU objdump on a host build has no
    reason to know the cross target the object was compiled for, and a disassembler that decodes
    nothing would make an empty function indistinguishable from an unreadable one."""
    env = _env("MERLIN_OBJDUMP")
    if env:
        return Path(env)
    local = llvm_install() / "bin" / "llvm-objdump"
    if local.exists():
        return local
    return Path(clang()).parent / "llvm-objdump"


def _llvm_sibling(env_key: str, tool: str) -> Path:
    """An LLVM binutil from the same install as :func:`clang`, env-overridable -- the objdump rule."""
    env = _env(env_key)
    if env:
        return Path(env)
    local = llvm_install() / "bin" / tool
    if local.exists():
        return local
    return Path(clang()).parent / tool


def objcopy() -> Path:
    """LLVM ``objcopy``, from the same install as :func:`clang`. Env-overridable.

    Used to give each of several package-emitted kernels its own symbol so they can link into one
    program: every one of them defines the entry the target's contract declares. Taken from the install
    that compiled the objects, for the same reason :func:`objdump` is."""
    return _llvm_sibling("MERLIN_OBJCOPY", "llvm-objcopy")


def nm() -> Path:
    """LLVM ``nm``, from the same install as :func:`clang`. Env-overridable."""
    return _llvm_sibling("MERLIN_NM", "llvm-nm")


def readelf() -> Path:
    """LLVM ``readelf``, from the same install as :func:`clang`. Env-overridable."""
    return _llvm_sibling("MERLIN_READELF", "llvm-readelf")
