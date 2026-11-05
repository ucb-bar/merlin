"""Import the pinned atlas-kernels submodule's generators by bare module name.

Kernel generators and their shared helpers (``atlas_program``, ``atlas_execution``,
``atlas_path_guard`` and the rest) live in ``third_party/atlas-kernels``. Import
this module before any of them; it is idempotent. ``module_path(name)`` returns
the library file for a module file name such as ``atlas_program.py``, and
``source_path(name)`` resolves a former ``tools/`` file name in this checkout.
"""

from pathlib import Path
import sys

LIBRARY = Path(__file__).absolute().parents[1] / "third_party" / "atlas-kernels"
if not (LIBRARY / "atlas_kernels_path.py").is_file():
    raise ImportError("atlas-kernels submodule is missing; run: git submodule update --init third_party/atlas-kernels")
if str(LIBRARY) not in sys.path:
    sys.path.append(str(LIBRARY))

from atlas_kernels_path import DIRECTORIES, module_path  # noqa: E402,F401

ROOT = LIBRARY.parents[1]
# smol_fusion_program imports chipyard.arena, which imports elf_image from tools/host_iss; the generator
# put that directory on sys.path itself when it lived in tools/.
if str(ROOT / "tools" / "host_iss") not in sys.path:
    sys.path.append(str(ROOT / "tools" / "host_iss"))

# The file names atlas_program.code_sources hashes, in its order. That function hashes siblings of
# atlas_program.py, which only works in a flat copy; code_sources below resolves each name in this checkout.
PROGRAM_SOURCES = ("atlas_program.py", "atlas_program_checks.py", "atlas_epilogue.py", "atlas_rne_add.py",
                   "atlas_resnet_input.py", "atlas_program_reuse_common.py", "run_model_tile.py",
                   "atlas_tile_arithmetic.py", "atlas_weight_source.py")


def relative_path(name):
    """Path below the atlas-mlir root of a former ``tools/`` file name, following moves into atlas-kernels."""
    try:
        return module_path(name).relative_to(ROOT)
    except ValueError:
        return Path("tools") / name


def source_path(name):
    """Absolute path in this checkout of a former ``tools/`` file name such as ``atlas_program.py``."""
    return ROOT / relative_path(name)


def moved_path(path):
    """Follow a missing ``<root>/tools/<module>.py`` path into atlas-kernels; other paths are returned unchanged."""
    path = Path(path)
    if path.parent == ROOT / "tools" and not path.exists():
        try:
            return module_path(path.name)
        except ValueError:
            pass
    return path


def source_paths(*, traffic_reuse=False):
    """The files atlas_program.code_sources names, as paths in this checkout."""
    import atlas_program
    names = PROGRAM_SOURCES
    if atlas_program._reuse_enabled(traffic_reuse):
        names += tuple(module + ".py" for module in dict.fromkeys(atlas_program._TRAFFIC_REUSE_MODULES.values()))
    return {name: source_path(name) for name in names}


def code_sources(*, traffic_reuse=False):
    """atlas_program.code_sources for this checkout: file name to SHA-256 of the file in use."""
    from atlas_tooling import sha256
    return {name: sha256(path) for name, path in source_paths(traffic_reuse=traffic_reuse).items()}
