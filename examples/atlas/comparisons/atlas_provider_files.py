"""Explicit companion source contracts shared by replay, export and ELF builds.

Update provider file names here when selecting a new Atlas source epoch. The
producer tuple mirrors run_workload_contraction.code_sources; packing also binds
its transitive tooling and the runtime header read by atlas_host_layout.
"""

PRODUCER_ID = "workload_contraction_v1"
PRODUCER_FILE = "run_workload_contraction.py"
TILE_FILE = "run_model_tile.py"
PACKING_FILE = "compile_workload_inputs.py"

# Native substitution requires an archived Atlas cold-contraction producer
# root with the interface below, which current atlas-mlir no longer ships.
LEGACY_VALIDATOR_FILE = "validate_model_contraction.py"
LEGACY_HELPER_MODULES = ("atlas_tile_arithmetic", "run_model_tile")
LEGACY_PRODUCER_FILES = ("run_model_contraction.py", *(name + ".py" for name in LEGACY_HELPER_MODULES))
LEGACY_PROVIDER_REQUIREMENT = (
    "Native-substitution replay requires an archived Atlas cold-contraction producer root "
    "providing run_model_contraction.py and validate_model_contraction.py with "
    "bind/recheck/admit_contraction, which current atlas-mlir no longer ships"
)

PRODUCER_FILES = (
    PRODUCER_FILE,
    "atlas_program.py",
    TILE_FILE,
    "atlas_tile_arithmetic.py",
    "atlas_vector_arithmetic.py",
)
PACKING_FILES = (PACKING_FILE, *PRODUCER_FILES,
                 "atlas_path_guard.py", "atlas_tooling.py", "atlas_host_layout.py")
LAYOUT_HEADER = "atlas_host_layout.h"
RUNTIME_COMMON = ("atlas_host.c", "atlas_host.h", "atlas_resnet_graph_math.c")
RUNTIME_FILES = {
    "tiny": (*RUNTIME_COMMON, "tiny_projection.c", "tiny_projection.h"),
    "smol": (*RUNTIME_COMMON, "smol_block_raw_adapter.c", "smol_block_raw_adapter.h"),
}


def packing_paths(tools):
    """Include the header that the imported Python layout module actually reads."""
    return (*(tools / name for name in PACKING_FILES), tools.parent / "runtime" / LAYOUT_HEADER)
