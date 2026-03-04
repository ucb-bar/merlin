import os
import lit.formats

config.name = "MerlinGemmini"
config.test_format = lit.formats.ShTest(execute_external=True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

# Find the IREE build directory from the environment, or infer it
iree_binary_dir = os.environ.get("IREE_BINARY_DIR", "")
if not iree_binary_dir:
    # Try to find iree-compile relative to the lit binary
    import shutil
    iree_compile = shutil.which("iree-compile")
    if iree_compile:
        iree_binary_dir = os.path.dirname(os.path.dirname(iree_compile))

if iree_binary_dir:
    config.environment["PATH"] = (
        os.path.join(iree_binary_dir, "tools")
        + os.pathsep
        + os.environ.get("PATH", "")
    )
    # Also add llvm-project/bin for FileCheck
    llvm_bin = os.path.join(iree_binary_dir, "llvm-project", "bin")
    if os.path.isdir(llvm_bin):
        config.environment["PATH"] = (
            llvm_bin + os.pathsep + config.environment["PATH"]
        )

config.environment["HOME"] = os.environ.get("HOME", "")
