"""Compile the portable Smol host primitives and expose their array ABI."""
import ctypes
import subprocess

import numpy as np

from merlin.common.paths import repo_root


def load_host_library(directory, name, *sources, flags=(), strict_fp=True, math=True):
    source = repo_root() / "examples/atlas/comparisons"
    output = directory / f"{name}.so"
    command = ["cc", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic"]
    if strict_fp:
        command += ["-ffp-contract=off", "-fno-fast-math"]
    command += ["-fPIC", "-shared", *flags, *(str(source / item) for item in sources)]
    command += (["-lm"] if math else []) + ["-o", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return ctypes.CDLL(str(output))


def pointer(array):
    return array.ctypes.data_as(ctypes.c_void_p)


def words(array):
    return (np.asarray(array, np.float32).view(np.uint32) >> 16).astype(np.uint16)
