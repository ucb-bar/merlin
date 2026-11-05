"""The ordinary immutable-layout helper is usable from the reviewed SDK alone."""

import os
import subprocess
import sys

from merlin.compile.scheduling.export import export_shared_scheduling, public_modules


def test_constant_panel_source_preparation_has_closed_sdk_export(tmp_path):
    name = "merlin.llvmlower.constant_panel_layout"
    assert name in public_modules(24)
    assert name not in public_modules(23)
    root = tmp_path / "sdk"
    export_shared_scheduling(root, version=24, source_binding=True)
    program = """
import pathlib, sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from xdsl.dialects.builtin import ModuleOp, i8
from merlin.llvmlower import constant_panel_layout as layout
assert pathlib.Path(layout.__file__).is_relative_to(pathlib.Path(sys.argv[1]))
source = dict(op='conv2d', kh=3, kw=3, ci=2, N=6)
original = np.arange(108,dtype=np.int8).reshape(18,6)
module = ModuleOp([])
layout.request_constant_panels(module,3,minimum_input_channels=2)
assert layout.constant_panel_request(module) == 3
assert layout.constant_panel_minimum_input_channels(module) == 2
packed, storage = layout.prepare_constant_panels(source,original,3,i8,minimum_input_channels=2)
assert packed.shape == (2,18,3)
assert storage.unpack_bytes(packed.tobytes(),max_storage_bytes=original.nbytes) == original.tobytes()
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(root)],
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
