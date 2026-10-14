"""The two MLP whole-model bundles (``mlp_basic``, ``mlp_basic_triton``) and their forward arguments.

Regenerate the bundles with ``build_tools/scripts/gen_mlp_kernel_fixtures.py``. They share weights,
input and golden; ``mlp_basic_triton`` runs fc1+relu+fc2+relu as one external-kernel call.
"""

from __future__ import annotations

import json

import numpy as np

from merlin.common.paths import merlin_dir

BUNDLES = merlin_dir() / "tests" / "data" / "mlp_bundles"
MLP_BASIC = BUNDLES / "mlp_basic"
MLP_BASIC_TRITON = BUNDLES / "mlp_basic_triton"


def forward_args(bundle) -> list[np.ndarray]:
    """@forward's arguments in signature order: weights from the safetensors blob (by manifest name),
    the input from ``inputs.npz``."""
    from merlin.llvmlower.weights_pack import load_safetensors_header

    header, payload_off = load_safetensors_header(bundle / "weights.safetensors")
    blob = (bundle / "weights.safetensors").read_bytes()[payload_off:]
    manifest = json.loads((bundle / "weights.safetensors.manifest.json").read_text())
    x = np.load(bundle / "inputs.npz")["in0"]
    args = []
    for i in range(len(manifest)):
        meta = manifest[str(i)]
        if meta["kind"] != "param":
            args.append(np.ascontiguousarray(x))
            continue
        begin, end = header[meta["weight"]]["data_offsets"]
        args.append(np.frombuffer(blob[begin:end], dtype=np.float32).reshape(meta["shape"]).copy())
    return args


def golden(bundle) -> np.ndarray:
    return np.load(bundle / "golden.npy")
