"""Build forward() args for a self-consistent capture (inputs+golden+MLIR+weights all
from ONE seeded model instance).

The smolVLA loader builds a *randomly-initialized* model with *random* inputs and no
seed, so a reference is only valid when captured in the same process as the MLIR it
checks. This runner consumes that consistent bundle:

  inputs.npz   in0..in5 = (img, img_mask, lang_tokens, lang_masks, state, noise)
  weights.safetensors(+manifest)  the exact (quantized) weights of that instance
  golden.npy   torch forward() of that instance on those inputs

Rotary inv_freq buffers are deterministic (derived from rope base/dim, not trained),
so they're taken from any prior capture's reference npz.
"""
from __future__ import annotations

import json
from pathlib import Path

from .model_runner import ModelIO, parse_forward_signature
from .weights_pack import load_safetensors_header

# manifest input name -> index into inputs.npz (loader tuple order).
INPUT_ORDER = {"img": 0, "img_mask": 1, "lang_tokens": 2, "lang_masks": 3,
               "state": 4, "noise": 5}


def build_consistent_io(bundle_dir: str | Path, mlir_path: str | Path,
                        lifted_npz: str | Path, rotary_npz: str | Path) -> ModelIO:
    import numpy as np

    b = Path(bundle_dir)
    sig = parse_forward_signature(mlir_path)
    manifest = json.loads((b / "weights.safetensors.manifest.json").read_text())
    header, payload_off = load_safetensors_header(b / "weights.safetensors")
    blob = np.fromfile(b / "weights.safetensors", dtype=np.uint8)[payload_off:]
    inputs = np.load(b / "inputs.npz")
    lifted = np.load(lifted_npz)
    rotary = np.load(rotary_npz)
    lifted_names = sorted(lifted.files)

    def rotary_lookup(name: str):
        # b_model_vlm_..._inv_freq  ->  buf::model.vlm_...inv_freq
        for k in rotary.files:
            if k.startswith("buf::") and \
                    ("b_" + k[len("buf::"):].replace(".", "_")) == name:
                return rotary[k]
        raise KeyError(name)

    arrays: dict = {}
    args: list = []
    li = 0
    for idx in range(len(sig)):
        meta = manifest[str(idx)]
        shape, _ = sig[idx]
        kind = meta["kind"]
        if kind == "param":
            name = meta["weight"]
            begin, _ = header[name]["data_offsets"]
            args.append((blob.ctypes.data + begin, shape))
            continue
        if kind == "buffer":
            arr = np.ascontiguousarray(rotary_lookup(meta["name"]))
        elif meta["name"].startswith("c_lifted_tensor_"):
            arr = np.ascontiguousarray(lifted[lifted_names[li]]); li += 1
        else:
            arr = np.ascontiguousarray(inputs[f"in{INPUT_ORDER[meta['name']]}"])
        arrays[idx] = arr
        args.append((arr.ctypes.data, shape))

    out = np.zeros((1, 50, 32), dtype=np.float32)
    arrays[-1] = out
    arrays[-2] = blob
    args.append((out.ctypes.data, [1, 50, 32]))
    return ModelIO(arrays=arrays, arg_list=args, output=out)
