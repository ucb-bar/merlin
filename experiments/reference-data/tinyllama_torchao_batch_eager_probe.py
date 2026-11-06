"""Check full eager TorchAO TinyLlama B8 output against eight B1 requests.

Run from the Merlin repository with model2MLIR on PYTHONPATH. This checks
source batching and integer matmul shapes; it does not certify an MLIR capture.
"""

import hashlib
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torchao
import transformers
from m2m.capture.torchao_pipeline import QuantizationConfig, apply_quantization
from torch.utils._python_dispatch import TorchDispatchMode

os.environ.update(
    CAUSAL_LM_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    CAUSAL_LM_ENTRY="ids",
    M2M_BATCH="8",
    M2M_SEQ="8",
    CAUSAL_LM_ATTENTION="eager",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)
torch.set_num_threads(1)
torch.manual_seed(0)
spec = importlib.util.spec_from_file_location("causal_section", "examples/workloads/causal_lm_section/loader.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load the causal LM source loader")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
model, (ids,) = loader.get_model_and_inputs()
model = apply_quantization(model, QuantizationConfig(scheme="int8_dyn_act_int8_weight"))
seen = []


class Observe(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._int_mm.default:
            a, b = args[:2]
            seen.append((tuple(a.shape), tuple(b.shape)))
        return func(*args, **(kwargs or {}))


start = time.monotonic()
with torch.no_grad(), Observe():
    batch = model(ids).detach().cpu().numpy()
with torch.no_grad():
    serial = np.stack([model(ids[i : i + 1]).detach().cpu().numpy()[0] for i in range(8)])
diff = np.abs(batch.astype(np.float64) - serial.astype(np.float64))
record = {
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "scheme": "int8_dyn_act_int8_weight",
    "seed": 0,
    "shape": list(batch.shape),
    "int_mm_count": len(seen),
    "int_mm_lhs_shapes": {str(s[0]): sum(1 for x in seen if x[0] == s[0]) for s in seen},
    "all_int_mm_M64": all(a[0] == 64 for a, b in seen),
    "input_sha256": hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),
    "unique_input_rows": len({tuple(row) for row in ids.cpu().numpy().tolist()}),
    "torch": torch.__version__,
    "torchao": torchao.__version__,
    "transformers": transformers.__version__,
    "batch_sha256": hashlib.sha256(batch.tobytes()).hexdigest(),
    "serial_sha256": hashlib.sha256(serial.tobytes()).hexdigest(),
    "bitwise_equal": bool(np.array_equal(batch, serial)),
    "max_abs": float(diff.max()),
    "relative_l2": float(np.linalg.norm(diff.ravel()) / np.linalg.norm(serial.astype(np.float64).ravel())),
    "argmax_equal_per_token": bool(np.array_equal(np.argmax(batch, axis=-1), np.argmax(serial, axis=-1))),
    "elapsed_s": time.monotonic() - start,
}
print(json.dumps(record, sort_keys=True), flush=True)
