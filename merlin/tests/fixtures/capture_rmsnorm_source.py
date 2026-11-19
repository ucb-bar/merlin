"""Regenerate tests/data/rmsnorm_source with the pinned model2MLIR capture environment.

Usage: python merlin/tests/fixtures/capture_rmsnorm_source.py OUTPUT_DIRECTORY
Select model2MLIR's capture pin from software_pins.yaml on PYTHONPATH. These are
standalone source-expression captures, not model or target numerical qualification.
"""

import importlib.metadata
import json
from hashlib import sha256
from pathlib import Path
import subprocess
import sys

import m2m
import torch

from merlin.integrations.model2mlir import capture_pin


class Norm(torch.nn.Module):
    def __init__(self, reverse):
        super().__init__()
        self.reverse = reverse

    def forward(self, hidden, gamma):
        normalized = hidden * torch.rsqrt((hidden * hidden).mean(-1, keepdim=True) + 1e-5)
        return normalized * gamma if self.reverse else gamma * normalized


m2m_root = Path(m2m.__file__).resolve().parents[1]
pin = capture_pin(m2m_root)
assert pin["ok"], pin["citation"]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=False)
torch.use_deterministic_algorithms(True)
torch.manual_seed(0)
members = {}
for case, shape in (
    ("decode1-input", (1, 2048)), ("decode1-post", (1, 2048)),
    ("prefill0-input", (128, 2048)), ("prefill0-post", (128, 2048)),
):
    result = m2m.convert(Norm(reverse=case.endswith("post")).eval(), (torch.randn(shape), torch.randn(2048)),
                         backend="fx_importer", level="linalg-on-tensors")
    assert result.ok, result.diagnostics
    raw = result.mlir_text.encode()
    (out / (case + ".mlir")).write_bytes(raw)
    members[case] = sha256(raw).hexdigest()
receipt = {
    "model2mlir_revision": subprocess.check_output(
        ["git", "-C", str(m2m_root), "rev-parse", "HEAD"], text=True).strip(),
    "torch_version": torch.__version__, "xdsl_version": importlib.metadata.version("xdsl"),
    "generator_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "members": members,
}
(out / "capture.json").write_text(json.dumps(receipt, indent=2) + "\n")
