"""Generate the target-side Merlin runtime ADAPTER scaffold (real, executable).

Emits:
  runtime/adapter/adapter.py            RuntimeAdapter: lower / encode / run_simulator / normalize
  runtime/adapter/command_encoding.yaml how the abstract command buffer maps to this target
  runtime/adapter/metrics_mapping.yaml  raw counters -> common Merlin metrics
  runtime/simulator/semantics.py        target opcode semantics (delegates to merlin.runtime)
  runtime/command_buffer/example_repeated_rhs.json   executable Merlin command buffer

The adapter performs REAL execution: it runs the command buffer through the Merlin-owned
runtime (`merlin.runtime`, which does actual integer tensor math), recomputes an independent
reference, checks correctness, and writes simulator_output / reference_output / metrics / trace
JSON. ToyNPU is a Merlin runtime *adapter*, not its own runtime.
"""
from __future__ import annotations

import json
from typing import Any

from ...common.artifacts import Artifact, yaml_artifact

_ADAPTER_PY = '''"""Merlin runtime adapter for `{target}` (generated, executable).

Implements the Merlin-owned runtime ABI for `{target}` by adapting the Merlin command-buffer /
metrics contract. Real execution is delegated to the Merlin-owned runtime engine
(`merlin.runtime`); install Merlin to run it (`pip install -e .` at the Merlin repo root).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

# opcode -> target-encoded mnemonic (used by encode_command_buffer)
_ENCODE = {{
    "RES_PACK": "{TGT}_RES_PACK",
    "MATMUL_RESIDENT": "{TGT}_MATMUL",
    "COMMIT": "{TGT}_COMMIT",
    "EVICT": "{TGT}_EVICT",
}}


def _load_semantics():
    here = Path(__file__).resolve().parent
    sem_path = here.parent / "simulator" / "semantics.py"
    spec = importlib.util.spec_from_file_location("{target}_semantics", sem_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


class RuntimeAdapter:
    """Adapter implementing the Merlin runtime ABI for `{target}`."""

    target_name = "{target}"

    def lower_target_ir_to_runtime(self, target_ir_path):
        """Load a runtime module. Current lowering: the module IS a command-buffer JSON."""
        return json.loads(Path(target_ir_path).read_text(encoding="utf-8"))

    def encode_command_buffer(self, runtime_module):
        """Encode the abstract command buffer into this target's command stream."""
        stream = []
        for cmd in runtime_module.get("commands", []):
            stream.append({{
                "op": _ENCODE.get(cmd["opcode"], cmd["opcode"]),
                "operands": cmd.get("operands", {{}}),
                "attributes": cmd.get("attributes", {{}}),
            }})
        return {{"target": self.target_name, "stream": stream}}

    def run_simulator(self, command_buffer_path, inputs_path=None, out_dir=None):
        """Execute the command buffer, recompute a reference, check correctness, emit artifacts."""
        cb = json.loads(Path(command_buffer_path).read_text(encoding="utf-8"))
        inputs = json.loads(Path(inputs_path).read_text(encoding="utf-8")) if inputs_path else None
        sem = _load_semantics()
        result = sem.simulate(cb, inputs)
        reference = sem.reference(cb, inputs)
        result["reference"] = reference
        result["correct"] = (result["outputs"] == reference)
        if out_dir:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "simulator_output.json").write_text(
                json.dumps(result["outputs"], indent=2, sort_keys=True), encoding="utf-8")
            (d / "reference_output.json").write_text(
                json.dumps(reference, indent=2, sort_keys=True), encoding="utf-8")
            (d / "metrics.json").write_text(
                json.dumps(result["metrics"], indent=2, sort_keys=True), encoding="utf-8")
            (d / "trace.json").write_text(
                json.dumps(result["trace"], indent=2, sort_keys=True), encoding="utf-8")
        return result

    def run_spike(self, command_buffer_path, harts=4, out_dir=None):
        """Execute the command buffer on spike as a bare-metal multicore RVV CPU.

        Delegates to the Merlin-owned backend (`merlin.runtime.backends.spike`):
        RVV codegen + chipyard toolchain + reference-equality gate. Requires the
        chipyard toolchain (MERLIN_CHIPYARD); raises a clear error otherwise.
        """
        from merlin.runtime.backends import spike as _spike
        if not _spike.available():
            raise RuntimeError(
                "spike backend unavailable: set MERLIN_CHIPYARD to a chipyard "
                "checkout with riscv-tools (gcc + spike)")
        cb = json.loads(Path(command_buffer_path).read_text(encoding="utf-8"))
        result = _spike.run_command_buffer(cb, harts=harts)
        if out_dir:
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            (d / "spike_output.json").write_text(
                json.dumps(result["outputs"], indent=2, sort_keys=True), encoding="utf-8")
            (d / "spike_metrics.json").write_text(
                json.dumps(result["metrics"], indent=2, sort_keys=True), encoding="utf-8")
        return result

    def normalize_metrics(self, raw_metrics):
        """Raw metrics are already in common form; return a copy."""
        return dict(raw_metrics)


if __name__ == "__main__":
    import sys
    here = Path(__file__).resolve()
    cb = sys.argv[1] if len(sys.argv) > 1 else str(
        here.parents[1] / "command_buffer" / "example_repeated_rhs.json")
    res = RuntimeAdapter().run_simulator(cb, out_dir=str(here.parents[1] / "run"))
    print(json.dumps({{"correct": res["correct"], "metrics": res["metrics"]}},
                     indent=2, sort_keys=True))
'''

_SEMANTICS_PY = '''"""Simulator semantics for `{target}` (generated).

ToyNPU-class targets use the standard resident-pack / matmul / commit / evict semantics
provided by the Merlin-owned runtime, which performs REAL integer tensor math. This module
names the target's opcode set + requant policy and delegates execution to `merlin.runtime`.
"""
from __future__ import annotations

from merlin.runtime import simulate as _simulate, reference_outputs as _reference

OPCODES = ("RES_PACK", "MATMUL_RESIDENT", "COMMIT", "EVICT")
DEFAULT_REQUANT_SHIFT = 4


def simulate(command_buffer, inputs=None):
    """Execute the command buffer (real arithmetic) -> {{outputs, metrics, trace}}."""
    return _simulate(command_buffer, inputs)


def reference(command_buffer, inputs=None):
    """Independent reference recomputation (bypasses the resident store)."""
    return _reference(command_buffer, inputs)
'''


def _example_command_buffer(target: str, reuse: int = 4) -> dict[str, Any]:
    tensors: dict[str, Any] = {
        "W": {"shape": [8, 6], "dtype": "i8", "role": "weight"},
        "bias": {"shape": [6], "dtype": "i32", "role": "bias"},
    }
    commands: list[dict[str, Any]] = [
        {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"},
         "attributes": {"layout": "packed_rhs"}},
    ]
    for i in range(reuse):
        tensors[f"A{i}"] = {"shape": [5, 8], "dtype": "i8", "role": "input"}
        commands.append({"opcode": "MATMUL_RESIDENT",
                         "operands": {"lhs": f"A{i}", "rhs": "W_res", "dst": f"acc{i}"}})
        commands.append({"opcode": "COMMIT",
                         "operands": {"src": f"acc{i}", "dst": f"Y{i}", "bias": "bias"},
                         "attributes": {"epilogue": ["bias_add", "requant", "relu"],
                                        "requant_shift": 4, "output_dtype": "i8"}})
    commands.append({"opcode": "EVICT", "operands": {"handle": "W_res"}})
    return {
        "abi_version": "0.1",
        "target": target,
        "backend": "simulator",
        "tensors": tensors,
        "commands": commands,
        "params": {"requant_shift": 4},
        "resources": {"handles": ["W_res"] + [f"acc{i}" for i in range(reuse)]},
        "metrics_requested": ["cycles", "bytes_moved", "command_count", "pack_count",
                              "resident_hits", "evictions", "accumulator_commits"],
    }


def generate(runtime_adapter_plan: dict[str, Any]) -> list[Artifact]:
    """Return runtime/ adapter + simulator + example artifacts."""
    target = runtime_adapter_plan.get("target", "target")
    tgt_upper = target.upper()
    command_encoding = {
        "target": target,
        "format": runtime_adapter_plan.get("command_encoding", {}).get("format", "command_stream"),
        "opcodes": {
            "RES_PACK": f"{tgt_upper}_RES_PACK",
            "MATMUL_RESIDENT": f"{tgt_upper}_MATMUL",
            "COMMIT": f"{tgt_upper}_COMMIT",
            "EVICT": f"{tgt_upper}_EVICT",
        },
        "notes": "Maps Merlin command-buffer opcodes to this target's encoding.",
    }
    metrics_mapping = runtime_adapter_plan.get("metrics", {"maps_to_common": {}, "target_specific": []})
    example = _example_command_buffer(target)
    return [
        Artifact("runtime/adapter/adapter.py",
                 _ADAPTER_PY.format(target=target, TGT=tgt_upper)),
        yaml_artifact("runtime/adapter/command_encoding.yaml", command_encoding,
                      header="Generated command encoding."),
        yaml_artifact("runtime/adapter/metrics_mapping.yaml", metrics_mapping,
                      header="Raw-counter -> common-metric mapping."),
        Artifact("runtime/simulator/semantics.py", _SEMANTICS_PY.format(target=target)),
        Artifact("runtime/command_buffer/example_repeated_rhs.json",
                 json.dumps(example, indent=2, sort_keys=True) + "\n"),
    ]
