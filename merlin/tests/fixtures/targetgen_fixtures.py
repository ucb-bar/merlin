"""Drivers shared by the native component admission tests (merlin/tests/targetgen)."""

import hashlib
import json
import os
import subprocess

import numpy as np

from merlin.common.paths import python_import_roots
from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components


class RefusingProvider:
    """A provider that fails the test if admission asks it for a device proof."""

    def admit_numeric_schedule(self, *args, **kwargs):
        raise AssertionError("a refused source must not request a device proof")


class DecliningProvider:
    """A provider that records each schedule request and grants no device implementation."""

    def __init__(self):
        self.called = []

    def admit_numeric_schedule(self, source, **kwargs):
        self.called.append(source)
        return dict(status="declined", reason="test grants no device implementation")


def eager_record(abi, output_bytes, post_input_bytes, post_indices):
    """A bundled case carrying an untransformed native CPU eager boundary (hex words)."""
    return dict(
        status="bundled",
        input_abi=abi,
        output_indices=[0],
        semantic_boundary=dict(post_indices=post_indices),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager", output_bytes=output_bytes, post_input_bytes=post_input_bytes
        ),
    )


def prepare_components(tmp_path, module, cases, provider=None):
    return prepare_integer_components(
        tmp_path,
        dict(cases=cases),
        dict(spec={}, host={}),
        provider or RefusingProvider(),
        None,
        module,
        limb_attribute="test.limb",
    )


def assert_host_refusal(tmp_path, source, inputs, cases, code, provider=None):
    """Admission keeps ``source`` (module or MLIR text) on the host with refusal ``code``.

    Writes model.mlir and inputs.npz (``inputs`` become in0, in1, ...), checks that admission
    returns None, leaves model.mlir byte-identical and records ``code``; returns the refusal.
    """
    text = source if isinstance(source, str) else str(source) + "\n"
    module = parse_mlir_text(source) if isinstance(source, str) else source
    (tmp_path / "model.mlir").write_text(text)
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": value for i, value in enumerate(inputs)})
    assert prepare_components(tmp_path, module, cases, provider) is None
    assert (tmp_path / "model.mlir").read_text() == text
    refusal = json.loads((tmp_path / "exact-dtype-refusal.json").read_text())
    assert refusal["code"] == code
    return refusal


def run_m2m_script(script):
    """Run ``script`` under the declared torch interpreter (MERLIN_M2M_PYTHON) on this checkout."""
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in python_import_roots())},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def selector_catalog(module, symbol, *bindings):
    """A provider catalog binding ``symbol``, a closed static element selector, at each (root_ordinal, operands)."""
    numeric = dict(family="F1", quantization="none", exact_arithmetic=True, byte_order="little")
    return dict(
        source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
        bindings=[dict(root_ordinal=r, symbol=symbol, operands=o, numeric=dict(numeric)) for r, o in bindings],
        kernels=[dict(symbol=symbol, meta=dict(proof="closed_static_element_selectors", readout="scratchpad_bytes"))],
    )


def measured_routing(case_id, target="gemmini", instructions=1, operations=({},)):
    """Device routing with single-case measured execution evidence for ``case_id``."""
    evidence = dict(case_id=case_id, status="measured", attribution="single_case_shard", target=target)
    evidence["executed_instructions"] = instructions
    routing = dict(lane="device", routed=True, target=target, routed_operations=list(operations))
    return dict(routing, execution_evidence=evidence)


def capture(directory, source, inputs, golden, input_abi=(), output_abi=(), status=None, **meta):
    """Write one capture directory: the capsule source, capture.json, inputs.json and golden.json."""
    directory.mkdir(parents=True)
    (directory / "capsule.linalg.mlir").write_text(source, encoding="utf-8")
    meta = dict(ok=True, opaque=0, input_abi=list(input_abi), output_abi=list(output_abi), **meta)
    document = dict(capture_meta=meta) if status is None else dict(status=status, capture_meta=meta)
    (directory / "capture.json").write_text(json.dumps(document), encoding="utf-8")
    (directory / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
    (directory / "golden.json").write_text(json.dumps(golden), encoding="utf-8")
