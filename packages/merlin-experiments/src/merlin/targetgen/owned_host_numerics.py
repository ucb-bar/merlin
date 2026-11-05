"""Compare complete owned-model outputs against the frozen numerical policy."""

from __future__ import annotations

import json
from pathlib import Path


def _numeric_full_output(capsule, record, console):
    """Use the unchanged public numeric policy on every typed forward result."""
    import numpy as np
    import yaml

    from merlin.compile.baremetal_model import read_full_outputs
    from merlin.runtime.out_bin import binary_console_diagnostics
    from merlin.targetgen.capsule_golden import compare
    from merlin.targetgen.declared_group_reference import owned_model_reference
    from merlin.targetgen.golden_store import load_golden

    text = binary_console_diagnostics(console).decode("utf-8", errors="replace")
    if sum(line.strip() == "DONE" for line in text.splitlines()) != 1:
        raise ValueError("owned model full console has no unique successful completion")
    values, metrics = read_full_outputs(
        console, Path(record["runtime_inputs"]), record["build"]["output"]["build_hash"]
    )
    order = json.loads((Path(record["runtime_inputs"]) / "output_order.json").read_text())
    capsule_root = Path(capsule["__dir__"])
    declaration = yaml.safe_load((capsule_root / "capsule.yaml").read_text())
    golden = load_golden(capsule_root)
    expected = golden["outputs"]
    if len(order) != len(values) or set(order) != set(expected) or len(order) != len(set(order)):
        raise ValueError("actual owned model output roster differs from frozen full golden")
    observed = {}
    for name, value in zip(order, values, strict=True):
        shape = np.asarray(expected[name]).shape
        if value.size != np.asarray(expected[name]).size:
            raise ValueError("owned full result extent differs from frozen golden")
        observed[name] = value.reshape(shape).tolist()
    policy, source = declaration["numeric_policy"], str(golden.get("golden_source") or "independent_capsule")
    report = compare(expected, observed, policy, golden_source=source)
    # An eager mismatch may still be the reference the placed groups' declared bounds admit.
    report = owned_model_reference(
        report, expected, observed, policy, record=record, order=order, golden_source=source, compare=compare
    )
    return report, metrics
