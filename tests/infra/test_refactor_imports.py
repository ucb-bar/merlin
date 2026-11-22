"""Moved source owners import independently of earlier pytest imports."""

import os
import subprocess
import sys

from merlin.common.paths import module_source_path


MODULE_PAIRS = (
        (
            "merlin.development.targetgen.conformance.gaps",
            "merlin.development.targetgen.conformance.requirements",
        ),
        (
            "merlin.compiler.dialects.lowering.analyses",
            "merlin.compiler.dialects.lowering.pipeline",
        ),
        (
            "merlin.runtime.execution.model_inputs",
            "merlin.runtime.dispatch.executor",
        ),
        (
            "merlin.verification.lean.host_checks",
            "merlin.verification.lean.checker",
        ),
        (
            "merlin.target.contract.harness.readout",
            "merlin.target.contract.harness.harness_render",
        ),
        (
            "merlin.development.testgen.evidence.coverage_commitment",
            "merlin.development.testgen.evidence.coverage_obligations_coverage_commitment",
        ),
        (
            "merlin.development.testgen.evidence.evidence",
            "merlin.development.targetgen.sandbox",
        ),
)
def test_moved_owners_import_in_both_fresh_process_orders(tmp_path):
    source = module_source_path("merlin").parent.parent
    environment = {**os.environ, "PYTHONPATH": str(source)}
    # Both orders matter: pytest's shared module cache can conceal a cycle.
    orders = [order for modules in MODULE_PAIRS for order in (modules, tuple(reversed(modules)))]
    for order in orders:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; "
                "[importlib.import_module(name) for name in sys.argv[1:]]",
                *order,
            ],
            cwd=tmp_path,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
