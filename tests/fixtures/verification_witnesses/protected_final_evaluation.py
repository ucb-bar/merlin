from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from types import ModuleType, SimpleNamespace

import pytest

from merlin.development.backend.optimization import protected_final_evaluation as F
from merlin.development.backend.optimization import protected_verifier_qualification as Q
from merlin.development.backend.optimization.contracts import StageGateError, sha256_file
from merlin.development.backend.optimization.numerical_readback import ReadbackFiles
from verification_witnesses.numerical_readback import _rebind
from verification_witnesses.numerical_readback import native_case as _native_case

native_case = _native_case
VERIFIER = '\nfrom dataclasses import replace\nfrom pathlib import Path\nimport json\nfrom merlin.development.backend.optimization.protected_final_evaluation import FinalExecutionWitness, REQUIRED_WITNESSES\nfrom merlin.development.backend.optimization.contracts import sha256_file\nfrom merlin.development.backend.optimization.protected_verifier_qualification import ProtectedArmRefusal\n\ndef verify(*, binding, arm, files, original_reference):\n    root = Path(__file__).parent\n    config = json.loads((root / "options.json").read_text())\n    controlled = binding.member.startswith("control/")\n    if controlled:\n        path = root / "controls" / (binding.member.split("/")[-1] + ".json")\n        control = json.loads(path.read_text())\n        if control["defect"]:\n            if config.get("control_fault") == "crash":\n                raise RuntimeError("synthetic verifier crashed; no diagnosed refusal")\n            if config.get("control_fault") == "accept":\n                control["defect"] = None\n        if control["defect"]:\n            raise ProtectedArmRefusal(witness_kind=control["defect"], binding_sha256=binding.identity(),\n                                      arm=arm, evidence=((path, sha256_file(path)),),\n                                      reason="synthetic control has explicit defective authority")\n    evidence = tuple((root / (role + ".json"), sha256_file(root / (role + ".json"))) for role in REQUIRED_WITNESSES)\n    result = FinalExecutionWitness(\n        binding.identity(), arm,\n        binding.reference_compiler_sha256 if arm == "reference" else binding.candidate_compiler_sha256,\n        sha256_file(files.executable), sha256_file(files.executable), sha256_file(files.console),\n        binding.hardware_sha256, binding.inputs_sha256, binding.timer_scope_sha256,\n        (100 if arm == "reference" else 99) if controlled else config[arm + "_cycles"],\n        sha256_file(original_reference.console),\n        tuple((role, digest) for role, (_path, digest) in zip(REQUIRED_WITNESSES, evidence)), evidence,\n    )\n    mode = None if controlled else config.get("mode")\n    if arm == "candidate" and mode == "hardware":\n        result = replace(result, hardware_sha256="9" * 64)\n    elif arm == "candidate" and mode == "staged":\n        result = replace(result, staged_executable_sha256="9" * 64)\n    elif arm == "candidate" and mode == "missing":\n        result = replace(result, witnesses=result.witnesses[:-1])\n    elif arm == "candidate" and mode == "unbound":\n        result = replace(result, witnesses=tuple((role, "9" * 64) for role, _digest in result.witnesses))\n    elif arm == "candidate" and mode == "mutate":\n        files.console.write_text(files.console.read_text() + "altered")\n    elif arm == "candidate" and mode == "dict":\n        return vars(result)\n    return result\n'


def _json(path, record):
    path.write_text(json.dumps(record))


def _request(
    case,
    *,
    mode=None,
    candidate_cycles=99,
    reference=None,
    original_reference=None,
    control_fault=None,
    omit_control=False,
):
    root = case.private / "execution"
    root.mkdir()
    owner = root / "verifier.py"
    owner.write_text(VERIFIER)
    options = root / "options.json"
    _json(
        options, dict(reference_cycles=100, candidate_cycles=candidate_cycles, mode=mode, control_fault=control_fault)
    )
    records = []
    for role in F.REQUIRED_WITNESSES:
        record = root / (role + ".json")
        _json(record, dict(test_fixture_only=True, role=role))
        records.append(record)
    module = ModuleType("synthetic_protected_final_verifier")
    module.__file__ = str(owner)
    sys.modules[module.__name__] = module
    exec(compile(owner.read_bytes(), str(owner), "exec"), module.__dict__)
    code_sha = F.callable_code_sha256(module.verify)
    dependencies = tuple(((path, sha256_file(path)) for path in (owner, options, *records)))
    verifier = F.ProtectedExecutionVerifier(module.verify, owner, dependencies, None, None, code_sha)
    oracle = original_reference or case.arms[0]
    binding = F.FinalExecutionBinding(
        member="independent-final",
        reference_compiler_sha256="1" * 64,
        candidate_compiler_sha256="2" * 64,
        program_sha256="3" * 64,
        inputs_sha256="4" * 64,
        quality_budget_sha256=sha256_file(case.private / "budget.json"),
        original_reference_console_sha256=sha256_file(oracle.console),
        runtime_sha256="5" * 64,
        toolchain_sha256="6" * 64,
        hardware_sha256="7" * 64,
        timer_scope_sha256="8" * 64,
    )
    binding_file = root / "binding.json"
    _json(binding_file, vars(binding))
    _json(
        root / "campaign.json",
        dict(
            schema="merlin.protected_final_campaign.v1",
            members=[dict(member=binding.member, binding_sha256=binding.identity())],
        ),
    )
    controls = []
    control_root = root / "controls"
    control_root.mkdir()
    for kind in F.REQUIRED_WITNESSES:
        for expected in ("observe", "refuse"):
            name = kind + "_" + expected
            control_file = control_root / (name + ".json")
            _json(control_file, dict(test_fixture_only=True, defect=kind if expected == "refuse" else None))
            control_binding = replace(binding, member="control/" + name)
            original = control_root / (name + "_binding.json")
            _json(original, vars(control_binding))
            controls.append(
                Q.ProtectedVerifierControl(
                    name,
                    kind,
                    expected,
                    control_binding,
                    original,
                    "candidate",
                    case.arms[1],
                    oracle,
                    tuple(((path, sha256_file(path)) for path in (*records, control_file))),
                )
            )
    if omit_control:
        controls.pop()
    plan = root / "control_plan.json"
    _json(
        plan,
        dict(
            schema=Q.CONTROL_SCHEMA,
            scope="synthetic joins only; no actual source or hardware qualification",
            dependencies=[[str(path), digest] for (path, digest) in dependencies],
            controls=[row.record() for row in controls],
        ),
    )
    inputs = case.freeze()
    verifier = Q.qualify_protected_execution_verifier(
        selection=verifier,
        original_plan=plan,
        controls=tuple(controls),
        run_dir=inputs["run_dir"],
        environment=inputs["environment"],
        receipt=inputs["run_dir"] / "private-qualification/receipt.json",
    )
    if reference is not None:
        inputs["reference"] = reference
    return F.ProtectedFinalMember(
        **inputs,
        binding=binding,
        original_binding=binding_file,
        verifier=verifier,
        original_reference=oracle,
        original_reference_build=case.service,
    )


def _recompile(case, arm):
    source = arm.harness.parent / "kernel.c"
    source.write_text(source.read_text().replace("h[0]=", "f[3] += 0.03f; h[0]="))
    subprocess.run(
        case.recipe.compile_command(source=source, output=arm.kernel_object), check=True, capture_output=True
    )
    subprocess.run(
        case.recipe.link_command(
            objects=[arm.kernel_object, arm.harness.parent / "harness.o"],
            output=arm.executable,
            link_script=case.recipe.link_script,
        ),
        check=True,
        capture_output=True,
    )
    arm.console.write_bytes(subprocess.run([arm.executable], check=True, capture_output=True).stdout)


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "F",
    "ModuleType",
    "Q",
    "ReadbackFiles",
    "SimpleNamespace",
    "StageGateError",
    "VERIFIER",
    "_json",
    "_native_case",
    "_rebind",
    "_recompile",
    "_request",
    "json",
    "native_case",
    "pytest",
    "replace",
    "sha256_file",
    "shutil",
    "subprocess",
    "sys",
]
