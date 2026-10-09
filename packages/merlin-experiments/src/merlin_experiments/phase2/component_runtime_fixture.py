"""Private original-source control fixtures; absent from every author grant.

These independently declared tensor tests are evaluator mutation support, not a
seed target compiler, runtime qualification bypass or workload implementation.
"""

from pathlib import Path

from merlin_experiments.phase1.component_witness import REQUIRED_EXECUTION_EFFECTS

from . import component_runtime_controls as controls
from . import component_runtime_copy_controls as copy_controls
from .component_runtime_qualification import RuntimeControlFixture
from .contracts import StageGateError, exact_tree_record, sha256_file, write_json


def prepare_source_control(*, name, root, build_service, contract_root, target_descriptor, copy_support=None):
    mechanism, _, direction = name.partition(".")
    if root.exists():
        raise StageGateError("private runtime control needs a fresh evidence destination")
    root.mkdir(parents=True, mode=0o700)
    candidate, capsule = root / "candidate", root / "capsule"
    candidate.mkdir()
    capsule.mkdir()
    source = (
        "module { func.func @main(%a: tensor<1x3xf32>, %b: tensor<1x3xf32>) "
        "-> (tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32>) { "
        "%sum = arith.addf %a, %b : tensor<1x3xf32> "
        "func.return %sum, %b, %a : tensor<1x3xf32>, tensor<1x3xf32>, tensor<1x3xf32> } }\n"
    )
    shape, dtype = [1, 3], "f32"
    output_names = ["sum", "copy", "identity"]
    source_inputs = [
        ("a", {"shape": shape, "decoded": [1.25, -3.5, 0.125]}),
        ("b", {"shape": shape, "decoded": [2.0, 0.5, -0.25]}),
    ]
    expected_outputs = {"sum": [[3.25, -3.0, -0.125]], "copy": [[2.0, 0.5, -0.25]], "identity": [[1.25, -3.5, 0.125]]}
    if copy_support is not None:
        import math

        import numpy as np

        shape, dtype = list(copy_support.shape), copy_support.dtype
        tensor_type = "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"
        source = (
            "module { func.func @main(%a: "
            + tensor_type
            + ", %b: "
            + tensor_type
            + ") -> ("
            + tensor_type
            + ", "
            + tensor_type
            + ") { "
        )
        for ordinal, operand in enumerate(("a", "b")):
            source += f"%d{ordinal} = tensor.empty() : {tensor_type} "
            source += (
                f"%r{ordinal} = linalg.copy ins(%{operand} : {tensor_type}) "
                f"outs(%d{ordinal} : {tensor_type}) -> {tensor_type} "
            )
        source += f"func.return %r0, %r1 : {tensor_type}, {tensor_type} }} }}\n"
        copy_controls.parse_copy(source)
        output_names = ["first", "second"]
        source_inputs = [
            (name, {"shape": shape, "decoded": [value] * math.prod(shape)}) for name, value in (("a", -17), ("b", 31))
        ]
        expected_outputs = {
            name: np.asarray(source_inputs[index][1]["decoded"]).reshape(shape).tolist()
            for index, name in enumerate(output_names)
        }
    (capsule / "source.mlir").write_text(source)
    policy = {"compare": "tolerance_float", "dtype": "f32", "atol": 0.0, "rtol": 0.0}
    if copy_support is not None:
        policy = {"compare": "exact_int", "dtype": dtype}
    write_json(root / "original_policy.json", policy)
    declaration = {
        "name": "private_runtime_control",
        "kind": "model" if copy_support is not None else "model_slice",
        "source_role": "handauthored_compiler_test",
        "label": "hidden",
        "interface_mlir": "source.mlir",
        "operation": {"op": "model" if copy_support is not None else "add"},
        "inputs": [
            {"name": tensor, "shape": shape, "dtype": dtype, "role": role}
            for tensor, role in (
                *((name, "input") for name, _ in source_inputs),
                *((name, "output") for name in output_names),
            )
        ],
        "numeric_policy": policy.copy(),
        "expected": {"instruction_classes": []},
        "required_oracle_tiers": ["L2"],
    }
    if direction == "negative" and mechanism == "original_numeric_gate":
        declaration["numeric_policy"]["atol"] = 100.0
    write_json(capsule / "capsule.yaml", declaration)
    from merlin.targetgen.golden_store import write_golden

    write_golden(
        capsule,
        {
            "golden_source": "private_upstream_control",
            "outputs": expected_outputs,
            "oracle_provenance": {"inputs": dict(source_inputs)},
        },
    )
    driver = Path(copy_controls.__file__ if copy_support is not None else controls.__file__).read_text()
    if direction == "negative" and mechanism == "source_correspondence":
        driver = driver.replace(
            '("add", *(expressions[value] for value in op.operands))', "expressions[op.operands[0]]", 1
        )
    if direction == "negative" and mechanism == "original_output_roster":
        driver = driver.replace(
            "PrimitiveProgram(shape, len(function.body.block.args), returned)",
            "PrimitiveProgram(shape, len(function.body.block.args), returned[:-1])",
            1,
        )
    (candidate / "driver.py").write_text(driver)
    target, entry = build_service.target, build_service.recipe.require_kernel_stack_frame().entry_symbol
    commands = {
        command: {"argv": ["python3", "driver.py", command, "{input_mlir}", target, entry]}
        for command in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
    }
    if copy_support is not None:
        for command in commands.values():
            command["argv"].append(copy_support.callee_symbol)
    commands["emit_command_buffer"]["argv"].append("{output_json}")
    write_json(
        candidate / "manifest.yaml",
        {
            "artifact_type": "mlir_oot_target_backend",
            "target": target,
            "language": "python",
            "authoring": {"mode": "hand_curated", "scope": "private primitive transport control only"},
            "integrity_exempt": False,
            "entrypoints": {"tool": "driver.py"},
            "commands": commands,
        },
    )
    member = {
        "name": declaration["name"],
        "program_sha256": sha256_file(capsule / "source.mlir"),
        "sha256": exact_tree_record(capsule)["sha256"],
        "output_roster": output_names,
    }
    runs = root / "grade"
    fixture = RuntimeControlFixture(
        name,
        {
            "package_dir": candidate,
            "capsules_root": [capsule],
            "runs_root": runs,
            "contract": contract_root,
            "target": target,
            "timeout": 60,
            "max_workers": 1,
        },
        {
            "result_path": runs / declaration["name"] / "capsule_result.json",
            "member": member,
            "candidate_root": candidate,
            "compiler_snapshot": candidate,
            "candidate_sha256": exact_tree_record(candidate)["sha256"],
            "capsule_root": capsule,
            "evidence_root": root,
            "target_descriptor": target_descriptor,
            "frontend": "mlir",
            "required_effects": REQUIRED_EXECUTION_EFFECTS,
            "timeout_s": 60,
        },
        member,
        "mlir",
        exact_tree_record(candidate)["sha256"],
        sha256_file(target_descriptor),
        capsule,
        root,
        REQUIRED_EXECUTION_EFFECTS,
        (
            "component_source_lowering",
            "parse",
            "lower_interface_to_target",
            "emit_command_buffer",
            "emit_target_artifact",
        ),
    )
    return fixture
