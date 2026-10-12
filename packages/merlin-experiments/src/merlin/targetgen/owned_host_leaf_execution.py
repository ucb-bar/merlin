"""Compile host-only frozen leaves with the selected Merlin host provider.

Candidate entrypoints are never invoked. Source routing is freshly checked before
materialization; numerical results qualify platform infrastructure, not candidate
host arithmetic or accelerator support.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from merlin.compile.materialized_model_inputs import MaterializedModelInputs
from merlin.compile.model_execution_inputs import file_sha256


def selected_host_leaf(capsule):
    if capsule.get("kind") not in {"layer", "model_slice"}:
        return False
    screen = capsule.get("software_screen") or {}
    decisions = screen.get("decisions") or []
    return (
        bool(decisions)
        and screen.get("status") == "admitted"
        and all(row.get("placement") == "host" and row.get("status") == "admitted" for row in decisions)
    )


def _original_host_source(capsule, *, target):
    from .native_model_execution import independent_frozen_source_eligibility

    root = Path(capsule["__dir__"])
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("provided host leaf source is indirect")
    declaration = yaml.safe_load((root / "capsule.yaml").read_text())
    if not selected_host_leaf(declaration) or (declaration.get("semantic") or {}).get("must_accelerate"):
        raise ValueError("provided host leaf lacks exact reviewed host-only admission")
    source_name = declaration.get("linalg_mlir") or declaration.get("interface_mlir")
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise ValueError("provided host leaf needs a capsule-local original linalg source")
    source = root / source_name
    if source.is_symlink() or not source.is_file():
        raise ValueError("provided host leaf original source is absent or indirect")
    attrs = dict((declaration.get("operation") or {}).get("attributes") or {})
    attrs.setdefault("dtype", "fp32")
    checked = {
        **declaration,
        "__dir__": str(root),
        "interface_mlir": source_name,
        "operation": {**declaration["operation"], "attributes": attrs},
    }
    certificate = independent_frozen_source_eligibility(checked, target=target)
    regions = certificate.get("regions")
    if not isinstance(regions, list) or not regions or any(row.get("target_eligible") is not False for row in regions):
        raise ValueError("provided host leaf contains eligible or unclassified accelerator computation")
    return declaration, source, certificate


def materialize_host_leaf(capsule, *, target, output):
    """Build an exact typed saved-model ABI from original IR and independent values."""
    from merlin.common.mlir_query import forward_signature
    from merlin.targetgen.golden_store import load_golden

    from .capsule_runner import _content_identity, _numpy_dtype, _write_deterministic_npz

    declaration, source, certificate = _original_host_source(capsule, target=target)
    root = source.parent
    files = {"source": source, "capsule": root / "capsule.yaml", "golden": root / "golden.yaml"}
    from merlin.targetgen import golden_store

    arrays = root / golden_store.ARRAYS
    if arrays.exists():
        files["golden_arrays"] = arrays
    if any(path.is_symlink() or not path.is_file() for path in files.values()):
        raise ValueError("provided host leaf source/golden files are indirect")
    identity = _content_identity(files, root=root)
    document = load_golden(root)
    inputs = declaration.get("inputs") or []
    signature, results = forward_signature(source)
    values = (document.get("oracle_provenance") or {}).get("inputs") or {}
    outputs = document.get("outputs") or {}
    names = (document.get("oracle_provenance") or {}).get("arg_order") or []
    input_names = [row["name"] for row in inputs]
    if len(signature) != len(inputs) or names[: len(inputs)] != input_names or set(values) != set(input_names):
        raise ValueError("provided host leaf source/input/golden ABI differs")
    output_names = names[len(inputs) :]
    if len(results) != len(output_names) or set(output_names) != set(outputs):
        raise ValueError("provided host leaf full output roster differs")
    packed_inputs, packed_outputs = {}, {}
    for index, (declared, (shape, dtype)) in enumerate(zip(inputs, signature, strict=True)):
        value = values[declared["name"]]
        if declared.get("shape") != list(shape) or declared.get("dtype") != dtype or value.get("shape") != list(shape):
            raise ValueError("provided host leaf input typed extent differs")
        array = np.asarray(value["decoded"], dtype=_numpy_dtype(dtype))
        if array.size != int(np.prod(shape)):
            raise ValueError("provided host leaf lacks complete input values")
        packed_inputs[f"in{index}"] = array.reshape(shape)
    for name, (shape, dtype) in zip(output_names, results, strict=True):
        array = np.asarray(outputs[name], dtype=_numpy_dtype(dtype))
        if array.size != int(np.prod(shape)):
            raise ValueError("provided host leaf lacks complete golden values")
        packed_outputs[name] = array.reshape(shape)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("provided host leaf materialization needs a fresh directory")
    output.mkdir(parents=True)
    (output / "model.mlir").write_bytes(source.read_bytes())
    # A parameter-free source has no stored tensors; all original arguments
    # remain runtime inputs. Emit the standard empty, 8-byte-aligned container.
    import struct

    header = b"{}      "
    (output / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)
    (output / "weights.safetensors.manifest.json").write_text(
        json.dumps({str(index): {"kind": "input", "name": name} for index, name in enumerate(input_names)})
    )
    _write_deterministic_npz(output / "inputs.npz", packed_inputs)
    (output / "input_order.json").write_text(json.dumps({name: index for index, name in enumerate(input_names)}))
    _write_deterministic_npz(output / "goldens.npz", packed_outputs)
    (output / "output_order.json").write_text(json.dumps(output_names))
    np.save(output / "golden.npy", packed_outputs[output_names[0]], allow_pickle=False)
    artifacts = {
        path.name: {"sha256": file_sha256(path), "size_bytes": path.stat().st_size}
        for path in output.iterdir()
        if path.is_file()
    }
    materialized = MaterializedModelInputs.bind(
        output,
        artifacts=artifacts,
        validator_identity={"worker_sha256": file_sha256(Path(__file__))},
        source_identity=identity,
    )
    if _content_identity(files, root=root) != identity:
        raise ValueError("provided host leaf source changed during materialization")
    return materialized, certificate, identity


def _verify_source_identity(capsule, identity):
    from .capsule_runner import _content_identity

    root = Path(capsule["__dir__"])
    files = {}
    for role, record in identity["artifacts"].items():
        name = record["path"]
        if Path(name).name != name:
            raise ValueError("provided host source identity has an unsafe artifact path")
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("provided host source identity is indirect")
        files[role] = path
    if _content_identity(files, root=root) != identity:
        raise ValueError("provided host leaf original source or full golden changed")


def _host_leaf_structure(receipt, *, target, facts):
    """Reopen actual host compilation/link custody and forbid accelerator words."""
    from merlin.common.facts_view import interface
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.compilation_recipe import verify_completed_recipe
    from merlin.runtime.backends.spike import gcc_path
    from merlin.targetgen import elf_lanes

    from .owned_host_model_execution import _pinned

    output = receipt["output"]
    elf = Path(output["elf"])
    if file_sha256(elf) != output["elf_sha256"]:
        raise ValueError("provided host leaf ELF changed")
    recipe = output["compilation_recipe"]
    completed = verify_completed_recipe(Path(recipe["path"]), executable=elf, expected_recipe_sha256=recipe["sha256"])
    ir = _pinned(output["host_artifacts"], "llvm_ir")
    obj = _pinned(output["host_artifacts"], "object")
    commands = [row for row in completed["commands"] if row["output"]["path"] == str(obj)]
    if (
        len(commands) != 1
        or Path(commands[0]["executable"]["path"]) != toolchain.clang().resolve()
        or "-c" not in commands[0]["argv"]
    ):
        raise ValueError("provided host leaf lacks its exact selected compiler invocation")
    if not any(row["path"] == str(ir) and row["sha256"] == file_sha256(ir) for row in commands[0]["inputs"]):
        raise ValueError("provided host leaf LLVM is not its actual object source")
    final = completed["commands"][-1]
    if Path(final["executable"]["path"]) != gcc_path().resolve() or not any(
        row["path"] == str(obj) and row["sha256"] == file_sha256(obj) for row in final["inputs"]
    ):
        raise ValueError("provided host leaf object is not its actual final link input")
    table = interface(json.loads(Path(facts).read_text()).get("facts"), "funct_decode_table") or {}
    opcode = table.get("custom_opcode")
    if type(opcode) is not int:
        raise ValueError("provided host leaf cannot establish the selected accelerator opcode")
    blob = elf.read_bytes()
    sections = elf_lanes.executable_sections(blob)
    if not sections or any(
        word & 0x7F == opcode
        for _, off, size, address in sections
        for _, word in elf_lanes.instruction_words(blob[off : off + size], address)
    ):
        raise ValueError("provided host leaf ELF contains accelerator work or no auditable executable sections")
    return {
        "elf": str(elf),
        "elf_sha256": output["elf_sha256"],
        "recipe": recipe,
        "scope": "provided host compiler custody and complete ELF accelerator prohibition",
    }


def grade_owned_host_leaf(
    capsule, *, target, paths, cfg, run_id, timeout, no_oracle, readback_policy=None,
    reachable_oracle_tiers=None,
):
    """Execute one provided-host image at every required tier without candidate compilation."""
    import os

    from merlin.compile.baremetal_model import compile_saved_model
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.targetgen import elf_lanes
    from merlin.targetgen.contract.readback_policy import FULL_VALUES_BIN, selected
    from merlin.targetgen.oracle_policy import selected_l3_engine_report
    from merlin.targetgen.rtl_engine_policy import ELABORATED_RTL

    from .capsule_runner import TierResult, _finalize_capsule_result, _resolve_model_host_lane
    from .owned_host_model_execution import (
        _functional_engine_for_facts,
        _run_selected_functional,
        _numeric_full_output,
        _run_selected_rtl,
        _selected_facts,
        _selected_file,
    )

    record, tiers, failure = {}, {}, None
    status, numeric = "incomplete", {"status": "skipped"}
    required = set(capsule.get("required_oracle_tiers") or [])
    try:
        root = Path(capsule["__dir__"])
        frozen = yaml.safe_load((root / "capsule.yaml").read_text())
        capsule = {**frozen, "__dir__": str(root)}
        required = set(capsule.get("required_oracle_tiers") or [])
        if not required or required - {"L0", "L1", "L2", "L3"}:
            raise ValueError("provided host leaf has no complete supported frozen tier selection")
        readback_policy = selected(readback_policy)
        if readback_policy is not None and readback_policy.transport != FULL_VALUES_BIN:
            raise ValueError("provided host leaf requires full binary values")
        selection_record = json.loads(os.environ["MERLIN_MODEL_OWNED_HOST_BUILD_SELECTION"])
        selection_path = _selected_file({"selection": selection_record}, "selection")
        selection = json.loads(selection_path.read_text())
        catalog, dts, facts = (_selected_file(selection, role) for role in ("board_catalog", "dts", "rtl_facts"))
        _, host, host_identity = _resolve_model_host_lane(target, None)
        host_bytes = strict_tree_sha256(host)
        from merlin.common.artifacts import new_product

        product = new_product(
            "owned-host-selfcheck", version=1, target=target, update_latest=False,
            sources=[str(paths.run_path)], notes="Provided-host leaf build and native evidence custody",
        )
        output = product.path
        materialized, source, source_identity = materialize_host_leaf(capsule, target=target, output=output / "inputs")
        receipt = compile_saved_model(
            capture=materialized.root,
            package=host,
            board_catalog=catalog,
            board=selection["board"],
            dts=dts,
            output=output / "model",
            target=target,
            run="none",
            arena_mb=selection["arena_mb"],
            timeout_s=timeout,
            readback="full",
            materialized_inputs=materialized,
            respect_captured_quantization_scope=True,
        )
        if receipt.get("status") != "compiled":
            raise ValueError("provided host compilation failed: " + str(receipt.get("failure")))
        structure = _host_leaf_structure(receipt, target=target, facts=facts)
        record = {
            "runtime_inputs": str(materialized.root),
            "build": receipt,
            "host_provider": host_identity,
            "source_eligibility": source,
            "source_identity": source_identity,
            "structure": structure,
            "scope": "platform host infrastructure qualification; no candidate host code or accelerator credit",
        }
        record_path = output / "qualification.json"
        elf = Path(receipt["output"]["elf"])
        before = file_sha256(elf)
        for tier in ("L0", "L1"):
            tiers[tier] = TierResult(
                tier=tier,
                status="pass",
                mandatory=tier in required,
                evidence=str(record_path),
                reason="provided host compiler/ELF structural qualification",
            )
        protected_facts = _selected_facts(facts, target=target)
        from merlin.runtime.boards import load_boards

        board = load_boards(catalog).get(selection["board"])
        if board is None or board.target != target or board.rtl_sim_config != protected_facts["config"]:
            raise ValueError("provided host leaf board and RTL facts identify different configurations")
        for tier in sorted(required - {"L0", "L1"}):
            if no_oracle or (reachable_oracle_tiers is not None and tier not in reachable_oracle_tiers):
                tiers[tier] = TierResult(
                    tier=tier, status="unavailable", mandatory=True,
                    reason="required provided-host tier is not selected in this grading request",
                )
                failure = {
                    "plane": "oracle_availability", "category": "NOT_RUN_IS_NOT_PASS",
                    "tier": tier, "tier_status": "unavailable",
                    "detail": "required provided-host tier is not selected in this grading request",
                }
                continue
            if no_oracle:
                tiers[tier] = TierResult(tier=tier, status="unavailable", mandatory=True, reason="no oracle requested")
                continue
            if tier == "L2":
                backend, citation, verify = _functional_engine_for_facts(target, protected_facts)
                verify()
                console = _run_selected_functional(backend, elf, timeout=timeout)
                verify()
                engine = {"engine": "spike", "engine_citation": citation}
            elif tier == "L3":
                chosen = selected_l3_engine_report(target)
                if not chosen.get("available") or (
                    os.environ.get("MERLIN_REQUIRED_RTL_ENGINE")
                    and chosen.get("engine") != os.environ["MERLIN_REQUIRED_RTL_ENGINE"]
                ):
                    raise ValueError("provided host leaf required RTL engine unavailable")
                engine = _run_selected_rtl(
                    elf,
                    target=target,
                    engine=chosen["engine"],
                    facts=protected_facts,
                    timeout=timeout,
                    elf_sha256=before,
                )
                console = engine.pop("console")
            else:
                raise ValueError("provided host leaf has an unsupported required tier")
            if not isinstance(console, bytes):
                raise ValueError("provided host leaf native readback did not preserve bytes")
            console_path = output / f"{tier}-console.bin"
            console_path.write_bytes(console)
            product.add_artifact(console_path.name)
            _verify_source_identity(capsule, source_identity)
            numeric, metrics = _numeric_full_output(capsule, record, console)
            _verify_source_identity(capsule, source_identity)
            tiers[tier] = TierResult(
                tier=tier,
                status=numeric["status"],
                mandatory=True,
                evidence=str(console_path),
                derived_from_rtl=(tier == "L3" and chosen.get("fidelity") == ELABORATED_RTL),
                cycle_accurate=False,
            )
            record.setdefault("native", {})[tier] = {
                **engine,
                "numeric": numeric,
                "metrics": metrics,
                "elf_sha256": before,
                "console_sha256": file_sha256(console_path),
            }
            if file_sha256(elf) != before or strict_tree_sha256(host) != host_bytes:
                raise ValueError("provided host leaf compiler or image changed")
            materialized.verify(materialized.root)
            if _original_host_source(capsule, target=target)[2] != source:
                raise ValueError("provided host leaf source routing changed")
        for role in ("board_catalog", "dts", "rtl_facts"):
            _selected_file(selection, role)
        _selected_file({"selection": selection_record}, "selection")
        if strict_tree_sha256(host) != host_bytes or file_sha256(elf) != before:
            raise ValueError("provided host leaf compiler or image changed before qualification")
        _verify_source_identity(capsule, source_identity)
        record_path.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
        product.add_artifact(str(record_path.relative_to(output)))
        product.add_artifact("model")
        product.add_artifact("inputs")
        product.write_manifest()
        status = (
            "fail" if any(row.status == "fail" for row in tiers.values())
            else "incomplete" if any(row.status == "unavailable" for row in tiers.values())
            else "pass"
        )
        record["lane_report"] = elf_lanes.lane_report_from_elf(capsule, elf, target=target)
    except Exception as exc:
        status = "incomplete"
        failure = {
            "plane": "platform_host",
            "category": "NOT_RUN_IS_NOT_PASS",
            "detail": f"{type(exc).__name__}: {exc}",
        }
    return _finalize_capsule_result(
        name=capsule["name"],
        capsule=capsule,
        status=status,
        failure=failure,
        tiers=tiers,
        trace_check_res={"status": "skipped", "violations": []},
        numeric=numeric,
        required=required,
        no_oracle=no_oracle,
        eff_target=target,
        paths=paths,
        run_id=run_id,
        cfg=cfg,
        extra={
            "provided_host_qualification": record,
            **({"lane_report": record["lane_report"]} if "lane_report" in record else {}),
        },
    )
