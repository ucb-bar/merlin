"""Construct operation capsules from their captured PyTorch source and host reference."""

from __future__ import annotations

import math
from pathlib import Path

from merlin.capture.artifacts import M2MUnavailable, _portable_weights_reference, _shape_of
from merlin.capture.demands import linalg_to_iface, verify_exact_application_int_mm
from merlin.capture.loaders import _FUSED_OP_INPUT_NAMES, _OP_INPUT_NAMES
from merlin.capture.pytorch import PytorchRefSource
from merlin.capture.receipts import _flatten, _model_output_numeric_policy, _tol
from merlin.capture.spec import SpecRefSource
from merlin.development.targetgen.corpus import components as component_sources
from merlin.development.targetgen.corpus.declarations import _capture_spec
from merlin.development.targetgen.grading.golden_store import write_golden
from merlin.experiments.corpus.capsules.model_sources import (
    _freeze_selected_m2m_tool,
    _write_frontend_evidence,
)

# ---- pytorch_sources ---------------------------------------------------------------------------


def _host_eager_golden(art, names, out_name, binding, *, interface: str, arg_order, exact_integer: bool = False):
    """The host torch-eager golden.yaml payload shared by the mapped and fused paths."""
    if len(art.inputs) < len(names):
        raise M2MUnavailable(f"op {art.op!r} produced {len(art.inputs)} inputs, expected {len(names)}")
    prov = {nm: {"shape": _shape_of(art.inputs[i]), "decoded": _flatten(art.inputs[i])} for i, nm in enumerate(names)}
    if exact_integer:
        # The source operation's torch-eager operands, not a second seeded stimulus, are the
        # byte-level grading inputs. The worker recorded their dtype before JSON conversion.
        abi = (art.meta or {}).get("input_abi")
        if not isinstance(abi, list) or len(abi) != len(names):
            raise M2MUnavailable("exact integer source capture has no complete typed input ABI")
        for i, name in enumerate(names):
            declared = abi[i]
            if (
                not isinstance(declared, dict)
                or declared.get("dtype") != "i8"
                or declared.get("shape") != prov[name]["shape"]
            ):
                raise M2MUnavailable(f"exact integer source input {name} disagrees with captured i8 ABI")
            values = prov[name]["decoded"]
            if any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or int(value) != value
                or not -128 <= value <= 127
                for value in values
            ):
                raise M2MUnavailable(f"exact integer source input {name} has invalid i8 values")
            prov[name]["dtype"] = "i8"
            prov[name]["integer_bytes_hex"] = bytes(int(value) & 0xFF for value in values).hex()
    return {
        "golden_source": "host_torch_eager",
        "oracle_provenance": {
            "engine": "PyTorch host eager reference; model2MLIR fx_importer produced the separately verified linalg",
            "operand_dtype": binding.cap_dtype(binding.operand_dtype),
            "output_dtype": binding.cap_dtype(binding.accum_dtype),
            "note": "INDEPENDENT of the target RTL; the PyTorch frontend + host eval are the reference.",
            "grade_policy": {"compare": binding.compare, "atol": _tol(binding)[0], "rtol": _tol(binding)[1]},
            "interface": interface,
            "arg_order": list(arg_order),
            "pytorch_source": "capsule.pytorch.py",
            "linalg_mlir": "capsule.linalg.mlir",
            "path_taken": art.meta.get("path_taken"),
            "inputs": prov,
        },
        "outputs": {out_name: art.golden},
    }


def _fused_capsule_yaml(entry: dict, binding, art, names: list[str]) -> dict:
    """A schema-valid capsule.yaml for a FUSED op whose interface is the linalg module directly (positional
    args). No merlin_iface builder / instruction classes — the agent compiles the standard-dialect linalg."""
    idt = binding.cap_dtype(binding.operand_dtype)
    out_name = entry.get("out", "Y0")
    inputs = [
        {"name": nm, "role": "input", "shape": _shape_of(art.inputs[i]), "dtype": idt} for i, nm in enumerate(names)
    ]
    return {
        "name": entry["name"],
        "kind": entry.get("kind", "model_slice"),
        "source_role": entry.get("source_role", "pytorch_model_slice"),
        "source_reference": entry.get("source_reference", ""),
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": inputs,
        "operation": {
            "op": entry["op"],
            "attributes": {"out": out_name, "arg_order": names, "causal": bool(entry.get("causal", False))},
        },
        "numeric_policy": {
            "compare": binding.compare,
            "dtype": idt,
            "atol": _tol(binding)[0],
            "rtol": _tol(binding)[1],
        },
        "expected": {"instruction_classes": [], "modes": dict(entry.get("modes", {}))},
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
        # Same per-target tier declaration the direct-MLIR builders carry (corpus_spec.build): a capsule
        # authored through a different source must not silently lose it, or the same tier that is honestly
        # N/A for the rest of the corpus fails forever on this one.
        **(
            {"inapplicable_oracle_tiers": dict(binding.inapplicable_tiers)}
            if getattr(binding, "inapplicable_tiers", None)
            else {}
        ),
        "pytorch_ref": {"op": entry["op"], "dtype": idt, "loader": "capsule.pytorch.py"},
        "linalg_mlir": "capsule.interface.mlir",
    }


def write_pytorch_capsule(entry: dict, binding, out_root, *, source: PytorchRefSource | None = None):
    """Materialize a full capsule dir from a PyTorch-defined op. For a merlin_iface-mapped op (matmul/
    linear/attention_qk/rmsnorm) the agent-facing interface + expected coverage are DERIVED-AND-VERIFIED
    from the captured linalg (``linalg_to_iface``: the lowering must actually contain the op + shapes) via
    the existing ``corpus_spec`` builder. For a FUSED op (attention_full/softmax/layernorm/geglu/rope) the
    interface IS the m2m linalg module (the agent compiles standard-dialect linalg), fed positionally. In
    BOTH cases the
    golden + canonical inputs come from the host torch-eager run and the pytorch loader + linalg are written
    as visible grounding. Unknown ops fail closed."""
    import yaml

    op = entry.get("op", "matmul")
    if op not in _OP_INPUT_NAMES and op not in _FUSED_OP_INPUT_NAMES:
        raise ValueError(
            f"write_pytorch_capsule: unknown op {op!r} "
            f"(mapped {sorted(_OP_INPUT_NAMES)}; fused {sorted(_FUSED_OP_INPUT_NAMES)})"
        )
    src = source or PytorchRefSource()
    spec = _capture_spec(entry, binding)
    art = src.capture(spec)
    exact_match = None
    if entry.get("application_signature_match"):
        if spec["op"] != "int_matmul" or spec["dtype"] != "fp32" or entry.get("op") != "matmul":
            raise M2MUnavailable("exact application integer slice must capture torch._int_mm without requantization")
        exact_match = verify_exact_application_int_mm(art.linalg_mlir, entry["application_signature_match"])
    out_name = entry.get("out", "Y0")
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    portable_linalg, _ = _portable_weights_reference(art.linalg_mlir, art.weights_path, sidecar=False)

    if op in _OP_INPUT_NAMES:  # merlin_iface interface
        names = _OP_INPUT_NAMES[op]
        cap, mlir = linalg_to_iface(portable_linalg, entry, binding)  # derive-and-verify from the lowering
        if exact_match is not None:
            cap["application_signature_match"] = exact_match
        cap["source_role"] = entry.get("source_role", "pytorch_model_slice")
        cap["pytorch_ref"] = {
            "op": op,
            "dtype": binding.cap_dtype(binding.operand_dtype),
            "loader": "capsule.pytorch.py",
        }
        cap["linalg_mlir"] = "capsule.linalg.mlir"
        golden = _host_eager_golden(
            art,
            names,
            out_name,
            binding,
            interface="merlin_iface",
            arg_order=names,
            exact_integer=exact_match is not None,
        )
        (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    elif op in _FUSED_OP_INPUT_NAMES:  # linalg interface (positional)
        names = _FUSED_OP_INPUT_NAMES[op]
        cap = _fused_capsule_yaml(entry, binding, art, names)
        golden = _host_eager_golden(
            art, names, out_name, binding, interface="linalg_positional", arg_order=names + [out_name]
        )
        from merlin.development.targetgen.applications.native_storage import publish_native_storage

        try:
            publish_native_storage(spec, art, cap, golden, binding)
        except ValueError as exc:
            raise M2MUnavailable(str(exc)) from exc
        if op == "producer_quantizer_observer" or len((art.meta or {}).get("output_abi") or []) > 1:
            from merlin.development.targetgen.corpus.declarations import publication

            try:
                outputs = publication(art, entry)
            except ValueError as exc:
                raise M2MUnavailable(str(exc)) from exc
            out_names = list(outputs)
            cap["operation"]["attributes"].update(out=out_names[0], outs=out_names)
            cap["numeric_policy"] = _model_output_numeric_policy(
                art.meta["output_abi"], list(outputs.values()), binding
            )
            golden["outputs"] = outputs
            golden["oracle_provenance"]["arg_order"] = names + out_names
        if op == "producer_quantizer_observer":
            cap["operation"]["attributes"].update(component_sources.parameters(spec))
        # the linalg module IS the interface the agent compiles
        (d / "capsule.interface.mlir").write_text(portable_linalg, encoding="utf-8")
    else:
        raise ValueError(
            f"write_pytorch_capsule: unknown op {op!r} "
            f"(mapped {sorted(_OP_INPUT_NAMES)}; fused {sorted(_FUSED_OP_INPUT_NAMES)})"
        )

    frontend_evidence = _write_frontend_evidence(art, d, portable_linalg, packaged_path="capsule.linalg.mlir")
    if frontend_evidence is not None:
        cap["frontend_trace"] = frontend_evidence
    cap["capture_tool"] = _freeze_selected_m2m_tool(src.m2m_dir, d)
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8"
    )
    (d / "capsule.pytorch.py").write_text(art.pytorch_src, encoding="utf-8")
    (d / "capsule.linalg.mlir").write_text(portable_linalg, encoding="utf-8")
    write_golden(d, golden)
    return d


# ---- spec_sources ------------------------------------------------------------------------------
# Construct capsules from the authored SpecIR program and its independent golden.


def write_spec_capsule(entry: dict, binding, out_root, *, source: SpecRefSource | None = None):
    """Materialize a capsule from a specir verification spec (``spec_ref: '<gen>:op.<name>'``): the agent
    compiles the merlin_iface interface for the op; the golden + the exact operands come from the spec's own
    command-buffer program (bit-exact refmodel, INDEPENDENT of the target RTL); the spec's command buffer +
    coverage_goal ship as grounding. Fails closed if specir has no program emitter for the gen."""
    import json

    import yaml

    from merlin.development.targetgen.corpus import binding as CS
    from merlin.development.targetgen.corpus import model_builders as _merlin_model_builders_1

    src = source or SpecRefSource()
    D = binding.tile_dim
    M = entry.get("M", entry.get("M_tiles", 1) * D)
    K = entry.get("K", entry.get("K_tiles", 1) * D)
    N = entry.get("N", entry.get("N_tiles", 1) * D)
    art = src.capture(
        entry["spec_ref"],
        workload=(M, N, K),
        tile_dim=D,
        program_emitter=entry.get("spec_program_emitter"),
    )

    cap, mlir = _merlin_model_builders_1.build(entry, binding)  # merlin_iface interface for the op (names align)
    cap["spec_ref"] = entry["spec_ref"]
    cap["spec_program_emitter"] = art.program_emitter
    out_name = entry.get("out", "Y0")
    d = Path(out_root) / entry["cat"] / entry["name"]
    d.mkdir(parents=True, exist_ok=True)
    # map the spec's role-keyed operands onto the interface's input names (lhs -> the activation input,
    # weight -> the resident weight) so the grader feeds the exact values the golden was computed over.
    role_to_name = {"lhs": entry.get("lhs", "A0"), "weight": entry.get("weight", "W")}
    prov_inputs = {
        role_to_name.get(role, role): {"shape": _shape_of(rows), "decoded": rows} for role, rows in art.operands.items()
    }
    golden = {
        "golden_source": f"specir_program_{art.gen}",
        "oracle_provenance": {
            "engine": "specir spec program + refmodel golden (INDEPENDENT of the target RTL; the spec is "
            "the reference)",
            "spec_ref": entry["spec_ref"],
            "program_emitter": art.program_emitter,
            "workload": list(art.workload),
            "compare": art.compare,
            "coverage_goal": art.coverage_goal,
            "opcode_backed_by_spec_rocc": art.opcode_backing,
            "program": "capsule.command_buffer.json",
            "inputs": prov_inputs,
        },
        "outputs": {out_name: art.golden.get("out")},
    }
    (d / "capsule.interface.mlir").write_text(mlir, encoding="utf-8")
    # the program ships as VISIBLE grounding — strip any embedded golden (the MXU/SIMT programs carry the
    # output values under "golden"; the answer lives ONLY in the masked golden files, never in a visible one)
    program = {k: v for k, v in art.command_buffer.items() if k != "golden"}
    (d / "capsule.command_buffer.json").write_text(json.dumps(program, indent=1), encoding="utf-8")
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8"
    )
    write_golden(d, golden)
    return d
