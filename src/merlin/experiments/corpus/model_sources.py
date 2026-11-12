"""Construct complete model capsules with the original input, output and source evidence."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from merlin.capture.artifacts import (
    CapsuleArtifacts,
    M2MUnavailable,
    _portable_weights_reference,
    input_provenance_record,
    model_input_abi,
)
from merlin.capture.numerics import _flatten, _model_output_numeric_policy, _tol, compile_dtype, derived_recipe
from merlin.capture.pytorch import PytorchRefSource
from merlin.development.targetgen.golden_store import write_golden
from merlin.experiments.corpus.model_inputs import (
    freeze_materialized_loader_profile,
    freeze_model_loader_dependencies,
    model_capture_declaration_identity,
    model_capture_env,
    model_capture_python,
    resolve_model_loader,
    resolve_model_workload,
)
from merlin.experiments.corpus.model_requirements import _checked_lanes, _model_semantic_block, model_accelerator_demand
from merlin.experiments.corpus.source_evidence import _freeze_selected_m2m_tool, _write_frontend_evidence


def write_model_capsule(
    entry: dict,
    binding,
    out_root,
    *,
    source: PytorchRefSource | None = None,
    artifact: CapsuleArtifacts | None = None,
):
    """Materialize a whole-model capsule: the model is lowered end-to-end via model2MLIR (the linalg IS
    the interface), weights are externalized alongside, and the golden is the host torch-eager output.
    A ``gate`` (default ``after_op_pass_fraction: 0.8``) defers scheduling until the op suite passes."""
    import shutil

    import yaml

    # only a whole-model capsule may ship externalized weights (a compile INPUT); guard so an op capsule
    # can never route here and leak a weight-derived answer (see CapsuleArtifacts). A model capsule is
    # identified by kind/op/cat == "model"; op capsules never carry any of those.
    if "model" not in (entry.get("kind"), entry.get("op"), entry.get("cat")):
        raise ValueError(
            f"write_model_capsule requires a model capsule (kind/op/cat == 'model'); got "
            f"kind={entry.get('kind')!r} op={entry.get('op')!r} cat={entry.get('cat')!r} "
            f"— weights ship only for whole-model capsules"
        )
    dtype = entry.get("operand_dtype") or binding.operand_dtype
    src = (source or PytorchRefSource()) if artifact is None else None
    loader = resolve_model_loader(entry, src.m2m_dir) if src is not None else None
    # THE MODEL'S OWN CAPTURE DECLARATION. A roster model states next to its loader how it must be
    # captured -- which input stream to read, which checkpoint, which interpreter its dependencies live
    # in -- and this path used to invoke the loader with none of it. Every loader that declines to
    # invent its inputs then raised, and the corpus recorded the model as one it could not build: a
    # fact about the invocation, published as a fact about the compiler's reach.
    workload = resolve_model_workload(entry, src.m2m_dir) if src is not None else None
    capture_declaration = model_capture_declaration_identity(workload, src.m2m_dir) if src is not None else None
    capture_env = model_capture_env(workload, src.m2m_dir) if src is not None else {}
    capture_quantization = entry.get("capture_quantization")
    if capture_quantization not in (None, "already_materialized"):
        raise ValueError(f"unknown model capture_quantization {capture_quantization!r}")
    already_quantized = capture_quantization == "already_materialized"
    if already_quantized and (entry.get("quant_scheme") or entry.get("quant_recipe")):
        raise ValueError("already-materialized model capture cannot also declare a quant scheme or recipe")
    # The capsule's declared scheme, not the dtype default. `quant_scheme` is how an entry says which
    # arithmetic it means by "int8"; dropping it here silently substituted weight-only quantization.
    # A recipe derived from the TARGET outranks a scheme named for a dtype: the scheme's defaults are
    # a framework's (per-channel weight scales, say, on a readout that holds one scale per command),
    # and the recipe's are what this hardware's readout holds. It is resolved HERE, at capture, and
    # not written into the synthesized profile: whether it can be derived depends on the target's
    # extracted facts, and a committed profile must not change with what a machine has extracted.
    # CAPTURE CAPABILITIES. An entry may ask the capture to also integerize the activation x
    # activation contractions (attention scores, value mixing) and the nonlinears between them. Both
    # replace the stock PT2E flow, so a target-derived recipe does not apply to them.
    act_contractions = bool(entry.get("quant_activation_contractions"))
    int_nonlinear = bool(entry.get("quant_integer_nonlinear"))
    if (act_contractions or int_nonlinear) and already_quantized:
        raise ValueError("an already-materialized capture cannot also ask for capture capabilities")
    recipe = None
    if not already_quantized:
        recipe = entry.get("quant_recipe")
        if recipe is None and not act_contractions:
            recipe = derived_recipe(getattr(binding, "target", None), dtype)
    art = (
        artifact
        if artifact is not None
        else src.capture_loader(
            loader,
            dtype,
            scheme=entry.get("quant_scheme"),
            env=capture_env,
            python=model_capture_python(workload, src.m2m_dir),
            recipe=recipe,
            already_quantized=already_quantized,
            agreement_tolerance=_tol(binding),
            **({"activation_contractions": True} if act_contractions else {}),
            **({"integer_nonlinear": True} if int_nonlinear else {}),
        )
    )
    # WHERE THE INPUTS CAME FROM -- recorded unconditionally, tri-state, and never inferred here.
    if src is not None and model_capture_declaration_identity(workload, src.m2m_dir) != capture_declaration:
        raise M2MUnavailable("selected Model2MLIR capture declaration changed during model capture")
    provenance = input_provenance_record(workload, capture_env, art.meta, capture_declaration)

    # Validate declared external payloads before touching a destination. A
    # missing source must not publish an ABI that names a nonexistent sidecar,
    # or partially overwrite an existing capsule with unrelated capture bytes.
    wsrc = Path(art.weights_path)
    if not wsrc.is_file():
        raise M2MUnavailable("model capture has no external weights file; refusing an incomplete capsule")
    manifest_src = Path((art.meta or {}).get("weights_manifest") or (str(wsrc) + ".manifest.json"))
    if not manifest_src.is_file():
        raise M2MUnavailable(
            "model capture has no external-weight argument manifest; refusing to reconstruct the "
            "quantized graph ABI from a later torch.export"
        )

    d = Path(out_root) / entry["cat"] / entry["name"]
    input_abi = model_input_abi(art)
    in_names = [f"I{i}" for i in range(len(input_abi))]
    idt = binding.cap_dtype(binding.operand_dtype)
    inputs = [
        {"name": nm, "role": "input", "shape": input_abi[i]["shape"], "dtype": input_abi[i]["dtype"]}
        for i, nm in enumerate(in_names)
    ]
    output_abi = list((art.meta or {}).get("output_abi") or [])
    if not output_abi:
        raise M2MUnavailable("model capture has no output_abi; refusing to guess result cardinality")
    golden_values = list(art.golden) if len(output_abi) > 1 else [art.golden]
    out_names = [str(x) for x in (entry.get("outs") or [])]
    if not out_names:
        out_names = [str(entry.get("out", "Y0"))] if len(output_abi) == 1 else [f"Y{i}" for i in range(len(output_abi))]
    if len(out_names) != len(output_abi) or len(set(out_names)) != len(out_names):
        raise M2MUnavailable(f"model declares outputs {out_names!r}, but capture has {len(output_abi)} tensor results")
    typed_public = (entry.get("public_boundary") or {}).get("schema") == "merlin.phase0.public_joint_boundaries.v2"
    output_selection = entry.get("output_numeric_policy")
    if output_selection not in (None, "captured_per_output_v1"):
        raise M2MUnavailable("model output numerical selection must be captured_per_output_v1")
    numeric_policy = _model_output_numeric_policy(
        output_abi,
        golden_values,
        binding,
        output_names=out_names if typed_public or output_selection == "captured_per_output_v1" else None,
    )
    out_name = out_names[0]
    gate = entry.get("gate") or {"after_op_pass_fraction": 0.8}

    # self-contained weights: copy in + rewrite the linalg's absolute prov.weights_file to a relative name
    linalg = art.linalg_mlir
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(wsrc, d / "capsule.weights.safetensors")
    original_weight = art.meta["materialized_weights_reference"] if artifact is not None else str(wsrc)
    linalg, _ = _portable_weights_reference(linalg, original_weight, sidecar=True)
    shutil.copyfile(manifest_src, d / "capsule.weights.safetensors.manifest.json")
    # Leaf constants a capture capability introduced (lookup tables, integer scales): a compile input
    # like the weights, shipped beside them under a fixed member name.
    extra_name = None
    extra_src = Path(str((art.meta or {}).get("extra") or ""))
    if (art.meta or {}).get("extra"):
        if not extra_src.is_file():
            raise M2MUnavailable("model capture declares leaf constants but the file is missing")
        extra_name = "capsule.extra.npz"
        shutil.copyfile(extra_src, d / extra_name)
    if artifact is not None:
        for original, member in (
            ("weights.safetensors", "capsule.weights.safetensors"),
            ("weights.safetensors.manifest.json", "capsule.weights.safetensors.manifest.json"),
        ):
            payload = (d / member).read_bytes()
            declared = art.meta["materialized_artifact_identities"][original]
            if len(payload) != declared["bytes"] or hashlib.sha256(payload).hexdigest() != declared["sha256"]:
                raise M2MUnavailable(f"materialized external payload changed during capsule copying: {original}")
    loader_profile = (
        freeze_materialized_loader_profile(art.meta["materialized_capture"], d) if artifact is not None else {}
    )
    loader_dependencies = (
        freeze_model_loader_dependencies(workload, d, art.meta or {}, src.m2m_dir) if artifact is None else []
    )

    # What this model owes the accelerator, derived from its own captured linalg and this target's role
    # census. Without it the capstone is vacuous: no required classes and no must_accelerate means a
    # submission that ran the whole network on the CPU and got the right numbers PASSES the one capsule
    # the entire suite builds toward.
    _model_family, _model_classes = model_accelerator_demand(linalg, binding)

    cap = {
        "name": entry["name"],
        "kind": "model",
        **({"materialized_capture": art.meta["materialized_capture"]} if artifact is not None else {}),
        "source_role": (
            "materialized_performance_capture"
            if artifact is not None and art.meta["materialized_capture"].get("workload_role") == "performance_scale"
            else "model_derived"
            if artifact is not None
            else entry.get("source_role", "pytorch_model_slice")
        ),
        "source_reference": entry.get("source_reference", f"whole model {entry.get('model', '')}"),
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "inputs": inputs,
        "operation": {
            "op": "model",
            "attributes": {
                "model": entry.get("model", ""),
                "dtype": idt,
                "out": out_name,
                **({"outs": out_names} if len(out_names) > 1 else {}),
                "bundle_abi_version": 2,
                "quant_scheme": (art.meta or {}).get("scheme"),
                **(
                    {
                        "capture_quantization": "already_materialized",
                        "materialized_contractions": (art.meta or {}).get("materialized_contractions"),
                    }
                    if already_quantized
                    else {}
                ),
                **(
                    {
                        "quant_recipe_sha256": (art.meta or {}).get("recipe_sha256"),
                        "quant_recipe_agreement": (art.meta or {}).get("recipe_agreement"),
                    }
                    if (art.meta or {}).get("recipe_sha256")
                    else {}
                ),
                **(
                    {"integerization_receipt": (art.meta or {})["integerization_receipt"]}
                    if (art.meta or {}).get("integerization_receipt")
                    else {}
                ),
                **(
                    {"quant_activation_contractions": (art.meta or {})["activation_contractions"]}
                    if (art.meta or {}).get("activation_contractions")
                    else {}
                ),
                **(
                    {"quant_integer_nonlinear": (art.meta or {})["integer_nonlinear"]}
                    if (art.meta or {}).get("integer_nonlinear")
                    else {}
                ),
                "torch_seed": int((art.meta or {}).get("torch_seed", 0)),
                **({"loader_dependencies": loader_dependencies} if loader_dependencies else {}),
                **loader_profile,
                "compile_dtype": compile_dtype(binding.operand_dtype),
                "arg_order": in_names + out_names,
                "weights": "capsule.weights.safetensors",
                "weights_manifest": "capsule.weights.safetensors.manifest.json",
                **({"extra": extra_name} if extra_name else {}),
            },
        },
        # The output ABI, not the target operand, determines readback width and comparison regime.
        "numeric_policy": numeric_policy,
        "expected": {"instruction_classes": _model_classes},
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
        "gate": gate,
        # INTEROP: the lanes this capsule requires to have carried work (routing-plan keys, e.g.
        # `on_mesh` + `scalar_rvv_lane`). Present only when the profile declares it. A capsule that names
        # lanes is asserting COMPOSITION -- part of the model on the accelerator, part on the lane the
        # target also owns -- so host-lane work is the behaviour under test rather than a fallback
        # failure, and must_accelerate is withheld below for the same reason.
        **({"lanes": {"require": _checked_lanes(entry, binding)}} if (entry.get("lanes") or {}).get("require") else {}),
        "pytorch_ref": {
            "op": "model",
            "dtype": idt,
            **(
                {"loader": "capsule.pytorch.py"}
                if artifact is None or art.meta["materialized_capture"].get("loader_sha256")
                else {}
            ),
        },
        # WHAT THIS CAPSULE'S INPUTS WERE. On the capsule rather than only in the golden, because the
        # capsule is what a reader has in hand when they quote the result, and a pass on seeded
        # synthetic inputs proves the compiler reproduces the reference -- not that the model is
        # accurate. `accuracy_claim_supported` is False unless the loader positively says otherwise.
        "input_provenance": provenance,
        "linalg_mlir": "capsule.interface.mlir",
        # A model capsule carried no semantic block, so the eligibility oracle could not resolve it
        # (`model` is not a family and never will be -- a model is a composition). Naming the family it
        # OWES is what makes it resolvable, and therefore what gives must_accelerate something to bite on:
        # an eligible region that falls back to the CPU is a hard failure. Only asserted when the demand
        # was actually grounded; an ungrounded assertion would fail a conformant submission.
        "semantic": _model_semantic_block(entry, _model_family, _model_classes),
    }
    prov = {nm: {"shape": input_abi[i]["shape"], "decoded": _flatten(art.inputs[i])} for i, nm in enumerate(in_names)}
    golden = {
        "golden_source": "host_torch_eager",
        "oracle_provenance": {
            "engine": "model2MLIR whole-model linalg-on-tensors + host torch-eager",
            "model": entry.get("model", ""),
            "output_dtype": "f32",
            "grade_policy": {"compare": binding.compare, "atol": _tol(binding)[0], "rtol": _tol(binding)[1]},
            "interface": "linalg_positional",
            "arg_order": in_names + out_names,
            **({"pytorch_source": "capsule.pytorch.py"} if cap["pytorch_ref"].get("loader") else {}),
            "linalg_mlir": "capsule.interface.mlir",
            # The same record the capsule carries, on the oracle side too: what the reference was
            # computed over is part of where the reference came from, and a grader reading only the
            # golden must not have to infer it.
            "input_provenance": provenance,
            "inputs": prov,
        },
        "outputs": dict(zip(out_names, golden_values, strict=True)),
    }
    if getattr(art, "float_reference", None) is not None:
        # The float model's outputs, before the capture's quantization or precision cast: the reference
        # the whole-model gate judges accuracy against (:mod:`merlin.analysis.perf.float_accuracy`).
        floats = list(art.float_reference) if len(output_abi) > 1 else [art.float_reference]
        golden["float_reference"] = {
            "source": "the loader's model before quantization or precision conversion, host torch-eager",
            "outputs": dict(zip(out_names, floats, strict=True)),
        }
    frontend_evidence = _write_frontend_evidence(art, d, linalg)
    if frontend_evidence is not None:
        cap["frontend_trace"] = frontend_evidence
    if entry.get("performance") is not None:
        cap["performance"] = copy.deepcopy(entry["performance"])
    if src is not None:
        cap["capture_tool"] = _freeze_selected_m2m_tool(src.m2m_dir, d)
    if artifact is not None and art.meta["materialized_capture"].get("workload_role") == "performance_scale":
        from merlin.development.targetgen.capsule_privacy import capture_reference_closure

        cap["reference_closure"] = capture_reference_closure(
            art.meta["materialized_artifact_identities"],
            source_mode=art.meta["materialized_capture"].get("source_mode"),
        )
    (d / "capsule.yaml").write_text(yaml.safe_dump(cap, sort_keys=False), encoding="utf-8")
    (d / "capsule.interface.mlir").write_text(linalg, encoding="utf-8")
    (d / "expected_instruction_coverage.yaml").write_text(
        yaml.safe_dump(cap["expected"], sort_keys=False), encoding="utf-8"
    )
    if artifact is None:
        (d / "capsule.pytorch.py").write_text(art.pytorch_src, encoding="utf-8")
    else:
        (d / "source-capture-receipt.json").write_bytes(art.meta["materialized_receipt_bytes"])
        if art.meta["materialized_capture"].get("loader_sha256"):
            selected_loader = Path(art.meta["materialized_capture"]["loader_path"])
            loader_bytes = selected_loader.read_bytes()
            if hashlib.sha256(loader_bytes).hexdigest() != art.meta["materialized_capture"]["loader_sha256"]:
                raise M2MUnavailable("materialized loader changed during capsule copying")
            (d / "capsule.pytorch.py").write_bytes(loader_bytes)
        if art.meta["materialized_capture"].get("workload_role") == "performance_scale":
            origin = Path(art.meta["materialized_capture"]["path"]).parent
            from merlin.development.targetgen.materialized_integer_capture import reference_members, selected
            from merlin.development.targetgen.materialized_reference import reference_members as float_reference_members

            references = (
                reference_members(art.meta["materialized_artifact_identities"])
                if selected(art.meta["materialized_capture"])
                else float_reference_members(art.meta["materialized_artifact_identities"])
            )
            for member in ("meta.json", "inputs.npz", "input_order.json", *references):
                raw = (origin / member).read_bytes()
                declared = art.meta["materialized_artifact_identities"][member]
                if len(raw) != declared["bytes"] or hashlib.sha256(raw).hexdigest() != declared["sha256"]:
                    raise M2MUnavailable("public performance capture reference closure changed")
                (d / ("source-capture-" + member)).write_bytes(raw)
    write_golden(d, golden)
    return d


def model_gate_satisfied(cap: dict, op_pass_fraction: float) -> bool:
    """True when a (model) capsule may be scheduled: its ``gate.after_op_pass_fraction`` is met, or absent.
    ``op_pass_fraction`` is the fraction of graded OP-level capsules that passed (0..1)."""
    thr = (cap.get("gate") or {}).get("after_op_pass_fraction")
    return thr is None or op_pass_fraction >= float(thr)
