"""Prepare portable, opt-in complete-model demo manifests without compilation.

Three existing tracked model capsules retain their exact shapes and operations.
The separately constructed MLP has twelve live runtime-weight layers. An
original small-Llama recapture is optional and must be explicitly available.
Two larger stored model slices can be explicitly included without changing them.
Preparation supplies neither source admission nor proof/native/model accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

from larger_model_gate_demo import SCHEMA, bound, file_ref, load_manifest, write
from larger_model_mlp import fixture
from larger_model_native_demo import validate_contract

CAPSULES = ("M0_small_llama_gemmini", "M2_microvit_gemmini", "M4_smolvla_denoise_gemmini")
LARGE_CAPTURES = ("SY_model_resnet50", "SY_model_tiny_llama")
MLP_PARAMETERS = {"layers": 12, "width": 256, "batch": 8, "classes": 31}
NATIVE_SCOPE = "finite deterministic baseline-versus-gated execution, no modelaccuracyclaim"


def _contract(source):
    """Read every original @forward argument/result from its actual typed IR."""
    from xdsl.dialects import builtin, func

    from merlin.llvmlower.passes_xdsl import parse_mlir_text

    before = file_ref(source)
    raw = Path(before["path"]).read_bytes()
    if len(raw) > 32 << 20:
        raise ValueError("complete demo source exceeds the explicit32MiB preparation policy")
    module = parse_mlir_text(raw.decode("utf-8"))
    functions = [
        operation
        for operation in module.walk()
        if isinstance(operation, func.FuncOp) and operation.sym_name.data == "forward"
    ]
    if len(functions) != 1 or not functions[0].body.blocks:
        raise ValueError("original model must define exactly one @forward entry")
    aliases = {
        "f16": "float16",
        "f32": "float32",
        "f64": "float64",
        "i1": "bool",
        "i8": "int8",
        "i16": "int16",
        "i32": "int32",
        "i64": "int64",
    }
    contract = {"entry": "forward", "arguments": [], "outputs": [], "scope": NATIVE_SCOPE}
    for key, types in (
        ("arguments", functions[0].function_type.inputs),
        ("outputs", functions[0].function_type.outputs),
    ):
        for index, actual in enumerate(types):
            if not isinstance(actual, builtin.TensorType) or str(actual.element_type) not in aliases:
                raise ValueError("demo native metadata requires supported static ranked tensor types")
            shape = list(actual.get_shape())
            if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
                raise ValueError("demo native metadata requires positive static extents")
            name = ("arg" if key == "arguments" else "result") + str(index)
            row = {"name": name, "shape": shape, "dtype": aliases[str(actual.element_type)]}
            if key == "arguments":
                row["canonical_input_name"] = name
            contract[key].append(row)
    if validate_contract(source, contract) != before or file_ref(source) != before:
        raise ValueError("complete model source changed during native ABI inspection")
    return before, contract


def _recapture_root(selected, environment):
    explicit = selected is not None
    candidate = selected if explicit else environment.get("MERLIN_RECAPTURES")
    if not candidate:
        return None
    root = Path(candidate).expanduser().resolve()
    source = root / "small_llama_int8_consistent" / "model.mlir"
    if not source.is_file():
        if explicit:
            raise ValueError("explicit recapture root lacks small_llama_int8_consistent/model.mlir")
        return None
    for name in ("input_order.json", "weights.safetensors.manifest.json"):
        if not (source.parent / name).is_file():
            raise ValueError("selected original recapture lacks provenance file " + name)
    return root


def _large_caption(folder):
    """Carry the stored source's declarations without upgrading its provenance."""
    import yaml

    path = folder / "capsule.yaml"
    pinned = file_ref(path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pinned["sha256"] or file_ref(path) != pinned:
        raise ValueError("original model provenance changed while reading")
    metadata = yaml.safe_load(raw)
    provenance = metadata["input_provenance"]
    caption = (
        "Original stored int8 ResNet50 slice at 224x224; synthesized/random capture; "
        "the original metadata selects M2M_RESNET_RANDOM=1 and seeded synthetic inputs."
        if folder.name == "SY_model_resnet50"
        else "Original stored int8 TinyLlama PyTorch model slice; synthesized roster capture; "
        "the original input provenance is undeclared."
    )
    return {
        "source_role": metadata["source_role"],
        "source_reference": metadata["source_reference"],
        "source_caption": caption + " No pretrained-model or model-accuracy claim.",
        "original_input_provenance": {
            key: provenance[key]
            for key in (
                "status",
                "synthetic_inputs",
                "paper_ready",
                "accuracy_claim_supported",
                "accuracy_claim_withheld_because",
                "loader_env",
            )
            if key in provenance
        },
        "pretrained_model_claim": False,
        "model_accuracy_claim": False,
    }


def prepare(destination, *, recaptures=None, environment=None, include_large_captures=False):
    """Prepare 8/10 cases, or 12/14 with large slices; generate no tensor arrays."""
    from merlin.common.paths import repo_root

    if type(include_large_captures) is not bool:
        raise ValueError("large stored captures require an explicit boolean selection")
    root = repo_root()
    environment = os.environ if environment is None else environment
    recapture = _recapture_root(recaptures, environment)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    cases = destination / "cases"
    cases.mkdir()
    helper_paths = [
        Path(__file__).resolve(),
        root / "merlin/tests/ir/larger_model_gate_demo.py",
        root / "merlin/tests/ir/larger_model_mlp.py",
        root / "merlin/tests/ir/larger_model_native_demo.py",
    ]
    helper_refs = [file_ref(path) for path in helper_paths]
    inputs = []
    if recapture is not None:
        folder = recapture / "small_llama_int8_consistent"
        inputs.append(
            (
                "small_llama_int8_consistent",
                folder / "model.mlir",
                [folder / "input_order.json", folder / "weights.safetensors.manifest.json"],
                "original_capture",
                {},
                "lower_quant_ext",
                None,
            )
        )
    for name in CAPSULES:
        folder = root / "merlin/contract/capsules/model" / name
        inputs.append(
            (
                name,
                folder / "capsule.interface.mlir",
                [folder / "capsule.yaml", folder / "capsule.pytorch.py", folder / "README.md"],
                "existing_model_capsule",
                {},
                "lower_quant_ext",
                None,
            )
        )
    value = fixture(**MLP_PARAMETERS)
    generated = destination / "constructed-models/MLP12_F16"
    generated.mkdir(parents=True)
    source = generated / "model.mlir"
    source.write_bytes(value["source"].encode("utf-8"))
    model = generated / "MODEL.json"
    write(
        model,
        {
            "parameters": MLP_PARAMETERS,
            "model": value["model"],
            "definition": file_ref(root / "merlin/tests/ir/larger_model_mlp.py"),
            "source": file_ref(source),
            "inputs_or_outputs_generated": False,
        },
    )
    inputs.append(
        (
            "MLP12_F16",
            source,
            [root / "merlin/tests/ir/larger_model_mlp.py", model],
            "explicit_model_definition",
            {"macc_f32acc": True},
            "bf16_f32acc_draft_contract",
            MLP_PARAMETERS,
        )
    )
    captions = {}
    if include_large_captures:
        for name in LARGE_CAPTURES:
            folder = root / "merlin/contract/capsules/model" / name
            captions[name] = _large_caption(folder)
            inputs.append(
                (
                    name,
                    folder / "capsule.interface.mlir",
                    [folder / "capsule.yaml", folder / "capsule.pytorch.py", folder / "README.md"],
                    "existing_model_capsule",
                    {},
                    "lower_quant_ext",
                    None,
                )
            )
    pinned_inputs, written = [], []
    for name, source, provenance, source_kind, options, selector, parameters in inputs:
        source_ref, contract = _contract(source)
        original_refs = [file_ref(path) for path in provenance]
        pinned_inputs.extend((source_ref, *original_refs))
        for textual in (False, True):
            route = "textual" if textual else "xdsl"
            identity = name + ("_" if name == "MLP12_F16" else "-") + route
            manifest = {
                "schema": SCHEMA,
                "id": identity,
                "source": source_ref,
                "provenance": [*original_refs, *helper_refs],
                "source_kind": source_kind,
                "options": options,
                "textual": textual,
                "required_changed_selectors": [selector],
                "required_verified_selectors": [selector],
                "native_contract": contract,
            }
            if parameters is not None:
                manifest["model_parameters"] = dict(parameters)
            if name in captions:
                manifest.update(captions[name])
            path = cases / (identity + ".json")
            write(path, manifest)
            # Exercise the canonical harness intake only; it does not invoke
            # source recognition, proof, compiler lowering or native execution.
            load_manifest(path)
            written.append({"manifest": file_ref(path)})
    for reference in [*helper_refs, *pinned_inputs, *(row["manifest"] for row in written)]:
        bound(reference)
    roster = {
        "schema": "merlin.larger-model-campaign.v1",
        "cases": written,
        "scope": (
            "three existing complete scaled model capsules and a separately defined connected twelve-layer half MLP; "
            "two routes each; optional original small-Llama recapture"
        )
        + (
            "; two unchanged larger stored model slices, without a pretrained/accuracy claim"
            if include_large_captures
            else ""
        ),
        "preparation": {
            "helper_sources": helper_refs,
            "original_inputs": pinned_inputs,
            "recapture_selected": recapture is not None,
            "large_captures_selected": include_large_captures,
            "model_parameters": MLP_PARAMETERS,
            "source_operations_shapes_or_types_changed": False,
            "source_admission_or_proofs_run": False,
            "compiler_or_native_run": False,
            "reference_or_input_arrays_generated": False,
            "model_accuracy_claim": False,
        },
    }
    write(destination / "CAMPAIGN-MANIFEST.json", roster)
    return roster


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--recaptures", type=Path, help="explicit original recapture root; missing requested inputs fail"
    )
    parser.add_argument(
        "--include-large-captures",
        action="store_true",
        help="also include unchanged stored TinyLlama and 224x224 ResNet50 graphs on both routes; no accuracy claim",
    )
    args = parser.parse_args(argv)
    result = prepare(args.out, recaptures=args.recaptures, include_large_captures=args.include_large_captures)
    print(f"Prepared {len(result['cases'])} complete-model route manifests; no compilation or execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
