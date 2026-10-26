"""Calculate every declared output from fresh prepared ungrouped quantized IR.

Uses the existing Merlin normalizer, argument binding, numerical interpreter and
portable host math. Does not execute TVM or inherit historical oracle admission.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_selected_inputs import digest

from merlin.common.yaml import load_yaml
from merlin.model_check import catalog as CAT
from merlin.model_check import chain as CH
from merlin.model_check import models as MODELS
from merlin.runtime import linalg_numpy as LN
from merlin.runtime.backends.zephyr_model import _prepare_model_mlir
from merlin.runtime.host_math_reference import host_math_context, reference_math


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("tiny_llama", "smolvla"), required=True)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    model = MODELS.load(args.models).model(args.model)
    declarations = load_yaml(args.protocol)["models"][args.model]["stages"]
    policy = "portable_exp_trig" if args.model == "tiny_llama" else "portable_neural_f32"
    with np.load(args.inputs, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data}
    catalogs = {}
    sources = [args.models, args.protocol, args.inputs, Path(__file__)]
    for program in model.programs:
        stage = args.capture / "stages" / program.name
        if not (stage / "model.mlir").is_file():
            raise ValueError("Missing actual quantized stage: " + str(stage))
        directory = args.output / "prepared" / program.name
        directory.mkdir(parents=True)
        print("Preparing ungrouped reference " + program.name, flush=True)
        prepared = _prepare_model_mlir(
            stage / "model.mlir", directory, int8_compute=True, respect_captured_quantization_scope=True
        )
        catalogs[program.name] = CAT.ProgramCatalog("gemmini", stage, prepared, prepared, {}, {}, {})
        sources.extend([stage / "model.mlir", prepared])
        sources.extend(p for p in stage.iterdir() if p.is_file() and p.name != "program.pt2")
    evaluator = reference_math(policy, args.output / "host-math")
    cache = CH.ChainCache()

    def progress(program, step, catalog, arguments):
        record = dict(status="evaluating", program=program, step=step)
        (args.output / "progress.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"Reference {program} step {step}", flush=True)

    with host_math_context(evaluator):
        contract = LN.numeric_contract()
        run = CH.run_reference(model, catalogs, arrays, args.output / "arguments", record=progress, cache=cache)
        reference, rows = {}, []
        for program in model.programs:
            elements = CH.result_elements(cache.module(catalogs[program.name].ref_prepared))
            declared = declarations[program.name]["outputs"]
            for step, values in enumerate(run.steps[program.name]):
                if len(values) != len(elements):
                    raise ValueError("Declared output arity differs")
                for index, (value, element) in enumerate(zip(values, elements, strict=True)):
                    stored = LN._store(np.asarray(value), element)
                    specification = declared[index]
                    dtype = {"f32": "float32", "i64": "int64", "i1": "bool"}[specification["dtype"]]
                    if list(stored.shape) != specification["shape"] or stored.dtype != np.dtype(dtype):
                        raise ValueError(f"Protocol output shape/dtype mismatch: {program.name}#{step} output {index}")
                    if not np.isfinite(stored).all():
                        raise ValueError("Nonfinite reference array")
                    key = f"{program.name}_{step:02d}_output_{index}"
                    reference[key] = stored
                    rows.append(dict(key=key, shape=list(stored.shape), dtype=str(stored.dtype), element=element))
        expected = 99 if args.model == "tiny_llama" else 33
        if len(reference) != expected or LN.numeric_contract() != contract:
            raise ValueError("Reference roster or numerical identity changed")
        np.savez_compressed(args.output / "all-outputs.npz", **reference)
        report = dict(
            schema="merlin.tvm.fresh_all_output_reference.v1",
            model=args.model,
            outputs=rows,
            output_arrays=len(rows),
            source_pins=[dict(path=str(p.resolve()), sha256=digest(p)) for p in dict.fromkeys(sources)],
            evaluator="merlin.runtime.linalg_numpy on prepared ungrouped quantized IR",
            numeric_contract=contract,
            host_math=evaluator.identity(),
            reference_sha256=digest(args.output / "all-outputs.npz"),
            historical_admission="Not inherited; fresh local numerical identity",
            tvm_execution="NOT_RUN",
            hardware_execution="NOT_RUN",
            model_quality="NOT_RUN",
        )
        (args.output / "reference.json").write_text(json.dumps(report, indent=2) + "\n")
        (args.output / "progress.json").write_text(json.dumps(dict(status="complete", arrays=len(rows))) + "\n")
        print(json.dumps(dict(model=args.model, arrays=len(rows), status="complete")), flush=True)


if __name__ == "__main__":
    main()
