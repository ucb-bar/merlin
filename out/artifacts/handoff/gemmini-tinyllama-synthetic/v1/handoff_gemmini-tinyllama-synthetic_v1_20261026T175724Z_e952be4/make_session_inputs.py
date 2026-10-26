"""Adapt diagnostic references to the 33-call session builder; not selected EL4."""

import argparse
import json
from pathlib import Path

import numpy as np
from prepare_firesim_chunks import identity, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "generated", "prefill-package", "decode-package", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    tokens = np.asarray(json.loads((args.inputs / "inputs.json").read_text())["measurement"], dtype=np.int64)
    calls = [dict(stage="prefill", inputs=[dict(fixture=0)])]
    arrays = dict(input_0=tokens[:, :32])
    with np.load(args.inputs / "session-reference.npz", allow_pickle=False) as reference:
        for stage in range(33):
            if stage:
                arrays[f"input_{stage}"] = tokens[:, 31 + stage : 32 + stage]
                calls.append(
                    dict(
                        stage="decode",
                        inputs=[
                            dict(fixture=stage),
                            dict(output=[stage - 1, 1]),
                            dict(output=[stage - 1, 2]),
                        ],
                    )
                )
            for output in range(3):
                arrays[f"output_{3 * stage + output}"] = reference[f"stage_{stage:02d}_output_{output}"]
        np.savez_compressed(args.output_dir / "fixture.npz", **arrays)
    write_json(args.output_dir / "route.json", dict(version=1, calls=calls))
    write_json(
        args.output_dir / "stages.json",
        dict(
            version=1,
            stages=dict(
                prefill=str(args.prefill_package.resolve()),
                decode=str(args.decode_package.resolve()),
            ),
        ),
    )
    sources = [
        identity(args.inputs / name)
        for name in (
            "inputs.json",
            "calibration.json",
            "checkpoint-receipt.json",
            "session-reference.json",
            "session-reference.npz",
        )
    ]
    sources += [identity(args.generated / (phase + ".pt2")) for phase in ("prefill", "decode")]
    write_json(
        args.output_dir / "fixture-provenance.json",
        dict(
            fixture_sha256=identity(args.output_dir / "fixture.npz")["sha256"],
            route=identity(args.output_dir / "route.json"),
            reference_sources=sources,
            reference_contract=("Frozen diagnostic Torch INT32/FP32 session, independent of TVM; all99 arrays"),
            selected_el4_conformance="FAIL; see EL4_INPUT_AUDIT.md",
            model_quality="NOT_RUN",
            hardware_execution="NOT_RUN",
            state_routing="Actual prior candidate cache/position outputs; only tokens come from input fixtures",
        ),
    )
    print(json.dumps(dict(calls=33, outputs=99, input_fixtures=33, candidate_state_carried=True)))


if __name__ == "__main__":
    main()
