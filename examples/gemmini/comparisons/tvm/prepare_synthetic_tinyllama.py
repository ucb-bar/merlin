"""Prepare diagnostic synthetic TinyLlama numerics and real-shape chunks.

This source does not conform to the selected EL4 recipe or cache ABI. See
EL4_INPUT_AUDIT.md; do not substitute its outputs into selected-profile campaigns.
No FPGA execution, schedules, new compiler passes or quality claims. The normal
mode calibrates separately, freezes scales, computes a full 32+32 source session
and exports its prefill/decode. Replay mode rebuilds cheap chunk inputs from the
published fixture bundle and the pinned checkpoint, without recalibration.
"""

import argparse
import gc
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from prepare_firesim_chunks import identity, write_json
from safetensors import safe_open
from synthetic_tinyllama import Chunk, Decode, Layer, Prefill, SourceModel, calibrate, rms

CHECKPOINT_SHA256 = "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933"


def save_arrays(path, arrays):
    np.savez_compressed(path, **arrays)


def array_identity(value):
    array = value.detach().contiguous().numpy()
    return dict(shape=list(array.shape), dtype=array.dtype.name, sha256=hashlib.sha256(array.tobytes()).hexdigest())


def outputs(value):
    return tuple(value) if isinstance(value, (tuple, list)) else (value,)


def freeze_chunk(root, name, module, arguments, sources, frozen_fixture=None):
    import tvm
    from program_io import import_program

    folder = root / "chunks" / name
    folder.mkdir(parents=True)
    if frozen_fixture is not None:
        with np.load(frozen_fixture, allow_pickle=False) as fixture:
            arguments = tuple(torch.from_numpy(fixture[f"input_{i}"].copy()) for i in range(len(arguments)))
            expected = tuple(
                torch.from_numpy(fixture[key].copy()) for key in sorted(fixture.files) if key.startswith("expected")
            )
        sources = [*sources, identity(frozen_fixture)]
        original_proof = frozen_fixture.with_name("fixture-provenance.json")
        if original_proof.is_file():
            sources.append(identity(original_proof))
    else:
        expected = outputs(module.run(arguments, oracle=True))
    actual = outputs(module(*arguments))
    errors = []
    for observed, reference in zip(actual, expected, strict=True):
        if frozen_fixture is not None and observed.dtype.is_floating_point:
            torch.testing.assert_close(observed, reference, rtol=1e-5, atol=2e-6)
        else:
            torch.testing.assert_close(observed, reference, rtol=0, atol=0)
        errors.append(float((observed.double() - reference.double()).abs().max()))
    program = torch.export.export(module, arguments, strict=True)
    torch.export.save(program, folder / "program.pt2")
    semantic = import_program(folder / "program.pt2")
    (folder / "program.json").write_text(tvm.ir.save_json(semantic))
    arrays = {f"input_{i}": x.contiguous().numpy() for i, x in enumerate(arguments)}
    arrays.update({"expected" if len(expected) == 1 else f"expected_{i}": x.numpy() for i, x in enumerate(expected)})
    if frozen_fixture is not None:
        shutil.copy2(frozen_fixture, folder / "fixture.npz")
    else:
        save_arrays(folder / "fixture.npz", arrays)
    proof = dict(
        program=identity(folder / "program.json"),
        fixture_sha256=identity(folder / "fixture.npz")["sha256"],
        reference_sources=sources,
        reference_contract="NumPy INT64 contractions and floor/parity RNE; source Torch FP32 non-matmul work",
        comparison=dict(rtol=1e-5, atol=2e-6),
        numerical_scope="Nonconforming synthetic diagnostic; not selected EL4 numerics/ABI",
        model_quality="NOT_RUN",
        current_run_policy="Static operands/group16 arithmetic; synthetic calibration, not team release",
        reference_mode="Frozen independent fixture copied unchanged" if frozen_fixture else "Fresh independent oracle",
    )
    write_json(folder / "fixture-provenance.json", proof)
    row = dict(
        name=name,
        kind=module.kind,
        phase="prefill" if module.prefill else "decode",
        source_layer=0,
        input_shapes=[list(a.shape) for a in arguments],
        output_shapes=[list(x.shape) for x in expected],
        oracle_max_abs_errors=errors,
        program=proof["program"],
        fixture=identity(folder / "fixture.npz"),
        reference=identity(folder / "fixture-provenance.json"),
        firesim="NOT_RUN",
    )
    write_json(folder / "chunk.json", row)
    print("prepared " + name, flush=True)
    return row


def arguments_for(layer, kind, trace):
    if kind == "qkv":
        return trace["hidden"], trace["positions"]
    if kind == "o":
        return (trace["attended"],)
    if kind == "mlp":
        return (trace["mlp_hidden"],)
    if kind == "down":
        normalized = rms(trace["mlp_hidden"], layer.post_norm)
        gated = torch.nn.functional.silu(layer.gate(normalized)) * layer.up(normalized)
        return (gated,)
    if kind == "qk":
        return trace["query"][0], trace["keys"][0]
    if kind == "pv":
        return trace["probabilities"][0], trace["values"][0]
    if kind == "attention":
        return trace["hidden"], trace["cache"], trace["position"]
    raise ValueError("Unknown chunk")


def emit_chunks(root, archive, config, calibration, bundle):
    layer = Layer(archive, 0, calibration["layers"][0], config).eval()
    sources = [
        identity(path)
        for path in (
            root / "calibration.json",
            root / "inputs.json",
            root / "checkpoint-receipt.json",
            Path(__file__),
            Path(__file__).with_name("synthetic_tinyllama.py"),
        )
    ]
    rows = []
    for phase in ("prefill", "decode"):
        with np.load(bundle / f"{phase}-boundaries.npz", allow_pickle=False) as data:
            trace = {key: torch.from_numpy(data[key].copy()) for key in data.files}
        for kind in ("qkv", "o", "mlp", "down", "qk", "pv", "attention"):
            name = f"tinyllama-{phase}-{kind}"
            frozen = bundle / "chunks" / name / "fixture.npz"
            rows.append(
                freeze_chunk(
                    root,
                    name,
                    Chunk(layer, kind, phase == "prefill"),
                    arguments_for(layer, kind, trace),
                    sources,
                    frozen_fixture=frozen if frozen.is_file() else None,
                )
            )
    write_json(
        root / "chunks.json",
        dict(chunks=rows, scope="14 meaningful full-N/K chunks, real layer0 computation; no schedules chosen"),
    )


def reference_session(root, model, tokens):
    references, checks, snapshots = {}, [], {}
    model.prefill = True
    trace = {}
    values = model.run(tokens[:, :32], traces=trace)
    save_arrays(root / "prefill-boundaries.npz", {name: value.numpy() for name, value in trace.items()})
    snapshots["prefill_tokens"] = tokens[:, :32].numpy()
    for stage in range(33):
        if stage:
            previous = values[1].clone()
            incoming_position = values[2].clone()
            model.prefill = False
            trace = {} if stage in (1, 17, 32) else None
            values = model.run(tokens[:, 31 + stage : 32 + stage], values[1], values[2], traces=trace)
            used = int(incoming_position[0])
            assert torch.equal(previous[..., :used, :], values[1][..., :used, :])
            if trace is not None:
                filename = "decode-boundaries.npz" if stage == 1 else f"decode-position-{used}-boundaries.npz"
                save_arrays(root / filename, {name: value.numpy() for name, value in trace.items()})
            if stage == 1:
                snapshots.update(
                    decode_tokens=tokens[:, 32:33].numpy(), cache=previous.numpy(), position=incoming_position.numpy()
                )
        assert int(values[2][0]) == 32 + stage
        assert not torch.count_nonzero(values[1][..., 32 + stage :, :])
        for output, value in enumerate(values):
            assert torch.isfinite(value).all()
            references[f"stage_{stage:02d}_output_{output}"] = value.numpy().copy()
        checks.append(dict(stage=stage, position=int(values[2][0]), outputs=[array_identity(x) for x in values]))
        print(f"source session stage {stage}/32", flush=True)
    save_arrays(root / "session-reference.npz", references)
    save_arrays(root / "session-inputs.npz", snapshots)
    write_json(
        root / "session-reference.json",
        dict(
            calls=33,
            output_arrays=len(references),
            prefill_tokens=32,
            decode_tokens=32,
            cache_capacity=64,
            cache_dtype="int8",
            cache_shape=list(values[1].shape),
            logits_shape=list(values[0].shape),
            checks=checks,
            reference=(
                "Independent of TVM: source Torch INT32 GEMMs and FP32 host work; "
                "chunk integer work also checked with NumPy INT64"
            ),
            state="Source cache carried through all calls; inactive tails zero; previous prefixes unchanged",
            model_quality="NOT_RUN",
            target_execution="NOT_RUN",
        ),
    )
    return snapshots


def full_exports(root, model, snapshots):
    model.prefill = True
    with torch.no_grad():
        exported = torch.export.export(Prefill(model), (torch.from_numpy(snapshots["prefill_tokens"]),), strict=True)
    torch.export.save(exported, root / "prefill.pt2")
    del exported
    gc.collect()
    print("saved full 22-layer prefill export", flush=True)
    model.prefill = False
    args = tuple(torch.from_numpy(snapshots[k]) for k in ("decode_tokens", "cache", "position"))
    exported = torch.export.export(Decode(model), args, strict=True)
    torch.export.save(exported, root / "decode.pt2")
    del exported
    gc.collect()
    print("saved full 22-layer decode export", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--replay-bundle", type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--calibration-sequences", type=int, default=3)
    parser.add_argument("--skip-full-export", action="store_true")
    args = parser.parse_args()
    if args.threads < 1 or args.calibration_sequences < 1:
        parser.error("Use positive thread and calibration counts")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    checkpoint = identity(args.checkpoint)
    if checkpoint["sha256"] != CHECKPOINT_SHA256:
        parser.error("Checkpoint differs from pinned TinyLlama 1.1B Chat weights")
    config = json.loads(args.config.read_text())
    expected = dict(
        hidden_size=2048,
        intermediate_size=5632,
        num_attention_heads=32,
        num_key_value_heads=4,
        num_hidden_layers=22,
        vocab_size=32000,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="silu",
        tie_word_embeddings=False,
        max_position_embeddings=2048,
        bos_token_id=1,
    )
    if any(config[k] != v for k, v in expected.items()) or config.get("attention_bias"):
        parser.error("Checkpoint geometry/bias policy differs from pinned source")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "protocol-conformance.json",
        dict(
            status="FAIL_SELECTED_EL4_PROFILE",
            scope="Diagnostic only; exclude from selected-profile DSE and E2E claims",
            audit="examples/gemmini/comparisons/tvm/EL4_INPUT_AUDIT.md",
            differences=["calibration", "recipe transforms/planes", "cache ABI", "host math", "reference producer"],
        ),
    )
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as archive:
        if args.replay_bundle:
            bundle = args.replay_bundle.resolve()
            for name in ("calibration.json", "inputs.json", "checkpoint-receipt.json", "frozen-buffers.json"):
                shutil.copy2(bundle / name, root / name)
            for path in bundle.glob("*.npz"):
                shutil.copy2(path, root / path.name)
            shutil.copy2(bundle / "session-reference.json", root / "session-reference.json")
            calibration = json.loads((root / "calibration.json").read_text())
            if not args.skip_full_export:
                model = SourceModel(archive, calibration, config, True).eval()
                with np.load(bundle / "session-inputs.npz", allow_pickle=False) as inputs:
                    snapshots = {name: inputs[name].copy() for name in inputs}
                full_exports(root, model, snapshots)
                del model
                gc.collect()
            emit_chunks(root, archive, config, calibration, bundle)
            return
        generator = np.random.default_rng(args.seed)
        calibration_tokens = generator.integers(
            3, config["vocab_size"], (args.calibration_sequences, 64), dtype=np.int64
        )
        measurement_tokens = np.random.default_rng(args.seed + 1).integers(
            3, config["vocab_size"], (1, 64), dtype=np.int64
        )
        calibration_tokens[:, 0] = measurement_tokens[:, 0] = config["bos_token_id"]
        inputs = dict(
            seed=args.seed,
            calibration=calibration_tokens.tolist(),
            measurement=measurement_tokens.tolist(),
            scope="Local synthetic tokens; separate calibration/measurement rosters; not team v00 or WikiText",
        )
        write_json(root / "inputs.json", inputs)
        write_json(
            root / "checkpoint-receipt.json",
            dict(
                checkpoint=checkpoint,
                config=identity(args.config),
                source="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                revision="fe8a4ea1ffedaf415f4da2f062534de366a451e6",
            ),
        )
        calibration = calibrate(archive, config, torch.from_numpy(calibration_tokens))
        calibration.update(
            weight_group=16,
            weight_range=[-127, 127],
            activation_range=[-128, 127],
            zero_point=0,
            rounding="nearest ties even",
            accumulation="int32",
            bias="none in this checkpoint",
            calibration_input=identity(root / "inputs.json"),
            scale_policy="Frozen per tensor after independent synthetic calibration",
            attention="QK/PV: INT8 x INT8 to INT32; fixed capacity64 including masked work; 32 Q/4 KV heads",
            rope="Source FP32 table frozen for positions0..63; explicit numerical source, no compiler transform",
            host_math="FP32 RMSNorm, RoPE, softmax, SiLU; all eligible matmuls integer",
        )
        write_json(root / "calibration.json", calibration)
        model = SourceModel(archive, calibration, config, True).eval()
        ledger = [{"name": name, **array_identity(value)} for name, value in model.named_buffers()]
        write_json(root / "frozen-buffers.json", dict(buffers=ledger, model_quality="NOT_RUN"))
        snapshots = reference_session(root, model, torch.from_numpy(measurement_tokens))
        if not args.skip_full_export:
            full_exports(root, model, snapshots)
        del model
        gc.collect()
        emit_chunks(root, archive, config, calibration, root)


if __name__ == "__main__":
    main()
