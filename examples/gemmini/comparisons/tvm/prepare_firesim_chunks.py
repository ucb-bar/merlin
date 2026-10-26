"""Prepare cheap, full-reduction TinyLlama probes or slice an actual export.

Checkpoint probes use trained weights and v00-derived first-layer boundaries.
Their fresh dynamic W8A8 recipe is diagnostic, not the calibrated EL4 release.
The slice command retains the supplied export's exact expressions and shapes.
No scheduling decisions or performance labels are generated here.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def checkpoint_boundaries(checkpoint, tokens):
    """A single FP32 source layer, only to obtain realistic probe live-ins.

    This is not a teacher-forced full-model reference or an EL4 quantizer.
    Attention uses all 33 selected prefix tokens, all heads and causal masking.
    """
    import torch
    from safetensors import safe_open

    torch.set_num_threads(1)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:

        def weight(name):
            return archive.get_tensor("model.layers.0." + name).float()

        hidden = archive.get_tensor("model.embed_tokens.weight")[tokens[:33]].float()

        def norm(x, name):
            return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + 1e-5) * weight(name)

        normalized = norm(hidden, "input_layernorm.weight")
        q = (normalized @ weight("self_attn.q_proj.weight").T).reshape(33, 32, 64).transpose(0, 1)
        k = (normalized @ weight("self_attn.k_proj.weight").T).reshape(33, 4, 64).transpose(0, 1)
        v = (normalized @ weight("self_attn.v_proj.weight").T).reshape(33, 4, 64).transpose(0, 1)
        frequencies = torch.arange(33).float()[:, None] / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
        angles = torch.cat((frequencies, frequencies), -1)

        def rotary(x):
            return x * angles.cos() + torch.cat((-x[..., 32:], x[..., :32]), -1) * angles.sin()

        q, k = rotary(q), rotary(k).repeat_interleave(8, 0)
        scores = (q @ k.transpose(-1, -2)) * 0.125
        scores.masked_fill_(torch.triu(torch.ones(33, 33, dtype=torch.bool), diagonal=1), -float("inf"))
        attention = (scores.softmax(-1) @ v.repeat_interleave(8, 0)).transpose(0, 1).reshape(33, 2048)
        post = norm(hidden + attention @ weight("self_attn.o_proj.weight").T, "post_attention_layernorm.weight")
        gated = torch.nn.functional.silu(post @ weight("mlp.gate_proj.weight").T)
        gated *= post @ weight("mlp.up_proj.weight").T
        projections = {
            name: weight(name).numpy()
            for name in ("self_attn.k_proj.weight", "self_attn.v_proj.weight", "mlp.down_proj.weight")
        }
    return normalized.numpy(), gated.numpy(), projections


def semantic_projection_chunk(values, weights):
    """Build mathematical Relax quantize -> GEMM -> scale; independent NumPy oracle."""
    import tvm
    from tinyllama_integer import independent_quantize_rows
    from tvm import relax

    codes, activation_scale = independent_quantize_rows(values)
    bb = relax.BlockBuilder()
    x = relax.Var("boundary", relax.TensorStructInfo(values.shape, "float32"))
    outputs, expected = [], []
    with bb.function("main", [x]):
        with bb.dataflow():
            maximum = bb.emit(relax.op.max(bb.emit(relax.op.abs(x)), axis=-1, keepdims=True))
            scale = bb.emit(
                relax.op.where(
                    bb.emit(relax.op.equal(maximum, relax.const(0.0, "float32"))),
                    relax.const(1.0, "float32"),
                    bb.emit(
                        relax.op.maximum(
                            bb.emit(relax.op.divide(maximum, relax.const(127.0, "float32"))),
                            relax.const(np.finfo(np.float32).tiny, "float32"),
                        )
                    ),
                )
            )
            packed = bb.emit(
                relax.op.astype(
                    bb.emit(relax.op.clip(bb.emit(relax.op.round(bb.emit(relax.op.divide(x, scale)))), -127, 127)),
                    "int8",
                )
            )
            for source_weight in weights:
                packed_weight, weight_scale = independent_quantize_rows(source_weight)
                product = bb.emit(relax.op.matmul(packed, relax.const(packed_weight.T.copy()), out_dtype="int32"))
                result = bb.emit(
                    relax.op.multiply(
                        bb.emit(relax.op.multiply(bb.emit(relax.op.astype(product, "float32")), scale)),
                        relax.const(weight_scale[:, 0]),
                    )
                )
                outputs.append(result)
                # INT64 dot is independent of TVM's lowering/target accumulator.
                accumulator = codes.astype(np.int64) @ packed_weight.T.astype(np.int64)
                if np.abs(accumulator).max() > np.iinfo(np.int32).max:
                    raise ValueError("Probe accumulator overflows int32")
                expected.append((accumulator.astype(np.float32) * activation_scale) * weight_scale[:, 0])
            out = bb.emit_output(relax.Tuple(outputs))
        bb.emit_func_output(out)
    module = bb.get()
    assert relax.analysis.well_formed(module)
    return module, expected


def checkpoint_probes(args):
    import tinyllama_integer
    import tvm

    selected = json.loads(args.tokens.read_text())["v00"]["tokens"]
    if len(selected) < 33:
        raise ValueError("Require the selected 32-token prefill and first decode token")
    normalized, gated, weights = checkpoint_boundaries(args.checkpoint, selected)
    sources = [
        identity(args.checkpoint),
        identity(args.tokens),
        identity(__file__),
        identity(tinyllama_integer.__file__),
    ]
    cases = [
        (
            "decode-kv",
            normalized[32:33],
            [weights["self_attn.k_proj.weight"], weights["self_attn.v_proj.weight"]],
            dict(parent_mnk=[1, 256, 2048], projections=2, output_panel=None),
        ),
        (
            "prefill-kv",
            normalized[:32],
            [weights["self_attn.k_proj.weight"], weights["self_attn.v_proj.weight"]],
            dict(parent_mnk=[32, 256, 2048], projections=2, output_panel=None),
        ),
        (
            "decode-down-panel",
            gated[32:33],
            [weights["mlp.down_proj.weight"][:128]],
            dict(parent_mnk=[1, 2048, 5632], projections=1, output_panel=[0, 128]),
        ),
    ]
    rows = []
    for name, values, selected_weights, description in cases:
        folder = args.output_dir / name
        folder.mkdir()
        module, expected = semantic_projection_chunk(values, selected_weights)
        (folder / "program.json").write_text(tvm.ir.save_json(module))
        arrays = {"input_0": values}
        arrays.update(
            {"expected" if len(expected) == 1 else f"expected_{i}": value for i, value in enumerate(expected)}
        )
        np.savez(folder / "fixture.npz", **arrays)
        proof = dict(
            fixture_sha256=identity(folder / "fixture.npz")["sha256"],
            program=identity(folder / "program.json"),
            reference_sources=sources,
            reference_contract="Diagnostic W8A8: independent floor/parity RNE, INT64 NumPy dot, FP32 scale products; every output",
            comparison=dict(rtol=1e-5, atol=2e-6),
            numerical_release="fresh checkpoint probe; not calibrated EL4",
        )
        write_json(folder / "fixture-provenance.json", proof)
        m, k = values.shape
        ns = [w.shape[0] for w in selected_weights]
        row = dict(
            name=name,
            **description,
            executed_mnk=[[m, n, k] for n in ns],
            multiply_accumulates=sum(m * n * k for n in ns),
            int8_weight_bytes=sum(w.size for w in selected_weights),
            full_reduction=True,
            input_shape=list(values.shape),
            output_shapes=[list(a.shape) for a in expected],
            boundary="layer 0 FP32 source computation on v00 tokens 0..32, then fresh dynamic W8A8 projections",
            reference_scope="trained-weight diagnostic, not exact EL4 export or model quality qualification",
            promotion=(
                "exact structural match and fresh FireSim measurement required"
                if description["output_panel"] is None
                else "panel records cannot match parent N=2048; confirm proposals on full shape before deployment"
            ),
        )
        write_json(folder / "chunk.json", row)
        rows.append(row)
    write_json(
        args.output_dir / "chunks.json",
        dict(
            schema="gemmini_firesim_small_chunks_v1", chunks=rows, target_cycle_measurements="NOT_RUN", sources=sources
        ),
    )


def exact_slice(args):
    import tvm
    from program_io import import_program
    from workload_chunks import inventory, slice_program

    module = import_program(args.program)
    if args.command == "inventory":
        write_json(args.output_dir / "inventory.json", dict(program=identity(args.program), bindings=inventory(module)))
        return
    chunk, boundary = slice_program(module, args.start, args.stop)
    proof = json.loads(args.fixture_provenance.read_text())
    if (
        proof.get("fixture_sha256") != identity(args.fixture)["sha256"]
        or not proof.get("reference_sources")
        or not proof.get("reference_contract")
    ):
        raise ValueError("Require independently identified actual boundary inputs AND all live-out references")
    for source in proof["reference_sources"]:
        if identity(source["path"])["sha256"] != source["sha256"]:
            raise ValueError("Boundary reference source changed")
    with np.load(args.fixture, allow_pickle=False) as arrays:
        function = chunk["main"]
        outputs = function.ret_struct_info.fields
        for key, info in [(f"input_{i}", p.struct_info) for i, p in enumerate(function.params)] + [
            ("expected" if len(outputs) == 1 else f"expected_{i}", info) for i, info in enumerate(outputs)
        ]:
            value = arrays[key]
            if list(value.shape) != [int(x) for x in info.shape] or value.dtype.name != info.dtype:
                raise ValueError("Boundary fixture differs from exact slice: " + key)
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                raise ValueError("Nonfinite boundary fixture")
    (args.output_dir / "program.json").write_text(tvm.ir.save_json(chunk))
    (args.output_dir / "fixture.npz").write_bytes(args.fixture.read_bytes())
    proof = dict(proof, program=identity(args.output_dir / "program.json"), parent_program=identity(args.program))
    proof["reference_sources"] = [
        *proof["reference_sources"],
        identity(args.program),
        identity(args.fixture_provenance),
    ]
    write_json(args.output_dir / "fixture-provenance.json", proof)
    write_json(
        args.output_dir / "chunk.json",
        dict(
            **boundary,
            parent_program=identity(args.program),
            numerical_scope="exact supplied semantic export slice; independent boundary fixture required",
            target_cycle_measurements="NOT_RUN",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("checkpoint-probes")
    probe.add_argument("--checkpoint", required=True, type=Path)
    probe.add_argument("--tokens", required=True, type=Path)
    for name in ("inventory", "slice"):
        command = commands.add_parser(name)
        command.add_argument("--program", required=True, type=Path)
        if name == "slice":
            command.add_argument("--start", required=True, type=int)
            command.add_argument("--stop", required=True, type=int)
            command.add_argument("--fixture", required=True, type=Path)
            command.add_argument("--fixture-provenance", required=True, type=Path)
    for command in (probe, *[commands.choices[name] for name in ("inventory", "slice")]):
        command.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.command == "checkpoint-probes":
        checkpoint_probes(args)
    else:
        exact_slice(args)


if __name__ == "__main__":
    main()
