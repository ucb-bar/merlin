"""Persistent JSONL projection and legacy attention bridge to original RTL numerics."""

import hashlib
import importlib.util
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).absolute().parent
MERLIN_ROOT = HERE.parents[2]
ATLAS_ROOT = Path(os.environ["ATLAS_ROOT"]).absolute()
ROOT = Path(os.environ["TINY_WORK_ROOT"]).absolute()
ATLAS_HELPERS = ATLAS_ROOT / "tools/tiny_full"
MERLIN_HELPERS = MERLIN_ROOT / "examples/atlas/comparisons"
MODEL_ROOT = Path(os.environ["ATLAS_NPU_MODEL_ROOT"]).absolute()
TOOLS = ATLAS_ROOT / "tools"
sys.path[:0] = [str(TOOLS), str(TOOLS / "host_iss")]
import kernel_library  # noqa: E402,F401  (atlas-mlir: kernels live in third_party/atlas-kernels)
import numpy as np
from atlas_attention_program import AttentionConfig, assembly, pack_tiles, stage, unpack_tiles
from tiny_full.fusion import prepare_projection, unpack_bf16_tiles, unpack_fp8_tiles

spec = importlib.util.spec_from_file_location("d1_model", ATLAS_HELPERS / "d1-model.py")
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def harness_sources():
    paths = {model.safe_path(__file__), model.safe_path(ATLAS_HELPERS / "d1-model.py")}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).absolute()
            if path.suffix == ".py" and path.is_relative_to(TOOLS):
                paths.add(model.safe_path(path))
    return {str(path): digest(path.read_bytes()) for path in sorted(paths)}


def run(request):
    root, out = model.safe_path(request["root"]), model.safe_path(request["out"])
    if not out.is_relative_to(root):
        raise ValueError("worker outputs must be beneath a D capture directory")
    out.mkdir(parents=True, exist_ok=False)

    def descriptor(name, data, shape=None, dtype="uint8"):
        path = out / name
        path.write_bytes(data)
        return dict(
            file=str(path.relative_to(root)),
            shape=shape or [len(data)],
            dtype=dtype,
            sha256=digest(data),
            bytes=len(data),
        )

    def read(desc):
        member = Path(desc["file"])
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("worker input member escapes root")
        path = model.safe_path(root / member)
        data = path.read_bytes()
        if len(data) != desc["bytes"] or digest(data) != desc["sha256"]:
            raise ValueError("worker input identity differs")
        return data

    before = model.source_binding()
    if request["command"] == "projection":
        m, k, n = (request[key] for key in ("m", "k", "n"))
        rows = request.get("logical_rows", m)
        if type(rows) is not int or not 0 < rows <= m:
            raise ValueError("logical projection row count differs")
        role = request["role"]
        operand = read(request["operand_tiles"]) if request.get("operand_tiles") is not None else None
        prepared = prepare_projection(
            m,
            k,
            n,
            request["rescale"],
            role,
            request["column"],
            request["width"],
            read(request["A_fp8"]),
            read(request["B_fp8"]),
            operand,
            request.get("output_scale", 0),
        )
        source, initial = prepared["source"], prepared["initial"]
        offset, extent = prepared["output_offset"], prepared["output_bytes"]
        program, program_sha = model.assemble(source)
        sources_before = harness_sources()
        executed = model.execute(
            source,
            initial,
            offset,
            extent,
            max_instructions=request.get("max_instructions", 20_000_000),
            clip_exponent=request.get("output_scale", 0) if role == 5 else None,
        )
        output = executed.output
        compact = b"".join(
            output[span["offset"] : span["offset"] + span["bytes"]] for span in prepared["written_spans"]
        )
        preserved = bytearray(initial)
        for span in prepared["written_spans"]:
            start, size = offset + span["offset"], span["bytes"]
            preserved[start : start + size] = executed.arena[start : start + size]
        if bytes(preserved) != executed.arena:
            raise AssertionError("projection changed bytes outside requested written spans")
        if role == 5:
            raw = unpack_fp8_tiles(compact, m, n)
            tensor = model.torch.from_numpy(raw.copy()).view(model.torch.float8_e4m3fn).float()
            logical = (
                (tensor * 2.0 ** request.get("output_scale", 0))
                .to(model.torch.bfloat16)
                .view(model.torch.uint16)
                .numpy()
            )
            clipping = executed.observed.get("fp8_output", {})
            if clipping.get("values") != m * n:
                raise AssertionError("FP8 clipping instrumentation did not cover every output value")
            clipping.update(
                fp8_nan_encodings=int(((raw & 127) == 127).sum()),
                emitted_nonfinite=int((~np.isfinite(tensor.numpy())).sum()),
            )
        else:
            logical = unpack_bf16_tiles(compact, m, n)
            clipping = None
        if (
            executed.observed.get("load_bytes") != prepared["dma_load_bytes"]
            or executed.observed.get("store_bytes") != prepared["dma_store_bytes"]
        ):
            raise AssertionError("observed projection DMA differs from declared bytes")
        record = dict(
            kind="projection",
            m=m,
            k=k,
            n=n,
            logical_rows=rows,
            role=role,
            column=request["column"],
            width=request["width"],
            rescale=request["rescale"],
            a_scale=request.get("a_scale"),
            b_scale=request.get("b_scale"),
            output_scale=request.get("output_scale", 0),
            dram_base=0x90000000,
            output_offset=offset,
            output_bytes=extent,
            arena_bytes=len(initial),
            requested_dma=dict(load_bytes=prepared["dma_load_bytes"], store_bytes=prepared["dma_store_bytes"]),
            dma_store_bytes=prepared["dma_store_bytes"],
            epilogue=prepared["epilogue"],
            extras=prepared["extras"],
            written_spans=prepared["written_spans"],
            output_dtype=prepared["output_dtype"],
            output_row_stride=prepared["output_row_stride"],
            output_column_tile=prepared["output_column_tile"],
            a_fp8=request["A_fp8"],
            b_fp8=request["B_fp8"],
            operand_tiles=request.get("operand_tiles"),
            program_sha256=program_sha,
            execution=executed.metadata(),
            output_quantization=clipping,
        )
        record["output_bf16"] = descriptor(
            "output-bf16.bin", logical[:rows].astype("<u2").tobytes(), [rows, n], "uint16"
        )
        if role == 5:
            record["output_fp8"] = descriptor("output-fp8.bin", compact)
            record["preclip_bf16_tiles"] = descriptor(
                "preclip-bf16-tiles.bin", b"".join(model.CLIP_TILES), [m * n], "uint16"
            )
        for key in ("step", "site", "panel", "layer", "name"):
            if key in request:
                record[key] = request[key]
    elif request["command"] == "attention":
        config = AttentionConfig(**request["config"])
        q, k, v = [
            np.frombuffer(read(request[key + "_bf16"]), dtype="<u2").reshape(request[key + "_bf16"]["shape"]).copy()
            for key in ("q", "k", "v")
        ]
        initial, offsets = stage(config, q, k, v)
        source = assembly(config, offsets)
        offset, extent = offsets["output"], offsets["output_bytes"]
        program, program_sha = model.assemble(source)
        sources_before = harness_sources()
        executed = model.execute(source, initial, offset, extent)
        output = executed.output
        logical = unpack_tiles(output, config.query_heads, config.query_rows)
        record = dict(
            kind="attention",
            config=asdict(config),
            offsets=offsets,
            dram_base=0x90000000,
            output_offset=offset,
            output_bytes=extent,
            arena_bytes=len(initial),
            output_layout="bf16_tiled",
            program_sha256=program_sha,
            execution=executed.metadata(),
            requested_dma=dict(
                load_bytes=executed.observed["load_bytes"], store_bytes=executed.observed["store_bytes"]
            ),
        )
        for key, value in zip(("q", "k", "v"), (q, k, v)):
            record[key + "_bf16"] = request[key + "_bf16"]
            record[key + "_tiled"] = descriptor(key + "-tiled.bin", pack_tiles(value))
        record["output_bf16"] = descriptor(
            "output-bf16.bin", logical.astype("<u2").tobytes(), list(logical.shape), "uint16"
        )
    else:
        raise ValueError("unsupported worker command")
    after = model.source_binding()
    if before != after:
        raise AssertionError("model source bookends differ")
    sources_after = harness_sources()
    if sources_before != sources_after:
        raise AssertionError("harness source bookends differ")
    record.update(
        schema="atlas.d1.original_capture.v1",
        status="PASS",
        exact=True,
        full_arena_exact=True,
        model_commit=model.MODEL_PIN,
        numerics="rtl",
        accelerated=False,
        native_dispatch_enabled=False,
        model_sources_before=before,
        model_sources_after=after,
        harness_sources_before=sources_before,
        harness_sources_after=sources_after,
        implementation_sources=[
            dict(path=path, sha256=sha, bytes=Path(path).stat().st_size) for path, sha in sources_before.items()
        ],
    )
    for key, name, raw in (
        ("program_source", "program.S", source.encode()),
        ("program_binary", "program.bin", b"".join(word.to_bytes(4, "little") for word in program.assemble())),
        ("arena_initial", "arena-initial.bin", initial),
        ("arena_expected", "arena-expected.bin", executed.arena),
        ("output_tiled", "output-tiled.bin", output),
    ):
        record[key] = descriptor(name, raw)
    (out / "record.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("command") == "close":
                print(json.dumps(dict(status="closed")), flush=True)
                break
            record = run(request)
            print(json.dumps(dict(status="ok", record=record)), flush=True)
        except BaseException as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(status="error", error=str(error), error_type=type(error).__name__)), flush=True)
