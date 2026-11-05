"""Bind fused Tiny projections to the exact programs executed during capture."""
import hashlib
import io
import json
from pathlib import Path
import struct

from tiny_full.fusion import ARENA_BASE, prepare_projection, unpack_bf16_tiles, unpack_fp8_tiles
from run_workload_contraction import admit_program

ORIGINAL_MODEL = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
NATIVE_MODEL = "70e2e7ceb81dac86210c8b3cca54024909c57d14"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def original_sources(capture, record):
    """Authenticate recorded execution dependencies against original HEAD47."""
    import kernel_library  # noqa: E402,F401
    capture.pin(kernel_library.module_path("atlas_execution.py"))
    from atlas_execution import Executor
    import npu_model
    if any(key in record for key in ("native_backend", "native_kind", "native_kind_sha256", "native_admission",
                                     "base_model_commit", "qualification_candidate")):
        raise ValueError("native provenance cannot be relabeled as original-model execution")
    before = record["model_sources_before"]
    if not before or before != record["model_sources_after"]:
        raise ValueError("original-model source bookends differ")
    root = Path(npu_model.__file__).absolute().parent.parent
    normalized = {str(capture.pin(path, sha)): sha for path, sha in before.items()}
    if len(normalized) != len(before):
        raise ValueError("original-model source relocation aliases distinct dependencies")
    binding = Executor(root).binding(normalized)
    actual = {str(root / name): sha for name, sha in binding["source_sha256"].items()}
    if normalized != actual:
        raise ValueError("original-model capture source closure differs from authenticated HEAD47")


def _surrogate_projection_kernel(capture, panel, assets, assemble, *, step, layer, role, site, site_index,
                      rows, width, index, native_checker=None):
    """Regenerate staging and instructions, then retain captured whole-arena results."""
    record = panel["launch"]
    native = record["model_commit"] == NATIVE_MODEL
    proof = None
    if native:
        if (native_checker is None or record.get("base_model_commit") != ORIGINAL_MODEL
                or record.get("qualification_candidate") is not False
                or any(record.get(key) is not True for key in
                       ("accelerated", "native_dispatch_enabled", "requires_original_backfill"))):
            raise ValueError("native capture requires explicit admitted provenance and original-model backfill")
        proof = native_checker.verify(capture.root, record)
        record_sha = digest(json.dumps(record, sort_keys=True, separators=(",", ":")).encode())
        if (proof.get("status") != "PASS_NATIVE_RECORD_ADMITTED_REQUIRES_ORIGINAL47_BACKFILL"
                or proof.get("execution_model") != NATIVE_MODEL or proof.get("reference_model") != ORIGINAL_MODEL
                or proof.get("requires_original_backfill") is not True
                or proof.get("original_final_graph_numerics_qualified") is not False
                or proof.get("native_library_loaded_by_checker") is not False
                or proof.get("model_instruction_execution_by_checker") is not False
                or proof.get("record_sha256") != record_sha
                or proof.get("program_sha256") != record["program_sha256"]
                or proof.get("native_kind_sha256") != record["native_kind_sha256"]
                or proof.get("native_admission") != record["native_admission"]
                or not proof.get("source_records")):
            raise ValueError("native capture provenance proof differs from its actual record")
        for source in proof["source_records"]:
            capture.pin(source["path"], source["sha256"])
    elif (record["model_commit"] != ORIGINAL_MODEL or record.get("accelerated") is not False
            or record.get("native_dispatch_enabled") is True or record.get("requires_original_backfill") is True):
        raise ValueError("fused capture requires the original model or an explicit native proof")
    else:
        original_sources(capture, record)
    m, k, n = panel["physical_mkn"]
    column, end = panel["columns"]
    if (end - column != n or panel["logical_mkn"] != [rows, k, n]
            or record["numerics"] != "rtl"
            or record["dram_base"] != ARENA_BASE):
        raise ValueError("fused capture geometry or original-model arithmetic differs")
    activation = capture.raw(record["a_fp8"])
    weight = capture.raw(record["b_fp8"])
    operand = capture.raw(record["operand_tiles"]) if record.get("operand_tiles") else None
    prepared = prepare_projection(m, k, n, panel["rescale"], role, column, width, activation, weight,
                                  operand, record.get("output_scale", 0))
    source, initial = prepared["source"], prepared["initial"]
    words = list(assemble(io.StringIO(source)).assemble())
    admit_program(words)
    program = struct.pack(f"<{len(words)}I", *words)
    requested = dict(load_bytes=prepared["dma_load_bytes"], store_bytes=prepared["dma_store_bytes"])
    if (source.encode() != capture.raw(record["program_source"])
            or program != capture.raw(record["program_binary"])
            or initial != capture.raw(record["arena_initial"])
            or requested != record["requested_dma"]
            or prepared["output_offset"] != record["output_offset"]
            or prepared["output_bytes"] != record["output_bytes"]
            or json.loads(json.dumps(prepared["epilogue"])) != record["epilogue"]
            or prepared["written_spans"] != record["written_spans"]):
        raise ValueError("fused capture differs from regenerated producer program or staging")
    execution = record["execution"]
    if (execution["program_sha256"] != digest(program)
            or {key: execution["observed"].get(key, 0) for key in requested} != requested
            or not record["model_sources_before"] or not record["harness_sources_before"]
            or record["model_sources_before"] != record["model_sources_after"]
            or record["harness_sources_before"] != record["harness_sources_after"]):
        raise ValueError("fused capture execution, DMA, or source bookends differ")
    for binding in (record["model_sources_before"], record["harness_sources_before"]):
        for path, expected_digest in binding.items():
            capture.pin(path, expected_digest)
    final, output = capture.raw(record["arena_expected"]), capture.raw(record["output_tiled"])
    offset, extent = prepared["output_offset"], prepared["output_bytes"]
    if (len(final) != len(initial) or len(output) != extent or final[offset:offset + extent] != output
            or final[:offset] != initial[:offset] or final[offset + extent:] != initial[offset + extent:]):
        raise ValueError("fused capture full arena or output extent differs")
    # Untouched holes are authenticated staging bytes, including leading shard holes.
    cursor = 0
    for span in prepared["written_spans"]:
        begin, size = span["offset"], span["bytes"]
        if begin < cursor or begin + size > extent or output[cursor:begin] != initial[offset + cursor:offset + begin]:
            raise ValueError("fused capture changed an unwritten output hole")
        cursor = begin + size
    if output[cursor:] != initial[offset + cursor:offset + extent]:
        raise ValueError("fused capture changed the output extent tail")
    import numpy as np
    compact = b''.join(output[span["offset"]:span["offset"] + span["bytes"]]
                       for span in prepared["written_spans"])
    if role == 5:
        import torch
        encoded = unpack_fp8_tiles(compact, m, n)
        logical = (torch.from_numpy(encoded.copy()).view(torch.float8_e4m3fn).float()
                   * 2.0**prepared["output_scale"]).to(torch.bfloat16).view(torch.uint16).numpy()
        if capture.raw(record["output_fp8"]) != compact:
            raise ValueError("fused capture compact FP8 output differs from written spans")
    else:
        logical = unpack_bf16_tiles(compact, m, n)
    if not np.array_equal(logical, capture.array(panel["expected_bf16"])):
        raise ValueError("fused capture logical output differs from actual stored output")
    if not np.array_equal(logical[:rows], capture.array(record["output_bf16"])):
        raise ValueError("fused capture graph value differs from actual stored output")
    item = dict(index=index, kind="projection", step=step, layer=layer, logical=role, site=site,
                site_index=site_index, columns=[column, end], width=width, logical_rows=rows,
                physical_mkn=[m, k, n], A_scale=panel["A_scale"], B_scale=panel["B_scale"],
                rescale=panel["rescale"], input_layout="fp8_tiled" if role == 6 else
                "bf16_tiled" if role == 3 else "bf16_row_major",
                output_layout="fp8_tiled_strided" if role == 5 else "bf16_tiled" if role <= 2 or role == 4
                else "bf16_row_major", output_dtype=prepared["output_dtype"],
                source=assets.put("programs", source.encode(), ".S"), original=assets.put("programs", program),
                activation=assets.put("activations", activation), weights=assets.put("weights", weight),
                initial=assets.put("arenas", initial), expected=assets.put("expected", output),
                expected_arena=assets.put("arenas", final), initial_sha256=digest(initial),
                program_sha256=digest(program), expected_sha256=digest(output), output_offset=offset,
                output_extent_bytes=extent, total_dram_bytes=len(initial), dram_base=ARENA_BASE,
                requested_dma=requested, program_policy="traffic-reuse-fused-v1", epilogue=prepared["epilogue"],
                written_spans=prepared["written_spans"], output_scale=prepared["output_scale"],
                output_row_stride=prepared["output_row_stride"], output_column_tile=prepared["output_column_tile"],
                fusion_role=role if role in (3, 4, 5, 6) else 0,
                operand_offset=prepared["extras"][0]["offset"] if prepared["extras"] else 0,
                capture_execution=record["execution"], model_commit=record["model_commit"], numerics="rtl")
    if proof is not None:
        item.update(base_model_commit=ORIGINAL_MODEL, accelerated=True, requires_original_backfill=True,
                    original_final_graph_numerics_qualified=False, native_proof=proof)
    if operand is not None:
        item["operand"] = assets.put("operands", operand)
    if record.get("output_quantization") is not None:
        item["output_quantization"] = record["output_quantization"]
    if role == 3:
        item["input_attention"] = step * 22 + layer
    return item


def projection_kernel(capture, panel, assets, assemble, *, step, layer, role, site, site_index, rows, width, index,
                      native_checker=None, device_residual=False, rope_tables=None):
    """Authenticate the admitted native capture, then export the final o/down and prefill q/k programs.

    Capture uses the admitted residual kernel with an all-zero operand, and the plain q/k producer, solely as
    numerical surrogates. Decode o/down export D2 plain programs. With device_residual, prefill o/down export
    the exact RNE residual epilogue staged with the real operand; with rope_tables, prefill q/k export the
    exact RoPE post-pass. Their expected outputs are the capture's host-formula surrogate results, and every
    exported launch still requires original47.
    """
    item = _surrogate_projection_kernel(capture, panel, assets, assemble, step=step, layer=layer, role=role,
        site=site, site_index=site_index, rows=rows, width=width, index=index, native_checker=native_checker)
    exact = device_residual and step == 0 and role in (3, 6)
    rope = rope_tables if rope_tables is not None and step == 0 and role in (0, 1) else None
    f2 = panel.get('f2')
    kind = 'prefill_rope_rne' if rope is not None else 'prefill_residual_rne' if exact else None
    if (f2 or {}).get('kind') != kind or f2 is not None and (f2.get('surrogate') is not True
            or f2.get('requires_original47_backfill') is not True):
        raise ValueError('F2 device surrogate record differs from the exported prefill program kind')
    if role not in (3, 6) and rope is None:
        return item
    record = panel['launch']
    if role in (3, 6) and any(capture.raw(record['operand_tiles'])):
        raise ValueError('D2 residual surrogate operand must be entirely positive zero')
    m,k,n = panel['physical_mkn'];column=panel['columns'][0]
    operand = capture.raw(f2['operand_tiles']) if exact else None
    prepared = prepare_projection(m,k,n,panel['rescale'],role,column,width,
        capture.raw(record['a_fp8']),capture.raw(record['b_fp8']),operand,residual_adds=exact,
        exact_residual=exact,rope=rope)
    source=prepared['source'];words=list(assemble(io.StringIO(source)).assemble());admit_program(words)
    binary=struct.pack(f'<{len(words)}I',*words);initial=prepared['initial']
    output=capture.raw(f2['output_tiled'] if f2 else record['output_tiled'])
    if len(output)!=prepared['output_bytes']:
        raise ValueError('final producer output extent differs')
    offset=prepared['output_offset'];final=initial[:offset]+output+initial[offset+len(output):]
    policy = ('traffic-reuse-F2-prefill-rope-rne-v1' if rope is not None else
              'traffic-reuse-F2-prefill-residual-rne-v1' if exact else 'traffic-reuse-D2-host-residual-v1')
    item.update(source=assets.put('programs',source.encode(),'.S'),original=assets.put('programs',binary),
        initial=assets.put('arenas',initial),expected=assets.put('expected',output),
        expected_arena=assets.put('arenas',final),initial_sha256=digest(initial),program_sha256=digest(binary),
        expected_sha256=digest(output),total_dram_bytes=len(initial),output_offset=offset,output_extent_bytes=len(output),
        requested_dma=dict(load_bytes=prepared['dma_load_bytes'],store_bytes=prepared['dma_store_bytes']),
        epilogue=prepared['epilogue'],operand_offset=prepared['extras'][0]['offset'] if prepared['extras'] else 0,
        program_policy=policy,surrogate_capture_execution=item.pop('capture_execution'),capture_execution={},
        surrogate_program_sha256=record['program_sha256'],capture_program_matches_final=False,requires_original_backfill=True,original_final_graph_numerics_qualified=False)
    if role in (3, 6):
        item['residual_add'] = exact
    item.pop('operand',None)
    if exact:
        item.update(operand=assets.put('operands',operand),residual_rounding='rne')
    if rope is not None:
        item.update(rope_tables=assets.put('operands',rope),rope_rounding='rne')
    if f2:
        item['device_surrogate'] = dict(f2, expected_from='capture host-formula surrogate')
    item['surrogate_native_proof'] = item.pop('native_proof', None)
    return item
