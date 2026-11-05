"""Export captured Atlas attention with explicit interleaved launch ordering."""
from __future__ import annotations

from dataclasses import asdict
from contextlib import ExitStack
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools/host_iss")]
import kernel_library  # noqa: E402,F401
from atlas_attention_program import AttentionConfig, assembly as attention_assembly, pack_tiles, stage, unpack_tiles
from atlas_tooling import sha256
from atlas_path_guard import lexical_absolute_path
from atlas_program import assembly as projection_assembly, dma_load_bytes
from chipyard.arena import DEFAULT_ARENA, relocate_assembly, window
from export_capture import Assets, ExactEncoder, attention_execution_mode, checked, pack_output, prepare_encoded
from compile_workload_inputs import pack_weights
from run_workload_contraction import admit_program

MODEL_PIN = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
ARENA_BASE = 0x90000000
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def projection_activation(capture, call, logical, role, encoder, *, require_physical=False):
    """Match the host's physical attention scale scan and logical-row packing."""
    rows, k = logical.shape
    m = 128 if rows == 128 else 32
    physical = logical if m == rows else np.concatenate((logical, np.zeros((m-rows, k), np.uint16)))
    if role == 3 and call.get("A_physical_bf16"):
        physical = capture.array(call["A_physical_bf16"])
        if physical.shape != (m, k) or not np.array_equal(physical[:rows], logical):
            raise ValueError("physical attention input differs from the logical projection input")
    elif role == 3 and rows != m and require_physical:
        raise ValueError("C decode capture requires the complete physical attention input")
    encoded, scale = encoder.encode(physical)
    if role == 3 and rows != m:
        # tiny_attention_pack validates all BF16 words but explicitly zeros
        # inactive FP8 rows after the scale scan over the physical extent.
        encoded = encoded.copy()
        encoded[rows:] = 0
    return physical, encoded, scale


def model_setup(root):
    root = checked(root)
    pin = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if pin != MODEL_PIN:
        raise ValueError("attention export requires the selected 47b8599 npu_model")
    sys.path.insert(0, str(root))
    import torch
    from npu_model.configs import isa_definition
    from npu_model.configs.numerics import RtlNumerics
    from npu_model.util.converter import input_to_program
    torch.set_num_threads(1)

    def binding():
        sources = {}
        for name, module in list(sys.modules.items()):
            if name == "npu_model" or name.startswith("npu_model."):
                path = getattr(module, "__file__", None)
                if path:
                    path = checked(path)
                    if not path.is_relative_to(root):
                        raise ValueError("model imported outside selected clone")
                    sources[str(path.relative_to(root))] = sha256(path)
        return {"revision": pin, "numerics": "rtl", "imported_source_sha256": sources,
                "curve_sha256": sha256(root / "npu_model/configs/memory_curves/ee290sim_vcs_probe.json")}
    return torch, RtlNumerics.to_fp8, input_to_program, binding


class Capture:
    def __init__(self, path, expected_digest, source_archive=None, relocations=()):
        self.path = checked(path)
        self.root = self.path.parent
        self.pins = {}
        self.archived_sources = {}
        self.source_substitutions = []
        self.relocations = []
        for value in relocations:
            old, separator, new = value.partition("=")
            if not separator:
                raise ValueError("capture source map requires OLD=NEW")
            self.relocations.append((lexical_absolute_path(old), lexical_absolute_path(new)))
        if source_archive is not None:
            archive = checked(source_archive)
            self.pins[str(archive)] = sha256(archive)
            data = json.loads(archive.read_text())
            if data.get("schema") != "atlas.capture_source_archive.v1":
                raise ValueError("unsupported capture source archive")
            for item in data["sources"]:
                name = Path(item["file"])
                if name.is_absolute() or ".." in name.parts:
                    raise ValueError("archived source must be relative")
                self.archived_sources[str(lexical_absolute_path(item["original_path"])), item["sha256"]] = checked(archive.parent / name)
        self.pin(self.path, expected_digest)
        self.data = json.loads(self.path.read_text())
        for item in self.data["source_records"]:
            self.pin(item["path"], item["sha256"])

    def pin(self, path, expected=None):
        original = lexical_absolute_path(path)
        matches = [(old, new) for old, new in self.relocations if original == old or old in original.parents]
        old, new = max(matches, key=lambda pair: len(pair[0].parts)) if matches else (original, original)
        # Records made before the kernel move name tools/<module>.py; follow it into atlas-kernels.
        path = checked(kernel_library.moved_path(new / original.relative_to(old)))
        try:
            actual = sha256(path)
        except FileNotFoundError:
            if expected is None:
                raise
            actual = None
        if expected is not None and actual != expected:
            archived = self.archived_sources.get((str(original), expected))
            if archived is None or sha256(archived) != expected:
                raise ValueError("capture source changed or missing without an exact archived copy: " + str(path))
            self.source_substitutions.append({"original_path": str(original), "archived_path": str(archived),
                "captured_sha256": expected, "current_original_sha256": actual})
            path, actual = archived, expected
        elif path != original:
            self.source_substitutions.append({"original_path": str(original), "relocated_path": str(path),
                "captured_sha256": actual})
        if str(path) in self.pins and self.pins[str(path)] != actual:
            raise ValueError("capture or export source identity changed: " + str(path))
        self.pins[str(path)] = actual
        return path

    def raw(self, desc):
        member = desc["file"]
        if not isinstance(member, str) or Path(member).is_absolute() or ".." in Path(member).parts:
            raise ValueError("capture payload must be relative")
        path = self.pin(self.root / member, desc["sha256"])
        raw = path.read_bytes()
        if len(raw) != desc["bytes"]:
            raise ValueError("capture payload extent changed")
        return raw

    def array(self, desc):
        raw = self.raw(desc)
        dtype = np.dtype(desc["dtype"])
        shape = tuple(desc["shape"])
        if dtype not in (np.dtype("uint16"), np.dtype("float32"), np.dtype("int64")) or any(
                type(n) is not int or n <= 0 for n in shape) or len(raw) != int(np.prod(shape)) * dtype.itemsize:
            raise ValueError("capture array geometry changed")
        return np.frombuffer(raw, dtype=dtype).reshape(shape)


def capture_codec(capture):
    records = [row for row in capture.data["source_records"] if Path(row["path"]).name == "tiny_fp8_pack.py"]
    if len(records) != 1:
        raise ValueError("one capture-bound FP8 codec required")
    record = records[0]
    codec_path = capture.pin(record["path"], record["sha256"])
    spec = importlib.util.spec_from_file_location("tiny_attention_export_codec", codec_path)
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    return codec


def platform_placement(args, launches=()):
    base, size = getattr(args, "dram_base", 0x80000000), getattr(args, "dram_bytes", None)
    if (type(base) is not int or type(size) is not int or base < 0 or base % 64
            or size <= 0 or size % 64 or base + size > 1 << 64):
        raise ValueError("attention export requires an aligned physical --dram-base and positive --dram-bytes")
    if not base <= ARENA_BASE < base + size:
        raise ValueError("attention arena is outside platform DRAM")
    for launch in launches:
        arena, extent = launch["dram_base"], launch["total_dram_bytes"]
        window(arena, extent)
        if not base <= arena <= base + size - extent:
            raise ValueError("attention launch arena is outside platform DRAM")
    return {"arena_base": ARENA_BASE, "dram_base": base, "dram_bytes": size}


def qualification_row(item, execution, assets):
    return {"index": item["index"], "kind": item["kind"], "status": "PASS", "exact": True,
        "full_arena_exact": True, "accelerated": False, "native_dispatch_enabled": False,
        "model": {"commit": execution.model["revision"], "root": execution.model["root"], "numerics": "rtl"},
        "source_sha256": {str(checked(Path(execution.model["root"]) / name)): value
            for name, value in execution.model["source_sha256"].items()},
        "harness_source_sha256": execution.implementation_source_sha256,
        "initial_sha256": assets.sums[item["initial"]], "output_sha256": assets.sums[item["expected"]],
        "final_arena_sha256": assets.sums[item["expected_arena"]],
        "program_source_sha256": assets.sums[item["source"]], "execution": execution.metadata()}


def attention_kernel(capture, record, assets, assemble):
    if record.get("schema") == "atlas.tiny.attention_launch.v2":
        from attention_v2_export import attention_kernel_v2
        return attention_kernel_v2(capture, record, assets, assemble, ARENA_BASE)
    config = AttentionConfig(**record["config"])
    if record["numerics"] != "rtl" or record["model_commit"] != MODEL_PIN or record["dram_base"] != ARENA_BASE:
        raise ValueError("attention arithmetic binding differs")
    if record["output_layout"] != "bf16_tiled" or config.head_dim != 64:
        raise ValueError("attention output layout differs")
    q, k, v = (capture.array(record[name + "_bf16"]) for name in ("q", "k", "v"))
    if q.shape != (config.query_heads, config.query_rows, 64) or k.shape != (config.kv_heads, config.keys, 64) or v.shape != k.shape:
        raise ValueError("attention logical input geometry differs")
    for name, value in zip(("q", "k", "v"), (q, k, v)):
        if capture.raw(record[name + "_tiled"]) != pack_tiles(value):
            raise ValueError("attention native input packing differs")
    initial, offsets = stage(config, q, k, v)
    if (offsets != record["offsets"] or initial != capture.raw(record["arena_initial"])
            or len(initial) != record["arena_bytes"] or offsets["output"] != record["output_offset"]
            or offsets["output_bytes"] != record["output_bytes"]):
        raise ValueError("captured attention staging differs from generator")
    source = attention_assembly(config, offsets, dram_base=ARENA_BASE)
    words = list(assemble(io.StringIO(source)).assemble())
    admit_program(words)
    program = struct.pack(f"<{len(words)}I", *words)
    if source.encode() != capture.raw(record["program_source"]) or program != capture.raw(record["program_binary"]):
        raise ValueError("captured attention program differs from generator")
    final = capture.raw(record["arena_expected"])
    lo, size = offsets["output"], offsets["output_bytes"]
    output = capture.raw(record["output_tiled"])
    if (len(final) != len(initial) or final[:lo] != initial[:lo] or final[lo + size:] != initial[lo + size:]
            or final[lo:lo + size] != output or len(output) != size
            or not np.array_equal(unpack_tiles(output, config.query_heads, config.query_rows),
                                  capture.array(record["output_bf16"]))):
        raise ValueError("captured attention full arena or output identity differs")
    item = {"kind": "attention", "step": record["step"], "layer": record["layer"], "site": record["site"],
        "config": asdict(config), "offsets": offsets, "input_layout": "bf16_tiled", "output_layout": "bf16_tiled",
        "source": assets.put("programs", source.encode(), ".S"), "original": assets.put("programs", program),
        "initial": assets.put("arenas", initial), "expected": assets.put("expected", output),
        "expected_arena": assets.put("arenas", final), "initial_sha256": digest(initial),
        "program_sha256": digest(program), "expected_sha256": digest(output),
        "output_offset": lo, "total_dram_bytes": len(initial), "dram_base": ARENA_BASE,
        "requested_dma": {name: record["requested_dma"][name] for name in ("load_bytes", "store_bytes")},
        "model_commit": MODEL_PIN, "numerics": "rtl", "capture_execution": record.get("execution", {})}
    logical = unpack_tiles(output, config.query_heads, config.query_rows).transpose(1, 0, 2).reshape(config.query_rows, -1)
    return item, logical


def emit_c(assets, data, host=None):
    """Emit the existing projection table and the separate 176-byte attention ABI."""
    host = assets.root / "host" if host is None else checked(host)
    host.mkdir(exist_ok=True)
    declarations, image, seen = [], [], {}
    asset_path = getattr(assets, "path", lambda member: assets.root / member)

    def blob(symbol, member):
        declarations.append(f"extern const uint8_t {symbol}[];")
        raw = asset_path(member).read_bytes()
        if member in seen:
            image.extend([f".global {symbol}", f".set {symbol},{seen[member]}"])
        else:
            seen[member] = symbol
            image.extend(['.section .rodata,"a",@progbits', '.balign 8', f'.global {symbol}',
                f'.type {symbol},@object', symbol + ':', '.incbin ' + json.dumps(str(asset_path(member))),
                f'.size {symbol},.-{symbol}'])
        return symbol

    for item in data["host_parameters"].values():
        blob(item["symbol"], item["file"])
    panel_table, panel_expected = [], []
    for index, item in enumerate(data["projection_kernels"]):
        weight = blob(f"tiny_full_weights_{index}", item["weights"])
        program = blob(f"tiny_full_program_{index}", item["original"])
        panel_expected.append(blob(f"tiny_full_panel_ref_{index}", item["expected"]))
        m, k, n = item["physical_mkn"]
        fields = [item["logical_rows"], m, k, n, item["columns"][0], item["width"], item["logical"], item["A_scale"],
            weight, k * n, "(const uint32_t*)" + program, len(asset_path(item["original"]).read_bytes()) // 4,
            item["output_offset"], item["total_dram_bytes"], item["requested_dma"]["load_bytes"],
            item["requested_dma"]["store_bytes"]]
        panel_table.append("{" + ",".join(map(str, fields)) + "}")
    attention_table = []
    for index, item in enumerate(data["attention_kernels"]):
        program = blob(f"tiny_full_attention_program_{index}", item["original"])
        initial = blob(f"tiny_full_attention_initial_{index}", item["initial"])
        expected = blob(f"tiny_full_attention_expected_{index}", item["expected"])
        c, o = item["config"], item["offsets"]
        fields = [item["step"], item["layer"], c["query_rows"], c["keys"], c["query_heads"], c["kv_heads"],
            c["q_exponent"], c["k_exponent"], c["v_exponent"], c["p_exponent"], "(const uint32_t*)" + program,
            len(asset_path(item["original"]).read_bytes()) // 4, o["q"], o["k"], o["v"], o["descriptors"],
            o["descriptor_bytes"], o["output"], o["output_bytes"], item["total_dram_bytes"],
            item["requested_dma"]["load_bytes"], item["requested_dma"]["store_bytes"], initial, expected]
        attention_table.append("{" + ",".join(map(str, fields)) + "}")
    norm_table = []
    for index, item in enumerate(data.get("norm_kernels", [])):
        program = blob(f"tiny_full_norm_program_{index}", item["original"])
        mask = blob(f"tiny_full_norm_mask_{index}", item["mask"])
        weight = blob(f"tiny_full_norm_weight_{index}", item["weight"])
        expected = blob(f"tiny_full_norm_expected_{index}", item["expected"])
        fields = [1, item["step"], item["layer"], item["logical_rows"], item["physical_rows"],
            item["output_offset"], item["total_dram_bytes"], item["mask_offset"], item["weight_offset"],
            item["requested_dma"]["load_bytes"], item["requested_dma"]["store_bytes"],
            "(const uint32_t*)" + program, len(asset_path(item["original"]).read_bytes()) // 4,
            mask, weight, expected, item["output_scale"], item["flags"]]
        norm_table.append("{" + ",".join(map(str, fields)) + "}")
    refs = data["reference_buffers"]
    actuals = []
    for item in refs:
        blob(item["expected_symbol"], item["file"])
        if item.get("field_kind") != "tokens":
            actuals.append(f'uint8_t {item["actual_symbol"]}[{item["bytes"]}];')
            declarations.append(f'extern uint8_t {item["actual_symbol"]}[{item["bytes"]}];')
    header = ['#ifndef TINY_FULL_ASSETS_H', '#define TINY_FULL_ASSETS_H', '#include "tiny_full_model.h"',
              '#include "tiny_full_projection.h"', '#include "tiny_attention.h"']
    for name, value in (("KERNELS", len(data["projection_kernels"])), ("ATTENTION_KERNELS", len(data["attention_kernels"])),
            ("LAUNCHES", len(data["launches"])), ("STEPS", data["decode_count"] + 1), ("LAYER_COUNT", data["layer_count"]),
            ("DRAM_BYTES", max(item["total_dram_bytes"] for item in data["launches"])),
            ("FUSIONS", int(data.get("fusions", False)))):
        header.append(f"#define TINY_FULL_{name} {value}")
    header += [f"#define TINY_FULL_ARENA_BASE UINT64_C({ARENA_BASE})",
        "typedef struct { uint32_t kind,index; } tiny_full_launch_entry;",
        "extern const tiny_full_launch_entry tiny_full_launch_plan[TINY_FULL_LAUNCHES];",
        "extern const tiny_panel tiny_full_panels[TINY_FULL_KERNELS];",
        "extern const uint8_t *const tiny_full_panel_expected[TINY_FULL_KERNELS];",
        "extern const tiny_attention_panel tiny_full_attention_panels[TINY_FULL_ATTENTION_KERNELS];",
        "extern const tiny_full_parameters tiny_full_parameters_global;",
        "extern tiny_full_state tiny_full_state_global;", "extern const int64_t *const tiny_full_prefill_tokens;",
        "extern uint32_t tiny_full_greedy_tokens[TINY_FULL_STEPS],tiny_full_decode_inputs[(TINY_FULL_STEPS>1)?TINY_FULL_STEPS-1:1];",
        "extern volatile uint64_t tohost,fromhost;", "int tiny_full_save_check_step(unsigned,const uint16_t *);",
        "#define TINY_FULL_LOGITS_LAYOUT_CHECK 1",
        "int tiny_full_save_check_step_layout(unsigned,const uint16_t *,size_t,uint64_t);"]
    if data.get("post_o_norm"):
        header += ["#define TINY_FULL_POST_O_NORM 1", f"#define TINY_FULL_NORM_KERNELS {len(norm_table)}",
            "extern const tiny_norm_panel tiny_full_norm_panels[TINY_FULL_NORM_KERNELS];"]
    if data.get("pre_attention_norm"):
        header += ["#define TINY_FULL_PRE_ATTENTION_NORM 1"]
    if data.get("prefill_device_rope"):
        if data.get("decode_device_rope"):
            tables_blob = data["rope_tables_blob"]
        else:
            tables = {item["rope_tables"] for item in data["projection_kernels"] if "rope_tables" in item}
            if len(tables) != 1:
                raise ValueError("prefill RoPE requires one shared table blob")
            tables_blob = tables.pop()
        blob("tiny_full_rope_tables", tables_blob)
        header += ["#define TINY_FULL_PREFILL_ROPE 1", "extern const uint8_t tiny_full_rope_tables[];"]
        if data.get("decode_device_rope"):
            header += ["#define TINY_FULL_DECODE_ROPE 1"]
    header += [line for line in declarations if "_expected" in line or "extern uint8_t" in line] + ["#endif"]
    params = data["host_parameters"]
    word = lambda key: "(const uint16_t*)" + params[key]["symbol"]
    norms = lambda suffix: "{" + ",".join(word(f"model.layers.{i}.{suffix}.weight") for i in range(22)) + "}"
    parameter = "{" + ",".join((word("model.embed_tokens.weight"), norms("input_layernorm"),
        norms("post_attention_layernorm"), word("model.norm.weight"), word("rotary_cosine"), word("rotary_sine"),
        "0x1.4f8b588e368f1p-17f", "tiny_full_attend",
        "tiny_full_fused_project" if data.get("fusions") else "0",
        "tiny_full_normalize" if data.get("post_o_norm") else "0",
        "|".join(name for name, enabled in (("TINY_FULL_CAP_POST_O_NORM", data.get("post_o_norm")),
            ("TINY_FULL_CAP_PREFILL_RESIDUAL", data.get("prefill_device_residual")),
            ("TINY_FULL_CAP_PREFILL_ROPE", data.get("prefill_device_rope")),
            ("TINY_FULL_CAP_PRE_ATTENTION_NORM", data.get("pre_attention_norm")),
            ("TINY_FULL_CAP_DECODE_ROPE", data.get("decode_device_rope"))) if enabled) or "0")) + "}"
    checks = []
    for step in range(data["decode_count"] + 1):
        checks.append(f"case {step}:")
        for item in refs:
            if item.get("step") != step:
                continue
            name = item["actual_symbol"]
            if "field" in item:
                checks.append(f"memcpy({name},tiny_full_state_global.{item['field']}"
                    f"[tiny_full_state_global.bank]+{item['layer']}*TINY_FULL_CACHE_WORDS,{item['bytes']});")
            elif item.get("field_kind") == "logits":
                checks.append("for(size_t i=0;i<32000;++i){uint32_t bits=(uint32_t)"
                    "logits[tiny_full_logits_index(row,i,(capabilities&TINY_FULL_CAP_TILED_LOGITS)!=0)]<<16;"
                    f"memcpy({name}+4*i,&bits,4);}}")
            else:
                continue  # Block hidden buffers are populated by the prefix runner.
            checks.append(f"if(memcmp({name},{item['expected_symbol']},{item['bytes']}))return 1;")
        if data.get("greedy_tokens"):
            checks.append(f"if(tiny_full_greedy_tokens[{step}]!="
                f"((const uint32_t*)tiny_full_greedy_tokens_expected)[{step}])return 1;")
            if step:
                checks.append(f"if(tiny_full_decode_inputs[{step - 1}]!="
                    f"((const uint32_t*)tiny_full_decode_inputs_expected)[{step - 1}])return 1;")
        checks.append("return 0;")
    fusion_lines = []
    if data.get("fusions"):
        header.insert(-1, "extern const tiny_fusion_panel tiny_full_fusion_panels[TINY_FULL_KERNELS];")
        fusion_lines = ["const tiny_fusion_panel tiny_full_fusion_panels[TINY_FULL_KERNELS]={" + ",\n".join(
            "{" + ",".join(map(str, (item["output_extent_bytes"], item["operand_offset"],
                item["output_scale"], item["fusion_role"]))) + "}" for item in data["projection_kernels"]) + "};"]
    attention_v2_lines = []
    if data.get("attention_version", 1) == 2:
        from attention_v2_export import companion_c
        header.insert(-1, "#define TINY_FULL_ATTENTION_V2 1")
        header.insert(-1, "extern const tiny_attention_v2_panel tiny_full_attention_v2_panels[TINY_FULL_ATTENTION_KERNELS];")
        attention_v2_lines = companion_c(data["attention_kernels"])
    block_lines = []
    if data.get("block_only"):
        header.insert(-1, "int tiny_full_save_check_block(unsigned,const void*,size_t);")
        block_lines = ["int tiny_full_save_check_block(unsigned layers,const void*workspace,size_t bytes){",
            "tiny_full_live_layout layout;",
            f"if(layers!={data['layer_count']}||!workspace||tiny_full_state_global.bank>1||"
            "tiny_full_get_live_layout(&layout)||layout.version!=1||bytes<layout.workspace_bytes)return 1;",
            "unsigned bank=1-tiny_full_state_global.bank;",
            "size_t offset=layout.hidden_offset[layers%2];"]
        for item in refs:
            name, size = item["actual_symbol"], item["bytes"]
            if item.get("field_kind") == "hidden":
                block_lines += [f"if({size}>layout.hidden_bytes||offset>bytes||{size}>bytes-offset)return 1;",
                    f"memcpy({name},(const uint8_t*)workspace+offset,{size});"]
            elif item.get("field") in ("keys", "values") and 0 <= item["layer"] < data["layer_count"]:
                block_lines.append(f"memcpy({name},tiny_full_state_global.{item['field']}[bank]"
                    f"+{item['layer']}*TINY_FULL_CACHE_WORDS,{size});")
            else:
                raise ValueError("block capture requires only hidden and selected-layer cache references")
            block_lines.append(f"if(memcmp({name},{item['expected_symbol']},{size}))return 1;")
        block_lines += ["return 0;}"]
    norm_lines = (["const tiny_norm_panel tiny_full_norm_panels[TINY_FULL_NORM_KERNELS]={" +
        ",\n".join(norm_table) + "};"] if data.get("post_o_norm") else [])
    lines = ['#include "tiny_full_assets.h"', '#include "tiny_full_tiles.h"', '#include <string.h>', *declarations, *actuals, *fusion_lines, *attention_v2_lines, *norm_lines,
        "const tiny_panel tiny_full_panels[TINY_FULL_KERNELS]={" + ",\n".join(panel_table) + "};",
        "const uint8_t *const tiny_full_panel_expected[TINY_FULL_KERNELS]={" + ",".join(panel_expected) + "};",
        "const tiny_attention_panel tiny_full_attention_panels[TINY_FULL_ATTENTION_KERNELS]={" + ",\n".join(attention_table) + "};",
        "const tiny_full_launch_entry tiny_full_launch_plan[TINY_FULL_LAUNCHES]={" + ",".join(
            "{" + str({"projection": 0, "attention": 1, "norm": 2}[entry["kind"]]) + "," + str(entry["index"]) + "}"
            for entry in data["launch_plan"]) + "};",
        "const tiny_full_parameters tiny_full_parameters_global=" + parameter + ";",
        "static uint16_t keys[2][22*TINY_FULL_CACHE_WORDS],values[2][22*TINY_FULL_CACHE_WORDS];",
        "tiny_full_state tiny_full_state_global={{keys[0],keys[1]},{values[0],values[1]},0,0,0};",
        "const int64_t *const tiny_full_prefill_tokens=(const int64_t*)" + params["prefill_tokens"]["symbol"] + ";",
        "uint32_t tiny_full_greedy_tokens[TINY_FULL_STEPS],tiny_full_decode_inputs[(TINY_FULL_STEPS>1)?TINY_FULL_STEPS-1:1];",
        "int tiny_full_save_check_step_layout(unsigned step,const uint16_t*logits,size_t row,uint64_t capabilities){"
        "(void)logits;if(row>=128||(capabilities&~TINY_FULL_CAPABILITIES))return 1;switch(step){" +
        "\n".join(checks) + "default:return 1;}}",
        "int tiny_full_save_check_step(unsigned step,const uint16_t*logits){"
        "return tiny_full_save_check_step_layout(step,logits,0,0);}", *block_lines]
    for name, raw in (("tiny_full_assets.h", "\n".join(header) + "\n"), ("tiny_full_assets.c", "\n".join(lines) + "\n"),
            ("images.S", "\n".join(image) + '\n.section .note.GNU-stack,"",@progbits\n')):
        (host / name).write_text(raw)
        assets.sums["host/" + name] = digest(raw.encode())


def export_attention(args):
    """Keep native validation isolated from the original-model assembler imports."""
    options = [getattr(args, name, None) for name in
               ("native_proof_script", "native_proof_sha256", "native_proof_python")]
    with ExitStack() as stack:
        checker = None
        if any(option is not None for option in options):
            if any(option is None for option in options):
                raise ValueError("native proof requires its script, SHA256, and Python interpreter")
            path, expected, python = options
            path = checked(path)
            if sha256(path) != expected:
                raise ValueError("native proof script identity differs")
            spec = importlib.util.spec_from_file_location("tiny_export_native_proof", path)
            provider = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(provider)
            checker = stack.enter_context(provider.ProofClient(python))
            proof_source = (path, expected)
        else:
            proof_source = None
        return _export_attention(args, checker, proof_source)


def _export_attention(args, native_checker=None, proof_source=None):
    capture = Capture(args.capture, args.capture_sha256, getattr(args, "capture_source_archive", None),
                      getattr(args, "relocate", ()))
    if proof_source is not None:
        capture.pin(*proof_source)
    report = capture.data
    if not attention_execution_mode(report):
        raise ValueError("attention exporter requires enabled instruction execution")
    platform_placement(args)
    block = report.get("block_only", False)
    layers, decode = report.get("layer_count", 22), report["decode_count"]
    include_head = report.get("include_lm_head", not block)
    fused = report.get("fusions") is True
    c_capture = report.get("host_reference") == "compiled_runtime_c"
    attention_version = report.get("attention_version", 1)
    post_o_norm = report.get("post_o_norm", False)
    if (type(post_o_norm) is not bool or post_o_norm and not (fused and c_capture and attention_version == 2)
            or not post_o_norm and report.get("norm_calls")):
        raise ValueError("post-o norm requires an explicit fused C attention-v2 capture")
    pre_norm = report.get("pre_attention_norm", False)
    if type(pre_norm) is not bool or pre_norm and (not post_o_norm or report.get("pre_attention_norm_headroom") not in (0, 1)):
        raise ValueError("pre-attention norm requires the post-o norm C capture and a headroom of zero or one exponent step")
    sites = 2 if pre_norm else 1
    device_residual = report.get("prefill_device_residual", False)
    device_rope = report.get("prefill_device_rope", False)
    if any(type(flag) is not bool or flag and not post_o_norm for flag in (device_residual, device_rope)):
        raise ValueError("prefill device residual and RoPE require the post-o norm C capture")
    if type(attention_version) is not int or attention_version not in (1, 2) or attention_version == 2 and not (fused and c_capture):
        raise ValueError("unsupported typed Tiny attention version")
    native_capture = report["status"] in (
        "PASS_TINY_FUSED_C_NATIVE_BLOCK_CAPTURE_REPLAY_REQUIRES_ORIGINAL47_BACKFILL",
        "PASS_FULL_TINY_FUSED_C_NATIVE_CAPTURE_REPLAY_REQUIRES_ORIGINAL47_BACKFILL")
    if (report["status"] not in ("PASS_TINY_ATTENTION_BLOCK_CAPTURE_REPLAY", "PASS_FULL_TINY_ARITHMETIC_CAPTURE_REPLAY",
            "PASS_TINY_FUSED_BLOCK_CAPTURE_REPLAY", "PASS_FULL_TINY_FUSED_CAPTURE_REPLAY",
            "PASS_TINY_FUSED_C_BLOCK_CAPTURE_REPLAY", "PASS_FULL_TINY_FUSED_C_CAPTURE_REPLAY",
            "PASS_TINY_FUSED_C_NATIVE_BLOCK_CAPTURE_REPLAY_REQUIRES_ORIGINAL47_BACKFILL",
            "PASS_FULL_TINY_FUSED_C_NATIVE_CAPTURE_REPLAY_REQUIRES_ORIGINAL47_BACKFILL")
            or any(report.get(name) is not True for name in ("source_bookends_verified", "target_replay_exact"))
            or (not c_capture and report.get("native_restored_exact") is not True)
            or (c_capture and (not fused or report.get("host_c_sources_verified") is not True))
            or (native_capture and (not c_capture or native_checker is None
                or report.get("requires_original_backfill") is not True
                or report.get("final_numerics_qualified") is not False))
            or not report.get("cache_prefixes_exact", report.get("all22_cache_prefixes_exact", False))
            or type(layers) is not int or not 1 <= layers <= 22 or type(decode) is not int or not 0 <= decode <= 32
            or block and (layers != 1 or decode != 0 or include_head)):
        raise ValueError("complete selected Tiny attention capture required")
    out = checked(args.out)
    if out.exists():
        raise ValueError("fresh attention bundle required")
    torch, to_fp8, assemble, binding = model_setup(args.model_root)
    model_before = binding()
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).absolute()
            if path.is_relative_to(ROOT / "tools"):
                capture.pin(path)
    encoder = ExactEncoder(capture_codec(capture), torch, to_fp8)
    rope_tables = None
    if device_rope:
        from tiny_full.fusion import rope_tables as pack_rope_tables
        rope_tables = pack_rope_tables(*(capture.array(report["host_parameters"][name])
                                         for name in ("rotary_cosine", "rotary_sine")))
    out.mkdir(parents=True)
    assets = Assets(out)
    attention, logical_attention = [], {}
    for index, record in enumerate(report["attention_calls"]):
        step, layer = divmod(index, layers)
        if record["index"] != index or (record["step"], record["layer"]) != (step, layer):
            raise ValueError("attention capture invocation order changed")
        if (record.get("schema") == "atlas.tiny.attention_launch.v2") != (attention_version == 2):
            raise ValueError("mixed or missing attention-v2 record discriminator")
        item, logical = attention_kernel(capture, record, assets, assemble)
        item["index"] = index
        attention.append(item)
        logical_attention[step, layer] = logical
    if len(attention) != layers * (decode + 1):
        raise ValueError("attention layer/step coverage differs")
    norms = []
    if post_o_norm:
        from norm_export import norm_kernel
        # With the pre-attention norm, each layer and step runs input_layernorm, then post_attention_layernorm.
        for index, record in enumerate(report["norm_calls"]):
            step, layer = divmod(index // sites, layers)
            kind = "input_layernorm" if pre_norm and index % 2 == 0 else "post_attention_layernorm"
            if ((record.get("index"), record.get("step"), record.get("layer"), record.get("site"))
                    != (index, step, layer, f"model.layers.{layer}.{kind}")):
                raise ValueError("norm capture invocation order changed")
            norms.append(norm_kernel(capture, record, assets, assemble, index))
        if len(norms) != sites * layers * (decode + 1):
            raise ValueError("norm layer/step coverage differs")
    projections, launches, launch_plan = [], [], []

    def launch(kind, item):
        table_index = item["index"]
        launch_plan.append({"launch_index": len(launches), "kind": kind, "index": table_index})
        launches.append(dict(item, index=len(launches), table_index=table_index))

    calls_per_step = layers * 7 + int(include_head)
    for call_index, call in enumerate(report["calls"]):
        step, index = divmod(call_index, calls_per_step)
        layer, role = divmod(index, 7) if index < layers * 7 else (layers, 7)
        site = f"model.layers.{layer}.{ROLES[role]}" if role < 7 else "lm_head"
        if (call["step"], call["site_index"], call["site"]) != (step, index, site) or step > decode:
            raise ValueError("projection invocation order changed")
        a, b = capture.array(call["A_bf16"]), capture.array(report["parameters"][site]["B_bf16"])
        rows = 128 if step == 0 else 1
        if a.shape != (rows, b.shape[0]):
            raise ValueError("projection logical geometry differs")
        if role == 3:
            if not np.array_equal(a, logical_attention[step, layer]):
                raise ValueError("o_proj did not consume the actual Atlas attention output")
            launch("attention", attention[step * layers + layer])
        norm_input = post_o_norm and role in (4, 5) or pre_norm and role in (0, 1, 2)
        if norm_input:
            producer = norms[(step * layers + layer) * sites + (sites - 1 if role >= 4 else 0)]
            if (call.get("input_layout") != "fp8_tiled" or call.get("input_norm") != producer["index"]
                    or call.get("norm_scale") != producer["output_scale"] or call["A_scale"] != producer["output_scale"]
                    or call.get("norm_h_sha256") != producer["norm_h_sha256"]
                    or capture.raw(call["A_fp8"]) != (out / producer["expected"]).read_bytes()):
                raise ValueError("q/k/v or gate/up did not consume its exact typed norm FP8 producer")
            from tiny_full.fusion_layout import unpack_fp8_tiles
            encoded = unpack_fp8_tiles(capture.raw(call["A_fp8"]), 128 if rows == 128 else 32, 2048)
            placeholder = ((torch.from_numpy(encoded.copy()).view(torch.float8_e4m3fn).float()
                           * 2.0**producer["output_scale"]).to(torch.bfloat16).view(torch.uint16).numpy())[:rows]
            if not np.array_equal(a, placeholder):
                raise ValueError("norm BF16 placeholder does not decode the actual FP8 producer")
            if role in (0, 4):
                launch("norm", producer)
        physical_a = a if rows == 128 else np.concatenate((a, np.zeros((31, a.shape[1]), np.uint16)))
        opaque_fp8 = fused and (norm_input or role == 6 or role == 3 and attention_version == 2)
        if not opaque_fp8:
            physical_a, encoded_a, a_scale = projection_activation(capture, call, a, role, encoder,
                                                                  require_physical=fused and c_capture)
        column = 0
        for panel in call["panels"]:
            lo, hi = panel["columns"]
            m, k, n = panel["physical_mkn"]
            expected = capture.array(panel["expected_bf16"])
            if (lo != column or hi <= lo or n != hi - lo or k != a.shape[1] or m != physical_a.shape[0]
                    or expected.shape != (m, n) or panel["logical_mkn"] != [rows, k, n]):
                raise ValueError("projection panel coverage differs")
            if fused:
                from fusion import pack_fp8_tiles
                from fusion_export import projection_kernel
                encoded_b, b_scale = encoder.encode(np.ascontiguousarray(b[:, lo:hi]))
                actual_a = capture.raw(call["A_fp8"]) if opaque_fp8 else pack_fp8_tiles(encoded_a)
                actual_scale = call["A_scale"] if opaque_fp8 else a_scale
                if role == 3 and attention_version == 2:
                    producer = attention[step * layers + layer]
                    if (call.get("input_layout") != "fp8_tiled" or call.get("producer_index") != producer["index"]
                            or call.get("input_attention") != producer["index"]
                            or actual_scale != producer["output_scale"]
                            or actual_a != (out / producer["expected"]).read_bytes() or len(actual_a) != m * k):
                        raise ValueError("o projection did not consume the exact typed attention FP8 extent")
                if (capture.raw(panel["launch"]["a_fp8"]) != actual_a
                        or capture.raw(panel["launch"]["b_fp8"]) != pack_weights(encoded_b)
                        or (actual_scale, b_scale, actual_scale + b_scale) !=
                           (panel["A_scale"], panel["B_scale"], panel["rescale"])):
                    raise ValueError("fused producer input or compile-time scale differs")
                item = projection_kernel(capture, panel, assets, assemble, step=step, layer=layer, role=role,
                    site=site, site_index=index, rows=rows, width=b.shape[1], index=len(projections),
                    native_checker=native_checker, device_residual=device_residual, rope_tables=rope_tables)
                if role == 3 and attention_version == 2:
                    item.update(input_layout="fp8_tiled", input_attention_version=2,
                                input_attention=step * layers + layer)
                if norm_input:
                    item.update(input_layout="fp8_tiled", input_norm=producer["index"],
                        norm_scale=producer["output_scale"], norm_h_sha256=producer["norm_h_sha256"])
                projections.append(item)
                launch("projection", item)
                column = hi
                continue
            encoded_b, b_scale = encoder.encode(np.ascontiguousarray(b[:, lo:hi]))
            if (a_scale, b_scale, a_scale + b_scale) != (panel["A_scale"], panel["B_scale"], panel["rescale"]):
                raise ValueError("projection capture scales differ")
            prepared = prepare_encoded(encoded_a, encoded_b, a_scale + b_scale, assemble)
            if not block:
                prepared["source"] = projection_assembly(m, k, n, prepared["memory"]["output_offset"],
                    a_scale + b_scale, traffic_reuse=True)
                prepared["requested_dma"] = {"load_bytes": dma_load_bytes(m, k, n, traffic_reuse=True),
                    "store_bytes": 2 * m * n}
            source = relocate_assembly(prepared["source"], DEFAULT_ARENA, ARENA_BASE)
            words = list(assemble(io.StringIO(source)).assemble())
            admit_program(words)
            program = struct.pack(f"<{len(words)}I", *words)
            initial, output = prepared["initial"], pack_output(expected)
            offset = prepared["memory"]["output_offset"]
            final = initial[:offset] + output + initial[offset + len(output):]
            item = {"index": len(projections), "kind": "projection", "step": step, "layer": layer, "logical": role,
                "site": site, "site_index": index, "columns": [lo, hi], "width": b.shape[1], "logical_rows": rows,
                "physical_mkn": [m, k, n], "A_scale": a_scale, "B_scale": b_scale, "rescale": a_scale + b_scale,
                "input_layout": "bf16_tiled" if role == 3 else "bf16_row_major",
                "output_layout": "bf16_tiled" if role < 3 else "bf16_row_major",
                "source": assets.put("programs", source.encode(), ".S"), "original": assets.put("programs", program),
                "activation": assets.put("activations", prepared["activation"]), "weights": assets.put("weights", prepared["weights"]),
                "initial": assets.put("arenas", initial), "expected": assets.put("expected", output),
                "expected_arena": assets.put("arenas", final), "initial_sha256": digest(initial),
                "program_sha256": digest(program), "expected_sha256": digest(output),
                "output_offset": offset, "total_dram_bytes": len(initial), "dram_base": ARENA_BASE,
                "requested_dma": prepared["requested_dma"]}
            item["program_policy"] = "original" if block else "traffic-reuse-v1"
            if role == 3:
                item["input_attention"] = step * layers + layer
            projections.append(item)
            launch("projection", item)
            column = hi
        if column != b.shape[1]:
            raise ValueError("incomplete projection output columns")
    if len(report["calls"]) != calls_per_step * (decode + 1):
        raise ValueError("projection logical call count differs")
    host = {}
    for index, name in enumerate(sorted(report["host_parameters"])):
        desc = report["host_parameters"][name]
        host[name] = dict(desc, file=assets.put("host-data", capture.raw(desc)), symbol=f"tiny_full_param_{index:03d}")
    references = []
    for trace in report["target_trace"]:
        step = trace["step"]
        for desc in trace["cache"]:
            logical = capture.array(desc).reshape(4, 128 + step, 64)
            padded = np.zeros((4, 160, 64), dtype=np.uint16)
            padded[:, :128 + step] = logical
            raw = pack_tiles(padded)
            name = f"tiny_full_cache_{step:02d}_{desc['layer']:02d}_{desc['field']}"
            references.append({"actual_symbol": name, "expected_symbol": name + "_expected",
                "file": assets.put("references", raw), "bytes": len(raw), "sha256": digest(raw),
                "dtype": "uint16", "shape": [1, 4, 160, 64], "logical_shape": [1, 4, 128 + step, 64],
                "layout": "bf16_tiled_cache_capacity160", "step": step, "layer": desc["layer"], "field": desc["field"]})
        if "hidden_payload" in trace:
            raw = capture.raw(trace["hidden_payload"])
            references.append({"actual_symbol": f"tiny_full_hidden_{step:02d}",
                "expected_symbol": f"tiny_full_hidden_{step:02d}_expected", "file": assets.put("references", raw),
                "bytes": len(raw), "sha256": digest(raw), "layout": "bf16_row_major", "step": step,
                "dtype": "uint16", "shape": trace["hidden_payload"]["shape"], "field_kind": "hidden"})
        if "logits_payload" in trace:
            raw = capture.raw(trace["logits_payload"])
            references.append({"actual_symbol": f"tiny_full_logits_{step:02d}",
                "expected_symbol": f"tiny_full_logits_{step:02d}_expected", "file": assets.put("references", raw),
                "bytes": len(raw), "sha256": digest(raw), "layout": "float32_bf16_values", "step": step,
                "field_kind": "logits", "dtype": "float32", "shape": [32000]})
    tokens = []
    if include_head:
        tokens = [trace["next_token"] for trace in report["target_trace"]]
        consumed = [trace["input_tokens"][0][0] for trace in report["target_trace"][1:]]
        if (len(tokens) != decode + 1 or consumed != tokens[:-1] or report.get("greedy_feedback_actual") is not True
                or any(type(token) is not int or not 0 <= token < 32000 for token in tokens)):
            raise ValueError("full attention capture greedy feedback differs")
        for name, values in (("tiny_full_greedy_tokens", tokens), ("tiny_full_decode_inputs", consumed)):
            raw = np.asarray(values, dtype="<u4").tobytes()
            references.append({"actual_symbol": name, "expected_symbol": name + "_expected",
                "file": assets.put("references", raw), "bytes": len(raw), "sha256": digest(raw),
                "dtype": "uint32", "shape": [len(values)], "field_kind": "tokens"})
    data = {"schema": "atlas.tiny_attention_contract.v1", "workload": "tiny-attention", "layer_count": layers,
        "decode_count": decode, "block_only": block, "include_lm_head": include_head,
        "projection_calls": len(report["calls"]), "projection_kernel_count": len(projections),
        "attention_kernel_count": len(attention), "kernel_count": len(launches), "projection_kernels": projections,
        "attention_kernels": attention, "launches": launches, "launch_plan": launch_plan,
        "host_parameters": host, "reference_buffers": references,
        "greedy_tokens": tokens, "program_policy": "traffic-reuse-fused-v1" if fused else "original" if block else "traffic-reuse-v1",
        "fusions": fused, "residual_adds": device_residual,
        "placement": platform_placement(args, launches)}
    if attention_version == 2:
        data["attention_version"] = 2
    if post_o_norm:
        data.update(post_o_norm=True, norm_kernel_count=len(norms), norm_kernels=norms)
    if pre_norm:
        data.update(pre_attention_norm=True, pre_attention_norm_headroom=report["pre_attention_norm_headroom"])
    if device_residual or device_rope:
        data.update(prefill_device_residual=device_residual, prefill_device_rope=device_rope,
                    device_rounding="rne", device_steps=[0])
    if fused:
        native_count = sum(item.get("accelerated") is True for item in projections)
        counts = {"original47_launches": len(launches) - native_count, "native70_launches": native_count}
        if native_capture:
            if not native_count or report.get("capture_execution_models") != counts:
                raise ValueError("native capture execution-model counts differ from actual records")
        elif native_count:
            raise ValueError("native records require an explicit native-capture status")
        data["capture_execution_models"] = {
            **counts, "native_capture_requires_original47_qualification": bool(native_count)}
    emit_c(assets, data)
    gate = None
    if getattr(args, "qualify_launches", False):
        from atlas_execution import Executor
        executor = Executor(args.model_root)
        receipts = []
        for item in launches:
            actual = executor.execute((out / item["source"]).read_text(), (out / item["initial"]).read_bytes(),
                item["output_offset"], len((out / item["expected"]).read_bytes()),
                expected_arena=(out / item["expected_arena"]).read_bytes(), requested_dma=item["requested_dma"],
                program_sha256=item["program_sha256"], dram_base=item["dram_base"])
            if actual.arena != (out / item["expected_arena"]).read_bytes():
                raise ValueError(f"launch {item['index']} differs from selected npu_model RTL")
            receipts.append(qualification_row(item, actual, assets))
            print(f"qualified launch {item['index'] + 1}/{len(launches)}", file=sys.stderr, flush=True)
        data["qualification"] = {"model": binding(), "all_launches_exact": True, "launches": receipts}
        gate = {"status": "PASS", "model_pin": MODEL_PIN, "model_root": actual.model["root"], "numerics": "rtl",
            "native_not_used": True, "attention_native_allowed": False,
            "all_launches_original_npu_model_rtl_exact": True, "launch_count": len(launches),
            "verified_indices": list(range(len(launches))), "launches": receipts}
    else:
        data["qualification"] = {"all_launches_exact": False, "attention_capture_executed": True,
            "projection_expected_source": "capture arithmetic; execution qualification required"}
    for path in (Path(__file__), Path(__file__).with_name("export_capture.py"),
                 kernel_library.source_path("atlas_attention_program.py"), ROOT / "tools/chipyard/arena.py"):
        capture.pin(path)
    if attention_version == 2:
        for path in (kernel_library.source_path("atlas_attention_v2_program.py"), Path(__file__).with_name("attention_v2_export.py"),
                     kernel_library.source_path("atlas_program_checks.py")):
            capture.pin(path)
    if fused:
        from kernel_library import code_sources
        for name in code_sources(traffic_reuse=True):
            capture.pin(kernel_library.source_path(name))
        for name in ("fusion.py", "fusion_export.py"):
            capture.pin(Path(__file__).with_name(name))
    if post_o_norm:
        for name in ("norm.py", "norm_export.py"):
            capture.pin(Path(__file__).with_name(name))
        for name in ("atlas_norm_postpass.py", "atlas_rmsnorm_postpass.py", "atlas_resident_output.py",
                     "atlas_rope_postpass.py"):
            capture.pin(kernel_library.source_path(name))
    data["model"] = binding()
    if any(data["model"]["imported_source_sha256"].get(name) != value
           for name, value in model_before["imported_source_sha256"].items()):
        raise ValueError("selected model source bookend changed")
    data["capture"] = {"path": str(capture.path), "sha256": args.capture_sha256}
    data["capture_source_substitutions"] = capture.source_substitutions
    data["capture_source_relocations"] = [{"old": str(old), "new": str(new)} for old, new in capture.relocations]
    data["source_sha256"] = capture.pins
    replay = {"schema": "atlas.v2.captured_reply.v1", "pass_string": "*** PASSED ***", "launches": launches,
        "final_buffers": [{"symbol": item["actual_symbol"], "expected": item["file"],
            "expected_sha256": item["sha256"]} for item in references if block or item.get("field_kind") != "hidden"]}
    for name, document in (("contract.json", data), ("kernels.json", data), ("provenance.json", data), ("replay.json", replay)):
        raw = (json.dumps(document, indent=2) + "\n").encode()
        (out / name).write_bytes(raw)
        assets.sums[name] = digest(raw)
    if gate is not None:
        gate["assets_replay_sha256"] = assets.sums["replay.json"]
        raw = (json.dumps(gate, indent=2) + "\n").encode()
        (out / "qualification.json").write_bytes(raw)
        assets.sums["qualification.json"] = digest(raw)
    if any(sha256(checked(path)) != value for path, value in capture.pins.items()):
        raise ValueError("attention export source bookend changed")
    (out / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in sorted(assets.sums.items())))
    return data
