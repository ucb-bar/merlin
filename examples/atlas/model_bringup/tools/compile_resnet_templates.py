#!/usr/bin/env python3
"""Compile the 54 ResNet templates and derive rescale patches without execution."""

from atlas_host_layout import fp8_output_offset, ATLAS_DRAM_BASE
import argparse
import io
import json
from pathlib import Path
import struct

import kernel_library  # noqa: E402,F401
from atlas_program import assembly, dma_load_bytes
from kernel_library import code_sources
from atlas_host_build import GRAPH_PROGRAM_WORDS, validate_graph_program
from atlas_path_guard import checked as safe_path
from atlas_tooling import sha256, write_json
from run_model_tile import configure_model, model_binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-root", "host-plan", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--traffic-reuse", action="store_true", help="select the qualified public traffic-reuse dispatch")
    args = parser.parse_args()
    out, plan_path = safe_path(args.out), safe_path(args.host_plan)
    if out.exists():
        raise ValueError("fresh output required")
    plan_id = {"path": str(plan_path), "sha256": sha256(plan_path)}
    plan = json.loads(plan_path.read_text())
    if plan.get("status") != "exported_compile_plan_only" or plan.get("contraction_count") != 54:
        raise ValueError("complete 54-contraction host plan required")
    own = safe_path(Path(__file__).absolute().parent)
    bindings = {str(path): sha256(path) for path in (Path(__file__).absolute(), kernel_library.source_path("atlas_program.py"),
        own / "run_model_tile.py", kernel_library.source_path("atlas_tile_arithmetic.py"))}
    bindings.update({str(kernel_library.source_path(name)): digest
                     for name, digest in code_sources(traffic_reuse=args.traffic_reuse).items()})
    bindings[str(own / "atlas_host_build.py")] = sha256(own / "atlas_host_build.py")
    root = safe_path(args.model_root)
    _, _, assemble, _ = configure_model(root)
    before = model_binding(root)

    def words(text):
        return list(assemble(io.StringIO(text)).assemble())

    table = []
    for exponent in range(-126, 128):
        encoded = words(f"vli.all m4, {(exponent + 127) << 7}\n")
        if len(encoded) != 1:
            raise ValueError("rescale must assemble as one word")
        table.append(encoded[0])
    out.mkdir(parents=True)
    (out / "rescale-words.bin").write_bytes(struct.pack("<254I", *table))
    write_json(out / "rescale-words.json",
        {"exponent_min": -126, "exponent_max": 127, "words_hex": [f"{word:08x}" for word in table],
        "instruction": "vli.all m4, BF16(2**exponent)",
        "encoding_source": "selected model assembler, no host bitfield constants"})
    kernels = []
    for node in plan["nodes"]:
        if node["partition"] != "device":
            continue
        name = node["target"]
        # Site names become a single directory component, never an input path.
        if (not isinstance(name, str)
            or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._" for c in name)):
            raise ValueError("invalid contraction site name")
        m, k, n = node["padded_mkn"]
        if any(type(x) is not int or x <= 0 or x % 32 for x in (m, k, n)) or k > 8192 or m * n > 1048576:
            raise ValueError("shape violates workload resource admission")
        offset = fp8_output_offset(m, k, n)
        total = offset + m * n * 2 + 256
        if total > 64 * 1024 * 1024:
            raise ValueError("DRAM resource bound exceeded")
        resident_b = not args.traffic_reuse and k * n <= 256 * 1024
        resident_a = not args.traffic_reuse and not resident_b and m * k <= 256 * 1024

        def source(exponent):
            return assembly(m, k, n, offset, exponent, resident_b=resident_b, pretranspose_resident_b=resident_b,
                resident_a=resident_a, traffic_reuse=args.traffic_reuse)

        text = source(0)
        base, alternate = words(text), words(source(1))
        positions = [i for i, (x, y) in enumerate(zip(base, alternate)) if x != y]
        if len(base) != len(alternate) or len(positions) != 1:
            raise ValueError("rescale patch is not one stable instruction word")
        index = positions[0]
        if base[index] != table[126] or alternate[index] != table[127]:
            raise ValueError("full program rescale differs from isolated instruction")
        validate_graph_program(struct.pack(f"<{len(base)}I", *base))
        for exponent in (-126, -1, 1, 127):
            patched = base.copy()
            patched[index] = table[exponent + 126]
            if patched != words(source(exponent)):
                raise ValueError("patched program differs from full reassembly")
        folder = safe_path(out / f"kernel-{len(kernels):02d}-{name}")
        folder.mkdir()
        (folder / "program.S").write_text(text)
        (folder / "program.bin").write_bytes(struct.pack(f"<{len(base)}I", *base))
        kernels.append({"site": name, "matrix_mkn": node["matrix_mkn"], "padded_mkn": [m, k, n], "program_words": len(base),
            "program": {"path": str(folder / "program.bin"), "sha256": sha256(folder / "program.bin")},
            "source": {"path": str(folder / "program.S"), "sha256": sha256(folder / "program.S")},
            "rescale_patch_word_index": index, "base_rescale_exponent": 0,
            "b_scale_exponent": node["weight"]["b_scale_exponent"],
            "required_runtime_rescale": "A_scale_exponent + B_scale_exponent; refuse outside [-126,127] before START",
            "traffic_reuse": args.traffic_reuse,
            "requested_dma": {"load_bytes": dma_load_bytes(m, k, n, resident_b=resident_b, resident_a=resident_a,
                traffic_reuse=args.traffic_reuse), "store_bytes": m * n * 2},
            "resident_b": resident_b, "resident_a": resident_a, "dram_base": ATLAS_DRAM_BASE, "input_bytes": m * k + k * n,
            "output_offset": offset, "output_bytes": m * n * 2, "total_dram_bytes": total, "weight": node["weight"],
            "zero_tail_contract": (
            "Host explicitly pads M/K/N with positive zero and crops M/N after complete " "output guard/exact check")})
    if len(kernels) != 54:
        raise ValueError("incomplete whole-graph contraction template set")
    if (model_binding(root) != before or sha256(plan_path) != plan_id["sha256"]
        or any(sha256(path) != digest for path, digest in bindings.items())):
        raise ValueError("input or source changed during compilation")
    report = {"status": "passed_compile_only", "scope": "54 declared dense contractions; no numerical execution",
        "traffic_reuse": args.traffic_reuse, "raw_program_word_limit": GRAPH_PROGRAM_WORDS,
        "host_plan": plan_id, "source_bindings": bindings, "model_binding": before, "kernels": kernels,
        "rescale_table_sha256": sha256(out / "rescale-words.bin"),
        "checks": {"all254_legal_exponents_assembled_as_one_word": True,
        "every_shape_boundary_patch_equals_full_reassembly": True, "all_shapes_fit_conservative_imem_vmem_dram_bounds": True,
        "source_bookends_unchanged": True},
        "not_claimed": ["instruction-model execution", "RTL execution", "host runtime correctness", "full SoC execution",
        "task-quality acceptance"]}
    write_json(out / "receipt.json", report)
    print(json.dumps({"status": report["status"], "kernels": len(kernels),
        "maximum_program_words": max(k["program_words"] for k in kernels)}))


if __name__ == "__main__":
    main()
