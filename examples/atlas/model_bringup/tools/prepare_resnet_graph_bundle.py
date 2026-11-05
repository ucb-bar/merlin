#!/usr/bin/env python3
"""Prepare a checksum-bound ResNet host and captured launch references without execution."""

import argparse
import io
import json
from pathlib import Path
import struct
import sys

from atlas_host_build import validate_graph_program
from atlas_host_layout import ATLAS_DRAM_BASE
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked
from atlas_program import assembly, dma_load_bytes
from kernel_library import code_sources
from atlas_tooling import Admission, sha, write_json
from build_resnet_graph_host import arena, relocate_program
from run_model_tile import configure_model, model_binding

sys.path.insert(0, str(Path(__file__).absolute().parent / "host_iss"))
from bundle import Bundle
from placement import arena_extent, placement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("host", "templates", "reference", "model-root", "out"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    for name, value in vars(args).items():
        setattr(args, name, checked(value))
    if args.out.exists():
        raise ValueError("fresh bundle output required")
    admission = Admission()
    template_bytes = admission.read(args.templates)
    templates = json.loads(template_bytes)
    reference_path = args.reference / "receipt.json"
    reference_bytes = admission.read(reference_path)
    reference = json.loads(reference_bytes)
    build_path = args.host / "build.json"
    build_bytes = admission.read(build_path)
    build = json.loads(build_bytes)
    if (templates.get("status") != "passed_compile_only" or templates.get("traffic_reuse") is not True
            or len(templates.get("kernels", [])) != 54 or len(reference.get("kernels", [])) != 54
            or build.get("returncode") != 0 or build.get("source_bookends_equal") is not True
            or build.get("kernel_count") != 54
            or build.get("inputs_sha256", {}).get(str(args.templates)) != sha(template_bytes)
            or build.get("inputs_sha256", {}).get(str(reference_path)) != sha(reference_bytes)):
        raise ValueError("matching complete reuse templates, reference and successful host build required")
    target = placement(build)
    arena_extent(build, build["dram_bytes"])
    own = Path(__file__).absolute().parent
    source_pins = {str(args.templates): sha(template_bytes), str(reference_path): sha(reference_bytes),
        str(build_path): sha(build_bytes)}
    for name, digest in code_sources(traffic_reuse=True).items():
        path = kernel_library.source_path(name)
        admission.read(path, digest)
        source_pins[str(path)] = digest
    for path in (Path(__file__).absolute(), own / "build_resnet_graph_host.py", own / "atlas_host_build.py",
            own / "chipyard/arena.py", own / "host_iss/bundle.py", own / "host_iss/placement.py",
            kernel_library.source_path("atlas_tooling.py"), kernel_library.source_path("atlas_path_guard.py")):
        source_pins[str(path)] = sha(admission.read(path))
    for path, digest in templates["source_bindings"].items():
        admission.read(path, digest)
    _, _, assemble, _ = configure_model(checked(args.model_root))
    model_before = model_binding(args.model_root)
    if templates["model_binding"] != model_before:
        raise ValueError("template model source differs")

    def encode(text):
        values = assemble(io.StringIO(text)).assemble()
        return b"".join(struct.pack("<I", word) for word in values)

    payloads = {"host/build.json": build_bytes}
    elf = admission.read(args.host / "host.elf", build["elf_sha256"])
    payloads["host/host.elf"] = elf
    for name, digest in build["generated_inputs_sha256"].items():
        path = checked(name)
        relative = path.relative_to(checked(args.host))
        payloads["host/" + relative.as_posix()] = admission.read(path, digest)
    final_logits = admission.read(args.reference / "target-logits.f32.bin", reference["final_logits"]["sha256"])
    if len(final_logits) != 4000 or payloads["host/graph_expected.bin"] != final_logits:
        raise ValueError("host expected logits differ from captured 1,000-word reference")
    kernel_receipts = []
    for index, (template, captured) in enumerate(zip(templates["kernels"], reference["kernels"], strict=True)):
        m, k, n = template["padded_mkn"]
        if (template.get("traffic_reuse") is not True or template["site"] != captured["site"]
                or template["padded_mkn"] != captured["padded_mkn"]
                or template["b_scale_exponent"] != captured["b_scale_exponent"]
                or captured["rescale_exponent"] != captured["a_scale_exponent"] + captured["b_scale_exponent"]):
            raise ValueError("captured launch geometry or scale differs")
        exponent = captured["rescale_exponent"]
        base = admission.read(template["program"]["path"], template["program"]["sha256"])
        retained_source = admission.read(template["source"]["path"], template["source"]["sha256"]).decode()
        if retained_source != assembly(m, k, n, template["output_offset"], 0, traffic_reuse=True):
            raise ValueError("template source differs from current public dispatch")
        source = assembly(m, k, n, template["output_offset"], exponent, traffic_reuse=True)
        raw = encode(source)
        patched = bytearray(base)
        struct.pack_into("<I", patched, template["rescale_patch_word_index"] * 4, ((exponent + 127) << 23) | 0x25f)
        if raw != bytes(patched):
            raise ValueError("fresh public program differs from template rescale patch")
        raw, _ = relocate_program(raw, template["rescale_patch_word_index"], target["arena_base"])
        source = arena.relocate_assembly(source, ATLAS_DRAM_BASE, target["arena_base"])
        if encode(source) != raw:
            raise ValueError("relocated source differs from relocated binary")
        validate_graph_program(raw)
        host_raw = payloads[f"host/graph_program_{index}.bin"]
        host_raw = bytearray(host_raw)
        _, host_patch = relocate_program(base, template["rescale_patch_word_index"], target["arena_base"])
        struct.pack_into("<I", host_raw, host_patch * 4, ((exponent + 127) << 23) | 0x25f)
        if bytes(host_raw) != raw:
            raise ValueError("host embedded program differs from launch fixture")
        activation = admission.read(captured["encoded_a_tiled"]["path"], captured["encoded_a_tiled"]["sha256"])
        weight = admission.read(template["weight"]["fp8_tiled"]["path"], template["weight"]["fp8_tiled"]["sha256"])
        expected = admission.read(captured["expected_physical_output"]["path"],
            captured["expected_physical_output"]["sha256"])
        if (len(activation) != m * k or len(weight) != k * n or len(expected) != m * n * 2
                or weight != payloads[f"host/graph_weight_{index}.bin"]):
            raise ValueError("captured launch payload extent or host weight differs")
        initial = activation + weight + b"\xa5" * 256 + b"\xcd" * len(expected) + b"\x5a" * 256
        if len(initial) != template["total_dram_bytes"]:
            raise ValueError("launch staging extent differs")
        prefix = f"kernels/{index:02d}/model/"
        artifacts = {"program.S": source.encode(), "program.bin": raw, "initial-dram.bin": initial,
            "expected-bf16-tiles.bin": expected}
        requested = dict(load_bytes=dma_load_bytes(m, k, n, traffic_reuse=True), store_bytes=len(expected))
        if requested != template["requested_dma"]:
            raise ValueError("template DMA accounting differs from public dispatch")
        receipt = dict(schema="atlas.resnet_kernel_preparation.v1", status="PREPARED_NOT_EXECUTED", index=index,
            site=template["site"], physical_mkn=[m, k, n], dram_base=target["arena_base"],
            output_offset=template["output_offset"], total_dram_bytes=len(initial), requested_dma=requested,
            rescale_exponent=exponent, source_inputs_sha256=source_pins,
            artifacts_sha256={name: sha(raw) for name, raw in artifacts.items()}, numerical_execution_performed=False)
        for name, raw in artifacts.items():
            payloads[prefix + name] = raw
        payloads[prefix + "receipt.json"] = (json.dumps(receipt, indent=2) + "\n").encode()
        kernel_receipts.append(dict(index=index, site=template["site"], shape=[m, k, n],
            raw_words=len(artifacts["program.bin"]) // 4, wrapped_words=len(artifacts["program.bin"]) // 4 + 5, requested_dma=requested))
    if (not admission.recheck() or model_binding(args.model_root) != model_before):
        raise ValueError("preparation source or payload changed")
    provenance = dict(schema="atlas.full_resnet_bundle.v3", status="PREPARED_NOT_EXECUTED", kernel_count=54,
        scope="Captured reference inputs and expected outputs for future live host execution",
        traffic_reuse=True, placement=target, model_binding=model_before, source_inputs_sha256=admission.inputs,
        host_elf_executed=False, numerical_execution_performed=False, full_soc_executed=False)
    payloads["provenance.json"] = (json.dumps(provenance, indent=2) + "\n").encode()
    args.out.mkdir(parents=True)
    for name, raw in payloads.items():
        path = checked(args.out / name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (args.out / "SHA256SUMS").write_text("".join(f"{sha(raw)}  {name}\n" for name, raw in sorted(payloads.items())))
    bundle = Bundle(args.out)
    bundle.verify_bookends()
    report = dict(status="PREPARED_NOT_EXECUTED", command=[sys.executable, *sys.argv], kernel_count=54, unique_shapes=len({tuple(t["shape"])
        for t in kernel_receipts}), kernels=kernel_receipts, source_bookends_equal=admission.recheck(),
        inputs_sha256=admission.inputs, bundle_checksums_sha256=bundle.checksums_sha256,
        host_elf_sha256=sha(elf), placement=target, all_1000_expected_logits_preserved=True,
        numerical_execution_performed=False)
    write_json(args.out.with_name(args.out.name + "-preparation.json"), report)
    print(json.dumps({key: report[key] for key in ("status", "kernel_count", "unique_shapes")}))


if __name__ == "__main__":
    main()
