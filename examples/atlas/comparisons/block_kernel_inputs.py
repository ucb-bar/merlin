"""Bind an explicit source-only provider and regenerate captured kernel inputs."""

import importlib
import json
import sys
import zipfile

import numpy as np

from atlas_provider_files import PACKING_FILE, PACKING_FILES, TILE_FILE, packing_paths


def admit_manifest(manifest, source, source_dtype):
    if manifest["source_dtype"] != source_dtype or manifest["npz_sha256"] != source["sha256"]:
        raise ValueError("Pinned manifest dtype or NPZ binding differs")


def regenerate(build, recipe):
    args = build.args
    names = PACKING_FILES
    for path in packing_paths(args.atlas_tools):
        build.pin(path)
    sys.path.insert(0, str(args.atlas_tools))
    for name in names:
        module = importlib.import_module(name[:-3])
        if build.pin(module.__file__) != build.pin(args.atlas_tools / name):
            raise ValueError("Provider helper imported from a different source: " + name)
    provider = importlib.import_module(PACKING_FILE[:-3])
    tile = importlib.import_module(TILE_FILE[:-3])
    _, _, assemble, _ = tile.configure_model(args.model_root)
    before = tile.model_binding(args.model_root)
    kernels, checks = [], []
    for index, original in enumerate(recipe["projections"]):
        source = original["source_npz"]
        path = build.pin(source["path"], source["sha256"])
        tile.preflight_archive(path)
        with zipfile.ZipFile(path) as archive:
            tile.validate_archive_members(archive.infolist())
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        manifest_record = original["source_manifest"]
        manifest = json.loads(build.pin(manifest_record["path"], manifest_record["sha256"]).read_text())
        admit_manifest(manifest, source, original["source_dtype"])
        prepared = provider.prepare(arrays, original["source_dtype"], assemble)
        if recipe["workload"] == "tiny":
            shape = [original[n] for n in ("physical_rows", "k", "n")]
            if prepared["memory"] != original["memory"]:
                raise ValueError("Prepared physical memory differs")
            if prepared["conversion"]["A"]["scale_exponent"] != original["a_scale"]:
                raise ValueError("Current activation scale differs")
        else:
            shape = [original[n] for n in ("physical_rows", "input_width", "output_width")]
            memory = prepared["memory"]
            if (
                memory["output_offset"] != original["output_offset"]
                or memory["output_bytes"] != original["output_bytes"]
                or memory["total_dram_bytes"] != original["dram_bytes"]
                or len(prepared["activation"]) != original["activation_bytes"]
                or len(prepared["weights"]) != original["weight_bytes"]
                or len(prepared["program"]) != original["program_words"] * 4
                or prepared["conversion"]["A"]["scale_exponent"] != original["activation_scale"]
                or prepared["conversion"]["B"]["scale_exponent"] != original["weight_scale"]
            ):
                raise ValueError("Prepared selected-block descriptor differs")
        if (
            prepared["shape"] != shape
            or prepared["rescale_exponent"] != original["rescale_exponent"]
            or prepared["requested_dma"] != original["requested_dma"]
        ):
            raise ValueError("Current shape/rescale/DMA metadata differs")
        generated = dict(original)
        outputs = {}
        for key, name in (
            ("program", "program.bin"),
            ("source", "program.S"),
            ("initial", "initial-dram.bin"),
            ("activation", "activation.bin"),
            ("weights", "weights.bin"),
        ):
            value = prepared[key]
            if isinstance(value, str):
                value = value.encode()
            reference_key = "program_source" if key == "source" else key
            if value != build.data(original[reference_key]):
                raise ValueError("Regenerated " + key + " differs at projection " + str(index))
            path = build.put("kernels/" + str(index) + "/" + name, value)
            outputs[reference_key] = {"path": str(path), "sha256": build.outputs["kernels/" + str(index) + "/" + name]}
            generated[reference_key] = outputs[reference_key]
        kernels.append(generated)
        checks.append(
            {
                "index": index,
                "shape": shape,
                "source_npz": source,
                "exact_program_source": True,
                "exact_program": True,
                "exact_initial": True,
                "exact_activation": True,
                "exact_weights": True,
                "outputs": outputs,
                "resource_plan": prepared["plan"],
                "conversion": prepared["conversion"],
                "requested_dma": prepared["requested_dma"],
            }
        )
    if tile.model_binding(args.model_root) != before:
        raise ValueError("ISA or imported assembler source changed during compilation")
    return kernels, {
        "source_only": True,
        "Core_instances": 0,
        "Core_ticks": 0,
        "model_binding": before,
        "projections": checks,
    }
