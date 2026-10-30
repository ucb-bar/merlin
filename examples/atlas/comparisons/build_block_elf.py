"""Regenerate captured block inputs and build an unexecuted RV64GC host ELF.

The recipe contains external tensors, weights, configuration and comparison
hashes. Host C comes from this checkout; device ABI and programs come from the
explicit companion provider. No device model, RTL, QEMU or workload is executed.
"""

from merlin.common.digest import sha256_bytes
from merlin.common.jsonio import write_pretty_json

import argparse
import ctypes
import hashlib
import json
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import time

import numpy as np

from capture_tinyllama_native import identity, safe_path, artifact_output, admit_archive_members
from workload_replay_admission import admit_executable
from atlas_provider_files import LAYOUT_HEADER, RUNTIME_FILES

SOURCE = Path(__file__).absolute().parent
RUNTIME = SOURCE / "block_runtime"


def admit_block_data(workload, data):
    """Require the fixed target harness extents before embedding any array."""
    if workload == "tiny":
        sizes = {
            "step0-entry": 128 * 2048 * 2,
            "step1-entry": 2048 * 2,
            "step0-expected": 128 * 2048 * 2,
            "step1-expected": 2048 * 2,
            "input-norm": 2048 * 2,
            "post-norm": 2048 * 2,
            "cosine": 160 * 64 * 2,
            "sine": 160 * 64 * 2,
            "step0-keys": 4 * 128 * 64 * 2,
            "step0-values": 4 * 128 * 64 * 2,
            "step1-keys": 4 * 129 * 64 * 2,
            "step1-values": 4 * 129 * 64 * 2,
        }
    elif workload == "smol":
        sizes = {
            "input": 177 * 960 * 2,
            "pad": 177,
            "attention_boundaries": 177,
            "input_norm_weight": 960 * 2,
            "post_norm_weight": 960 * 2,
            "expected_output": 177 * 960 * 2,
            "expected_key_cache": 177 * 320 * 2,
            "expected_value_cache": 177 * 320 * 2,
        }
        if not data["cosine"] or len(data["cosine"]) != len(data["sine"]) or len(data["cosine"]) % 128:
            raise ValueError("Smol rotary table extents differ")
    else:
        raise ValueError("Unsupported fixed block")
    for name, expected in sizes.items():
        if len(data[name]) != expected:
            raise ValueError("Fixed block data extent differs: " + name)


class Build:
    def __init__(self, args):
        self.args = args
        self.pins = {}
        self.commands = []
        self.outputs = {}

    def pin(self, path, expected=None):
        path = Path(path)
        if not path.is_absolute():
            path = self.args.recipe.parent / path
        item = identity(path, expected)
        previous = self.pins.setdefault(item["path"], item["sha256"])
        if previous != item["sha256"]:
            raise ValueError("Source changed during build: " + item["path"])
        return safe_path(item["path"])

    def data(self, record):
        path = self.pin(record["path"], record["sha256"])
        if "array" not in record:
            return path.read_bytes()
        import zipfile

        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                admit_archive_members(archive.infolist())
            with np.load(path, allow_pickle=False) as arrays:
                array = arrays[record["array"]].copy()
        elif path.suffix == ".npy" and record["array"] is None:
            array = np.load(path, allow_pickle=False)
        else:
            raise ValueError("Array input requires an admitted NPZ or NPY")
        if array.dtype not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32")):
            raise ValueError("Expected uint8, BF16 uint16, or FP32 input")
        return array.astype(array.dtype.newbyteorder("<")).tobytes()

    def put(self, relative, value):
        path = safe_path(self.args.out / relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value.encode() if isinstance(value, str) else value)
        self.outputs[relative] = identity(path)["sha256"]
        return path

    def copy(self, source, relative):
        return self.put(relative, self.pin(source).read_bytes())

    def run(self, command, label, cwd=None):
        start = time.monotonic()
        result = subprocess.run(command, cwd=cwd or self.args.out, capture_output=True, text=True, timeout=300)
        text = result.stdout + result.stderr
        if len(text) > 8 * 1024 * 1024:
            raise ValueError("Build command log cap")
        log = self.put("logs/" + label + ".log", text)
        self.commands.append(
            {
                "command": command,
                "cwd": str(cwd or self.args.out),
                "returncode": result.returncode,
                "elapsed_seconds": time.monotonic() - start,
                "log_sha256": identity(log)["sha256"],
            }
        )
        if result.returncode:
            raise ValueError("Build failed at " + label)
        return result.stdout

    def numeric_support(self, relative):
        args = self.args
        libm = self.pin(args.libm, args.libm_sha256)
        native = ctypes.CDLL(str(libm))
        native.fegetround.restype = ctypes.c_int
        if native.fegetround() != 0:
            raise ValueError("Parameter generation requires FE_TONEAREST")
        helper = self.pin(RUNTIME / "native_parameters.c")
        shared = args.out / "native-parameters.so"
        cc = admit_executable(shutil.which(args.native_cc) or args.native_cc)
        self.pin(cc["physical_binary"]["path"], cc["physical_binary"]["sha256"])
        self.run(
            [
                cc["entry"],
                "-std=c11",
                "-O2",
                "-fPIC",
                "-shared",
                "-ffreestanding",
                "-fno-builtin",
                "-ffp-contract=off",
                "-fno-fast-math",
                str(helper),
                "-o",
                str(shared),
            ],
            "native-parameters",
        )
        producer = ctypes.CDLL(str(shared))
        producer.exp_bits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
        producer.pow_bits.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint32)]
        table = (ctypes.c_uint32 * 65536)()
        producer.exp_bits(ctypes.cast(native.expf, ctypes.c_void_p), table)
        doubles, singles = (ctypes.c_uint64 * 2)(), (ctypes.c_uint32 * 2)()
        producer.pow_bits(ctypes.cast(native.pow, ctypes.c_void_p), doubles, singles)
        if native.fegetround() != 0 or tuple(doubles) != (0x3FC0000000000000, 0x3FB6A09E667F3BCD):
            raise ValueError("Native mathematical parameter policy differs")
        text = "\n".join(",".join("0x%08xu" % table[i + j] for j in range(8)) + "," for i in range(0, 65536, 8)) + "\n"
        path = self.put(relative + "expf_bf16_values.inc", text)
        self.pin(path, args.expf_table_sha256)
        for name in ("rv64_support.c", "include/math.h", "include/string.h", "include/fenv.h"):
            self.copy(RUNTIME / name, relative + name)
        return {
            "expf_table_sha256": identity(path)["sha256"],
            "libm": identity(libm),
            "native_compiler": cc,
            "pow_double_bits": list(doubles),
            "scope": "65536 raw BF16 encodings passed through C expf function pointer; "
            "preserves NaN payload bits; errno/exception flags are not emulated",
        }

    def smol_assets(self, recipe, kernels):
        data = recipe["data"]
        names = [
            "input",
            "pad",
            "attention_boundaries",
            "input_norm_weight",
            "post_norm_weight",
            "expected_output",
            "expected_key_cache",
            "expected_value_cache",
            "cosine",
            "sine",
        ]
        raw_data = {name: self.data(data[name]) for name in names}
        admit_block_data("smol", raw_data)
        blobs = {}
        for name in names:
            raw = raw_data[name]
            self.put("data/" + name + ".bin", raw)
            blobs[name] = len(raw)
        header = [
            "#ifndef SMOL_SELECTED_ASSETS_H",
            "#define SMOL_SELECTED_ASSETS_H",
            '#include "smol_block_raw_adapter.h"',
        ]
        assembly = ['.section .rodata.smol_assets,"a",@progbits']
        for i, kernel in enumerate(kernels):
            for key, suffix in (("program", "program"), ("weights", "weights")):
                name = suffix + "_" + str(i)
                raw = self.data(kernel[key])
                self.put("data/" + name + ".bin", raw)
                blobs[name] = len(raw)
            if blobs["program_" + str(i)] != kernel["program_words"] * 4:
                raise ValueError("Program extent differs")
            if blobs["weights_" + str(i)] != kernel["weight_bytes"]:
                raise ValueError("Weight extent differs")
        for name in blobs:
            header.append("extern const unsigned char smol_asset_" + name + "[];")
            assembly.extend(
                [
                    ".balign 16",
                    ".global smol_asset_" + name,
                    "smol_asset_" + name + ":",
                    '.incbin "data/' + name + '.bin"',
                ]
            )
        header.extend(
            [
                "extern const smol_raw_kernel smol_kernels[7];",
                "extern const smol_block_parameters smol_parameters;",
                "#endif",
            ]
        )
        self.put("source/smol_selected_assets.h", "\n".join(header) + "\n")
        self.put("source/images.S", "\n".join(assembly) + '\n.section .note.GNU-stack,"",@progbits\n')
        c = ['#include "smol_selected_assets.h"', "const smol_raw_kernel smol_kernels[7]={"]
        for i, kernel in enumerate(kernels):
            values = ["(const uint32_t *)smol_asset_program_" + str(i), "smol_asset_weights_" + str(i)]
            values += [
                str(kernel[n])
                for n in (
                    "program_words",
                    "rescale_word",
                    "input_width",
                    "output_width",
                    "activation_bytes",
                    "weight_bytes",
                    "output_bytes",
                    "output_offset",
                    "dram_bytes",
                    "activation_scale",
                    "weight_scale",
                )
            ]
            c.append("{" + ",".join(values) + "},")
        parameters = recipe["parameters"]
        c.extend(
            [
                "};",
                "const smol_block_parameters smol_parameters={",
                "(const uint16_t *)smol_asset_input_norm_weight,(const uint16_t *)smol_asset_post_norm_weight,"
                + float(parameters["input_epsilon"]).hex()
                + "f,"
                + float(parameters["post_epsilon"]).hex()
                + "f,",
                "(const float *)smol_asset_cosine,(const float *)smol_asset_sine,"
                + str(parameters["first_position"])
                + ","
                + str(blobs["cosine"] // 128),
                "};",
            ]
        )
        self.put("source/smol_selected_assets.c", "\n".join(c) + "\n")

    def tiny_assets(self, recipe, panels):
        data = {name: self.data(record) for name, record in recipe["data"].items()}
        admit_block_data("tiny", data)
        symbols, assembly, table, expected = {}, [], [], []

        def blob(raw, kind):
            digest = sha256_bytes(raw)
            key = kind + "_" + digest[:24]
            if key not in symbols:
                name = "assets/" + key + ".bin"
                self.put(name, raw)
                symbols[key] = {"file": name, "sha256": digest, "bytes": len(raw), "kind": kind}
                assembly.extend(
                    [
                        '.section .rodata,"a",@progbits',
                        ".balign 8",
                        ".global " + key,
                        key + ":",
                        '.incbin "' + name + '"',
                    ]
                )
            if symbols[key]["sha256"] != digest:
                raise ValueError("Asset digest prefix collision")
            return key

        for panel in panels:
            weight = self.data(panel["weights"])
            program = self.data(panel["program"])
            reply = self.data(panel["expected_output"])
            m, k, n = panel["physical_rows"], panel["k"], panel["n"]
            if len(weight) != k * n or len(reply) != m * n * 2 or len(program) % 4 or program[-4:] != b"\x73\0\0\0":
                raise ValueError("Tiny projection data extents differ")
            weights, code = blob(weight, "weight"), blob(program, "program")
            expected.append(blob(reply, "expected_panel"))
            memory, dma = panel["memory"], panel["requested_dma"]
            values = [
                panel[name] for name in ("rows", "physical_rows", "k", "n", "column", "width", "logical", "a_scale")
            ]
            row = "{" + ",".join(map(str, values)) + ", " + weights + ", " + str(k * n)
            row += ", (const uint32_t*)" + code + ", " + str(len(program) // 4) + ", "
            row += (
                ",".join(
                    map(
                        str,
                        [memory["output_offset"], memory["total_dram_bytes"], dma["load_bytes"], dma["store_bytes"]],
                    )
                )
                + "}"
            )
            table.append(row)
        external = {}
        for name in (
            "step0-entry",
            "step1-entry",
            "step0-expected",
            "step1-expected",
            "input-norm",
            "post-norm",
            "cosine",
            "sine",
        ):
            kind = (
                "parameter"
                if name in ("input-norm", "post-norm", "cosine", "sine")
                else ("entry" if "entry" in name else "expected_block")
            )
            external[name] = blob(data[name], kind)
        for phase in (0, 1):
            for kind in ("keys", "values"):
                name = "step" + str(phase) + "-" + kind
                external[name] = blob(data[name], "expected_cache")
        header = """#ifndef TINY_BUNDLE_ASSETS_H
#define TINY_BUNDLE_ASSETS_H
#include "tiny_projection.h"
extern const tiny_panel tiny_panels[24];
extern const uint8_t *const tiny_panel_expected[24];
extern const uint16_t *const tiny_entry[2],*const tiny_expected_output[2];
extern const uint16_t *const tiny_expected_keys[2],*const tiny_expected_values[2];
extern const uint16_t *const tiny_input_norm,*const tiny_post_norm,*const tiny_cosine,*const tiny_sine;
extern const float tiny_epsilon;
#endif
"""
        declarations = ["extern const uint8_t " + key + "[];" for key in symbols]
        text = '#include "tiny_bundle_assets.h"\n' + "\n".join(declarations)
        text += "\nconst tiny_panel tiny_panels[24]={\n" + ",\n".join(table) + "\n};\n"
        text += "const uint8_t *const tiny_panel_expected[24]={" + ",".join(expected) + "};\n"
        for name, keys in (
            ("tiny_entry", ("step0-entry", "step1-entry")),
            ("tiny_expected_output", ("step0-expected", "step1-expected")),
            ("tiny_expected_keys", ("step0-keys", "step1-keys")),
            ("tiny_expected_values", ("step0-values", "step1-values")),
        ):
            text += "const uint16_t *const " + name + "[2]={"
            text += ",".join("(const uint16_t*)" + external[k] for k in keys) + "};\n"
        for name, key in (
            ("tiny_input_norm", "input-norm"),
            ("tiny_post_norm", "post-norm"),
            ("tiny_cosine", "cosine"),
            ("tiny_sine", "sine"),
        ):
            text += "const uint16_t *const " + name + "=(const uint16_t*)" + external[key] + ";\n"
        text += "const float tiny_epsilon=" + float(recipe["parameters"]["epsilon"]).hex() + "f;\n"
        self.put("tiny_bundle_assets.h", header)
        self.put("tiny_bundle_assets.c", text)
        self.put("images.S", "\n".join(assembly) + '\n.section .note.GNU-stack,"",@progbits\n')
        return symbols

    def sources(self, workload):
        args = self.args
        shared = ["host_matrix_fp8pack.c", "host_matrix_fp8pack.h", "smol_host_bits.h", "smol_host_sleef_LICENSE.txt"]
        stems = ["ops", "softmax"] if workload == "tiny" else ["ops", "rope", "softmax", "attention", "controls"]
        for stem in stems:
            shared += ["smol_host_" + stem + ".c", "smol_host_" + stem + ".h"]
        if workload == "tiny":
            for stem in (
                "tiny_block",
                "llama_host_attention",
                "llama_host_layout",
                "llama_rotary_table",
                "llama_host_bmm",
            ):
                shared += [stem + ".c", stem + ".h"]
            launch = ["block_target.c"]
            prefix = ""
        else:
            shared += ["smol_vlm_block_prototype.c", "smol_vlm_block_prototype.h"]
            launch = ["smol_selected_workload.c", "smol_selected_workload.h", "smol_selected_elf.c"]
            prefix = "source/"
        provider = list(RUNTIME_FILES[workload])
        for name in shared:
            self.copy(SOURCE / name, prefix + name)
        # The layout header is new in the companion provider. Older providers
        # remain usable while sources that include it require its full closure.
        layout = safe_path(args.provider_runtime / LAYOUT_HEADER)
        if layout.exists():
            provider.append(LAYOUT_HEADER)
        self.provider_sources(provider, prefix, local_headers=shared)
        for name in launch:
            self.copy(RUNTIME / name, prefix + name)
        return prefix

    def provider_sources(self, names, prefix, *, local_headers=()):
        """Snapshot the explicit provider sources and their quoted Atlas headers."""
        pending = list(names)
        copied = set()
        while pending:
            name = pending.pop(0)
            if name in copied:
                continue
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Provider source must remain beneath its runtime root")
            source = self.pin(self.args.provider_runtime / relative)
            raw = source.read_bytes()
            self.put(prefix + name, raw)
            copied.add(name)
            for header in re.findall(r'^\s*#include\s+"([^"\n]+)"', raw.decode(), re.M):
                if header in local_headers:
                    continue
                # Merlin and generated headers are supplied separately. Atlas
                # runtime headers, including layout dependencies, belong here.
                if header.endswith(".h") and (relative.suffix == ".h" or Path(header).name.startswith("atlas_")):
                    pending.append(str(relative.parent / header))

    def compile(self, workload):
        args = self.args
        compiler = admit_executable(shutil.which(args.clang) or args.clang)
        linker = admit_executable(shutil.which(args.linker) or args.linker)
        for entry in (compiler, linker):
            self.pin(entry["physical_binary"]["path"], entry["physical_binary"]["sha256"])
        common = [
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-ffp-contract=off",
            "-fno-fast-math",
            "-ffreestanding",
            "-fno-builtin",
        ]
        if workload == "tiny":
            resource = (
                safe_path(self.run([compiler["entry"], "-print-resource-dir"], "resource-dir").strip()) / "include"
            )
            flags = common + ["-nostdinc", "-isystem", str(resource), "-I", "include", "-I", "."]
            names = [
                "tiny_block.c",
                "tiny_projection.c",
                "llama_host_attention.c",
                "llama_host_layout.c",
                "llama_rotary_table.c",
                "llama_host_bmm.c",
                "host_matrix_fp8pack.c",
                "smol_host_ops.c",
                "smol_host_softmax.c",
                "rv64_support.c",
                "atlas_host.c",
                "block_target.c",
                "tiny_bundle_assets.c",
                "images.S",
                "start.S",
            ]
            objects = []
            for i, name in enumerate(names):
                obj = "object-" + str(i) + ".o"
                objects.append(obj)
                selected_flags = flags + (["-DSMOL_HOST_EXCLUDE_MIXED_RESIDUAL"] if name == "smol_host_ops.c" else [])
                self.run(
                    [compiler["entry"], *selected_flags, "-M", "-MF", obj + ".before.d", name], "dependencies-" + str(i)
                )
                before = self.dependencies(args.out / (obj + ".before.d"))
                self.run(
                    [compiler["entry"], *selected_flags, "-MD", "-MF", obj + ".d", "-c", name, "-o", obj],
                    "compile-" + str(i),
                )
                if self.dependencies(args.out / (obj + ".d")) != before:
                    raise ValueError("Compiler dependency set changed during compilation")
            self.run(
                [
                    linker["entry"],
                    "-flavor",
                    "gnu",
                    "-m",
                    "elf64lriscv",
                    "-static",
                    "--no-undefined",
                    "-T",
                    "link.ld",
                    "-Map",
                    "host.map",
                    *objects,
                    "-o",
                    "host.elf",
                ],
                "link",
            )
            elf = args.out / "host.elf"
        else:
            folder = safe_path(args.out / "elf")
            flags = common + [
                "-fno-stack-protector",
                "-fno-pic",
                "-fno-vectorize",
                "-fno-slp-vectorize",
                "-I" + str(folder / "support/include"),
                "-I" + str(args.out / "source"),
            ]
            names = [
                "atlas_host",
                "smol_vlm_block_prototype",
                "smol_block_raw_adapter",
                "host_matrix_fp8pack",
                "smol_host_ops",
                "smol_host_rope",
                "smol_host_softmax",
                "smol_host_attention",
                "smol_host_controls",
                "smol_selected_assets",
                "smol_selected_workload",
                "smol_selected_elf",
            ]
            objects = []
            sources = [(args.out / "source" / (name + ".c"), name) for name in names]
            sources += [(folder / "support/rv64_support.c", "rv64_support")]
            for source, name in sources:
                obj = folder / (name + ".o")
                objects.append(str(obj))
                self.run(
                    [compiler["entry"], *flags, "-M", "-MF", str(obj.with_suffix(".before.d")), str(source)],
                    "dependencies-" + name,
                )
                before = self.dependencies(obj.with_suffix(".before.d"))
                self.run(
                    [
                        compiler["entry"],
                        *flags,
                        "-MD",
                        "-MF",
                        str(obj.with_suffix(".d")),
                        "-c",
                        str(source),
                        "-o",
                        str(obj),
                    ],
                    "compile-" + name,
                )
                if self.dependencies(obj.with_suffix(".d")) != before:
                    raise ValueError("Compiler dependency set changed during compilation")
            for source, name in ((folder / "start.S", "start"), (args.out / "source/images.S", "images")):
                obj = folder / (name + ".o")
                objects.append(str(obj))
                self.run([compiler["entry"], *flags, "-c", str(source), "-o", str(obj)], "compile-" + name)
            elf = folder / "host.elf"
            self.run(
                [
                    compiler["entry"],
                    "--target=riscv64-unknown-elf",
                    "-march=rv64gc",
                    "-mabi=lp64d",
                    "-mcmodel=medany",
                    "-nostdlib",
                    "--ld-path=" + linker["entry"],
                    *objects,
                    "-Wl,-T," + str(folder / "link.ld"),
                    "-Wl,--no-relax",
                    "-Wl,-Map," + str(folder / "host.map"),
                    "-o",
                    str(elf),
                ],
                "link",
            )
        item = identity(elf, args.expected_elf_sha256)
        return {"elf": item, "compiler": compiler, "linker": linker}

    def dependencies(self, depfile):
        words = safe_path(depfile).read_text().replace("\\\n", " ").split(":", 1)[1].split()
        paths = set()
        for value in words:
            path = Path(value)
            paths.add(self.pin(path if path.is_absolute() else self.args.out / path))
        return paths

    def build(self):
        args = self.args
        self.pin(__file__)
        self.pin(SOURCE / "capture_tinyllama_native.py")
        self.pin(SOURCE / "workload_replay_admission.py")
        self.pin(SOURCE / "atlas_provider_files.py")
        recipe = json.loads(self.pin(args.recipe, args.recipe_sha256).read_text())
        workload = recipe["workload"]
        if workload not in ("tiny", "smol"):
            raise ValueError("Only the captured first Tiny block or selected Smol block is supported")
        if len(recipe["projections"]) != (24 if workload == "tiny" else 7):
            raise ValueError("Projection count differs from bounded example")
        from block_kernel_inputs import regenerate

        self.pin(SOURCE / "block_kernel_inputs.py")
        kernels, kernel_checks = regenerate(self, recipe)
        prefix = self.sources(workload)
        folder = "elf/" if workload == "smol" else ""
        self.copy(args.startup, folder + "start.S")
        self.copy(args.linker_script, folder + "link.ld")
        support = self.numeric_support(folder + "support/" if workload == "smol" else "")
        if workload == "smol":
            self.smol_assets(recipe, kernels)
            symbols = None
        else:
            symbols = self.tiny_assets(recipe, kernels)
        for name, expected in recipe.get("expected_generated_sha256", {}).items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Generated comparison paths must remain relative")
            self.pin(args.out / name, expected)
        linked = self.compile(workload)
        for path, expected in tuple(self.pins.items()):
            identity(path, expected)
        for name, expected in self.outputs.items():
            identity(args.out / name, expected)
        return {
            "status": "built_unexecuted_RV64GC_block_ELF",
            "workload": workload,
            "source_bookends_unchanged": True,
            "kernel_regeneration": kernel_checks,
            "numeric_support": support,
            "assets": symbols,
            **linked,
            "ELF_executed": False,
            "model_RTL_or_QEMU_executed": False,
            "scope": "Captured block only; no full inference/policy, platform or timing qualification",
            "quality": "Integrated task-quality accuracy not yet measured",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "provider-runtime", "atlas-tools", "model-root", "startup", "linker-script", "libm"):
        parser.add_argument("--" + name, type=safe_path, required=True)
    for name in ("recipe-sha256", "libm-sha256", "expf-table-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("clang", "linker"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--native-cc", default=shutil.which("cc"))
    parser.add_argument("--expected-elf-sha256")
    parser.add_argument("--out", type=artifact_output, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    build = Build(args)
    result = {
        "status": "failed_build_only",
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "ELF_executed": False,
    }
    code = 1
    try:
        result.update(build.build())
        code = 0
    except Exception as error:
        result.update(
            error={"type": type(error).__name__, "message": str(error)},
            next_action="Preserve failed build and repair its first boundary; do not execute or relabel",
        )
    result.update(source_inputs_sha256=build.pins, commands=build.commands, outputs_sha256=build.outputs)
    write_pretty_json(args.out / "build.json", result)
    print(json.dumps({"status": result["status"], "error": result.get("error"), "elf": result.get("elf")}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
