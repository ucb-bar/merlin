"""Actual encoded program extents and early host-builder rejection."""

from pathlib import Path
import io
import json
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools/chipyard"))
import kernel_library  # noqa: E402,F401
import atlas_program
from atlas_host_build import GRAPH_PROGRAM_WORDS, validate_graph_program, wrap
from atlas_host_layout import ATLAS_IMEM_BYTES
import build_resnet_graph_host

def program(words):
    return struct.pack("<I", 0x13) * (words - 1) + struct.pack("<I", 0x73)


class ProgramBoundTest(unittest.TestCase):
    def test_all_public_dispatch_shapes_fit_hardware_and_graph(self):
        from npu_model.configs import isa_definition
        from npu_model.util.converter import input_to_program
        self.assertEqual(len(atlas_program._TRAFFIC_REUSE_MODULES), 58)
        for m, k, n in atlas_program._TRAFFIC_REUSE_MODULES:
            with self.subTest(shape=(m, k, n)):
                source = atlas_program.assembly(m, k, n, m*k+k*n+256, 0, traffic_reuse=True)
                words = input_to_program(io.StringIO(source)).assemble()
                raw = b"".join(struct.pack("<I", word & 0xffffffff) for word in words)
                self.assertEqual(validate_graph_program(raw, len(words)), len(words))
                self.assertLessEqual(len(wrap(raw)), ATLAS_IMEM_BYTES)

    def test_python_cap_matches_runtime_enum(self):
        header = (ROOT / "runtime/atlas_resnet_graph_device.h").read_text()
        cap, = re.findall(r"\bATLAS_GRAPH_PROGRAM_WORDS\s*=\s*(\d+)", header)
        self.assertEqual(GRAPH_PROGRAM_WORDS, int(cap))

    def test_actual_encoded_boundary_and_descriptor_lies(self):
        raw = program(GRAPH_PROGRAM_WORDS)
        self.assertEqual(validate_graph_program(raw), GRAPH_PROGRAM_WORDS)
        self.assertEqual(validate_graph_program(raw, GRAPH_PROGRAM_WORDS), GRAPH_PROGRAM_WORDS)
        self.assertEqual(len(wrap(raw)), len(raw) + 20)
        for declared in (0, GRAPH_PROGRAM_WORDS - 1, GRAPH_PROGRAM_WORDS + 1, None, True, float(GRAPH_PROGRAM_WORDS)):
            with self.subTest(declared=declared), self.assertRaises(ValueError):
                validate_graph_program(raw, declared)
        for declared in (GRAPH_PROGRAM_WORDS, GRAPH_PROGRAM_WORDS + 1):
            with self.subTest(oversize_declared=declared), self.assertRaises(ValueError):
                validate_graph_program(program(GRAPH_PROGRAM_WORDS + 1), declared)

    def test_structural_admission_is_preserved(self):
        for raw in (b"", b"x", struct.pack("<I", 0x13), struct.pack("<2I", 0x73, 0x73),
                    struct.pack("<2I", 0x00100073, 0x73), struct.pack("<2I", 0xc1001073, 0x73)):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validate_graph_program(raw)

    def test_full_host_rejects_bad_raw_or_descriptor_before_linking(self):
        with tempfile.TemporaryDirectory(prefix="atlas-full-program-bounds-") as directory:
            root = Path(directory)
            names = ("generated", "replay-gate", "reference", "templates", "rescale-table",
                     "atlas-root", "clang", "lld", "out")
            paths = {name: root / name for name in names}
            graphpath = paths["generated"] / "graph.json"
            receiptpath = paths["reference"] / "receipt.json"
            gate = {"status": "PASS", "source_bookends_equal": True,
                    "all_54_device_inputs_and_scales_exact": True, "final_1000_logit_words_exact": True,
                    "inputs_sha256": {str(graphpath): "graph", str(receiptpath): "reference"}}
            reference = {"nodes": [{"payload": {"path": str(root / "entry"), "sha256": "entry"}}],
                         "final_logits": {"sha256": "logits"}}
            template = {"base_rescale_exponent": 0, "program_words": 3,
                        "program": {"path": str(root / "program"), "sha256": "program"}}
            graph = {"input_sha256": {str(paths["templates"]): "templates"},
                     "generated_sha256": {}, "params_sha256": {},
                     "geometry_admission": ("all operators/shapes/arity/topology/54 contractions/"
                                            "groups 1 proven before generation"),
                     "kernels": [{"template": template, "kernel_index": i} for i in range(54)]}
            payloads = {paths["replay-gate"]: json.dumps(gate).encode(),
                        graphpath: json.dumps(graph).encode(), receiptpath: json.dumps(reference).encode(),
                        paths["templates"]: json.dumps({"kernels": [template] * 54,
                                                      "rescale_table_sha256": "scale"}).encode(),
                        root / "entry": bytes(3 * 224 * 224 * 4),
                        paths["reference"] / "target-logits.f32.bin": bytes(4000),
                        paths["rescale-table"]: struct.pack("<254I", *[((e + 127) << 23) | 0x25F
                                                                     for e in range(-126, 128)]),
                        root / "program": program(4)}
            argv = ["build_resnet_graph_host.py", *(arg for name in names
                     for arg in ("--" + name, str(paths[name])))]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(build_resnet_graph_host, "Admission") as admission, \
                 mock.patch.object(build_resnet_graph_host.subprocess, "run") as linker:
                admission.return_value.read.side_effect = lambda path, *args: payloads.get(Path(path), b"")
                with self.assertRaises(ValueError):
                    build_resnet_graph_host.main()
                linker.assert_not_called()
                self.assertFalse(paths["out"].exists())



if __name__ == "__main__":
    unittest.main()
