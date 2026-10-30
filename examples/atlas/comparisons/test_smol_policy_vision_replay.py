"""Genuine product vision C graph with checked target projection injection.

No device execution: each callback first checks the actual C predecessor and
then injects the authenticated Atlas arithmetic projection reference.
"""

import ctypes as ct
import hashlib
import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import test_smol_policy_graph as graph
import torch
from source_relocation import SourceRelocation
from source_relocation import physical_path as safe


class Layer(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in ("norm1_weight", "norm1_bias", "norm2_weight", "norm2_bias")]


class Parameters(ct.Structure):
    _fields_ = [
        ("position", ct.c_void_p),
        ("layers", Layer * 12),
        ("post_weight", ct.c_void_p),
        ("post_bias", ct.c_void_p),
        ("gelu", ct.c_void_p),
        ("epsilon", ct.c_float),
    ]


FIELDS = (
    "tokens",
    "normalized",
    "query",
    "key",
    "value",
    "attended",
    "projected",
    "large",
    "activated",
    "shuffle",
    "output",
)


class Frame(ct.Structure):
    _fields_ = [(name, ct.c_void_p) for name in FIELDS]


class State(ct.Structure):
    _fields_ = [(name, ct.c_uint) for name in ("started", "projections", "complete", "error")]


Callback = ct.CFUNCTYPE(
    ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p, ct.c_size_t, ct.c_size_t, ct.c_void_p, ct.c_size_t
)


@unittest.skipUnless(
    all(os.environ.get(name) for name in ("SMOL_REPLAY", "SMOL_CHECKPOINT", "SMOL_ATTENTION_HEADERS")),
    "Optional target-policy vision fixtures are not configured",
)
class PolicyVisionReplayTest(unittest.TestCase):
    def test_actual_both_images_all_vision_predecessors_and_prefix_rows(self):
        torch.set_num_threads(1)
        root = safe(os.environ["SMOL_REPLAY"])
        receipt_path = safe(root / "receipt.json")
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        self.assertEqual(receipt["status"], "PASS_ATLAS_ARITHMETIC_FULL_POLICY_REPLAY")
        pins = {str(receipt_path): hashlib.sha256(receipt_raw).hexdigest()}
        torch_library = safe(Path(torch.__file__).parent / "lib/libtorch_cpu.so")
        with torch_library.open("rb") as stream:
            torch_digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.assertEqual(torch_digest, receipt["source"]["torch_cpu_library"]["sha256"])
        pins[str(torch_library)] = torch_digest
        pins[str(safe(torch.__file__))] = hashlib.sha256(safe(torch.__file__).read_bytes()).hexdigest()

        def read(path, expected=None):
            path = safe(path)
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if expected is not None:
                self.assertEqual(digest, expected)
            self.assertIn(pins.get(str(path), digest), (digest,))
            pins[str(path)] = digest
            return raw

        checkpoint = safe(os.environ["SMOL_CHECKPOINT"])
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.assertEqual(digest, receipt["source"]["weights"]["sha256"])
        pins[str(checkpoint)] = digest
        with checkpoint.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        arrays = []

        def weight(name):
            row = header["model.vlm_with_expert.vlm.model.vision_model." + name]
            self.assertEqual(row["dtype"], "BF16")
            begin, end = row["data_offsets"]
            with checkpoint.open("rb") as stream:
                stream.seek(8 + size + begin)
                raw = stream.read(end - begin)
            value = np.frombuffer(raw, "<u2").copy().reshape(row["shape"])
            arrays.append(value)
            return value.ctypes.data

        position = weight("embeddings.position_embedding.weight")
        layers = (Layer * 12)(
            *[
                Layer(
                    *[
                        weight(f"encoder.layers.{i}.layer_norm{n}.{field}")
                        for n in (1, 2)
                        for field in ("weight", "bias")
                    ]
                )
                for i in range(12)
            ]
        )
        post_weight, post_bias = weight("post_layernorm.weight"), weight("post_layernorm.bias")
        words = torch.arange(65536, dtype=torch.int32).to(torch.uint16)
        gelu = (
            torch.nn.functional.gelu(words.view(torch.bfloat16), approximate="tanh").view(torch.uint16).numpy().copy()
        )
        gelu[(np.arange(65536, dtype=np.uint16) & 0x7F80) == 0x7F80] = 0
        parameters = Parameters(position, layers, post_weight, post_bias, gelu.ctypes.data, 1e-6)
        here = safe(__file__).parent
        attention = safe(os.environ["SMOL_ATTENTION_HEADERS"])
        temporary = Path(tempfile.mkdtemp(prefix="policy-vision-"))
        library = temporary / "vision.so"
        sources = [here / (stem + ".c") for stem in ("smol_vision_graph", "smol_vision_ops", "smol_host_ops")]
        sources.append(attention / "smol_vision_attention.c")
        for path in [
            safe(__file__),
            *sources,
            *[
                here / (stem + ".h")
                for stem in ("smol_vision_graph", "smol_vision_ops", "smol_host_bits", "smol_host_ops")
            ],
            attention / "smol_vision_attention.h",
            attention / "smol_host_bits.h",
        ]:
            read(path)
        subprocess.run(
            [
                "/usr/bin/cc",
                "-O2",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-I" + str(attention),
                *map(str, sources),
                "-lm",
                "-o",
                str(library),
            ],
            check=True,
        )
        owner = ct.CDLL(str(library))
        function = owner.smol_vision_image
        function.argtypes = [
            ct.c_void_p,
            ct.POINTER(Parameters),
            ct.POINTER(Frame),
            ct.POINTER(State),
            Callback,
            ct.c_void_p,
        ]
        trajectory = receipt["trajectory"]
        read(trajectory["path"], trajectory["sha256"])
        with np.load(safe(trajectory["path"]), allow_pickle=False) as archive:
            prefix = archive["prefix_embeddings"].copy()

        def tensor(row):
            path = Path(row["file"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid reference member")
            self.assertEqual(row["dtype"], "torch.bfloat16")
            return np.frombuffer(read(root / path, row["sha256"]), "<u2").copy().reshape(row["shape"])

        roles = ("patch_embedding", "q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "connector")
        checked_images = []
        for image_index in range(2):
            calls = receipt["calls"][74 * image_index : 74 * (image_index + 1)]
            image = tensor(calls[0]["input"])
            buffers = {
                name: np.empty(
                    (64 * 960 if name == "output" else 1024 * 3072 if name in ("large", "activated") else 1024 * 768),
                    np.uint16,
                )
                for name in FIELDS
            }
            frame = Frame(*[buffers[name].ctypes.data for name in FIELDS])
            events, errors = [], []

            @Callback
            def inject(context, layer, role, address, rows, k, destination, n):
                try:
                    record = calls[len(events)]
                    self.assertTrue(
                        record["module"].endswith(roles[role])
                        or role == 7
                        and record["module"].endswith("modality_projection.proj")
                    )
                    if role not in (0, 7):
                        self.assertIn(f".layers.{layer}.", record["module"])
                    expected = tensor(record["input"])
                    if role == 0:
                        expected = expected.reshape(3, 32, 16, 32, 16).transpose(1, 3, 0, 2, 4)
                    expected = np.ascontiguousarray(expected.reshape(rows, k))
                    actual = np.ctypeslib.as_array(ct.cast(address, ct.POINTER(ct.c_uint16)), shape=(rows * k,))
                    mismatch = int(np.count_nonzero(actual.reshape(rows, k) != expected))
                    events.append({"call": record["index"], "mismatches": mismatch})
                    self.assertEqual(mismatch, 0, record["module"])
                    output = tensor(record["output"])
                    if role == 0:
                        output = output.reshape(768, 1024).transpose(1, 0)
                    output = np.ascontiguousarray(output.reshape(rows, n))
                    ct.memmove(destination, output.ctypes.data, output.nbytes)
                    return 0
                except Exception as error:
                    errors.append(str(error))
                    return 71

            state = State()
            code = function(image.ctypes.data, ct.byref(parameters), ct.byref(frame), ct.byref(state), inject, None)
            self.assertEqual((code, state.complete, state.projections, errors), (0, 1, 74, []), events)
            actual = (buffers["output"].astype(np.uint32) << 16).view(np.float32).reshape(64, 960)
            expected = prefix[0, 64 * image_index : 64 * (image_index + 1)]
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            print(
                json.dumps(
                    {"image": image_index, "actual_C_predecessors": 74, "prefix_words": 64 * 960, "mismatches": 0}
                ),
                flush=True,
            )
            checked_images.append(dict(image=image_index, predecessors=events, complete=True, prefix_exact=True))
        for path, digest in pins.items():
            with safe(path).open("rb") as stream:
                self.assertEqual(hashlib.file_digest(stream, "sha256").hexdigest(), digest, path)
        if os.environ.get("SMOL_VISION_REPLAY_REPORT"):
            safe(os.environ["SMOL_VISION_REPLAY_REPORT"]).write_text(
                json.dumps(
                    dict(
                        status="PASS_NATIVE_C_VISION_CAPTURED_REPLIES",
                        images=checked_images,
                        source_sha256=pins,
                        source_bookends_verified=True,
                        device_execution=False,
                        raw_adapter_included=False,
                    ),
                    indent=2,
                )
                + "\n"
            )


class PolicyGraphReplayTest(graph.PolicyGraphTest):
    """Authenticate relocation and reporting around the unchanged graph control."""

    __test__ = False

    def test_actual_prefix_and_all_ten_expert_euler_updates(self):
        root = safe(os.environ["SMOL_REPLAY"])
        receipt_path = safe(root / "receipt.json")
        receipt_raw = receipt_path.read_bytes()
        receipt = json.loads(receipt_raw)
        command = receipt["command"]
        original = command[command.index("--observation-input") + 1]
        observation = safe(os.environ.get("SMOL_OBSERVATION", original))
        relocation = SourceRelocation([original + "=" + str(observation)])
        metadata = SimpleNamespace(loads=lambda raw: relocation.apply(json.loads(raw)), dumps=json.dumps)
        pins = {str(receipt_path): hashlib.sha256(receipt_raw).hexdigest()}

        def pin(path, expected=None):
            path = safe(path)
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            self.assertEqual(digest, expected if expected is not None else digest, str(path))
            self.assertEqual(pins.get(str(path), digest), digest, str(path))
            pins[str(path)] = digest

        observation_digest = next(row["sha256"] for row in receipt["source_bookends"] if row["path"] == original)
        pin(observation, observation_digest)
        pin(safe(Path(torch.__file__).parent / "lib/libtorch_cpu.so"), receipt["source"]["torch_cpu_library"]["sha256"])
        stems = (
            "smol_policy_graph",
            "smol_policy_block",
            "smol_policy_ops",
            "smol_host_ops",
            "smol_host_controls",
            "smol_host_rope",
            "smol_host_attention",
            "smol_host_softmax",
        )
        here = safe(__file__).parent
        for path in [
            *[here / (stem + suffix) for stem in stems for suffix in (".c", ".h")],
            here / "smol_host_bits.h",
            safe(__file__),
            safe(graph.__file__),
            safe(graph.block.__file__),
            safe(torch.__file__),
        ]:
            pin(path)
        pins[str(safe(os.environ["SMOL_CHECKPOINT"]))] = receipt["source"]["weights"]["sha256"]
        pins[str(safe(receipt["trajectory"]["path"]))] = receipt["trajectory"]["sha256"]
        rope_path = safe(os.environ["SMOL_ROPE_CONTRACT"])
        pin(rope_path)
        rope = json.loads(rope_path.read_bytes())
        for name in ("cosine", "sine"):
            row = rope["assets"][name]["file"]
            pins[str(safe(row["path"]))] = row["sha256"]
        import lerobot.policies.smolvla.modeling_smolvla as native

        pin(safe(native.__file__))
        # Only parsed metadata changes; the baseline authenticates the original receipt bytes.
        with patch.object(graph, "json", metadata), patch.object(graph.block, "json", metadata):
            super().test_actual_prefix_and_all_ten_expert_euler_updates()
        for path, digest in tuple(pins.items()):
            pin(path, digest)
        calls = receipt["calls"][148:]
        self.assertEqual(len(calls), 1273)
        if os.environ.get("SMOL_POLICY_REPLAY_REPORT"):
            safe(os.environ["SMOL_POLICY_REPLAY_REPORT"]).write_text(
                json.dumps(
                    dict(
                        status="PASS_NATIVE_C_POLICY_CAPTURED_REPLIES",
                        predecessors=[dict(call=row["index"], module=row["module"], mismatches=0) for row in calls],
                        steps=10,
                        projections=1273,
                        all_prefix_cache_exact=True,
                        final_actions_exact=True,
                        source_sha256=pins,
                        source_bookends_verified=True,
                        observation_original_path=original,
                        observation_actual_path=str(observation),
                        device_execution=False,
                        raw_adapter_included=False,
                        vision_inputs_captured=True,
                    ),
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    unittest.main()
