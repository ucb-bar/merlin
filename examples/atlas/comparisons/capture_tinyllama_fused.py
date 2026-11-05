"""Capture Tiny projections and fused pointwise outputs on the original RTL model."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path
from types import MethodType

import numpy as np
import tiny_workload_native as native
from capture_tinyllama_full import capture, padded_words
from capture_tinyllama_native import array_identity, identity, safe_path
from compare_tinyllama_attention import raw_array
from tiny_fp8_pack import encode
from tiny_workload_inventory import column_shards

MODEL_PIN = "47b8599d8ffbe5ddd96059ffe2b5be2427a410f5"
SCALE_CONTRACT_SHA = "a34fa90b3dac3a6e131c8f9d0bdf8da57978e2a3d76af1325d02af197e46b938"


def load_layout(path, sha256, records):
    record = identity(path, sha256)
    spec = importlib.util.spec_from_file_location("tiny_fused_layout", record["path"])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records.append(record)
    return module


def down_scales(path, records):
    record = identity(path, SCALE_CONTRACT_SHA)
    contract = json.loads(Path(record["path"]).read_bytes())
    result = {}
    for item in contract["projection_kernels"]:
        if item["site"].endswith(".mlp.down_proj"):
            key = (item["step"], item["site"])
            value = item["A_scale"]
            if type(value) is not int or not -126 <= value <= 127 or result.get(key, value) != value:
                raise ValueError("fixed down consumer scales disagree")
            result[key] = value
    if len(result) != 66:
        raise ValueError("require all 22 layers and three fixed down consumer scales")
    records.append(record)
    return result


def assemble_fp8(panels, read, m, width):
    """Copy only declared written spans, rejecting holes, overlap, and omissions."""
    result, written = bytearray(m * width), bytearray(m * width)
    for panel in panels:
        launch = panel["launch"]
        source = read(launch["output_tiled"])
        for span in launch["written_spans"]:
            start, extent = span["offset"], span["bytes"]
            end = start + extent
            if start < 0 or end > len(result) or end > len(source) or extent <= 0 or start % 1024 or extent % 1024:
                raise ValueError("FP8 written span outside consumer tiles")
            if any(written[start:end]):
                raise ValueError("overlapping FP8 producer spans")
            result[start:end] = source[start:end]
            written[start:end] = b"\1" * extent
    if not all(written):
        raise ValueError("FP8 producer spans do not cover down A")
    return bytes(result)


class ProjectionWorkers:
    """Three isolated persistent model interpreters, with deterministic replies."""

    def __init__(self, python, script, records, count=3):
        if type(count) is not int or not 1 <= count <= 3:
            raise ValueError("worker count must be one through three")
        records.append(identity(script))
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.update(PYTHONDONTWRITEBYTECODE="1", NPU_MODEL_RTL_NATIVE_MXU0="0")
        self.workers = []
        try:
            for _ in range(count):
                self.workers.append(
                    subprocess.Popen(
                        [str(python), "-B", str(script)],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                        env=env,
                    )
                )
        except BaseException:
            self.close()
            raise

    def run(self, requests):
        result = []
        for start in range(0, len(requests), len(self.workers)):
            batch = requests[start : start + len(self.workers)]
            for worker, request in zip(self.workers, batch):
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
            for worker in self.workers[: len(batch)]:
                line = worker.stdout.readline()
                if not line:
                    raise RuntimeError("projection worker terminated without a result")
                answer = json.loads(line)
                if answer.get("status") != "ok":
                    raise RuntimeError("projection worker failed: " + answer.get("error", "unknown"))
                result.append(answer["record"])
        return result

    def close(self):
        error = None
        for worker in self.workers:
            try:
                if worker.poll() is None:
                    reply, _ = worker.communicate(json.dumps(dict(command="close")) + "\n", timeout=30)
                    if not reply or json.loads(reply).get("status") != "closed":
                        raise RuntimeError("projection worker did not close cleanly")
            except BaseException as caught:
                worker.kill()
                worker.wait()
                error = error or caught
        if error is not None:
            raise error


class FusedProjectionCapture:
    def __init__(self, args, out, save, torch, model, sites, records, calls, parameters):
        self.out, self.save, self.torch, self.model = out, save, torch, model
        self.sites, self.records, self.calls, self.parameters = sites, records, calls, parameters
        self.layout = load_layout(args.atlas_layout, args.atlas_layout_sha256, records)
        self.scales = down_scales(args.scale_contract, records)
        records.append(identity(__file__))
        self.weights, self.saved, self.methods, self.context = {}, {}, [], None
        self.attention_capture = None
        self.workers = ProjectionWorkers(args.projection_worker_python, args.projection_worker_script, records)

    def read(self, desc):
        member = Path(desc["file"])
        if member.is_absolute() or ".." in member.parts:
            raise ValueError("projection descriptor escapes capture")
        path = safe_path(self.out / member)
        identity(path, desc["sha256"])
        data = path.read_bytes()
        if len(data) != desc["bytes"]:
            raise ValueError("projection descriptor byte extent changed")
        return data

    def save_bytes(self, relative, data):
        return self.save(relative, np.frombuffer(data, dtype=np.uint8))

    def begin_session(self, label):
        self.context = None

    def attach_attention(self, attention):
        self.attention_capture = attention

    def physical_o_input(self, a, step, site, rows, m):
        physical = self.context.get("o_input_physical")
        attention = getattr(self, "attention_capture", None)
        if physical is None and attention is not None:
            layer = int(site.split(".")[2])
            matches = [item for item in attention.calls if item["step"] == step and item["layer"] == layer]
            if len(matches) != 1:
                raise ValueError("actual attention physical output is unavailable")
            physical = self.layout.unpack_bf16_tiles(self.read(matches[0]["output_tiled"]), m, a.shape[1])
        if physical is None and rows == m:
            physical = a
        if (
            physical is None
            or physical.shape != (m, a.shape[1])
            or physical.dtype != np.uint16
            or not np.array_equal(physical[:rows], a)
        ):
            raise ValueError("actual o input physical BF16 words are required")
        return physical

    def install(self):
        owner = self
        for layer_index, layer in enumerate(self.model.model.layers[: int(len(self.sites) // 7)]):
            original = layer.forward
            signature = inspect.signature(original)

            def decoder(module, hidden_states, *args, _index=layer_index, _sig=signature, **kwargs):
                bound = _sig.bind(hidden_states, *args, **kwargs)
                bound.apply_defaults()
                forwarded = dict(bound.arguments)
                forwarded.pop("hidden_states")
                for name, parameter in _sig.parameters.items():
                    if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                        forwarded.update(forwarded.pop(name, {}))
                owner.context = dict(layer=_index, entry=raw_array(hidden_states[0], owner.torch))
                normalized = module.input_layernorm(hidden_states)
                hidden_states, _ = module.self_attn(hidden_states=normalized, **forwarded)
                owner.context["residual"] = raw_array(hidden_states[0], owner.torch)
                normalized = module.post_attention_layernorm(hidden_states)
                return module.mlp(normalized)

            def mlp(module, value):
                module.gate_proj(value)
                product = module.up_proj(value)
                return module.down_proj(product)

            for module, replacement in ((layer, decoder), (layer.mlp, mlp)):
                self.methods.append((module, "forward" in module.__dict__, module.__dict__.get("forward")))
                module.forward = MethodType(replacement, module)

    def restore(self):
        while self.methods:
            module, own, original = self.methods.pop()
            if own:
                module.forward = original
            else:
                del module.forward

    def tensor(self, words):
        return self.torch.from_numpy(words.copy()).view(self.torch.bfloat16).unsqueeze(0)

    def validate_launch(self, launch):
        if (
            launch.get("model_commit") != MODEL_PIN
            or launch.get("numerics") != "rtl"
            or launch.get("accelerated") is not False
            or launch.get("native_dispatch_enabled") is not False
        ):
            raise ValueError("projection worker is not the original RTL reference")

    def linear(self, site, module, a, step, index, mode):
        rows, k = a.shape
        m = 128 if rows == 128 else 32
        role = 7 if site == "lm_head" else native.PROJECTIONS.index(".".join(site.split(".")[-2:]))
        context = self.context
        if role != 7 and (context is None or context["layer"] != int(site.split(".")[2])):
            raise ValueError("projection lost its decoder residual context")
        operand = None
        if role == 3:
            operand = context["entry"]
        elif role == 5:
            operand = context["gate"]
            if array_identity(a) != context["gate_input"]:
                raise ValueError("gate and up normalized predecessors differ")
        elif role == 6:
            operand = context["residual"]
        physical = self.physical_o_input(a, step, site, rows, m) if role == 3 else None
        signature = (
            array_identity(a),
            None if operand is None else array_identity(operand),
            None if physical is None else array_identity(physical),
        )
        key = (step, site)
        if mode == "replay":
            old, words, emitted = self.saved[key]
            if signature != old:
                raise ValueError("replay fused predecessor or residual changed")
            if role == 6 and context["down_fp8"] != emitted:
                raise ValueError("replay emitted down FP8 predecessor changed")
        else:
            if site not in self.weights:
                raw_b = raw_array(module.weight.T, self.torch)
                self.parameters[site] = dict(B_bf16=self.save(f"parameters/{index:03d}-B.bin", raw_b))
                pieces = []
                for panel, shard in enumerate(column_shards(k, module.out_features)):
                    lo, hi = shard["columns"]
                    encoded, scale = encode(np.ascontiguousarray(raw_b[:, lo:hi]))
                    desc = self.save_bytes(
                        f"parameters/{index:03d}-{panel:02d}-B-fp8.bin", self.layout.pack_fp8_weights(encoded)
                    )
                    pieces.append((lo, hi, desc, scale))
                self.weights[site] = module.weight._version, module.weight.data_ptr(), pieces
            version, address, pieces = self.weights[site]
            if (version, address) != (module.weight._version, module.weight.data_ptr()):
                raise ValueError("cached parameter changed")
            stem = f"calls/{step:02d}-{index:03d}"
            down_scale = self.scales[(step, site.replace("up_proj", "down_proj"))] if role == 5 else 0
            if role == 6:
                a_scale, packed_a = context["down_scale"], context["down_fp8"]
            else:
                encoded, a_scale = encode(a if physical is None else physical)
                if physical is not None:
                    encoded[rows:] = 0
                elif rows == 1:
                    encoded = np.concatenate((encoded, np.zeros((31, k), np.uint8)))
                packed_a = self.layout.pack_fp8_tiles(encoded)
            a_desc = self.save_bytes(stem + "-A-fp8.bin", packed_a)
            call = dict(
                step=step,
                site=site,
                site_index=index,
                A_bf16=self.save(stem + "-A.bin", a),
                A_fp8=a_desc,
                A_scale=a_scale,
                panels=[],
            )
            if physical is not None:
                call["A_physical_bf16"] = self.save(stem + "-A-physical.bin", physical)
            requests = []
            for panel, (lo, hi, b_desc, b_scale) in enumerate(pieces):
                panel_stem = stem + f"-{panel:02d}"
                request = dict(
                    command="projection",
                    root=str(self.out),
                    out=str(self.out / panel_stem),
                    step=step,
                    site=site,
                    panel=panel,
                    layer=-1 if role == 7 else context["layer"],
                    m=m,
                    k=k,
                    n=hi - lo,
                    logical_rows=rows,
                    role=role,
                    column=lo,
                    width=module.out_features,
                    rescale=a_scale + b_scale,
                    a_scale=a_scale,
                    b_scale=b_scale,
                    output_scale=down_scale,
                    A_fp8=a_desc,
                    B_fp8=b_desc,
                )
                if operand is not None:
                    compact = np.ascontiguousarray(operand[:, lo:hi])
                    if compact.shape[0] == rows:
                        compact = padded_words(compact, rows)
                    elif compact.shape[0] != m:
                        raise ValueError("epilogue operand physical rows differ")
                    request["operand_tiles"] = self.save_bytes(
                        panel_stem + "-operand.bin", self.layout.pack_bf16_tiles(compact)
                    )
                requests.append(request)
            print(json.dumps(dict(milestone="projection_start", step=step, site=site)), flush=True)
            results = self.workers.run(requests)
            if len(results) != len(pieces):
                raise ValueError("incomplete projection panel replies")
            logical = []
            for panel, ((lo, hi, _, b_scale), launch) in enumerate(zip(pieces, results)):
                self.validate_launch(launch)
                for source in launch["implementation_sources"]:
                    if source not in self.records:
                        self.records.append(source)
                output = np.frombuffer(self.read(launch["output_bf16"]), dtype="<u2").reshape(rows, hi - lo).copy()
                if role == 5:
                    # FP8 producer BF16 is a Torch placeholder decoded by the worker, never a native product.
                    raw = self.layout.unpack_fp8_tiles(self.read(launch["output_fp8"]), m, hi - lo)
                    decoded = self.torch.from_numpy(raw.copy()).view(self.torch.float8_e4m3fn).float()
                    expected = (decoded * 2.0**down_scale).to(self.torch.bfloat16).view(self.torch.uint16).numpy()
                    if not np.array_equal(expected[:rows], output):
                        raise ValueError("FP8 worker placeholder differs from exact emitted bytes")
                else:
                    expected = self.layout.unpack_bf16_tiles(self.read(launch["output_tiled"]), m, hi - lo)
                logical.append(output)
                call["panels"].append(
                    dict(
                        columns=[lo, hi],
                        logical_mkn=[rows, k, hi - lo],
                        physical_mkn=[m, k, hi - lo],
                        A_scale=a_scale,
                        B_scale=b_scale,
                        rescale=a_scale + b_scale,
                        expected_bf16=self.save(stem + f"-{panel:02d}-expected.bin", expected),
                        launch=launch,
                    )
                )
                print(json.dumps(dict(milestone="projection_panel", step=step, site=site, panel=panel)), flush=True)
            words = np.concatenate(logical, axis=1)
            emitted = packed_a if role == 6 else None
            if role == 5:
                emitted = assemble_fp8(call["panels"], self.read, m, module.out_features)
                call["output_fp8"] = self.save_bytes(stem + "-output-fp8.bin", emitted)
                call["output_scale"] = down_scale
            self.saved[key] = signature, words.copy(), emitted
            self.calls.append(call)
        if role == 4:
            context["gate"], context["gate_input"] = words.copy(), array_identity(a)
        elif role == 5:
            context["down_fp8"] = emitted
            context["down_scale"] = self.scales[(step, site.replace("up_proj", "down_proj"))]
        return self.tensor(words)

    def summary(self):
        return dict(
            status="PASS_TINY_FUSED_BLOCK_CAPTURE_REPLAY"
            if len(self.sites) < 155
            else "PASS_FULL_TINY_FUSED_CAPTURE_REPLAY",
            fusions=True,
            atlas_instruction_execution=True,
            projection_reference_model=MODEL_PIN,
            projection_workers=3,
            projection_numerics="rtl",
            fixed_down_scale_contract_sha256=SCALE_CONTRACT_SHA,
        )

    def close(self):
        self.workers.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "historical",
        "atlas-arithmetic",
        "out",
        "attention-worker-script",
        "projection-worker-script",
        "scale-contract",
        "atlas-layout",
    ):
        parser.add_argument("--" + name, type=safe_path, required=True)
    for name in ("atlas-arithmetic-sha256", "atlas-layout-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("attention-worker-python", "projection-worker-python"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--attention-model-root", type=safe_path)
    parser.add_argument("--decode-count", type=int, default=2)
    parser.add_argument("--layer-count", type=int, default=22)
    parser.add_argument("--block-only", action="store_true")
    parser.add_argument("--relocate", action="append", default=[])
    args = parser.parse_args()
    capture(args, projection_factory=FusedProjectionCapture)


if __name__ == "__main__":
    main()
