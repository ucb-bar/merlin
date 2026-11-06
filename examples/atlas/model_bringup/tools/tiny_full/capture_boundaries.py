"""Prepare Tiny semantic boundaries using real native C and captured replies.

This command never constructs a device Core or emits an execution receipt.
The output has no ELF addresses until a segment-capable image admits its ABI.
"""
from __future__ import annotations

import argparse
import ctypes as ct
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
import kernel_library  # noqa: E402,F401
from atlas_path_guard import checked


class Parameters(ct.Structure):
    _fields_ = [("embedding", ct.c_void_p), ("input_norm", ct.c_void_p * 22),
                ("post_norm", ct.c_void_p * 22), ("final_norm", ct.c_void_p),
                ("cosine", ct.c_void_p), ("sine", ct.c_void_p), ("epsilon", ct.c_float),
                ("attention", ct.c_void_p), ("fused_project", ct.c_void_p),
                ("normalize", ct.c_void_p), ("capabilities", ct.c_uint64)]


class State(ct.Structure):
    _fields_ = [("keys", ct.c_void_p * 2), ("values", ct.c_void_p * 2),
                ("position", ct.c_size_t), ("bank", ct.c_uint), ("next_token", ct.c_uint32)]


class Phase(ct.Structure):
    _fields_ = [("position", ct.c_uint64)] + [(name, ct.c_uint32) for name in
                ("bank", "next_token", "next_layer", "rows", "active", "reserved")]


class Layout(ct.Structure):
    _fields_ = [("version", ct.c_uint64), ("workspace_bytes", ct.c_uint64),
                ("hidden_offset", ct.c_uint64 * 2)] + [(name, ct.c_uint64) for name in
                ("hidden_bytes", "cache_bank_bytes", "phase_bytes", "state_bytes")]


Projection = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_uint, ct.c_uint, ct.c_void_p,
                         ct.c_size_t, ct.c_size_t, ct.c_size_t, ct.c_void_p)
PROVIDER = ("smol_host_ops.c", "llama_host_attention.c", "llama_host_layout.c",
            "llama_rotary_table.c", "llama_host_bmm.c", "smol_host_softmax.c")
HEADERS = ("smol_host_ops.h", "smol_host_bits.h", "llama_host_attention.h", "llama_host_layout.h",
           "llama_rotary_table.h", "llama_host_bmm.h", "smol_host_softmax.h", "tiny_block.h")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bind_library(path):
    if sys.byteorder != "little" or ct.sizeof(ct.c_void_p) != 8:
        raise ValueError("native boundary capture requires a64-bit little-endian ABI")
    lib = ct.CDLL(str(path))
    lib.tiny_full_get_live_layout.argtypes = [ct.POINTER(Layout)]
    lib.tiny_full_model_begin.argtypes = [ct.POINTER(Parameters), ct.POINTER(State), ct.POINTER(Phase),
        ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    lib.tiny_full_model_layer.argtypes = [ct.POINTER(Parameters), ct.POINTER(State), ct.POINTER(Phase),
        ct.c_uint, Projection, ct.c_void_p, ct.c_void_p, ct.c_size_t]
    lib.tiny_full_model_finish.argtypes = [ct.POINTER(Parameters), ct.POINTER(State), ct.POINTER(Phase),
        Projection, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    lib.tiny_full_model_step.argtypes = [ct.POINTER(Parameters), ct.POINTER(State), ct.c_void_p,
        ct.c_size_t, Projection, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p, ct.c_size_t]
    layout = Layout()
    if lib.tiny_full_get_live_layout(ct.byref(layout)) or layout.version != 1:
        raise ValueError("unsupported compiled live layout")
    if (layout.phase_bytes != ct.sizeof(Phase) or layout.state_bytes != ct.sizeof(State)
            or layout.cache_bank_bytes != 22 * 4 * 160 * 64 * 2 or layout.hidden_bytes != 128 * 2048 * 2
            or not 0 <= layout.hidden_offset[0] < layout.hidden_offset[1]
            or layout.hidden_offset[0] + layout.hidden_bytes != layout.hidden_offset[1]
            or layout.hidden_offset[1] + layout.hidden_bytes > layout.workspace_bytes):
        raise ValueError("native/C live ABI differs")
    return lib, layout


def validate_inventory(data):
    if len(data["calls"]) != 465 or len(data["target_trace"]) != 3:
        raise ValueError("complete465-projection/three-step inventory required")
    widths = (2048, 256, 256, 2048, 5632, 5632, 2048, 32000)
    roles = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    for index, call in enumerate(data["calls"]):
        step, site = divmod(index, 155)
        layer, role = divmod(site, 7) if site < 154 else (22, 7)
        rows, k, width = (128 if step == 0 else 1), (5632 if role == 6 else 2048), widths[role]
        name = f"model.layers.{layer}.{roles[role]}" if layer < 22 else "lm_head"
        a = call["A_bf16"]
        if (type(call["step"]) is not int or call["step"] != step or type(call["site_index"]) is not int
                or call["site_index"] != site or call["site"] != name or a["shape"] != [rows, k]
                or a["dtype"] != "uint16" or a["bytes"] != rows * k * 2):
            raise ValueError("complete logical projection inventory differs")
        column, m = 0, 128 if step == 0 else 32
        for panel in call["panels"]:
            lo, hi = panel["columns"]
            n = min(1024 if k == 5632 else 2048, width - column)
            result = panel["expected_bf16"]
            if (lo != column or hi != column + n or n <= 0 or panel["logical_mkn"] != [rows, k, n]
                    or panel["physical_mkn"] != [m, k, n] or result["shape"] != [m, n]
                    or result["dtype"] != "uint16" or result["bytes"] != m * n * 2):
                raise ValueError("complete physical panel inventory differs")
            column += n
        if column != width:
            raise ValueError("missing projection panel")
    for step, trace in enumerate(data["target_trace"]):
        cache = trace["cache"]
        expected = {(layer, field) for layer in range(22) for field in ("keys", "values")}
        if (trace["step"] != step or len(cache) != 44
                or {(x["layer"], x["field"]) for x in cache} != expected
                or any(x["shape"] != [1, 4, 128 + step, 64] or x["dtype"] != "uint16"
                       or x["bytes"] != 4 * (128 + step) * 64 * 2 for x in cache)):
            raise ValueError("all22 layer KV inventory required at every step")
        logits = trace["logits_payload"]
        if logits["shape"] != [32000] or logits["dtype"] != "float32" or logits["bytes"] != 128000:
            raise ValueError("complete last-row logits inventory required")


class Capture:
    def __init__(self, path, expected, pins):
        self.path, self.pins, self.cache = checked(path), pins, {}
        self.data = json.loads(self.pin(self.path, expected))
        data = self.data
        if (data.get("status") != "PASS_FULL_TINY_ARITHMETIC_CAPTURE_REPLAY"
                or data.get("decode_count") != 2 or type(data["decode_count"]) is not int
                or any(data.get(key) is not True for key in ("source_bookends_verified", "target_replay_exact",
                    "greedy_feedback_actual", "all22_cache_prefixes_exact", "native_restored_exact"))
                or len(data.get("calls", [])) != 465 or len(data.get("target_trace", [])) != 3):
            raise ValueError("complete prefill128 plus two-decode capture required")
        validate_inventory(data)

    def pin(self, path, expected=None):
        path = checked(path)
        raw = path.read_bytes()
        value = digest(raw)
        if expected is not None and value != expected:
            raise ValueError("source/payload digest changed: " + str(path))
        if str(path) in self.pins and self.pins[str(path)] != value:
            raise ValueError("source/payload changed during capture")
        self.pins[str(path)] = value
        return raw

    def read(self, desc):
        name = desc["file"]
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("capture payload must be relative")
        identity = json.dumps(desc, sort_keys=True)
        if name not in self.cache:
            raw = self.pin(self.path.parent / name, desc["sha256"])
            dtype, shape = np.dtype(desc["dtype"]), tuple(desc["shape"])
            if (dtype not in (np.dtype("uint16"), np.dtype("float32"), np.dtype("int64"))
                    or any(type(n) is not int or n <= 0 for n in shape)
                    or len(raw) != desc["bytes"] or len(raw) != int(np.prod(shape)) * dtype.itemsize):
                raise ValueError("capture payload extent/type differs")
            self.cache[name] = (identity, np.frombuffer(raw, dtype=dtype).reshape(shape))
        elif self.cache[name][0] != identity:
            raise ValueError("conflicting capture descriptor")
        return self.cache[name][1]


def validate_gate(capture, raw):
    gate = json.loads(raw)
    if (gate.get("status") != "PASS_FULL_TINY_C_ARITHMETIC_REPLY_REPLAY"
            or gate.get("capture_sha256") != digest(capture.path.read_bytes())
            or any(gate.get(key) is not True for key in
                ("all_actual_predecessors_exact", "all22_KV_all_steps_exact", "source_bookends_verified"))
            or gate.get("greedy_tokens") != [step["next_token"] for step in capture.data["target_trace"]]
            or len(gate.get("ledger", [])) != 465):
        raise ValueError("complete original C replay gate required")
    for i, (call, row) in enumerate(zip(capture.data["calls"], gate["ledger"])):
        if (row.get("index") != i or type(row["index"]) is not int or row.get("step") != call["step"]
                or row.get("site") != call["site"] or row.get("actual_input_sha256") != call["A_bf16"]["sha256"]):
            raise ValueError("original C projection gate binding differs")
    return gate


class Runner:
    def __init__(self, capture, lib, layout):
        self.capture, self.lib, self.layout = capture, lib, layout
        host = {name: capture.read(desc) for name, desc in capture.data["host_parameters"].items()}
        self.host = host
        self.parameters = Parameters(host["model.embed_tokens.weight"].ctypes.data,
            (ct.c_void_p * 22)(*[host[f"model.layers.{i}.input_layernorm.weight"].ctypes.data for i in range(22)]),
            (ct.c_void_p * 22)(*[host[f"model.layers.{i}.post_attention_layernorm.weight"].ctypes.data for i in range(22)]),
            host["model.norm.weight"].ctypes.data, host["rotary_cosine"].ctypes.data,
            host["rotary_sine"].ctypes.data, 1e-5, None)
        self.keys = [np.zeros((22, 4 * 160 * 64), np.uint16) for _ in range(2)]
        self.values = [np.zeros_like(item) for item in self.keys]
        self.state = State((ct.c_void_p * 2)(*[v.ctypes.data for v in self.keys]),
                           (ct.c_void_p * 2)(*[v.ctypes.data for v in self.values]), 0, 0, 0)
        self.phase = Phase()
        self.workspace = np.zeros((layout.workspace_bytes + 7) // 8, np.uint64)
        self.logits = np.zeros((128, 32000), np.uint16)
        self.counter, self.launches, self.errors, self.ledger, self.references = 0, 0, [], [], {}
        for step, trace in enumerate(capture.data["target_trace"]):
            for desc in trace["cache"]:
                self.references[f"tiny_full_cache_{step:02d}_{desc['layer']:02d}_{desc['field']}"] = bytes(desc["bytes"])
            self.references[f"tiny_full_logits_{step:02d}"] = bytes(32000 * 4)
        self.references["tiny_full_greedy_tokens"] = bytes(12)
        self.references["tiny_full_decode_inputs"] = bytes(8)
        self.callback = Projection(self.reply)

    def reply(self, _, layer, logical, incoming, rows, k, width, output):
        try:
            call = self.capture.data["calls"][self.counter]
            site = call["site_index"]
            expected = divmod(site, 7) if site < 154 else (22, 7)
            actual = np.ctypeslib.as_array(ct.cast(incoming, ct.POINTER(ct.c_uint16)),
                                          shape=(rows * k,)).reshape(rows, k)
            wanted = self.capture.read(call["A_bf16"])
            if (layer, logical) != expected or actual.shape != wanted.shape or actual.tobytes() != wanted.tobytes():
                raise ValueError(f"actual C predecessor/order differs at call {self.counter}")
            pieces = [self.capture.read(panel["expected_bf16"])[:rows] for panel in call["panels"]]
            words = np.concatenate(pieces, axis=1)
            if words.shape != (rows, width):
                raise ValueError("projection reply geometry differs")
            ct.memmove(output, words.ctypes.data, words.nbytes)
            self.ledger.append(dict(index=self.counter, step=call["step"], site=call["site"],
                actual_input_sha256=digest(actual.tobytes()), reply_sha256=digest(words.tobytes())))
            self.counter += 1
            self.launches += len(call["panels"])
            return 0
        except Exception as error:
            self.errors.append(str(error))
            return -99

    def require(self, status):
        if status or self.errors:
            raise ValueError(f"native phase failed: status={status}, errors={self.errors}")

    def tokens(self, step):
        tokens = (self.host["prefill_tokens"].astype(np.uint32) if step == 0
                  else np.array([self.state.next_token], np.uint32))
        if tokens.tolist() != self.capture.data["target_trace"][step]["input_tokens"][0]:
            raise ValueError("actual greedy feedback differs")
        if step:
            data = bytearray(self.references["tiny_full_decode_inputs"])
            struct.pack_into("<I", data, (step - 1) * 4, int(tokens[0]))
            self.references["tiny_full_decode_inputs"] = bytes(data)
        return tokens

    def check_step(self, step):
        trace = self.capture.data["target_trace"][step]
        if (self.state.position, self.state.bank, self.state.next_token) != (128 + step, (step + 1) % 2, trace["next_token"]):
            raise ValueError("actual committed state differs")
        rows = 128 if step == 0 else 1
        actual = (self.logits[rows - 1].astype(np.uint32) << 16).view(np.float32).tobytes()
        if actual != self.capture.read(trace["logits_payload"]).tobytes():
            raise ValueError("actual last-row logits differ")
        self.references[f"tiny_full_logits_{step:02d}"] = actual
        for desc in trace["cache"]:
            banks = self.keys if desc["field"] == "keys" else self.values
            actual = banks[self.state.bank][desc["layer"], :4 * self.state.position * 64].tobytes()
            if actual != self.capture.read(desc).tobytes():
                raise ValueError("actual committed KV differs")
            self.references[f"tiny_full_cache_{step:02d}_{desc['layer']:02d}_{desc['field']}"] = actual
        data = bytearray(self.references["tiny_full_greedy_tokens"])
        struct.pack_into("<I", data, step * 4, self.state.next_token)
        self.references["tiny_full_greedy_tokens"] = bytes(data)

    def snapshot(self, step, next_layer):
        regions = {f"{name}_{bank}": values.tobytes() for name, banks in
                   (("keys", self.keys), ("values", self.values)) for bank, values in enumerate(banks)}
        for bank, offset in enumerate(self.layout.hidden_offset):
            regions[f"hidden_{bank}"] = ct.string_at(self.workspace.ctypes.data + offset, self.layout.hidden_bytes)
        regions["phase"] = bytes(self.phase)
        regions["control"] = struct.pack("<8Q", step, next_layer, self.state.position, self.state.bank,
                                        self.state.next_token, 128 if step == 0 else 1, self.counter, self.launches)
        regions.update(self.references)
        return regions

    def clear_scratch(self):
        ct.memset(self.workspace.ctypes.data, 0, self.layout.hidden_offset[0])
        end = self.layout.hidden_offset[1] + self.layout.hidden_bytes
        ct.memset(self.workspace.ctypes.data + end, 0, self.layout.workspace_bytes - end)
        self.logits.fill(0)

    def run(self, segmented=False, save=None):
        if save:
            save(0, self.snapshot(0, 0))
        for step in range(3):
            tokens, rows = self.tokens(step), 128 if step == 0 else 1
            common = (ct.byref(self.parameters), ct.byref(self.state))
            tail = (self.workspace.ctypes.data, self.workspace.nbytes)
            if not segmented:
                self.require(self.lib.tiny_full_model_step(*common, tokens.ctypes.data, rows,
                    self.callback, None, *tail, self.logits.ctypes.data, self.logits.size))
            else:
                self.require(self.lib.tiny_full_model_begin(*common, ct.byref(self.phase), tokens.ctypes.data, rows, *tail))
                for layer in range(22):
                    self.clear_scratch()
                    self.require(self.lib.tiny_full_model_layer(*common, ct.byref(self.phase), layer, self.callback, None, *tail))
                    if save:
                        save(self.launches, self.snapshot(step, layer + 1))
                self.clear_scratch()
                self.require(self.lib.tiny_full_model_finish(*common, ct.byref(self.phase), self.callback, None,
                    *tail, self.logits.ctypes.data, self.logits.size))
            self.check_step(step)
            if save:
                save(self.launches, self.snapshot(step + 1, 0))
        if self.counter != 465 or self.launches != 840:
            raise ValueError("incomplete native projection gate")
        return dict(all_actual_predecessors_exact=True, all22_KV_all_steps_exact=True,
                    greedy_tokens=[s["next_token"] for s in self.capture.data["target_trace"]], ledger=self.ledger)


def prepare(args):
    started, pins = time.monotonic(), {}
    capture = Capture(args.capture, args.capture_sha256, pins)
    gate = validate_gate(capture, capture.pin(args.gate, args.gate_sha256))
    out = checked(args.out)
    if out.exists():
        raise ValueError("fresh boundary preparation output required")
    out.mkdir(parents=True)
    report = dict(schema="atlas.tiny_boundary_preparation.v1", status="FAILED", reference_only=True,
        fresh_instruction_execution=False, continuous=False, hardware_execution=False,
        capture_sha256=args.capture_sha256, original_gate_sha256=args.gate_sha256)
    try:
        source = checked(Path(__file__).with_name("tiny_full_model.c"))
        runtime = checked(Path(__file__).absolute().parents[2] / "runtime")
        attention = runtime / "tiny_attention.c"
        provider = checked(args.merlin)
        compiler = checked(args.cc)
        for path in (Path(__file__).absolute(), Path(sys.modules[checked.__module__].__file__),
                     source, source.with_suffix(".h"), attention, runtime / "tiny_attention.h",
                     runtime / "atlas_numeric.h", compiler,
                     *(provider / name for name in PROVIDER + HEADERS)):
            capture.pin(path)
        for name in PROVIDER + HEADERS:
            prior = {h for path, h in gate["source_sha256"].items() if Path(path).name == name}
            if len(prior) != 1 or pins[str(provider / name)] not in prior:
                raise ValueError("host provider differs from original C gate: " + name)
        command = [str(compiler), "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic",
            "-ffp-contract=off", "-fno-fast-math", "-shared", "-fPIC", "-I", str(provider), "-I", str(runtime),
            str(source), str(attention), *(str(provider / name) for name in PROVIDER), "-lm", "-o", str(out / "native.so")]
        report["dependency_commands"] = []
        for unit in (source, attention, *(provider / name for name in PROVIDER)):
            dependencies = [str(compiler), "-std=c11", "-I", str(provider), "-I", str(runtime), "-M", str(unit)]
            result = subprocess.run(dependencies, capture_output=True, text=True, timeout=30, check=True)
            report["dependency_commands"].append(dependencies)
            for path in result.stdout.replace("\\\n", " ").split(":", 1)[1].split():
                capture.pin(path)
        report["compile_command"] = command
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out / "compile.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise ValueError("native host compile failed")
        capture.pin(out / "native.so")
        lib, layout = bind_library(out / "native.so")
        report["compiled_live_layout"] = {name: (list(getattr(layout, name)) if name == "hidden_offset"
            else getattr(layout, name)) for name, _ in Layout._fields_}
        full = Runner(capture, lib, layout).run()
        if full["ledger"] != gate["ledger"]:
            raise ValueError("new full-step path differs from original C predecessor/reply ledger")
        sums, boundaries, sizes = {}, [], None
        def save(cursor, regions):
            nonlocal sizes
            current = {name: len(raw) for name, raw in regions.items()}
            if sizes is not None and current != sizes:
                raise ValueError("boundary semantic region inventory changed")
            sizes = current
            payloads = {}
            for name, raw in regions.items():
                h = digest(raw)
                member = "payloads/" + h + ".bin"
                if member not in sums:
                    path = out / member
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(raw)
                    sums[member] = h
                payloads[name] = member
            boundaries.append(dict(index=len(boundaries), launch_cursor=cursor, regions=payloads))
        phased = Runner(capture, lib, layout).run(segmented=True, save=save)
        if phased != full:
            raise ValueError("phase/full-step native gates differ")
        expected = [0] + [cursor for step in range(3) for cursor in
                         [*(280 * step + 12 * layer for layer in range(1, 23)), 280 * (step + 1)]]
        if [item["launch_cursor"] for item in boundaries] != expected:
            raise ValueError("complete69-segment partition differs")
        for path, value in pins.items():
            if digest(checked(path).read_bytes()) != value:
                raise ValueError("native capture source/payload bookend changed")
        source_snapshots = {}
        for path, value in pins.items():
            if Path(path).is_relative_to(capture.path.parent):
                continue
            raw = checked(path).read_bytes()
            if digest(raw) != value:
                raise ValueError("source changed while snapshotting")
            member = "sources/" + value + ".bin"
            destination = out / member
            destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(raw)
            sums[member] = value
            source_snapshots[path] = member
        for member, path in (("capture.json", capture.path), ("original-gate.json", checked(args.gate)),
                             ("native.so", out / "native.so")):
            raw = path.read_bytes()
            if digest(raw) != pins[str(path)]:
                raise ValueError("receipt/binary changed while snapshotting")
            if member != "native.so":
                (out / member).write_bytes(raw)
            sums[member] = digest(raw)
        report.update(status="PREPARED_REFERENCE_BOUNDARIES_NOT_EXECUTED", source_bookends_verified=True,
            all_actual_predecessors_exact=True, all22_KV_all_steps_exact=True, all_logits_greedy_exact=True,
            full_step_and_phases_exact=True, scratch_reset_each_phase=True, segments=69, boundaries=70,
            projection_calls=465, physical_panels=840, source_sha256=pins, source_snapshots=source_snapshots,
            capture_member="capture.json", original_gate_member="original-gate.json", gate=phased,
            semantic_regions=sizes, wall_seconds=time.monotonic() - started)
        native_gate = dict(status="PASS_FULL_TINY_C_ARITHMETIC_REPLY_REPLAY", capture_sha256=args.capture_sha256,
            fresh_instruction_execution=False, quality_qualified=False, timing_qualified=False,
            source_bookends_verified=True, source_sha256=pins, **full)
        raw = (json.dumps(native_gate, indent=2) + "\n").encode()
        (out / "native-gate.json").write_bytes(raw)
        sums["native-gate.json"] = digest(raw)
        metadata = dict(schema="atlas.tiny_semantic_boundaries.v1", qualification="REFERENCE_BOUNDARIES_ONLY",
            workload="tiny-full", kernel_count=840, decode_count=2, elf_bound=False, byte_order="little",
            region_bytes=sizes, boundaries=boundaries, final_outputs=sorted(Runner(capture, lib, layout).references))
        raw = (json.dumps(metadata, indent=2) + "\n").encode()
        (out / "boundaries.json").write_bytes(raw)
        sums["boundaries.json"] = digest(raw)
        raw = (json.dumps(report, indent=2) + "\n").encode()
        (out / "receipt.json").write_bytes(raw)
        sums["receipt.json"] = digest(raw)
        (out / "SHA256SUMS").write_text("".join(h + "  " + name + "\n" for name, h in sorted(sums.items())))
        return report
    except BaseException as error:
        report.update(status="FAILED", error=dict(type=type(error).__name__, message=str(error)), wall_seconds=time.monotonic() - started)
        (out / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("capture", "gate", "merlin", "cc", "out"):
        parser.add_argument("--" + name, type=checked, required=True)
    for name in ("capture-sha256", "gate-sha256"):
        parser.add_argument("--" + name, required=True)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
