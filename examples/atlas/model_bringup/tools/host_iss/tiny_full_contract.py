"""Exact full Tiny host admission, all-layer KV and greedy completion checks."""
import hashlib
import json
import math
import struct

import numpy as np

from bundle import EPILOGUE
from tiny_contract import stop_sites as block_stop_sites
from placement import arena_extent, placement

CYCLE_CAP = 200_000_000
MAGIC = 0x54494E5946554C4C
ROLES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
WIDTHS = (2048, 256, 256, 2048, 5632, 5632, 2048)
CACHE_WORDS = 22 * 4 * 160 * 64


def stop_sites(image):
    if "tiny_full_trap_stop" not in image.symbols:
        return block_stop_sites(image)
    names = ["main"] + (["complete"] if "complete" in image.symbols else [])
    sites = {pc: length for name in names for pc, length, word in image.instructions(name)
             if length == 4 and word == 0x10500073}
    require(bool(sites), "Tiny full completion WFI missing")
    trap = image.symbols["tiny_full_trap_stop"]
    recorder = image.symbols["trap_entry"]
    require(recorder["size"] > 0 and recorder["address"] <= trap["address"]
            and trap["address"] + 2 <= recorder["address"] + recorder["size"]
            and image.bytes_at(trap["address"], 2) == b"\x01\xa0"
            and trap["address"] not in sites, "Tiny full named trap stop changed")
    sites[trap["address"]] = 2
    return sites


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def finite_bf16(raw):
    return len(raw) % 2 == 0 and not np.any((np.frombuffer(raw, dtype="<u2") & 0x7f80) == 0x7f80)


def plan(decode_count):
    require(type(decode_count) is int and 1 <= decode_count <= 32, "Tiny full decode extent")
    result = []
    for step in range(decode_count + 1):
        rows, m = (128, 128) if step == 0 else (1, 32)
        for index in range(155):
            layer, logical = divmod(index, 7) if index < 154 else (22, 7)
            site = f"model.layers.{layer}.{ROLES[logical]}" if index < 154 else "lm_head"
            k = 5632 if logical == 6 else 2048
            width = WIDTHS[logical] if logical < 7 else 32000
            maximum = 1024 if k == 5632 else 2048
            for column in range(0, width, maximum):
                n = min(maximum, width-column)
                result.append(dict(step=step, site=site, site_index=index, layer=layer, logical=logical,
                    logical_rows=rows, physical_mkn=[m, k, n], columns=[column, column+n], width=width))
    return result


def reference_geometry(count):
    result = {}
    for step in range(count + 1):
        for layer in range(22):
            for field in ("keys", "values"):
                result[f"tiny_full_cache_{step:02d}_{layer:02d}_{field}"] = ([1, 4, 128 + step, 64], "uint16")
        result[f"tiny_full_logits_{step:02d}"] = ([32000], "float32")
    result["tiny_full_greedy_tokens"] = ([count + 1], "uint32")
    result["tiny_full_decode_inputs"] = ([count], "uint32")
    return result


def parameter_geometry():
    result = {"model.embed_tokens.weight": ([32000, 2048], "uint16"),
              "model.norm.weight": ([2048], "uint16"),
              "rotary_cosine": ([160, 64], "uint16"), "rotary_sine": ([160, 64], "uint16"),
              "prefill_tokens": ([128], "int64")}
    for layer in range(22):
        for kind in ("input_layernorm", "post_attention_layernorm"):
            result[f"model.layers.{layer}.{kind}.weight"] = ([2048], "uint16")
    return result


def validate_plan(data, provenance):
    physical = placement(provenance)
    require(placement(data) == physical, "Tiny full contract placement binding")
    policy = provenance.get('program_policy','original')
    require(policy in ('original','traffic-reuse-v1') and data.get('program_policy','original') == policy,
            'Tiny full program policy binding')
    if policy == 'traffic-reuse-v1':
        import kernel_library  # noqa: E402,F401
        from atlas_program import dma_load_bytes
        require(isinstance(provenance.get('reuse'),dict) and data.get('reuse') == provenance['reuse']
                and provenance['reuse'].get('captured_data_unchanged') is True
                and provenance['reuse'].get('instruction_execution') is False,
                'Tiny full reuse preparation binding')
    count = data["decode_count"]
    expected = plan(count)
    require(data["schema"] == "atlas.tiny_full_contract.v1" and data["cycle_cap"] == CYCLE_CAP,
            "Tiny full contract schema or cycle bound")
    require(data["kernel_count"] == len(expected) == provenance["kernel_count"], "Tiny full launch count")
    require(data["projection_calls"] == 155 * (count + 1), "Tiny full projection count")
    require(data["kernels"] == provenance["kernels"] and len(data["kernels"]) == len(expected),
            "Tiny full descriptor binding")
    for index, (item, geometry) in enumerate(zip(data["kernels"], expected)):
        require(type(item["index"]) is int and item["index"] == index, "Tiny full panel index")
        require(all(item[name] == geometry[name] for name in
                    ("step", "site", "site_index", "logical_rows", "physical_mkn", "columns")),
                "Tiny full panel role or geometry")
        m, k, n = geometry["physical_mkn"]
        load = (m*k*(n//32) + k*n if k*n <= 262144 else
                m*k + k*n*(m//32) if m*k <= 262144 else (m//32)*(n//32)*(2*k//32*1024))
        if policy == 'traffic-reuse-v1':
            load = dma_load_bytes(m,k,n,traffic_reuse=True)
        require(item["dram_base"] == physical["arena_base"] and item["output_offset"] == m*k+k*n+256
                and item["total_dram_bytes"] == m*k+k*n+512+2*m*n
                and item["requested_dma"] == {"load_bytes": load, "store_bytes": 2*m*n},
                "Tiny full physical layout or DMA")
        arena_extent(provenance, item["total_dram_bytes"])
        require(all(type(item[name]) is int and -126 <= item[name] <= 127
                    for name in ("A_scale", "B_scale", "rescale"))
                and item["rescale"] == item["A_scale"] + item["B_scale"], "Tiny full scales")
    expected_refs = reference_geometry(count)
    references = data["reference_buffers"]
    require(len(references) == len(expected_refs)
            and {item["actual_symbol"] for item in references} == set(expected_refs),
            "Tiny full cache/logit/token reference coverage")
    for item in references:
        name = item["actual_symbol"]
        shape, dtype = expected_refs[name]
        elements = 1
        for size in shape:
            elements *= size
        require(item["shape"] == shape and item["dtype"] == dtype
                and item["bytes"] == elements * (2 if dtype == "uint16" else 4)
                and item["expected_symbol"] == name + "_expected", "Tiny full reference geometry")
    parameters = data["host_parameters"]
    require(set(parameters) == set(parameter_geometry()), "Tiny full host parameter inventory")
    for index, name in enumerate(sorted(parameters)):
        item = parameters[name]
        shape, dtype = parameter_geometry()[name]
        require(item["shape"] == shape and item["dtype"] == dtype
                and item["symbol"] == f"tiny_full_param_{index:03d}", "Tiny full host parameter geometry")
    require(len(data["greedy_tokens"]) == count + 1 and all(type(value) is int and 0 <= value < 32000
            for value in data["greedy_tokens"]), "Tiny full greedy token range")
    for key, value in (("mailbox_symbol", "tiny_full_mailbox"), ("panel_symbol", "tiny_full_panels"),
                       ("state_symbol", "tiny_full_state_global"), ("context_symbol", "tiny_full_projection_state")):
        require(data[key] == value, "Tiny full fixed host ABI")
    return expected


def contract(bundle):
    p = bundle.provenance
    require(p["schema"] == "atlas.full_bundle.v1" and p["workload"] == "tiny-full", "Tiny full workload")
    raw = bundle.read(p["contract"])
    require(digest(raw) == p["contract_sha256"], "Tiny full contract identity")
    data = json.loads(raw)
    validate_plan(data, p)
    return data


def blob(image, symbol, raw, *, actual=False):
    record = image.symbols[symbol]
    require(record["size"] == len(raw) if actual else record["size"] in (0, len(raw)),
            "Tiny full ELF symbol extent: " + symbol)
    if not actual:
        require(image.bytes_at(record["address"], len(raw)) == raw, "Tiny full embedded bytes: " + symbol)
    else:
        require(any(s["flags"] & 2 and s["address"] <= record["address"]
                    and record["address"] + len(raw) <= s["address"] + s["memory_bytes"]
                    for s in image.segments), "Tiny full writable actual buffer: " + symbol)
    return record["address"]


def disjoint(left, right):
    return left[0] + left[1] <= right[0] or right[0] + right[1] <= left[0]


def reference_locations(image, items, immutable=()):
    actual, expected = [], list(immutable)
    for item in items:
        name, size = item["actual_symbol"], item["bytes"]
        address = blob(image, name, bytes(size), actual=True)
        span = (address, size)
        require(all(disjoint(span, other) for other in actual), "Tiny full actual reference buffer alias")
        actual.append(span)
        expected.append((image.symbols[item["expected_symbol"]]["address"], size))
    require(all(disjoint(a, e) for a in actual for e in expected), "Tiny full actual/immutable reference alias")
    return actual


def validate_bundle(bundle, image):
    data = contract(bundle)
    build = bundle.build
    require(build["schema"] == "atlas.tiny_full_host_build.v1" and build["actual_returncode"] == 0
            and type(build["actual_returncode"]) is int and build["source_bookends_verified"] is True
            and build["elf"]["sha256"] == digest(image.raw)
            and build["contract_sha256"] == bundle.provenance["contract_sha256"], "Tiny full build identity")
    require(bool(build["source_sha256"]), "Tiny full build source pins")
    physical = placement(bundle.provenance)
    require(placement(build) == physical, "Tiny full build placement binding")
    require(build.get('program_policy','original') == bundle.provenance.get('program_policy','original'),
            'Tiny full build program policy binding')
    arena = physical["arena_base"]
    capacity = max(item["total_dram_bytes"] for item in data["kernels"])
    for segment in image.segments:
        require(segment["address"] == segment["physical"] and segment["address"] >= 0x80000000
                and physical["dram_base"] <= segment["address"]
                and segment["address"] + segment["memory_bytes"] <= physical["dram_base"] + physical["dram_bytes"]
                and disjoint((segment["address"], segment["memory_bytes"]), (arena, capacity)),
                "Tiny full ELF physical DRAM or host/DMA separation")
    for index, left in enumerate(image.segments):
        for right in image.segments[:index]:
            require(left["address"] + left["memory_bytes"] <= right["address"]
                    or right["address"] + right["memory_bytes"] <= left["address"], "Tiny full ELF load overlap")
    for name in ("tohost", "fromhost"):
        address = blob(image, name, bytes(8), actual=True)
        require(address % 64 == 0 and image.bytes_at(address, 8) == bytes(8), "Tiny full HTIF initialized/aligned")
        require(any(s["flags"] & 2 and s["address"] <= address and address + 8 <= s["address"] + s["memory_bytes"]
                    for s in image.segments), "Tiny full writable HTIF")
    require(image.symbols["tohost"]["address"] != image.symbols["fromhost"]["address"], "Tiny full distinct HTIF")
    require(image.symbols["tiny_full_mailbox"]["size"] == 128
            and image.symbols["tiny_full_panels"]["size"] == 120 * bundle.kernel_count
            and image.symbols["tiny_full_projection_state"]["size"] == 112
            and image.symbols["tiny_full_state_global"]["size"] == 48
            and image.symbols["tiny_full_host"]["size"] == 112, "Tiny full actual host ABI extents")
    require(image.bytes_at(image.symbols["tiny_full_projection_state"]["address"] + 72, 40) == bytes(40),
            "Tiny full legacy attention context must start zero")
    stop_sites(image)
    immutable = [(image.symbols[name]["address"], image.symbols[name]["size"]) for name in
                 ("tohost", "fromhost", "tiny_full_mailbox", "tiny_full_panels", "tiny_full_projection_state",
                  "tiny_full_state_global", "tiny_full_host")]
    for index, (item, geometry) in enumerate(zip(data["kernels"], plan(data["decode_count"]))):
        panel = struct.unpack("<6QIi8Q", image.bytes_at(image.symbols["tiny_full_panels"]["address"] + 120*index, 120))
        m, k, n = item["physical_mkn"]
        original, weights = bundle.read(item["original"]), bundle.read(item["weights"])
        immutable.extend(((panel[8], len(weights)), (panel[10], len(original))))
        require(panel[:8] == (item["logical_rows"], m, k, n, item["columns"][0], geometry["width"],
                             geometry["logical"], item["A_scale"]), "Tiny full embedded panel geometry")
        require(panel[9] == len(weights) and image.bytes_at(panel[8], len(weights)) == weights
                and image.bytes_at(panel[10], len(original)) == original
                and panel[11:] == (len(original)//4, item["output_offset"], item["total_dram_bytes"],
                                  item["requested_dma"]["load_bytes"], item["requested_dma"]["store_bytes"]),
                "Tiny full embedded program/weight/extent")
    for item in data["reference_buffers"]:
        raw = bundle.read(item["file"])
        require(digest(raw) == item["sha256"] and len(raw) == item["bytes"], "Tiny full reference payload")
        if item["dtype"] == "uint16":
            require(finite_bf16(raw), "Tiny full finite BF16 cache reference")
        if item["actual_symbol"] == "tiny_full_greedy_tokens":
            require(struct.unpack(f"<{data['decode_count']+1}I", raw) == tuple(data["greedy_tokens"]),
                    "Tiny full greedy token reference binding")
        elif item["actual_symbol"] == "tiny_full_decode_inputs":
            require(struct.unpack(f"<{data['decode_count']}I", raw) == tuple(data["greedy_tokens"][:-1]),
                    "Tiny full consumed-token reference binding")
        elif item["actual_symbol"].startswith("tiny_full_logits_"):
            values = struct.unpack("<32000f", raw)
            bits = struct.unpack("<32000I", raw)
            step = int(item["actual_symbol"].rsplit("_", 1)[1])
            require(all(math.isfinite(v) for v in values) and all((v & 65535) == 0 for v in bits)
                    and max(range(32000), key=values.__getitem__) == data["greedy_tokens"][step],
                    "Tiny full finite BF16-logit greedy reference")
        blob(image, item["expected_symbol"], raw)
        blob(image, item["actual_symbol"], raw, actual=True)
    for item in data["host_parameters"].values():
        raw = bundle.read(item["file"])
        require(digest(raw) == item["sha256"] and len(raw) == item["bytes"], "Tiny full parameter payload")
        if item["dtype"] == "uint16":
            require(finite_bf16(raw), "Tiny full finite BF16 host parameter")
        else:
            require(all(0 <= token < 32000 for token in struct.unpack("<128q", raw)),
                    "Tiny full prefill token range")
        blob(image, item["symbol"], raw)
        immutable.append((image.symbols[item["symbol"]]["address"], len(raw)))
    reference_locations(image, data["reference_buffers"], immutable)
    return data


def verify_completion(remote, image, device, bundle, out):
    data = contract(bundle)
    require(remote.reg(32) in {pc for pc, length in stop_sites(image).items() if length == 4}, "Tiny full normal WFI stop")
    count, kernels = data["decode_count"], bundle.kernel_count
    mailbox = remote.read(image.symbols["tiny_full_mailbox"]["address"], 128)
    require(len(mailbox) == 128, "Tiny full mailbox extent")
    words = struct.unpack("<16Q", mailbox)
    require(words == (MAGIC, 2, kernels, kernels-1, 0, 0, 1, count, 128+count,
                      data["greedy_tokens"][-1], 0, 0, 0, 0, 0, 0), "Tiny full mailbox completion")
    for name, value in (("tohost", 1), ("fromhost", 0)):
        require(remote.read(image.symbols[name]["address"], 8) == struct.pack("<Q", value), "Tiny full HTIF completion")
    require(len(device.completed) == kernels, "Tiny full completed launch count")
    for index, (summary, item) in enumerate(zip(device.completed, data["kernels"])):
        require(all(type(summary[name]) is int and summary[name] == value for name, value in
                    (("index", index), ("sequence", index+1), ("reset_count", 1)))
                and summary["exact"] is True and summary["drained"] is True
                and type(summary["cycles"]) is int and 0 < summary["cycles"] <= CYCLE_CAP,
                "Tiny full ordered exact/drained ledger")
        original, expected = bundle.read(item["original"]), bundle.read(item["expected"])
        requested = item["requested_dma"]
        require(summary["program_sha256"] == digest(original[:-4] + EPILOGUE)
                and summary["host_input_sha256"] == item["initial_sha256"]
                and summary["actual_output_sha256"] == summary["expected_output_sha256"] == digest(expected)
                and summary["observed_dma"] == dict(requested,
                    response_beats=(requested["load_bytes"]+requested["store_bytes"])//32),
                "Tiny full live program/input/output/DMA identity")
    state = struct.unpack("<5QII", remote.read(image.symbols["tiny_full_state_global"]["address"], 48))
    initial = struct.unpack("<5QII", image.bytes_at(image.symbols["tiny_full_state_global"]["address"], 48))
    require(state[:4] == initial[:4] and state[4:] == (128+count, (count+1)%2, data["greedy_tokens"][-1]),
            "Tiny full actual KV state")
    context = struct.unpack("<9Q", remote.read(image.symbols["tiny_full_projection_state"]["address"], 72))
    initial = struct.unpack("<9Q", image.bytes_at(image.symbols["tiny_full_projection_state"]["address"], 72))
    tail = image.symbols["tiny_full_projection_state"]["address"] + 72
    require(image.bytes_at(tail, 40) == bytes(40) and remote.read(tail, 40) == bytes(40),
            "Tiny full legacy attention context must remain zero")
    require(context[2] == kernels and context[7] == data["projection_calls"]
            and all(context[i] == initial[i] for i in (0, 1, 3, 4, 5, 6, 8)), "Tiny full actual projection cursor")
    host = remote.read(image.symbols["tiny_full_host"]["address"], 112)
    require(struct.unpack_from("<QQ", host, 72) == (0, 0)
            and struct.unpack_from("<6I", host, 88)[1:3] == (0, 0)
            and struct.unpack_from("<I", host, 108)[0] == kernels, "Tiny full drained host ownership")
    results = []
    for item in data["reference_buffers"]:
        expected = bundle.read(item["file"])
        actual = remote.read(image.symbols[item["actual_symbol"]]["address"], len(expected))
        require(actual == expected, "Tiny full actual cache/logit/greedy output: " + item["actual_symbol"])
        (out / (item["actual_symbol"] + ".bin")).write_bytes(actual)
        results.append(dict(symbol=item["actual_symbol"], bytes=len(actual), sha256=digest(actual), exact=True))
    (out / "mailbox.bin").write_bytes(mailbox)
    return dict(tohost=1, fromhost=0, mailbox_words=words, expected_starts=kernels,
                prefill_tokens=128, greedy_decode_iterations=count, all22_KV_exact=True,
                greedy_tokens=data["greedy_tokens"], final_guest_buffers=results, all_final_guest_buffers_exact=True)
