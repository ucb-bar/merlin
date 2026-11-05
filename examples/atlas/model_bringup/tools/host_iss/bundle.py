"""Read a relocated, checksum-bound ResNet bundle with explicit input paths."""

import kernel_library  # noqa: E402,F401
from atlas_tooling import sha256 as sha

import io
import json
from pathlib import Path, PurePosixPath
import re
import struct

from atlas_tooling import sha as digest
from atlas_path_guard import checked

EPILOGUE = struct.pack("<6I", 0x13, 0xC1102FF3, 0x13, 0xC10F9073, 0x13, 0x73)


def verify_kernel_receipt(summary, item, index, cycle_cap):
    if not (all(type(summary.get(name)) is int and summary[name] == value for name, value in
                (("index", index), ("sequence", index + 1), ("reset_count", 1)))
            and summary.get("exact") is True and summary.get("drained") is True
            and type(summary.get("cycles")) is int and 0 < summary["cycles"] <= cycle_cap):
        raise ValueError("ordered exact/drained kernel")
    requested = item["requested_dma"]
    if not (summary["program_sha256"] == digest(item["program"][:-4] + EPILOGUE)
            and summary["host_input_sha256"] == item["initial_sha256"]
            and summary["actual_output_sha256"] == summary["expected_output_sha256"] == item["expected_sha256"]
            and summary["observed_dma"] == dict(requested,
                response_beats=(requested["load_bytes"] + requested["store_bytes"]) // 32)):
        raise ValueError("kernel live payload/DMA identity")


def relative_name(name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
        or any(ord(c) < 32 or ord(c) == 127 for c in name) or name.startswith("/")
        or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("unsafe bundle member name")
    return PurePosixPath(name)


def overlay_paths(root, sums):
    """Resolve an explicitly checksum-bound, read-only inherited asset set."""
    storage = checked(root / "storage.json")
    if not storage.exists():
        return {}, {}
    raw = storage.read_bytes()
    if digest(raw) != sums.get("storage.json"):
        raise ValueError("unlisted or changed overlay storage descriptor")
    data = json.loads(raw)
    if (set(data) != {"schema", "base_root", "base_sha256sums_sha256", "inherit"}
            or data["schema"] != "atlas.asset_overlay.v1" or not Path(data["base_root"]).is_absolute()
            or not isinstance(data["inherit"], list) or len(data["inherit"]) != len(set(data["inherit"]))):
        raise ValueError("invalid asset overlay descriptor")
    base = checked(data["base_root"])
    manifest = checked(base / "SHA256SUMS")
    base_raw = manifest.read_bytes()
    if base == root or digest(base_raw) != data["base_sha256sums_sha256"]:
        raise ValueError("overlay base identity differs")
    base_sums = {}
    for line in base_raw.decode().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
        if match is None:
            raise ValueError("invalid overlay base checksum manifest")
        value, name = match.groups()
        relative_name(name)
        if name in base_sums or name == "SHA256SUMS":
            raise ValueError("duplicate overlay base checksum")
        base_sums[name] = value
    paths = {}
    for name in data["inherit"]:
        relative = relative_name(name)
        if (name in ("storage.json", "SHA256SUMS") or name not in sums or sums[name] != base_sums.get(name)
                or checked(root / relative).exists()):
            raise ValueError("overlay inheritance differs or has a local shadow")
        paths[name] = checked(base / relative)
    return paths, {str(storage): digest(raw), str(manifest): digest(base_raw)}


class ChecksumBundle:
    def __init__(self, root):
        self.root = checked(root)
        self.sums = {}
        self.inherited_paths = {}
        self.storage_pins = {}
        sums_path = self.path("SHA256SUMS")
        self.checksums_sha256 = sha(sums_path)
        for line in sums_path.read_text().splitlines():
            match = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", line)
            if match is None:
                raise ValueError("invalid SHA256SUMS line")
            digest, name = match.groups()
            relative_name(name)
            if name in self.sums or name == "SHA256SUMS":
                raise ValueError("duplicate or self-referential checksum")
            self.sums[name] = digest
        if not self.sums:
            raise ValueError("empty SHA256SUMS")
        self.inherited_paths, self.storage_pins = overlay_paths(self.root, self.sums)
        self.verify_bookends()

    def path(self, name):
        relative = relative_name(name)
        return self.inherited_paths.get(name, checked(self.root / relative))

    def read(self, name):
        if name not in self.sums:
            raise ValueError("unlisted bundle payload " + name)
        raw = self.path(name).read_bytes()
        if digest(raw) != self.sums[name]:
            raise ValueError("changed bundle payload " + name)
        return raw

    def verify_bookends(self):
        for path, expected in self.storage_pins.items():
            if sha(checked(path)) != expected:
                raise ValueError("changed overlay storage identity")
        if sha(self.path("SHA256SUMS")) != self.checksums_sha256:
            raise ValueError("changed SHA256SUMS")
        for name, expected in self.sums.items():
            if sha(self.path(name)) != expected:
                raise ValueError("changed bundle payload " + name)


class Bundle(ChecksumBundle):
    def __init__(self, root):
        super().__init__(root)
        provenance = json.loads(self.read("provenance.json"))
        if provenance.get("schema") != "atlas.full_resnet_bundle.v3" or provenance.get("kernel_count") != 54:
            raise ValueError("only the complete ResNet v3 development bundle is supported")
        from placement import arena_extent, placement
        self.placement = placement(provenance)
        self.kernel_count = 54
        build = json.loads(self.read("host/build.json"))
        self.build = build
        if (build.get("returncode") != 0 or build.get("source_bookends_equal") is not True or build.get("kernel_count") != 54
            or build.get("elf_sha256") != self.sums.get("host/host.elf")):
            raise ValueError("host build identity disagrees with bundle")
        if "placement" in provenance or "placement" in build:
            if placement(build) != self.placement:
                raise ValueError("ResNet build placement disagrees with bundle")
            arena_extent(provenance, build["dram_bytes"])
        for index in range(54):
            for name in ("receipt.json", "program.S", "program.bin", "initial-dram.bin", "expected-bf16-tiles.bin"):
                if f"kernels/{index:02d}/model/{name}" not in self.sums:
                    raise ValueError("missing model payload")

def runtime_program(source, original, assemble):
    """Reconstruct the exact six-word runtime completion epilogue."""
    if not source.rstrip().endswith("ecall") or len(original) < 4 or len(original) % 4:
        raise ValueError("program must end with an aligned ecall")
    if original[-4:] != struct.pack("<I", 0x73):
        raise ValueError("original binary must end with ecall")
    patched = source.rstrip()[:-5] + "nop\ncsrrs x31, 0xc11, x0\nnop\ncsrrw x0, 0xc10, x31\nnop\necall\n"
    program = assemble(io.StringIO(patched))
    words = b"".join(struct.pack("<I", word) for word in program.assemble())
    if words != original[:-4] + EPILOGUE:
        raise ValueError("runtime epilogue reconstruction mismatch")
    return program, words


def verify_completion(mailbox, tohost, logits, expected, completed):
    if len(mailbox) != 928 or len(logits) != 4000 or len(expected) != 4000:
        raise ValueError("completion payload extent")
    fields = struct.unpack_from("<10I", mailbox)
    assert tohost == 1 and fields[:5] == (0x41544C53, 3, 0, 0, 54)
    assert fields[6:8] == (0, 0) and fields[8] == 0xFFFFFFFF and fields[9] == 54
    assert struct.unpack_from("<54I", mailbox, 40) == tuple(range(1, 55))
    assert mailbox[904:928] == bytes(24), "trap mailbox nonzero"
    assert len(completed) == 54 and logits == expected
    assert [r["index"] for r in completed] == list(range(54))
    assert [r["sequence"] for r in completed] == list(range(1, 55))
