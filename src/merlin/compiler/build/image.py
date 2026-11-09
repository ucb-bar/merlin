"""Link selected synchronous C regions with an explicitly supplied image entry.

The caller owns startup, deployment and flags. This stage establishes a real
link and limited ELF header corroboration, not numerical correctness, selected
device execution, complete ISA coverage or platform startup qualification.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath

from merlin.compiler.artifact import Artifact, Compilation, Deployment, digest
from merlin.compiler.build.contextentry import CImageEntry


def _json(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class CImageLink:
    """An explicit compiler and caller-authored entry for a selected C dispatch.

    ``startup`` is an existing saved Compilation; ``link_inputs`` selects its
    C/object/archive members. All its artifacts are retained, including headers
    and linker scripts. Flags are argv tokens, never a shell command. Declaring
    a deployment does not establish that this startup invokes the dispatcher or
    initializes the selected hardware correctly.
    """

    compiler: Path
    flags: tuple[str, ...]
    startup: Compilation
    link_inputs: tuple[str, ...]
    deployment: Deployment
    timeout: float = 300
    context_entry: CImageEntry | None = None
    _identity: str = field(init=False, repr=False)
    _tool: Artifact = field(init=False, repr=False)
    _manifest: Artifact = field(init=False, repr=False)

    def __post_init__(self):
        if type(self.startup) is not Compilation or type(self.deployment) is not Deployment:
            raise ValueError("C image linking requires a saved startup Compilation and explicit Deployment")
        path = Path(self.compiler)
        if not path.is_absolute() or not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("C image compiler must be an explicit absolute executable path")
        object.__setattr__(self, "compiler", path.resolve())
        if (type(self.flags) is not tuple or not self.flags or
                any(type(flag) is not str or not flag.startswith("-") or "\x00" in flag
                    for flag in self.flags) or
                any(flag in {"-c", "-S", "-E", "-shared", "-r"} or flag.startswith("-o")
                    for flag in self.flags)):
            raise ValueError("C image flags must be explicit option tokens for executable linking")
        if (type(self.link_inputs) is not tuple or not self.link_inputs or
                len(set(self.link_inputs)) != len(self.link_inputs)):
            raise ValueError("C image requires distinct explicit startup link inputs")
        for member in self.link_inputs:
            if (type(member) is not str or not member or "\\" in member or ":" in member or
                    PurePosixPath(member).is_absolute() or ".." in PurePosixPath(member).parts or
                    PurePosixPath(member).as_posix() != member):
                raise ValueError("startup link inputs require portable relative paths")
        if (isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or
                not math.isfinite(self.timeout) or self.timeout <= 0):
            raise ValueError("C image timeout must be positive and finite")
        if self.deployment.platform not in {"linux", "baremetal", "zephyr"}:
            raise ValueError("C image needs an explicit Linux, baremetal or Zephyr deployment")
        if not (self.deployment.isa == "x86_64" and self.deployment.abi == "sysv" or
                self.deployment.isa.startswith("rv32") and self.deployment.abi in {"ilp32", "ilp32f", "ilp32d"} or
                self.deployment.isa.startswith("rv64") and self.deployment.abi in {"lp64", "lp64f", "lp64d", "lp64q"}):
            raise ValueError("C image has no ELF architecture admission for the selected deployment")
        self.deployment.__post_init__()
        if self.context_entry is not None and type(self.context_entry) is not CImageEntry:
            raise ValueError("C image context entry must be an original typed selection")
        object.__setattr__(self, "_entry", self.context_entry)
        object.__setattr__(self, "_tool", Artifact.from_file(self.compiler, kind="tool", role="compiler"))
        object.__setattr__(self, "_manifest", Artifact.from_file(self.startup.manifest, kind="json", role="startup"))
        object.__setattr__(self, "_identity", self._current())
        self.revalidate()

    def _current(self):
        return _json({"compiler": str(self.compiler), "flags": self.flags, "link_inputs": self.link_inputs,
                      "deployment": asdict(self.deployment), "timeout": self.timeout,
                      "startup_directory": str(self.startup.directory), "startup": self.startup.to_dict(),
                      **({"context_entry": self.context_entry.identity()} if self.context_entry is not None else {})})

    def revalidate(self):
        if self.context_entry is not self._entry:
            raise ValueError("original C image context entry changed")
        if self.context_entry is not None:
            self.context_entry.revalidate()
            self.context_entry.validate_flags(self.flags)
        if self._current() != self._identity:
            raise ValueError("C image selection changed after binding")
        self._tool.verify()
        self._manifest.verify()
        loaded = Compilation.load(self.startup.manifest)
        if loaded.to_dict() != self.startup.to_dict():
            raise ValueError("C image startup differs from its original saved receipt")
        self.startup.verify()
        products = {a.path.relative_to(self.startup.directory).as_posix(): a for a in self.startup.artifacts}
        if any(member not in products or products[member].kind not in {"c-source", "object", "archive"}
               for member in self.link_inputs):
            raise ValueError("C image startup link inputs must be actual C/object/archive products")
        if self.startup.deployment is not None and self.startup.deployment != self.deployment:
            raise ValueError("C image cannot override its startup deployment")


def _elf_contract(path, deployment, reservations=()):
    """Corroborate executable shape, architecture and the supported ELF ABI bits."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if (len(header) < 52 or header[:4] != b"\x7fELF" or header[4] not in (1, 2) or
                header[5:7] != b"\x01\x01"):
            raise ValueError("C image linker did not produce a supported little-endian ELF")
        bits = {1: 32, 2: 64}[header[4]]
        if bits == 64 and len(header) < 64:
            raise ValueError("C image linker produced a truncated ELF64")
        kind, machine, version = struct.unpack_from("<HHI", header, 16)
        entry, phoff = struct.unpack_from("<QQ" if bits == 64 else "<II", header, 24)
        flags, ehsize, phsize, phnum = struct.unpack_from("<IHHH", header, 48 if bits == 64 else 36)
        if (kind not in (2, 3) or version != 1 or not entry or ehsize != (64 if bits == 64 else 52) or
                phsize != (56 if bits == 64 else 32) or not 0 < phnum < 65535 or
                phoff < ehsize or phoff + phsize * phnum > path.stat().st_size):
            raise ValueError("C image requires an executable ELF with an entry and program headers")
        if deployment.isa == "x86_64":
            if bits != 64 or machine != 62 or deployment.abi != "sysv":
                raise ValueError("C image ELF differs from the selected x86_64/sysv contract")
        elif deployment.isa.startswith(f"rv{bits}") and machine == 243:
            abis = {32: {"ilp32": 0, "ilp32f": 2, "ilp32d": 4},
                    64: {"lp64": 0, "lp64f": 2, "lp64d": 4, "lp64q": 6}}
            if deployment.abi not in abis[bits] or flags & 6 != abis[bits][deployment.abi] or flags & 8:
                raise ValueError("C image ELF differs from the selected RISC-V psABI")
        else:
            raise ValueError("C image has no ELF architecture admission for the selected deployment")
        if reservations and kind != 2:
            raise ValueError("reserved-interval image admission requires a fixed-address executable ELF")
        loads = []
        for index in range(phnum):
            stream.seek(phoff + index * phsize)
            raw = stream.read(phsize)
            if bits == 64:
                ptype, permissions, offset, address, physical, size, memory, _ = struct.unpack("<IIQQQQQQ", raw)
            else:
                ptype, offset, address, physical, size, memory, permissions, _ = struct.unpack("<IIIIIIII", raw)
            if ptype == 1:
                if size > memory or offset + size > path.stat().st_size:
                    raise ValueError("C image ELF has an invalid load segment")
                if deployment.dram_base is not None and not (
                        deployment.dram_base <= address and
                        address + memory <= deployment.dram_base + deployment.memory_bytes):
                    raise ValueError("C image ELF load segment exceeds the selected deployment memory")
                if reservations and (address + memory > 1 << bits or physical + memory > 1 << bits):
                    raise ValueError("C image ELF load segment exceeds its declared address width")
                for reserved in reservations:
                    reserved.__post_init__()
                    base = physical if reserved.address_space == "physical" else address
                    if memory and base < reserved.address + reserved.bytes and reserved.address < base + memory:
                        raise ValueError("C image ELF load segment overlaps an admitted platform reservation")
                loads.append((address, memory, permissions))
        if not any(address <= entry < address + size and permissions & 1 for address, size, permissions in loads):
            raise ValueError("C image ELF entry is not inside an executable load segment")
    return {"bits": bits, "machine": machine, "type": kind, "entry": entry,
            "scope": "ELF structure, architecture and supported psABI bits; ISA extensions and startup unqualified"}


def link_selected_image(selection: CImageLink, partition, region_products, description, *, directory):
    """Link checked control and wrappers; retain original selected inputs around the process."""
    from merlin.compiler.build import process

    directory = Path(directory)
    if type(selection) is not CImageLink:
        raise ValueError("selected C image requires a typed link selection")
    original = _json(description)

    def check():
        selection.revalidate()
        partition.revalidate()
        for product in region_products:
            product.revalidate()
        if _json(description) != original:
            raise ValueError("selected C control description changed before image admission")

    check()
    entry = selection.context_entry
    if description["payloads"] and entry is None:
        raise ValueError("C image route has no admitted payload loading or embedding contract")
    actual = entry.admit(selection, partition, region_products, description, directory=directory) if entry else None
    root = directory / "image"
    root.mkdir()
    startup = root / "startup"
    startup.mkdir()
    artifacts = []
    for product in selection.startup.artifacts:
        path = startup / product.path.relative_to(selection.startup.directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(product.path, path)
        if digest(path) != product.sha256:
            raise ValueError("selected startup changed while copying")
        artifacts.append(Artifact.from_file(path, kind=product.kind, role="startup." + product.role))
    manifest = startup / "compilation.json"
    shutil.copyfile(selection.startup.manifest, manifest)
    if digest(manifest) != selection._manifest.sha256:
        raise ValueError("selected startup receipt changed while copying")
    artifacts.append(Artifact.from_file(manifest, kind="json", role="startup.manifest"))
    inputs = tuple(directory / member for member in description["link_inputs"])
    inputs += tuple(startup / member for member in selection.link_inputs)
    entry_artifacts, extra = [], []
    if entry is not None:
        entry_root = root / "context"
        entry_root.mkdir()
        header, source = entry.sources(actual, description)
        for member, text in (("entry.h", header), ("entry.c", source)):
            path = entry_root / member
            path.write_text(text, encoding="utf-8", newline="\n")
            entry_artifacts.append(Artifact.from_file(path, kind="c-header" if member.endswith(".h") else "c-source",
                                                       role="stateful_entry"))
        artifacts.extend(entry_artifacts)
        inputs += (entry_root / "entry.c",)
        extra.extend(("-I" + str(entry_root), "-I" + str(directory)))
        extra.extend("-Wl,-T," + str(startup / member) for member in entry.linker_scripts)
    copied = tuple(artifacts)
    control = tuple(Artifact.from_file(path, kind="data", role="link-input")
                    for path in dict.fromkeys((*inputs, *(directory / "control").iterdir())))
    image = root / "program.elf"
    command = [selection.compiler, *selection.flags, *extra, "-I" + str(directory / "control"), *inputs, "-o", image]
    check()
    admission_inputs = (() if entry is None else
                        (*entry._sources, *entry.evidence,
                         *(a for binding in entry.contexts for a in binding.platform.artifacts)))
    region_inputs = (() if entry is None else
                     tuple(a for product in region_products for a in product.compilation.artifacts))
    process.run(command, environment=dict(os.environ), timeout_limit_s=selection.timeout,
                timeout_s=selection.timeout, error=RuntimeError,
                inputs=tuple(dict.fromkeys(a.path for a in (*copied, *control, *region_inputs, *admission_inputs))),
                outputs=(image,), stage="selected_image_link")
    check()
    for artifact in (*copied, *control):
        artifact.verify()
    if entry is not None:
        entry.admit(selection, partition, region_products, description, directory=directory)
    elf = _elf_contract(image, selection.deployment, entry.reservations if entry is not None else ())
    artifacts.append(Artifact.from_file(image, kind="elf", role="executable"))
    record = root / "link.json"
    record.write_text(json.dumps({
        "schema": "merlin.selected_c_image.v1", "command": list(map(str, command)),
        "compiler": {"path": str(selection.compiler), "sha256": selection._tool.sha256},
        "deployment": asdict(selection.deployment), "elf": elf,
        "control": json.loads(original), "startup_source_sha256": selection.startup.source_sha256,
        "inputs": [{"path": a.path.relative_to(directory).as_posix(), "sha256": a.sha256} for a in control],
        "numerical_verification": "not_established", "placement_execution": "not_observed",
        "startup_dispatch_invocation": "caller_responsibility", "transitive_toolchain_closure": "not_established",
        **({"context_entry": {"selection": json.loads(entry.identity()),
             "sources": [{"path": str(a.path), "sha256": a.sha256} for a in entry._sources],
             "generated_sequence": "checked_initializer_context_table_dispatch; original_boot_invocation_unobserved",
             "platform_admission": "original_caller_conditional_contract; hardware_execution_unobserved",
             "error_lifetime": "persistent_entry_context_workspace; no_retry_or_reset",
             "remaining": ["boot_startup_invocation", "platform_effects_and_reachability",
                           "device_arithmetic", "execution_and_visibility_evidence"]}} if entry is not None else {}),
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    artifacts.append(Artifact.from_file(record, kind="json", role="link_record"))
    artifacts.extend(Artifact.from_file(path, kind="data", role="link_invocation")
                     for path in sorted((root / "invocations").rglob("*")) if path.is_file())
    check()
    return tuple(artifacts)
