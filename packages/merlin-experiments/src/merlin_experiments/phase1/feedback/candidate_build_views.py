"""Publish original linked candidate function bytes without exposing a harness ELF.

Only the existing host build/export owner writes the private publication index.
An author manifest is a locator, never producer authority. No code is linked or
executed here; the original recorded LLVM/object/link chain supplies the bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import struct
from contextlib import contextmanager
from pathlib import Path

from merlin.compile.analysis.artifact_reader import identity, open_ordinary
from merlin.verify.lean_contract import document_digest

SCHEMA = "merlin.candidate_build_view.v1"
MAX_BYTES = 32 << 20


def registry_root(bundle_manifest):
    if bundle_manifest is None:
        raise ValueError("candidate_build_original_owner_unavailable")
    return Path(bundle_manifest).absolute().parent / "candidate-build-views"


def _plain(path):
    path = Path(path).absolute()
    if path != Path(os.path.normpath(path)):
        raise ValueError("candidate_build_indirect_product")
    return path


def _read(path, expected=None):
    """Retain bytes from one ordinary FD, checking identity and the consumed digest."""
    path = _plain(path)
    fd = open_ordinary(path)
    try:
        before = os.fstat(fd)
        if before.st_size > MAX_BYTES:
            raise ValueError("candidate_build_product_unavailable_or_exceeds_bound")
        chunks, offset = [], 0
        while offset < before.st_size:
            chunk = os.pread(fd, min(1 << 20, before.st_size - offset), offset)
            if not chunk:
                raise ValueError("candidate_build_product_changed")
            chunks.append(chunk)
            offset += len(chunk)
        raw = b"".join(chunks)
        pin = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        if identity(before) != identity(os.fstat(fd)):
            raise ValueError("candidate_build_product_changed")
        if expected is not None and pin != expected:
            raise ValueError("candidate_build_published_leaf_changed")
        return raw, pin
    finally:
        os.close(fd)


def _pin(path):
    return _read(path)[1]


@contextmanager
def _directory(path):
    """Create/open an anchored parent without following any replaced ancestor."""
    path = _plain(path)
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = following
        yield fd
    finally:
        os.close(fd)


def _write(path, raw):
    """Publish through a pinned directory FD and a fresh exclusive temporary leaf."""
    path = _plain(path)
    with _directory(path.parent) as parent:
        temporary = ".candidate-build-" + secrets.token_hex(16)
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
            os.replace(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            try:
                os.unlink(temporary, dir_fd=parent)
            except FileNotFoundError:
                pass
    # If a public ancestor moved during publication, refuse the lost locator.
    _read(path, dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))


def publish(*, workspace, public_root, private_root, generated, compiler, build_root=None,
            entry_symbol=None, compiler_snapshot_sha256=None):
    """Retain the existing producer's selected function view and exact source joins.

    Call only after the original compilation/snapshot owner has completed its
    custody checks. This helper independently reopens the recorded native chain.
    Public leaves contain candidate source/IR/commands and function instructions,
    never the linked harness, input tensors, golden values or runtime sections.
    """
    from merlin.compile.analysis.artifact_reader import PinnedELF, open_ordinary
    from merlin.perf.pc_instruction_listing import normalized_text
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
    from merlin_experiments.phase1.brokers.selfcheck import _submission_digest

    from .candidate_observations import reopen_native_products

    workspace, public_root, private_root, generated, compiler = map(
        _plain, (workspace, public_root, private_root, generated, compiler)
    )
    build_root = _plain(build_root or generated)
    if not public_root.is_relative_to(workspace) or private_root.is_relative_to(workspace):
        raise ValueError("candidate_build_public_and_original_owners_overlap")
    compiler_sha = _submission_digest(compiler)
    if compiler_snapshot_sha256 is not None and compiler_sha != compiler_snapshot_sha256:
        raise ValueError("candidate_build_original_compiler_changed")
    emission = {role: _pin(generated / name) for role, name in (
        ("source_interface", "input.interface.mlir"), ("command_buffer", "command_buffer.json"),
        ("lowered_mlir", "lowered.llvm.mlir"))}
    elf_pin = _pin(build_root / PACKAGE_ELF_NAME)
    products, invocations = reopen_native_products(
        build_root, emission=emission, elf_pin=elf_pin, emission_root=generated
    )
    if entry_symbol is None:
        stack = json.loads(_read(build_root / "kernel.stack_frame.json")[0])
        entry_symbol = stack.get("entry_symbol")
    if not isinstance(entry_symbol, str) or not entry_symbol:
        raise ValueError("candidate_build_original_entry_unavailable")
    elf = PinnedELF(open_ordinary(products["elf"]), "original_linked_candidate")
    try:
        if elf.pin["sha256"] != elf_pin["sha256"]:
            raise ValueError("candidate_build_original_sources_changed")
        if struct.unpack_from("<H", elf.read_at(0, 64), 16)[0] != 2:
            raise ValueError("candidate_build_requires_original_linked_executable")
        extent = elf.function_extent(entry_symbol)
        rows = [row for row in elf.instructions(section=extent["section"])
                if extent["address"] <= row.addr < extent["address"] + extent["size_bytes"]]
        if not rows or any(row.relocations for row in rows):
            raise ValueError("candidate_build_linked_function_unresolved")
        text = (f'{extent["address"]:x} <{entry_symbol}>:\n' + normalized_text(rows)).encode()
        if len(text) > MAX_BYTES:
            raise ValueError("candidate_build_function_view_exceeds_bound")
    finally:
        elf.close()
    original_products = {role: _pin(path) for role, path in products.items()}
    initial = dict(source=emission["source_interface"], command_buffer=emission["command_buffer"],
                   emitted=emission["lowered_mlir"], elf=elf_pin)
    if (any(original_products[role] != expected for role, expected in initial.items())
            or _submission_digest(compiler) != compiler_sha):
        raise ValueError("candidate_build_original_sources_changed")
    receipts = [_read(path) for path in invocations]
    # Recheck the retained receipt bytes against the exact products being published.
    def consumes(stage, before, after):
        for raw, _ in receipts:
            row = json.loads(raw)
            if row.get("stage") != stage:
                continue
            inputs, outputs = row.get("inputs", ()), row.get("outputs", ())
            if (any(pin.get("sha256") == original_products[before]["sha256"] for pin in inputs)
                    and any((pin.get("path"), pin.get("sha256")) ==
                            (original_products[after]["path"], original_products[after]["sha256"])
                            for pin in outputs)):
                return True
        return False
    if not all(consumes(*edge) for edge in (("llvm_translation", "emitted", "llvm_ir"),
                                           ("object", "llvm_ir", "object"), ("elf", "object", "elf"))):
        raise ValueError("candidate_build_original_producer_chain_changed")
    leaves = {}
    for role, name in (("source", "input.interface.mlir"), ("command_buffer", "command_buffer.json"),
                       ("emitted", "lowered.llvm.mlir")):
        raw, _ = _read(products[role], original_products[role])
        destination = public_root / name
        _write(destination, raw)
        leaves[role] = _pin(destination)
    _write(public_root / "kernel.linked.dis", text)
    leaves["binary"] = _pin(public_root / "kernel.linked.dis")
    if (any(leaves[role]["sha256"] != original_products[role]["sha256"]
            for role in ("source", "command_buffer", "emitted"))
            or any(_pin(products[role]) != pin for role, pin in original_products.items())):
        raise ValueError("candidate_build_original_sources_changed")
    payload = dict(
        schema=SCHEMA, workspace=str(workspace), manifest=str(public_root / "candidate_build.json"),
        compiler_sha256=compiler_sha, entry=extent, leaves=leaves,
        compiler_emission_binding=("observed_original_owner_snapshot" if compiler_snapshot_sha256 is not None
                                   else "unknown"),
        original_products=original_products,
        producer_invocations=[dict(sha256=pin["sha256"], stage=json.loads(raw)["stage"])
                              for raw, pin in receipts],
        scope=("original compiler snapshot emission and LLVM/object/link byte consumption; "
               "no semantic or execution qualification" if compiler_snapshot_sha256 is not None else
               "recorded LLVM/object/link byte consumption; compiler/source-to-emission correspondence unknown"),
    )
    identity = document_digest(payload)
    _write(private_root / (identity + ".json"), json.dumps(payload, sort_keys=True).encode())
    _write(public_root / "candidate_build.json", json.dumps(dict(schema=SCHEMA, publication=identity)).encode())
    return dict(status="published", path=(public_root / "candidate_build.json").relative_to(workspace).as_posix(),
                format="candidate_build", publication=identity, compiler_sha256=compiler_sha,
                scope=payload["scope"], numerical_equivalence="unknown", performance="unknown")


def _load(path, *, workspace, bundle_manifest, manifest_sha256=None):
    """Retain verified bytes; later path replacement cannot inherit producer authority."""
    path, workspace = _plain(path), _plain(workspace)
    manifest, manifest_pin = _read(path)
    if manifest_sha256 is not None and manifest_pin["sha256"] != manifest_sha256:
        raise ValueError("candidate_build_publication_changed")
    raw = json.loads(manifest)
    identity = raw.get("publication")
    if (set(raw) != {"schema", "publication"} or raw["schema"] != SCHEMA
            or not isinstance(identity, str) or len(identity) != 64
            or any(char not in "0123456789abcdef" for char in identity)):
        raise ValueError("candidate_build_publication_invalid")
    registry = _plain(registry_root(bundle_manifest))
    if registry.is_relative_to(workspace):
        raise ValueError("candidate_build_original_owner_is_public")
    original = registry / (identity + ".json")
    record = json.loads(_read(original)[0])
    if (record.get("schema") != SCHEMA or document_digest(record) != identity
            or record.get("workspace") != str(workspace) or record.get("manifest") != str(path)):
        raise ValueError("candidate_build_original_publication_mismatch")
    payloads = {}
    for role, pin in record["leaves"].items():
        leaf = _plain(pin["path"])
        if leaf.parent != path.parent:
            raise ValueError("candidate_build_published_leaf_changed")
        payloads[role], _ = _read(leaf, pin)
    return record, payloads, identity


def load(path, *, workspace, bundle_manifest):
    """Reopen original host publication and every public leaf; JSON cannot register it."""
    return _load(path, workspace=workspace, bundle_manifest=bundle_manifest)[0]


def publish_feedback(**kwargs):
    """A missing diagnostic artifact cannot change an ordinary compilation verdict."""
    try:
        return publish(**kwargs)
    except Exception as error:
        code = str(error)
        return dict(status="unknown", reason=code if code.startswith("candidate_build_")
                    else "candidate_build_original_producer_chain_unavailable",
                    error_type=type(error).__name__, numerical_equivalence="unknown", performance="unknown")


def producer_binding(record):
    """Public byte joins from an original host publication, without private paths."""
    return dict(
        status=("observed_original_build_consumption"
                if record["compiler_emission_binding"] == "observed_original_owner_snapshot"
                else "observed_native_build_consumption"),
        compiler_emission_binding=record["compiler_emission_binding"], compiler_sha256=record["compiler_sha256"],
        source_sha256=record["leaves"]["source"]["sha256"],
        command_buffer_sha256=record["leaves"]["command_buffer"]["sha256"],
        emitted_sha256=record["leaves"]["emitted"]["sha256"],
        object_sha256=record["original_products"]["object"]["sha256"],
        linked_elf_sha256=record["original_products"]["elf"]["sha256"],
        function_view_sha256=record["leaves"]["binary"]["sha256"],
        retained_entry=record["entry"], producer_invocations=record["producer_invocations"],
        scope=record["scope"], semantic_correspondence="unknown", execution_multiplicity="unknown",
        numerical_equivalence="unknown", performance="unknown",
    )


def analyze(value, pin, *, workspace, bundle_manifest, facts, contract, target, descriptor=None):
    """Feed original resolved function bytes into the existing comparison owner."""
    from . import reference_analysis as A
    from .reference_representations import emitted_view
    from .reference_views import source_view

    record, payloads, publication = _load(
        workspace / pin["path"], workspace=workspace, bundle_manifest=bundle_manifest,
        manifest_sha256=pin["sha256"],
    )
    binary = record["leaves"]["binary"]
    label = Path(binary["path"]).relative_to(workspace).as_posix()
    report, cca = A._analyze_open(
        dict(path=label, format="disassembly", symbol=record["entry"]["symbol"],
             instruction_offset=value.get("instruction_offset", 0)),
        raw=payloads["binary"], pin=dict(binary, path=label), label=label, entry=None,
        workspace=workspace, reference_root=None, facts=facts, candidate=True,
        contract=contract, target=target, descriptor=descriptor,
    )
    source, command_buffer, emitted = (payloads[role] for role in ("source", "command_buffer", "emitted"))
    binding = producer_binding(record)
    decoder = A._operand_decoder(contract, facts, target, descriptor=descriptor)
    report["emitted_view"], _ = emitted_view(emitted, record["leaves"]["emitted"], decoder=decoder,
                                            symbol=record["entry"]["symbol"])
    try:
        report["source_view"] = source_view(source)
    except ValueError:
        from .reference_source_structure import source_structure

        report["source_view"] = source_structure(source, pin=record["leaves"]["source"])
    from .reference_schedule_receipts import inspect_schedule_receipts

    report["construction_receipts"] = inspect_schedule_receipts(
        source_mlir=source, command_buffer=command_buffer, emitted_mlir=emitted, producer_binding=binding)
    report["function_view_input"], report["input"] = report["input"], pin
    report["producer_binding"] = binding
    report["candidate_build_publication"] = publication
    report["artifact_to_current_compiler_binding"] = record["compiler_emission_binding"]
    return report, cca
