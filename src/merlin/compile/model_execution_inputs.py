"""Byte-bound inputs and engine identities for saved-model execution.

These helpers do not build or run an image and do not grant host capability.
The caller owns board selection, process success, numerical comparison, and receipts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from merlin.compile.host_lane import _isa_parts


class ModelExecutionInputError(ValueError):
    """A selected model, ELF, or elaborated engine lacks exact input evidence."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_tree_sha256(root: Path) -> dict[str, Any]:
    """Bind every byte below an input tree; never follow a symlink or skip a subtree."""
    if not root.is_dir() or root.is_symlink():
        raise ModelExecutionInputError(f"input tree is missing or is a symlink: {root}")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ModelExecutionInputError(f"input tree contains a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ModelExecutionInputError(f"input tree contains a non-file: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative + b"\0")
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        digest.update(b"\0")
        count += 1
    if count == 0:
        raise ModelExecutionInputError(f"input tree has no files: {root}")
    return {"sha256": digest.hexdigest(), "n_files": count}


def require_elf_isa_supported(arch: list[str], host_isa: str, *, require_scalar: bool = True) -> None:
    """Refuse ELF extensions outside the selected CPU, accounting for base-ISA aliases."""
    if not arch or not arch[0].startswith(("rv32i", "rv64i")):
        raise ModelExecutionInputError(f"ELF has no auditable RISC-V arch: {arch}")
    base = arch[0] + "".join(part for part in arch[1:] if len(part) == 1)
    named = [part for part in arch[1:] if len(part) > 1]
    elf_xlen, required = _isa_parts("_".join([base, *named]))
    host_xlen, available = _isa_parts(host_isa)
    # LLVM records some standard subsets separately even when the DTS names the
    # containing base extension. This is a finite, explicit implication list.
    if "m" in available:
        available.add("zmmul")
    if "a" in available:
        available.update({"zaamo", "zalrsc"})
    if "zfh" in available:
        available.add("zfhmin")
    if "c" in available:
        available.add("zca")
        if "d" in available:
            available.add("zcd")
    missing = sorted(required - available)
    if (
        elf_xlen != host_xlen
        or missing
        or (
            require_scalar
            and any(extension == "v" or extension.startswith(("zve", "zvl", "zv")) for extension in required)
        )
    ):
        compatibility = "scalar-compatible" if require_scalar else "compatible"
        raise ModelExecutionInputError(f"ELF ISA is not {compatibility} with selected CPU: {missing or arch}")


def selected_firrtl(path: str | Path, *, target: str, config: str, source_relocations=None) -> dict[str, str]:
    """Read one explicitly selected RTL facts document and recheck the FIRRTL ``config`` executes.

    Facts of ``config`` itself yield their own FIRRTL input. Facts of another configuration yield
    the FIRRTL of their verified simulation correspondence to ``config`` (see
    :func:`merlin.targetgen.rtl.unit_correspondence.execution_binding`), byte-verified on disk and
    marked with ``facts_config``; anything else refuses.
    """
    lexical = Path(path)
    if lexical.is_symlink() or not lexical.is_file():
        raise ModelExecutionInputError("selected RTL facts are missing or a symlink")
    source = lexical.resolve()
    raw = source.read_bytes()
    original_document = json.loads(raw)
    from merlin.targetgen.rtl.source_relocations import relocate_operand_sources

    try:
        document = relocate_operand_sources(original_document, source_relocations)
    except ValueError as exc:
        raise ModelExecutionInputError(str(exc)) from exc
    facts = document.get("facts") or {}
    origin = facts.get("source") or {}
    inputs = document.get("inputs") or {}
    rows = inputs.get("firrtl_inputs") or []
    if inputs.get("target") == target and origin.get("config") != config:
        # Cross-configuration correspondence retains its original independently
        # checked source selection. ISA/target-FIRRTL moves cannot rewrite it.
        result = _corresponding_firrtl(original_document, source, target=target, config=config)
        if source.read_bytes() != raw or relocate_operand_sources(original_document, source_relocations) != document:
            raise ModelExecutionInputError("selected source inputs changed before publication")
        return result
    if inputs.get("target") != target or origin.get("config") != config or len(rows) != 1:
        raise ModelExecutionInputError("selected RTL facts do not name the board target/config and one FIRRTL input")
    row = rows[0]
    firrtl = Path(str(row.get("path") or ""))
    digest = row.get("sha256")
    if (
        not firrtl.is_absolute()
        or firrtl.is_symlink()
        or not firrtl.is_file()
        or not isinstance(digest, str)
        or len(digest) != 64
        or digest != inputs.get("fir_sha256")
        or file_sha256(firrtl) != digest
    ):
        raise ModelExecutionInputError("selected RTL facts have no byte-verified FIRRTL input")
    result = {
        "path": str(source),
        "sha256": file_sha256(source),
        "firrtl": str(firrtl.resolve()),
        "firrtl_sha256": digest,
        "target": target,
        "config": config,
    }
    if source.read_bytes() != raw or relocate_operand_sources(original_document, source_relocations) != document:
        raise ModelExecutionInputError("selected source inputs changed before publication")
    return result


def _corresponding_firrtl(document: dict, source: Path, *, target: str, config: str) -> dict[str, str]:
    from merlin.targetgen.rtl.unit_correspondence import ExecutionBindingError, execution_binding

    try:
        binding = execution_binding(document, config=config)
    except ExecutionBindingError as exc:
        raise ModelExecutionInputError(
            "selected RTL facts do not name the board target/config and one FIRRTL input"
        ) from exc
    firrtl = Path(str(binding["firrtl"].get("path") or ""))
    digest = binding["firrtl"]["sha256"]
    if not firrtl.is_absolute() or firrtl.is_symlink() or not firrtl.is_file() or file_sha256(firrtl) != digest:
        raise ModelExecutionInputError("selected RTL facts have no byte-verified corresponding simulation FIRRTL")
    return {
        "path": str(source),
        "sha256": file_sha256(source),
        "firrtl": str(firrtl.resolve()),
        "firrtl_sha256": digest,
        "target": target,
        "config": config,
        "facts_config": binding["facts_config"],
    }


def selected_execution_firrtl(path: str | Path, *, target: str) -> dict[str, str]:
    """Select the bound runner's FIRRTL, retaining the original facts identity.

    Legacy backends without a bound runner retain their explicitly selected facts
    configuration. Cross-configuration admission is always owned by selected_firrtl.
    """
    from merlin.runtime.backends import base as backends
    from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend

    backend = backends.get_backend(target)
    if isinstance(backend, BoundRoCCBackend):
        config = backend._block()["config"]  # noqa: SLF001 -- original selected runner, not facts provenance
        source_relocations = backend.operand_source_relocations(path)
    else:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        config = ((document.get("facts") or {}).get("source") or {}).get("config")
        source_relocations = None
    if not isinstance(config, str) or not config:
        raise ModelExecutionInputError("selected execution has no explicit configuration")
    return selected_firrtl(path, target=target, config=config, source_relocations=source_relocations)


def native_engine(target: str, simulator: str, facts: dict[str, str]):
    """Resolve one public backend engine, cite exact lineage, and return a revalidator.

    Returns ``(backend, citation, revalidate, prepare_gsim_command_or_none)``.
    Call ``revalidate`` before and after execution. For GSIM, prepare the command
    with the selected ELF digest and citation and revalidate it across the run.
    """
    from merlin.runtime.backends import base as backends

    backend = backends.get_backend(target)
    if not backend.available(simulator):
        raise ModelExecutionInputError(f"{target} backend reports {simulator} unavailable")
    if simulator == "gsim":
        from merlin.targetgen import gsim_emulator

        env_var = getattr(backend, "GSIM_EMU_ENV", None)
        citation = gsim_emulator.citation(target, env_var=env_var, backend=backend)
        receipt = citation.get("receipt") or {}
        if (
            citation.get("available") is not True
            or citation.get("refused") is not False
            or citation.get("flavour") != "binary"
            or citation.get("receipt_status") != "bound"
            or receipt.get("schema_version") != gsim_emulator.STRICT_RECEIPT_SCHEMA
            or receipt.get("firrtl_sha256") != facts["firrtl_sha256"]
        ):
            raise ModelExecutionInputError("GSIM has no bound v3 receipt for the selected FIRRTL")
        engine_path = Path(str(citation.get("path") or ""))
        selected_path = getattr(backend, "gsim_path", None)
        prepare = getattr(backend, "prepare_gsim_command", None)
        from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend

        bound_execution = isinstance(backend, BoundRoCCBackend)
        if not callable(selected_path) or (not callable(prepare) and not bound_execution) or not engine_path.is_file():
            raise ModelExecutionInputError("GSIM backend lacks a citable engine or command revalidator")
        if Path(selected_path()).resolve() != engine_path.resolve() or file_sha256(engine_path) != citation.get(
            "binary_sha256"
        ):
            raise ModelExecutionInputError("GSIM backend would run different engine bytes than its citation")

        def revalidate() -> None:
            if bound_execution:
                backend.verify_execution_inputs()
            current = gsim_emulator.citation(target, env_var=env_var, backend=backend)
            if current != citation or file_sha256(engine_path) != citation["binary_sha256"]:
                raise ModelExecutionInputError("GSIM engine citation or binary changed during qualification")

        # The bound runner owns its exact argv/environment and checks the ELF
        # and engine identities across execution. Legacy runners instead supply
        # a separately prepared command for the same checks.
        revalidate()
        return backend, citation, revalidate, prepare if callable(prepare) else None

    if simulator == "verilator":
        from merlin.common import provenance

        artifacts = provenance.load_artifacts()
        selected = {}
        for role in ("verilator_binary", "verilator_firrtl"):
            matches = [item for item in artifacts.values() if item.target == target and item.role == role]
            if len(matches) != 1 or matches[0].config != facts["config"]:
                raise ModelExecutionInputError(f"{target} has no unique {role} pin for the selected config")
            check = provenance.verify_artifact(matches[0].name)
            if check.matches is not True:
                raise ModelExecutionInputError(f"{role} pin is absent or has drifted: {check.gaps}")
            selected[role] = matches[0]
        binary_pin, firrtl_pin = selected["verilator_binary"], selected["verilator_firrtl"]
        if firrtl_pin.name not in binary_pin.built_from or firrtl_pin.digest != facts["firrtl_sha256"]:
            raise ModelExecutionInputError("Verilator binary pin does not derive from selected FIRRTL")
        selected_path = getattr(backend, "verilator_path", None)
        if not callable(selected_path):
            raise ModelExecutionInputError("Verilator backend exposes no engine path")
        engine_path = Path(selected_path()).resolve()
        if not engine_path.is_file() or file_sha256(engine_path) != binary_pin.digest:
            raise ModelExecutionInputError("Verilator backend would run bytes outside its selected pin")
        citation = {
            "target": target,
            "engine": simulator,
            "path": str(engine_path),
            "binary_pin": binary_pin.name,
            "binary_sha256": binary_pin.digest,
            "firrtl_pin": firrtl_pin.name,
            "firrtl_sha256": firrtl_pin.digest,
            "config": facts["config"],
        }

        def revalidate() -> None:
            if (
                Path(selected_path()).resolve() != engine_path
                or file_sha256(engine_path) != binary_pin.digest
                or any(provenance.verify_artifact(item.name).matches is not True for item in selected.values())
            ):
                raise ModelExecutionInputError("Verilator engine or hardware pins changed during qualification")

        return backend, citation, revalidate, None
    raise ModelExecutionInputError(f"unsupported native simulator {simulator!r}")
