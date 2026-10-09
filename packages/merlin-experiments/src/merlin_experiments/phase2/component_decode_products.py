"""Join actual ordinary execution and coherent decoder products, without roles.

The private target observer still owns packet grammar, ELF object membership,
full histories and effects. These joins retain actual invocation correspondence;
they never construct a semantic witness or qualify callback/physical authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from merlin.common import invocation_record as I

from .contracts import StageGateError, mapping_file, sha256_file


def _reopen(pin, owner=None):
    if type(pin) is not dict or type(pin.get("path")) is not str or type(pin.get("sha256")) is not str:
        raise StageGateError("decoder product has no exact original file membership")
    path = Path(pin["path"])
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(value.is_symlink() for value in (path, *path.parents))
        or not path.is_file()
        or owner is not None
        and not path.is_relative_to(owner)
        or sha256_file(path) != pin["sha256"]
    ):
        raise StageGateError("decoder product differs from its actual private bytes")
    return path


def _member(path):
    if (
        not path.is_absolute()
        or path.resolve() != path
        or any(value.is_symlink() for value in (path, *path.parents))
        or not path.is_file()
    ):
        raise StageGateError("decoder product requires a canonical unlinked regular file before reading")
    return {"path": str(path), "sha256": sha256_file(path)}


@dataclass(frozen=True)
class ComponentDecodeProducts:
    """Observation only; not accepted by runtime or semantic stage issuers."""

    result: tuple[Path, str]
    execution_root: Path
    execution_record: tuple[Path, str]
    decoder_record: tuple[Path, str]
    decoder_product: tuple[Path, str]
    elf: tuple[Path, str]
    payload: tuple[Path, str]
    source_pins: tuple[tuple[Path, str], ...]
    unknown: tuple[str, ...] = (
        "packet_grammar",
        "original_elf_object_roster",
        "original_complete_per_call_histories",
        "observer_integrity",
        "general_execution_effects",
        "hardware_runtime_binding",
        "physical_timing",
    )

    def verify(self):
        for path, digest in (
            self.result,
            self.execution_record,
            self.decoder_record,
            self.decoder_product,
            self.elf,
            self.payload,
            *self.source_pins,
        ):
            _reopen({"path": str(path), "sha256": digest})
        if _join(result_path=self.result[0], execution_root=self.execution_root) != self:
            raise StageGateError("decoder product observation differs from its reopened actual joins")

    def record(self):
        self.verify()
        return {
            name: {"path": str(value[0]), "sha256": value[1]}
            for name, value in (
                ("result", self.result),
                ("execution_record", self.execution_record),
                ("decoder_record", self.decoder_record),
                ("decoder_product", self.decoder_product),
                ("elf", self.elf),
                ("payload", self.payload),
            )
        } | {
            "execution_root": str(self.execution_root),
            "source_pins": [{"path": str(path), "sha256": digest} for path, digest in self.source_pins],
            "unknown": list(self.unknown),
            "scope": "actual execution/decoder file correspondence only; no semantic/runtime/physical authority",
        }


def join_component_decode_products(*, result_path: Path, execution_root: Path) -> ComponentDecodeProducts:
    """Reopen actual same-ELF dispatch, console, decoder and declared packet."""
    observation = _join(result_path=result_path, execution_root=execution_root)
    observation.verify()
    return observation


def _join(*, result_path, execution_root):
    root = Path(execution_root)
    result_path = Path(result_path)
    if (
        not root.is_absolute()
        or root.resolve() != root
        or not root.is_dir()
        or any(part.is_symlink() for part in (root, *root.parents))
        or not result_path.is_relative_to(root)
    ):
        raise StageGateError("decoder joins require the actual canonical private execution owner")
    _reopen(_member(result_path), root)
    result = mapping_file(result_path)
    native = result.get("native")
    if type(native) is not dict or type(native.get("readback_observation")) is not dict:
        raise StageGateError("ordinary execution has no actual coherent decoder observation")
    elf = _reopen(result.get("elf"), root)
    elf_pin = _member(elf)
    if native.get("elf") != str(elf):
        raise StageGateError("ordinary execution and decoder select different ELFs")
    selection = native["readback_observation"]
    record_path = _reopen(selection.get("record"), root)
    product = _reopen(selection.get("product"), root)
    decoded = I.verify(record_path)
    if decoded.get("kind") != "python_call" or decoded.get("stage") != "coherent_memory_decode":
        raise StageGateError("decoder join has no actual selected ordinary callback invocation")
    if elf_pin not in decoded["inputs"] or _member(product) not in decoded["outputs"]:
        raise StageGateError("decoder invocation did not consume the actual ELF or produce its selected result")
    if decoded["stdout"]["sha256"] != sha256_file(product):
        raise StageGateError("decoder product differs from its actual recorded return")
    returned = mapping_file(product)
    memory = returned.get("memory_evidence")
    if type(memory) is not dict or memory.get("status") != "complete" or memory != native.get("readback_memory"):
        raise StageGateError("ordinary memory evidence differs from the actual decoder return")
    if (
        type(returned.get("outputs")) is not dict
        or not returned["outputs"]
        or returned["outputs"] != native.get("outputs")
    ):
        raise StageGateError("ordinary output values differ from the actual decoder return")
    if memory.get("original_elf") != elf_pin:
        raise StageGateError("decoder returned another original ELF identity")
    payload = _reopen(memory.get("payload"), root)
    if _member(payload) not in decoded["outputs"]:
        raise StageGateError("full packet is not an actual declared decoder product")
    console = _reopen(result.get("console"), root)
    if _member(console) not in decoded["inputs"]:
        raise StageGateError("decoder invocation did not consume the original complete console")
    executions = []
    for path in root.rglob("invocation.json"):
        # Inspect canonical membership before hashing/parsing any scanned file.
        # A supplied alias cannot make this observer read an excluded target.
        _reopen(_member(path), root)
        row = mapping_file(path)
        if row.get("kind") == "python_call" and row.get("stage") == "execution":
            row = I.verify(path)
            if elf_pin in row["inputs"]:
                executions.append((path, row))
    if len(executions) != 1 or executions[0][1]["stdout"]["sha256"] != sha256_file(console):
        raise StageGateError("decoder console does not join the unique actual same-ELF execution")
    execution_path, _ = executions[0]
    observation = ComponentDecodeProducts(
        (result_path, sha256_file(result_path)),
        root,
        (execution_path, sha256_file(execution_path)),
        (record_path, sha256_file(record_path)),
        (product, sha256_file(product)),
        (elf, elf_pin["sha256"]),
        (payload, sha256_file(payload)),
        tuple((Path(pin["path"]), pin["sha256"]) for pin in decoded["dependencies"]),
    )
    return observation
