"""Selected final-program checks on a declared section of the actual linked ELF.

This host-only bridge composes with the caller's existing admission policy.
Section selection and independently qualified decoding remain explicit inputs;
it grants no runtime, physical mapping, numerical or timing qualification.
"""

from __future__ import annotations

import importlib
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from merlin.common.digest import sha256_bytes
from merlin.compile.scheduling.library_sources import shared_library_sources
from merlin.compile.scheduling.program_limits import ProgramDecoder, ProgramLimits, finalize_program
from merlin.compile.scheduling.selected_contracts import SelectedSchedulingContracts
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.contract.execution_service import _callback_selection, _same_callback
from merlin.targetgen.elf_lanes import ElfUnreadable, read_program_section


@dataclass(frozen=True)
class _ProgramCheck:
    selected: SelectedSchedulingContracts
    decoder: ProgramDecoder
    section: str
    reader_max_bytes: int
    limits: ProgramLimits | None
    _state: str = field(init=False, repr=False)
    _decoder_selection: tuple = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "_state", self.state())
        object.__setattr__(self, "_decoder_selection", _callback_selection(self.decoder.decode))

    def state(self):
        return json.dumps(
            {
                "selected": self.selected.record(),
                "program": self.selected.program.declarations_json,
                "storage": self.selected.storage.declarations_json if self.selected.storage else None,
                "decoder": [self.decoder.configuration_identity, self.decoder.qualification_identity],
                "section": self.section,
                "reader_max_bytes": self.reader_max_bytes,
                "limits": asdict(self.limits) if self.limits else None,
            },
            sort_keys=True,
        )

    def revalidate(self):
        self.selected.revalidate()
        if self.state() != self._state or not _same_callback(
            _callback_selection(self.decoder.decode), self._decoder_selection
        ):
            raise ValueError("linked program selection or decoder changed; reselect and replan")

    def validate_report(self, stored):
        self.revalidate()
        if (
            stored.get("schema") != "merlin.model_check.program_admission.v1"
            or stored.get("selection") != json.loads(self._state)
            or stored.get("library_sources_sha256") != shared_library_sources()
        ):
            raise ValueError("linked program report belongs to a different policy context; reevaluate")
        self.revalidate()

    def evaluate(self, *, elf, evidence_root, prerequisite_result):
        self.revalidate()
        result = {
            "schema": "merlin.model_check.program_admission.v1",
            "status": "refused",
            "elf_sha256": prerequisite_result["elf_sha256"],
            "prerequisite": prerequisite_result,
            "selection": json.loads(self._state),
            "library_sources_sha256": shared_library_sources(),
            "scope": "declared linked section; runtime coverage and execution qualification not established",
        }
        if prerequisite_result["status"] == "refused":
            result["reason"] = "prerequisite linked-image policy refused"
        else:
            try:
                image, location = read_program_section(elf, self.section, max_bytes=self.reader_max_bytes)
                result["location"] = location
                result["image_sha256"] = sha256_bytes(image)
                checked = finalize_program(image, self.limits, self.decoder, selected_contracts=self.selected)
                # Admission cannot rewrite a linked program or its launcher.
                # Never approve a hypothetical segmented replacement.
                if checked.segments != (image,):
                    raise ValueError("linked program needs pre-link replanning; admission cannot segment it")
                result["program"] = checked.report
                result["status"] = "accepted"
            except (ValueError, ElfUnreadable) as error:
                result["reason"] = str(error)
        self.revalidate()
        evidence_root.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(result, indent=1, sort_keys=True) + "\n").encode()
        digest = sha256_bytes(payload)
        path = evidence_root / ("program-admission-" + digest + ".json")
        try:
            with path.open("xb") as stream:
                stream.write(payload)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError("existing linked program report changed; retain original evidence") from None
        return {**result, "report_path": str(path.resolve()), "report_sha256": digest}


def service(
    *,
    prerequisite: LinkedElfAdmissionService,
    selected_contracts: SelectedSchedulingContracts,
    decoder: ProgramDecoder,
    decoder_sources: tuple[tuple[str, str], ...],
    section: str,
    reader_max_bytes: int,
    limits: ProgramLimits | None = None,
) -> LinkedElfAdmissionService:
    """Add actual-byte checks to an original policy for ordinary build/cache/relink.

    ``decoder_sources`` must come from the original independently qualified
    decoder provider. Source pins record attribution, not qualification. The
    reader budget bounds extraction; it never enlarges selected hardware limits.
    No section discovery, backend import, segmentation or SDK grant is supplied.
    """
    if (
        type(selected_contracts) is not SelectedSchedulingContracts
        or selected_contracts.program is None
        or type(decoder) is not ProgramDecoder
        or type(prerequisite) is not LinkedElfAdmissionService
        or limits is not None
        and type(limits) is not ProgramLimits
    ):
        raise ValueError("linked program checks require original typed contracts, decoder and prerequisite")
    if (
        type(section) is not str
        or not section
        or "\0" in section
        or len(section.encode()) > 1024
        or type(reader_max_bytes) is not int
        or reader_max_bytes <= 0
    ):
        raise ValueError("linked program checks require an explicit section and positive reader byte budget")
    selected_contracts.revalidate()
    prerequisite.verify(selected_contracts.target)
    callback = _callback_selection(decoder.decode)
    if callback is None or type(decoder_sources) is not tuple or not decoder_sources:
        raise ValueError("linked program decoder requires an actual pinned function or bound method")
    owner = inspect.getsourcefile(callback[1])
    if owner is None or (str(Path(owner).resolve()), file_digest(Path(owner))) not in decoder_sources:
        raise ValueError("linked program decoder has no original pinned inspected source owner")
    # Use the existing reviewed shared roster. This host bridge does not export
    # evaluator code or infer new authoring access from module existence.
    pins = dict(decoder_sources)
    if len(pins) != len(decoder_sources):
        raise ValueError("linked program decoder source pins must be unique")
    for name, digest in shared_library_sources().items():
        path = Path(importlib.import_module(name).__file__).resolve()
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("linked program source selections disagree")
        pins[str(path)] = digest
    for name in (
        __name__,
        "merlin.targetgen.elf_lanes",
        "merlin.targetgen.contract.elf_admission",
        "merlin.targetgen.contract.execution_service",
        "merlin.targetgen.contract.build_service",
        "merlin.common.strict_json",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        digest = file_digest(path)
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("linked program source selections disagree")
        pins[str(path)] = digest
    for path in (selected_contracts.facts_path, selected_contracts.software_path):
        digest = file_digest(path)
        if str(path) in pins and pins[str(path)] != digest:
            raise ValueError("linked program source selections disagree")
        pins[str(path)] = digest
    check = _ProgramCheck(selected_contracts, decoder, section, reader_max_bytes, limits)
    result = LinkedElfAdmissionService(
        selected_contracts.target,
        check.evaluate,
        tuple(sorted(pins.items())),
        prerequisite=prerequisite,
        result_validator=check.validate_report,
    )
    result.verify(selected_contracts.target)
    return result
