"""Join explicit execution facts to an original live instruction selection.

This is source/configuration custody, not instruction effects, runtime fidelity
or numerical qualification. No receipt can construct the prepared owner.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from merlin.compile.model_execution_inputs import file_sha256, selected_firrtl

from . import private_instruction_coordinator as C
from .instruction_admission_preparation import InstructionAdmissionSelection, prepare_live


@dataclass(frozen=True)
class InstructionFactsSelection:
    public_effective: Path
    public_sha256: str
    input_facts: Path
    input_sha256: str
    config: str

    def verify(self, *, target: str) -> dict:
        if type(self) is not InstructionFactsSelection or type(self.config) is not str or not self.config:
            raise ValueError("instruction facts require their exact original typed configuration selection")
        result = {}
        for role, path, digest in (
            ("public_effective", self.public_effective, self.public_sha256),
            ("private_input", self.input_facts, self.input_sha256),
        ):
            if (
                not isinstance(path, Path)
                or not path.is_absolute()
                or path.resolve() != path
                or any(p.is_symlink() for p in (path, *path.parents))
                or not path.is_file()
                or type(digest) is not str
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or file_sha256(path) != digest
            ):
                raise ValueError("instruction facts differ from their explicit original byte selection")
            result[role] = selected_firrtl(path, target=target, config=self.config)
            if result[role]["sha256"] != digest:
                raise ValueError("instruction facts changed during source/configuration selection")
        if result["public_effective"]["firrtl_sha256"] != result["private_input"]["firrtl_sha256"]:
            raise ValueError("instruction facts select different execution FIRRTL")
        return result


@contextmanager
def selected_facts(prepared: C.PreparedInstructionSelection, selection: InstructionFactsSelection):
    """Reopen facts and call the original live join before and after use.

    The legacy role name ``private_input`` is the coordinator's existing API;
    public callers explicitly select their actual input facts without requiring
    private model metadata. No environment or private gate is replaced.
    """
    if type(prepared) is not C.PreparedInstructionSelection or type(selection) is not InstructionFactsSelection:
        raise ValueError("instruction facts need original live preparation and typed facts selection")
    target = prepared.service_selection[0]
    before = selection.verify(target=target)
    prepared.require_facts(before)
    try:
        yield prepared
    finally:
        after = selection.verify(target=target)
        if after != before:
            raise ValueError("instruction facts/source binding changed during live admission")
        prepared.require_facts(after)


@contextmanager
def prepare_for_facts(selection: InstructionAdmissionSelection, facts: InstructionFactsSelection):
    """Issue once, then retain the exact original owner through its checked use."""
    if type(selection) is not InstructionAdmissionSelection or type(facts) is not InstructionFactsSelection:
        raise ValueError("instruction preparation requires original typed admission and facts selections")
    # Fail before expensive host tools when the explicit facts are unavailable.
    before = facts.verify(target=selection.target)
    prepared = prepare_live(selection)
    with selected_facts(prepared, facts) as original:
        if facts.verify(target=selection.target) != before:
            raise ValueError("instruction facts changed during original preparation")
        yield original
