"""Explicit host-owned source choices through the existing broker transport.

This opt-in workflow compiles only an original caller's fixed, authorized
development source. It grants no author imports, access to another source,
numerical verdict, generated-compiler lineage or hardware qualification. The
ordinary authoring launcher does not select this workflow automatically.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
from pathlib import Path

from .broker import BrokerAction
from .broker_policy import SOURCE_CHOICES_V1, WorkflowPolicy, _record_host_refusal
from .contracts import StageGateError, canonical_json

ACTION = "compile-source-choice"
MAX_RESPONSE_BYTES = 64 << 10


def action() -> BrokerAction:
    return BrokerAction(
        ACTION,
        ("__host_owned_source_choice__", "{request}"),
        ("request",),
        "compile one original authorized development source with a predeclared shared-pass choice; "
        "return artifact identities only, without numerical, timing or hardware qualification",
        False,
    )


def _directory_identity(observation):
    return observation.st_dev, observation.st_ino, stat.S_IFMT(observation.st_mode)


def _request_bytes(candidate: Path, value: str, maximum: int, directory_identity) -> bytes:
    path = Path(value)
    if not path.name or (path.parent != candidate if path.is_absolute() else len(path.parts) != 1):
        raise StageGateError("source choice request must be a direct candidate-owned file")
    root_fd = os.open(candidate, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        if _directory_identity(os.fstat(root_fd)) != directory_identity:
            raise StageGateError("source choice original candidate directory changed")
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
        with os.fdopen(descriptor, "rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
                raise StageGateError("source choice request exceeds its original regular-file budget")
            payload = stream.read(maximum + 1)
            after = os.fstat(stream.fileno())
            current = os.stat(path.name, dir_fd=root_fd, follow_symlinks=False)
            if _directory_identity(os.stat(candidate, follow_symlinks=False)) != directory_identity or (
                current.st_dev,
                current.st_ino,
            ) != (before.st_dev, before.st_ino):
                raise StageGateError("source choice request ownership changed during its read")
    finally:
        os.close(root_fd)
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if len(payload) > maximum or any(getattr(before, field) != getattr(after, field) for field in fields):
        raise StageGateError("source choice request moved during its bounded read")
    return payload


def _closed_object(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise ValueError("source choice request repeats a JSON field")
        document[key] = value
    return document


class SourceChoicesPolicy(WorkflowPolicy):
    """One live original source owner; candidate requests select no authority."""

    workflow_id = SOURCE_CHOICES_V1

    def __init__(self, *, candidate, target_experiment, receipt_path, source_choices):
        from merlin.llvmlower.source_choices import SourceDenseChoices

        if type(source_choices) is not SourceDenseChoices:
            raise StageGateError("source choices require the exact live original owner")
        candidate, receipt_path = Path(candidate), Path(receipt_path)
        if (
            not candidate.is_absolute()
            or candidate.resolve() != candidate
            or candidate.is_symlink()
            or not candidate.is_dir()
            or not receipt_path.is_absolute()
            or receipt_path.resolve() != receipt_path
            or receipt_path.parent == candidate
            or receipt_path.parent.is_relative_to(candidate)
            or candidate.is_relative_to(receipt_path.parent)
            or not source_choices.output_root.is_relative_to(receipt_path.parent)
            or source_choices.output_root == receipt_path.parent
            or source_choices.output_root.is_relative_to(candidate)
            or candidate.is_relative_to(source_choices.output_root)
            or source_choices.source.is_relative_to(candidate)
        ):
            raise StageGateError("source choices require fixed private source and output ownership")
        source_choices.revalidate()
        super().__init__(candidate=candidate, target_experiment=target_experiment, receipt_path=receipt_path)
        self.source_choices = source_choices
        self._original_owner = source_choices
        self._candidate = candidate
        self._receipt_path = receipt_path
        self._candidate_directory = _directory_identity(os.stat(candidate, follow_symlinks=False))

    @property
    def unavailable(self):
        return {}

    def build_registry(self):
        return (action(),)

    def validate_actions(self, actions):
        original = action()
        if set(actions) != {ACTION} or actions[ACTION] != original:
            raise StageGateError("source choices require their exact fixed host action registry")
        self._verify_owner()

    def _selection_changed(self):
        return (
            self.source_choices is not self._original_owner
            or self.candidate != self._candidate
            or self.receipt_path != self._receipt_path
            or self.candidate.resolve() != self.candidate
            or self.candidate.is_symlink()
            or self.receipt_path.resolve() != self.receipt_path
            or _directory_identity(os.stat(self.candidate, follow_symlinks=False)) != self._candidate_directory
        )

    def _verify_owner(self):
        if self._selection_changed():
            raise StageGateError("source choice original workflow selection changed")
        self._original_owner.revalidate()
        if self._selection_changed():
            raise StageGateError("source choice original workflow selection changed during verification")

    def execute(self, request, action_name, rendered, call_index, timeout_s, started):
        if action_name != ACTION:
            raise StageGateError("source choice workflow cannot execute another action")
        try:
            self._verify_owner()
            payload = _request_bytes(
                self.candidate, rendered["request"], self._original_owner.max_request_bytes, self._candidate_directory
            )
            choice = json.loads(payload, object_pairs_hook=_closed_object)
            request_root = self.receipt_path.parent / "source_choice_requests"
            request_root.mkdir(mode=0o700, exist_ok=True)
            if request_root.resolve() != request_root or request_root.is_symlink():
                raise StageGateError("source choice private request directory changed")
            retained = request_root / f"{call_index:04d}.json"
            with retained.open("xb") as stream:
                stream.write(payload)
            retained.chmod(0o400)
            if time.monotonic() - started >= timeout_s:
                raise StageGateError("source choice admission deadline expired before compilation")
            result = self._original_owner.compile(choice)
            self._verify_owner()
            if time.monotonic() - started >= timeout_s:
                raise StageGateError("source choice compilation completed beyond its admission budget")
            document = canonical_json(result.record())
            if (
                self._selection_changed()
                or retained.is_symlink()
                or hashlib.sha256(retained.read_bytes()).digest() != hashlib.sha256(payload).digest()
            ):
                raise StageGateError("source choice request or workflow changed during compilation")
            if len(document) > MAX_RESPONSE_BYTES:
                raise StageGateError("source choice result exceeds its fixed public response budget")
            if time.monotonic() - started >= timeout_s:
                raise StageGateError("source choice result verification completed beyond its admission budget")
            response = dict(returncode=0, stdout=document.decode(), stderr="")
        except Exception as error:
            # Original paths, callback messages and private source products are
            # not candidate feedback. Actual failures remain ordinary refusals.
            _record_host_refusal(
                self, error, round_index=None, call_index=call_index, private_root=self._receipt_path.parent
            )
            response = dict(
                returncode=125,
                stdout="",
                stderr=f"original source choice refused ({type(error).__name__})",
            )
        return {**response, "elapsed_s": round(time.monotonic() - started, 3)}, None
