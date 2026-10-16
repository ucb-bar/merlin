"""Run-owned original selection for negative native result-format evidence."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import yaml

from merlin.common.digest import is_sha256

_ISSUED = object()
_RECORD = "native_result_format_selection"


def _bytes(path):
    path = Path(path).absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValueError("original native result selection must be an ordinary absolute file")
    return path.read_bytes()


@dataclass(frozen=True)
class FrozenSelection:
    path: Path
    source: Path
    sha256: str
    target: str
    _issuer: object

    def verify(self):
        if self._issuer is not _ISSUED or not is_sha256(self.sha256):
            raise ValueError("native result selection has no original run owner")
        raw = _bytes(self.path)
        if sha256(raw).hexdigest() != self.sha256 or _bytes(self.source) != raw:
            raise ValueError("original native result selection bytes changed")
        return raw

    def issue(self, *, original_contract_bytes, original_facts_bytes):
        from merlin.targetgen.rtl.result_format_ceiling import verify_original_selection

        return verify_original_selection(
            original_contract_bytes=original_contract_bytes, original_facts_bytes=original_facts_bytes,
            selection_bytes=self.verify(), expected_selection_sha256=self.sha256,
        )


def freeze_for_run(path, *, target, run_dir, prior=None, resuming=False):
    """Freeze explicit operator bytes privately; original hardware verification is later."""
    from merlin_experiments.corpus.preparation import copy_input

    if not path:
        if resuming and prior is not None:
            raise ValueError("resume refused: original native result selection removed")
        return None, None
    source = Path(path).absolute()
    raw = _bytes(source)
    if type(json.loads(raw)) is not dict:
        raise ValueError("native result selection must be an original JSON object")
    frozen = Path(run_dir).absolute() / "private_native_result_input" / "selection.json"
    record = {"target": target, "source": str(source), "sha256": sha256(raw).hexdigest(),
              "frozen_path": str(frozen)}
    if resuming:
        if prior != record or _bytes(frozen) != raw:
            raise ValueError("resume refused: original native result selection changed")
    else:
        copy_input(source, frozen, private=True, expected_sha256=record["sha256"])
        frozen.parent.chmod(0o700)
        frozen.chmod(0o400)
    selected = FrozenSelection(frozen, source, record["sha256"], target, _ISSUED)
    selected.verify()
    return selected, record


def from_run(run_dir, *, target):
    """Reopen only the normal preparation's protected immutable selection."""
    root = Path(run_dir).absolute()
    environment_path = root / "environment.yaml"
    if not environment_path.exists() and not environment_path.is_symlink():
        return None
    environment = yaml.safe_load(_bytes(environment_path))
    if type(environment) is not dict:
        raise ValueError("native result selection requires an original run environment")
    record = environment.get(_RECORD)
    if record is None:
        return None
    if (type(record) is not dict or set(record) != {"target", "source", "sha256", "frozen_path"}
        or record.get("target") != target or not is_sha256(record.get("sha256"))
        or record.get("frozen_path") != str(root / "private_native_result_input" / "selection.json")
        or type(record.get("source")) is not str):
        raise ValueError("native result selection differs from its original run owner")
    selected = FrozenSelection(Path(record["frozen_path"]), Path(record["source"]), record["sha256"], target, _ISSUED)
    selected.verify()
    return selected
