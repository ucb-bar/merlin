"""An open model's oracle, recomputed in a process of its own or read back from its content-keyed cache.

The oracle (:func:`merlin.analysis.perf.whole_model_open.oracle`: the model evaluated in numpy, every device
dispatch's operands and result digested) depends on the model and the target, never on the package
under test, so it starts with the build and is joined only where the build first needs it. For a
transformer it is minutes of interpreter-bound work that used to run after the device part, on the
build's critical path; recorded by the digest of everything it is a function of, a later build of the
same model (another candidate, a rebuild) reads it back.

The child recomputes which groups are device dispatches exactly as the build's cut does (the normalized
module's non-host groups, joined to the capture's groups by region), and the build refuses a result
whose dispatch set is not its own. Nothing here names a target.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: Recorded open-model oracles, by :func:`cache_key`.
NAMESPACE = "whole-model-open-oracle"


def device_indices(capsule, original_groups: Sequence[Any], *, target: str) -> list[int]:
    """The capture-group index of every device dispatch the build's cut makes, in the cut's order."""
    from merlin.analysis.perf import whole_model_open as WO
    from merlin.compiler.dialects.lowering import compute_groups as CG

    # The capture's own device groups, as the statement states them (the build checks that the two sets
    # agree before it uses the result).
    by_region = {WO._region_of(g): int(g.index) for g in original_groups if g.placement != CG.HOST}
    from merlin.common import mlir_query as mq

    module = mq.parse(Path(capsule.interface)).clone()  # a private copy of the shared parse, to normalize
    WO.normalize(module)
    indices = []
    for group in CG.form_groups(module, target):
        if group.placement == CG.HOST:
            continue
        index = by_region.get(WO._region_of(group))
        if index is None:
            raise WO.OpenModelError(f"device group {group.index} ({WO._region_of(group)}) is no group of the capture")
        indices.append(index)
    return indices


def cache_key(capsule, *, target: str, extra: str | Path | None) -> str:
    """The closed model's oracle key (model, code, driver, support, contract, facts) plus the leaves file
    an open build may be handed from outside the capsule, under its own label."""
    import hashlib

    from merlin.analysis.perf.whole_model_oracle import oracle_cache_key

    digest = hashlib.sha256(b"open-model-oracle\0")
    digest.update(oracle_cache_key(capsule, target=target).encode())
    digest.update(b"\0")
    digest.update(Path(extra).read_bytes() if extra is not None else b"<capsule-declared>")
    return digest.hexdigest()


def compute(capsule, *, target: str, extra: str | Path | None = None) -> tuple[dict[str, Any], list[int]]:
    """``(oracle, device indices)`` for ``capsule``, in this process."""
    from types import SimpleNamespace

    from merlin.analysis.perf import whole_model_open as WO
    from merlin.common import mlir_query as mq
    from merlin.compiler.dialects.lowering import compute_groups as CG
    from merlin.runtime.backends import base as backends

    arguments, _sources = WO.forward_arguments(capsule, extra=extra)
    # By path: one parse of the capture serves its groups here and, cloned, the normalized cut.
    original = mq.parse(Path(capsule.interface))
    groups = CG.form_groups(original, target)
    indices = device_indices(capsule, groups, target=target)
    result = WO.oracle(
        original,
        arguments,
        [SimpleNamespace(group=i) for i in indices],
        groups,
        group_digest=backends.whole_model_driver(target).program.group_digest,
    )
    return result, indices


def _write(directory: Path, result: dict[str, Any], indices: list[int]) -> None:
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "outputs.npz", *[np.asarray(o) for o in result["outputs"]])
    rows = [[int(k), v] for k, v in result["dispatch"].items()]
    (directory / "dispatch.json").write_text(json.dumps({"indices": indices, "rows": rows}), encoding="utf-8")
    # Written last: a directory holding done.json is a complete record.
    (directory / "done.json").write_text(json.dumps({"outputs": len(result["outputs"])}), encoding="utf-8")


def _read(directory: Path) -> tuple[dict[str, Any], list[int]] | None:
    import numpy as np

    if not (directory / "done.json").is_file():
        return None
    count = int(json.loads((directory / "done.json").read_text(encoding="utf-8"))["outputs"])
    with np.load(directory / "outputs.npz") as stored:
        outputs = [np.asarray(stored[f"arr_{i}"]) for i in range(count)]
    recorded = json.loads((directory / "dispatch.json").read_text(encoding="utf-8"))
    return {"outputs": outputs, "dispatch": {int(k): v for k, v in recorded["rows"]}}, list(recorded["indices"])


def _main(argv: list[str]) -> int:
    """``python -m merlin.analysis.perf.whole_model_open_oracle <capsule dir> <interface> <target> <extra|-> <out>``"""
    from merlin.analysis.perf import whole_model_build as WMB

    capsule_dir, interface, target, extra, out = argv
    capsule = dataclasses.replace(WMB.load_model_capsule(capsule_dir), interface=Path(interface))
    result, indices = compute(capsule, target=target, extra=None if extra == "-" else extra)
    _write(Path(out), result, indices)
    return 0


class OpenOracleJob:
    """The open oracle, started with the build in a child process, or read from its cache."""

    def __init__(self, capsule, *, target: str, extra: str | Path | None = None, cache: bool = True):
        from merlin.common.artifacts import cache_dir

        self.key = cache_key(capsule, target=target, extra=extra)
        self.entry = Path(cache_dir(NAMESPACE)) / self.key[:2] / self.key if cache else None
        self.state = "hit" if self.entry is not None and _read(self.entry) is not None else "miss"
        self._process = None
        if self.state == "miss":
            parent = self.entry.parent if self.entry is not None else Path(tempfile.gettempdir())
            parent.mkdir(parents=True, exist_ok=True)
            self._staged = Path(tempfile.mkdtemp(prefix=".oracle.", dir=parent))
            self._log = self._staged.with_suffix(".log")
            argv = [str(capsule.directory), str(capsule.interface), target, str(extra) if extra else "-"]
            with self._log.open("w", encoding="utf-8") as log:
                self._process = subprocess.Popen(
                    [sys.executable, "-m", __name__, *argv, str(self._staged)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=dict(os.environ),
                )

    def result(self, dispatch_groups: Sequence[int]) -> dict[str, Any]:
        """The oracle, refused unless it was computed for exactly ``dispatch_groups`` (the cut's own)."""
        from merlin.analysis.perf import whole_model_open as WO

        if self._process is None:
            recorded = _read(self.entry)
        else:
            if self._process.wait() != 0:
                tail = self._log.read_text(encoding="utf-8", errors="replace")[-1500:]
                raise WO.OpenModelError(f"the oracle could not be recomputed: {tail}")
            recorded = _read(self._staged)
            if self.entry is not None:
                try:
                    self._staged.rename(self.entry)
                except OSError:  # another build recorded the same oracle first; it is the same value
                    shutil.rmtree(self._staged, ignore_errors=True)
            else:
                shutil.rmtree(self._staged, ignore_errors=True)
            self._log.unlink(missing_ok=True)
        if recorded is None:
            raise WO.OpenModelError("the oracle job wrote no complete record")
        result, indices = recorded
        if sorted(indices) != sorted(int(g) for g in dispatch_groups):
            raise WO.OpenModelError("the oracle was computed for another set of device dispatches than the cut's")
        return result

    def cancel(self) -> None:
        """Stop a child that is still computing (the build failed before it needed the oracle)."""
        if self._process is not None and self._process.poll() is None:
            self._process.kill()
            self._process.wait()
        if self._process is not None:
            shutil.rmtree(self._staged, ignore_errors=True)
            self._log.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
