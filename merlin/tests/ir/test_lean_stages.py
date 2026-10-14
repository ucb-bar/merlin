"""Stage capture and the C1 checker on a real Merlin pass (design §8.1, §8.2, §9).

`prune_dead_pure_tensor_ops` runs on a fixture with dead ops; the recorded before/after pair must be
verified by C1, seeded faults on the real output must be rejected, and recording must not change
what the pass emits (byte-neutrality)."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import out_dir, repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean.export import export_file

FIXTURE = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dce_forward.mlir"
LEAN_EXE = out_dir() / "build" / "lean" / "build" / "bin" / "merlin-lean"
# The authoritative exporter (design D9) parses with the pipeline's MLIR in the model2MLIR env.
M2M_PYTHON = Path(os.environ.get("MERLIN_M2M_PYTHON", Path.home() / "Projects" / "model2MLIR" / ".venv" / "bin" / "python"))


def _export(path, tmp_path, tag):
    out = tmp_path / f"{tag}.json"
    backend = "mlir" if M2M_PYTHON.exists() else "xdsl"
    export_file(path, out, backend=backend, m2m_python=M2M_PYTHON)
    return json.loads(out.read_text())


def _record(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    return preprocess_text(FIXTURE.read_text())


def test_recording_is_byte_neutral(tmp_path, monkeypatch):
    plain, _ = preprocess_text(FIXTURE.read_text())
    recorded, _ = _record(tmp_path, monkeypatch)
    assert recorded == plain
    names = sorted(p.name.split("_", 1)[1] for p in (tmp_path / "stages").iterdir())
    assert names == ["xdsl-c-interface", "xdsl-pruned", "xdsl-quant-lowered"]


def _body(doc):
    return next(o for o in doc["module"]["ops"] if o["name"] == "func.func")["regions"][0]["blocks"][0]["ops"]


def _check(tmp_path, before, after, tag):
    (tmp_path / "b.json").write_text(json.dumps(before))
    (tmp_path / f"{tag}.json").write_text(json.dumps(after))
    proc = subprocess.run([str(LEAN_EXE), "check-c1", str(tmp_path / "b.json"), str(tmp_path / f"{tag}.json"), "forward"],
                          capture_output=True, text=True, check=False)
    return json.loads(proc.stdout)["verdict"]


@pytest.mark.skipif(not LEAN_EXE.exists(), reason="merlin-lean not built (cd merlin/lean && lake build)")
def test_c1_verifies_real_dce_and_rejects_faults(tmp_path, monkeypatch):
    _record(tmp_path, monkeypatch)
    stage = next(p for p in (tmp_path / "stages").iterdir() if p.name.endswith("xdsl-pruned"))
    assert json.loads((stage / "meta.json").read_text())["changed"]
    before = _export(stage / "before.mlir", tmp_path, "before")
    after = _export(stage / "after.mlir", tmp_path, "after")
    assert len(_body(after)) < len(_body(before))
    assert _check(tmp_path, before, after, "real") == "verified"

    def generic(doc):
        return next(o for o in _body(doc) if o["name"] == "linalg.generic")

    def drop_fill(d):
        ops = _body(d)
        del ops[next(i for i, o in enumerate(ops) if o["name"] == "linalg.fill")]

    def to_parallel(d):
        g = generic(d)
        (g["props"].get("iterator_types") or g["attrs"]["iterator_types"])["elements"][2]["value"] = "parallel"

    def swap_operands(d):
        body_ops = generic(d)["regions"][0]["blocks"][0]["ops"]
        next(o for o in body_ops if o["name"] == "arith.muli")["operands"].reverse()

    for name, fault in [("drop_fill", drop_fill), ("to_parallel", to_parallel), ("swap", swap_operands)]:
        mutated = copy.deepcopy(after)
        fault(mutated)
        assert _check(tmp_path, before, mutated, name) == "abstracted", name
