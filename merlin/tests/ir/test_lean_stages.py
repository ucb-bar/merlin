"""Stage capture and the C1 checker on a real Merlin pass (design §8.1, §8.2, §9).

`prune_dead_pure_tensor_ops` runs on a fixture with dead ops; the recorded before/after pair must be
verified by C1, seeded faults on the real output must be rejected, and recording must not change
what the pass emits (byte-neutrality)."""

from __future__ import annotations

import copy
import json
import subprocess

from merlin.common.paths import repo_root
from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import stages as lean_stages
from merlin.verify.lean.export import export_file

FIXTURE = repo_root() / "merlin" / "tests" / "fixtures" / "lean_dce_forward.mlir"


def _export(tools, path, tmp_path, tag):
    # The authoritative exporter (design D9): the pipeline's own MLIR, never a fallback reader.
    out = tmp_path / f"{tag}.json"
    export_file(path, out, backend="mlir", m2m_python=tools.m2m_python)
    return json.loads(out.read_text())


def _record(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))
    return preprocess_text(FIXTURE.read_text())


def test_recording_is_byte_neutral(tmp_path, monkeypatch):
    plain, _ = preprocess_text(FIXTURE.read_text())
    recorded, _ = _record(tmp_path, monkeypatch)
    assert recorded == plain
    names = sorted(p.name.split("_", 1)[1] for p in lean_stages.completed(tmp_path / "stages"))
    assert names == ["xdsl-c-interface", "xdsl-parse", "xdsl-pruned", "xdsl-quant-lowered"]


def _body(doc):
    return next(o for o in doc["module"]["ops"] if o["name"] == "func.func")["regions"][0]["blocks"][0]["ops"]


def _check(tools, tmp_path, before, after, tag):
    (tmp_path / "b.json").write_text(json.dumps(before))
    (tmp_path / f"{tag}.json").write_text(json.dumps(after))
    proc = subprocess.run([str(tools.lean_exe), "check-c1", str(tmp_path / "b.json"), str(tmp_path / f"{tag}.json"),
                           "forward"], capture_output=True, text=True, check=False)
    return json.loads(proc.stdout)["verdict"]


def test_c1_verifies_real_dce_and_rejects_faults(tmp_path, monkeypatch, lean_tools):
    _record(tmp_path, monkeypatch)
    stage = next(p for p in lean_stages.completed(tmp_path / "stages") if p.name.endswith("xdsl-pruned"))
    assert json.loads((stage / "meta.json").read_text())["changed"]
    before = _export(lean_tools, stage / "before.mlir", tmp_path, "before")
    after = _export(lean_tools, stage / "after.mlir", tmp_path, "after")
    assert len(_body(after)) < len(_body(before))
    assert _check(lean_tools, tmp_path, before, after, "real") == "verified"

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
        assert _check(lean_tools, tmp_path, before, mutated, name) == "abstracted", name


def test_receipts_for_real_preprocessing(tmp_path, monkeypatch, lean_tools):
    import jsonschema

    from merlin.verify.lean.check import check_stage

    _record(tmp_path, monkeypatch)
    stages = lean_stages.completed(tmp_path / "stages")
    # Chain continuity: each boundary consumes exactly what the previous one produced.
    for prev, nxt in zip(stages, stages[1:]):
        assert (prev / "after.mlir").read_bytes() == (nxt / "before.mlir").read_bytes()
    schema = json.loads((repo_root() / "merlin" / "contract" / "schemas" / "lean_receipt.schema.json").read_text())
    for stage in stages:
        receipt = check_stage(stage, m2m_python=lean_tools.m2m_python)
        jsonschema.validate(receipt, schema)
        assert receipt["verdict"] == "verified", receipt.get("reason")
        assert receipt["axioms"] == ["propext", "Classical.choice", "Quot.sound"]
    # A wrong after-file must not verify.
    pruned = next(s for s in stages if s.name.endswith("xdsl-pruned"))
    text = (pruned / "after.mlir").read_text()
    (pruned / "after.mlir").write_text(text.replace('"arith.muli"', '"arith.addi"', 1))
    assert check_stage(pruned, m2m_python=lean_tools.m2m_python)["verdict"] != "verified"


def test_concurrent_recorders_never_collide(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("MERLIN_LEAN_STAGES", str(tmp_path / "stages"))

    def record(worker):
        for k in range(20):
            lean_stages.record_boundary(f"w{worker}-{k}", "test", f"before {worker} {k}", f"after {worker} {k}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(record, range(8)))
    stages = lean_stages.completed(tmp_path / "stages")
    assert len(stages) == 160
    assert len({p.name.split("_", 1)[0] for p in stages}) == 160  # every sequence number used once
    for p in stages:
        meta = json.loads((p / "meta.json").read_text())
        worker, k = meta["boundary"][1:].split("-")
        assert (p / "before.mlir").read_text() == f"before {worker} {k}"
        assert meta["run"] == lean_stages.RUN

