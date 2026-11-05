"""Advisory admission records, seed-fallback identity and author-named shape sweeps."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback import abstraction_feedback as A
from merlin_experiments.phase1.feedback import public_scalability

from merlin.common import invocation_record
from merlin.common.paths import repo_root

REFUSE = (
    "from merlin.compile.scheduling.dense_panel_epochs import schedule_dense_panel_epochs as s\n"
    "try:\n    s(*[None] * 7, None)\nexcept ValueError:\n    pass\n"
)


def _environment():
    root = repo_root()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in (root / "src", root / "packages/merlin-experiments/src"))
    env.pop("MERLIN_SCHEDULING_ADMISSION_REPORT", None)
    return env


def _compile_call(generated, stage, code):
    invocation_record.run([sys.executable, "-B", "-c", code], directory=generated, stage=stage,
                          capture_output=True, text=True, env=_environment(), check=True)


def test_compiler_stderr_records_reach_the_case_row(tmp_path):
    generated = tmp_path / "generated"
    _compile_call(generated, "emit_analysis_bundle", REFUSE)
    # A non-compiler stage cannot add records, even with a well-formed line.
    forged = json.dumps({"schema": "merlin.scheduling_admission.v1", "id": 0, "api": "forged", "admitted": True})
    line = "merlin-scheduling-admission: " + forged
    _compile_call(generated, "selected_rocc_process", f"import sys; sys.stderr.write({line!r} + '\\n')")
    report = A.case_admission(generated, case="PUBLIC_CASE")
    assert report["status"] == "reported"
    (row,) = report["records"]
    assert row["api"] == "merlin.compile.scheduling.dense_panel_epochs.schedule_dense_panel_epochs"
    assert row["case"] == "PUBLIC_CASE" and row["admitted"] is False
    assert "typed epoch law" in row["failed_premise"]
    assert row["fallback_route"].startswith("caller-selected original route")
    assert row["stages"] == ["emit_analysis_bundle"]
    assert report["refused_apis"] == [row["api"]] and report["admitted_apis"] == []
    assert A.case_admission(tmp_path / "missing", case="x")["status"] == "unavailable"


def _seed(tmp_path):
    package = tmp_path / "seed"
    package.mkdir()
    (package / "manifest.yaml").write_text("name: seed\n")
    return A.select_seed(package, cache=tmp_path / "cache")


def _case(tmp_path, artifact=b"kernel", buffer=b"{}"):
    generated = tmp_path / "case" / "generated"
    generated.mkdir(parents=True)
    (generated / "input.interface.mlir").write_text("module {}\n")
    (generated / A.ARTIFACT).write_bytes(artifact)
    (generated / "command_buffer.json").write_bytes(buffer)
    return generated


@pytest.fixture
def seed_compiler(monkeypatch):
    calls = []

    def lower(package, interface, generated, *, contract, timeout, artifact_name):
        calls.append(Path(interface).read_text())
        generated = Path(generated)
        generated.mkdir(parents=True)
        (generated / artifact_name).write_bytes(b"kernel")
        (generated / "command_buffer.json").write_bytes(b"{}")

    import merlin.targetgen.capsule_common as common

    monkeypatch.setattr(common, "lower_interface", lower)
    monkeypatch.setattr(A, "_seed_package", lambda selection, contract: object())
    return calls


def test_seed_fallback_flags_byte_identical_kernels_and_caches(tmp_path, seed_compiler):
    selection = _seed(tmp_path)
    same = A.fallback_to_seed(_case(tmp_path), contract="c", timeout=10, selection=selection)
    assert same["fallback_to_seed"] is True and same["seed_identity"]["command_buffer_identical"] is True
    other = tmp_path / "other"
    changed = A.fallback_to_seed(_case(other, artifact=b"optimized"), contract="c", timeout=10, selection=selection)
    assert changed["fallback_to_seed"] is False
    assert len(seed_compiler) == 1  # one seed emission per exact input
    assert A.fallback_to_seed(_case(tmp_path / "x"), contract="c", timeout=10)["fallback_to_seed"] is None
    with A.seeded(selection):
        assert A.fallback_to_seed(_case(tmp_path / "y"), contract="c", timeout=10)["fallback_to_seed"] is True


def test_seed_refusal_and_changed_seed_copy_give_no_flag(tmp_path, monkeypatch):
    import merlin.targetgen.capsule_common as common

    def refuse(*args, **kwargs):
        raise ValueError("seed declines this shape")

    monkeypatch.setattr(common, "lower_interface", refuse)
    monkeypatch.setattr(A, "_seed_package", lambda selection, contract: object())
    selection = _seed(tmp_path)
    result = A.fallback_to_seed(_case(tmp_path), contract="c", timeout=10, selection=selection)
    assert result["fallback_to_seed"] is None and "seed declines" in result["seed_reason"]
    with pytest.raises(ValueError, match="differs"):
        A.select_seed(selection.package, cache=tmp_path / "cache", expected_sha256="0" * 64)


def test_attached_case_fields_reach_the_public_case_rows(tmp_path):
    runs = tmp_path / "runs"
    _compile_call(runs / "CASE" / "generated", "emit_analysis_bundle", REFUSE)
    observed = {"CASE": {"capsule": "CASE", "status": "pass", "tiers": {}}}
    A.attach_cases(observed, runs, contract="c", timeout=10)
    public = public_scalability.public_result({"observed": observed, "score": {}})
    (row,) = public["cases"]
    assert row["abstraction_admission"]["records"][0]["admitted"] is False
    assert row["fallback_to_seed"] is None and "no seed" in row["fallback_to_seed_basis"]
    late = {"CASE": {"capsule": "CASE"}}
    A.attach_cases(late, runs, contract="c", timeout=10, deadline=0)
    assert late["CASE"]["abstraction_admission"]["reason"] == "request deadline"


def test_shapes_are_author_named_bounded_integers():
    assert A.parse_shapes("m=24:k=2048, m=64") == [{"m": 24, "k": 2048}, {"m": 64}]
    for bad in ("", "m=0", "m=24:m=32", "m=x", "m=24,m=24", "m=1,m=2,m=3,m=4,m=5,m=6,m=7", "m=70000"):
        with pytest.raises(ValueError):
            A.parse_shapes(bad)


def _public_suite(tmp_path):
    cases = tmp_path / "train-suite" / "cases"
    template = cases / "PUBLIC_DENSE"
    template.mkdir(parents=True)
    (template / "capsule.interface.mlir").write_text("public template\n")
    (template / "capsule.yaml").write_text("name: PUBLIC_DENSE\nlabel: public\ninputs:\n- name: A\n  shape: [16, 64]\n")
    hidden = tmp_path / "heldout-suite" / "cases" / "HIDDEN_DENSE"
    hidden.mkdir(parents=True)
    (hidden / "capsule.interface.mlir").write_text("hidden\n")
    (hidden / "capsule.yaml").write_text("name: HIDDEN_DENSE\n")
    return cases


def test_sweep_builds_only_named_shapes_from_the_public_template(tmp_path, monkeypatch, seed_compiler):
    import merlin.targetgen.contract.interface_emit as emit
    import merlin.targetgen.package_runtime as P

    cases = _public_suite(tmp_path)
    read = []
    monkeypatch.setattr(emit, "parse_interface_mlir",
                        lambda text: read.append(text) or {"tensors": {"A": {"shape": [16, 64]}}})
    monkeypatch.setattr(emit, "emit_interface_mlir", lambda cb: json.dumps(cb, sort_keys=True))
    for name in ("load_package", "integrity_scan", "build_package"):
        monkeypatch.setattr(P, name, lambda *a, **k: object())
    built = []

    def lower(package, interface, generated, *, contract, timeout, artifact_name):
        built.append(json.loads(Path(interface).read_text()))
        _compile_call(generated, "emit_analysis_bundle", REFUSE)
        (generated / artifact_name).write_text("kernel")
        (generated / "command_buffer.json").write_text("{}")
        (generated / "input.interface.mlir").write_text(Path(interface).read_text())

    import merlin.targetgen.capsule_common as common

    monkeypatch.setattr(common, "lower_interface", lower)
    monkeypatch.setattr(A, "census", lambda text, *, target: {"fences": {"static_sites": 1, "dynamic": 3}})

    def resize(cb, shape):
        cb["tensors"]["A"]["shape"] = [shape["m"], 64]
        return cb

    context = SimpleNamespace(target="example")
    common_args = dict(submission=tmp_path / "submission", public_cases=cases, allowed_templates={"PUBLIC_DENSE"},
                       resize=resize, contract="c", context=context, timeout=60)
    for template in ("HIDDEN_DENSE", "../../heldout-suite/cases/HIDDEN_DENSE"):
        with pytest.raises(ValueError, match="public"):
            A.coverage_sweep(template=template, shapes=[{"m": 24}], output=tmp_path / "refused", **common_args)
    result = A.coverage_sweep(template="PUBLIC_DENSE", shapes=[{"m": 24}, {"m": 64}], output=tmp_path / "out",
                              **common_args)
    assert read == ["public template\n"]
    assert [b["tensors"]["A"]["shape"] for b in built] == [[24, 64], [64, 64]]
    assert [row["label"] for row in result["shapes"]] == ["sweep-00", "sweep-01"]
    first = result["shapes"][0]
    assert first["status"] == "compiled" and first["numerical"] == {"status": "not_requested"}
    assert first["refusals"][0]["api"].endswith("schedule_dense_panel_epochs")
    assert first["emitted"]["fences"]["dynamic"] == 3
    text = json.dumps(result)
    assert "HIDDEN" not in text and "heldout" not in text and str(tmp_path) not in text
    synthetic = (tmp_path / "out" / "cases" / "sweep-01" / "capsule.yaml").read_text()
    assert "sweep-01" in synthetic and "64" in synthetic
    with pytest.raises(ValueError, match="1 to 6"):
        A.coverage_sweep(template="PUBLIC_DENSE", shapes=[{"m": m} for m in range(1, 8)],
                         output=tmp_path / "many", **common_args)
