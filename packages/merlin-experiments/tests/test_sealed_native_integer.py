"""Explicit source semantics and read-only CAS reuse preserve original seal gates."""

import copy
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from merlin_experiments.capture_execution import native_integer as N
from merlin_experiments.capture_execution import readonly_runtime as R
from merlin_experiments.capture_execution import runtime_store as STORE
from merlin_experiments.capture_execution import sealed_m2m as M
from merlin_experiments.capture_execution.sealed_static import _file_digest


def fixture(tmp_path):
    (
        tmp_path / "model.mlir"
    ).write_text("""module { func.func @forward(%x: tensor<2xi8>) -> (tensor<2xi8>, tensor<2xi8>) {
      func.return %x, %x : tensor<2xi8>, tensor<2xi8>
    } }""")
    np.savez(tmp_path / "inputs.npz", in0=np.asarray([-2, 3], dtype=np.int8))
    (tmp_path / "golden.json").write_text("[[-2,3],[-2,3]]")
    (tmp_path / "frontend-trace.json").write_text(
        json.dumps(
            {
                "schema": "m2m.frontend_trace.v1",
                "graphs": {"original": {"schema": "m2m.frontend_graph.v1", "status": "complete"}},
            }
        )
    )
    meta = dict(
        capture_quantization="already_materialized",
        source_mode=N.MODE,
        quantization_stats={"applied": False},
        ok=True,
        opaque=0,
        input_abi=[dict(shape=[2], dtype="i8")],
        output_abi=[dict(shape=[2], dtype="i8"), dict(shape=[2], dtype="i8")],
    )
    plan = dict(
        dtype="int8",
        recipe=None,
        worker_options={"source_mode": N.MODE},
        native_integer_verifier_sha256=_file_digest(Path(N.__file__)),
    )
    return plan, meta


def test_native_source_mode_is_explicit_and_keeps_legacy_commands():
    output = Path("/capture-out")
    plain = M._command_v2(output, dtype="int8", recipe=True)
    assert plain == M._command_v2(output, dtype="int8", recipe=True, options={})
    with pytest.raises(M.SealedM2MError, match="selected recipe"):
        M._command_v2(output, dtype="int8", recipe=False)
    native = M._command_v2(output, dtype="int8", recipe=False, options={"source_mode": N.MODE})
    assert "--already-quantized" in native[-1] and "--recipe" not in native[-1]
    for dtype, recipe, extras in [
        ("fp32", False, {}),
        ("int8", True, {}),
        ("int8", False, {"stage_fp32": False}),
        ("int8", False, {"agreement_tolerance": [0, 0]}),
    ]:
        with pytest.raises(M.SealedM2MError, match="without recipe"):
            M._command_v2(output, dtype=dtype, recipe=recipe, options={"source_mode": N.MODE, **extras})
    with pytest.raises(M.SealedM2MError, match="source_mode"):
        M._worker_options({"source_mode": "native_integer_unknown"})


def test_native_source_validates_all_signed_outputs_and_refuses_numeric_mutations(tmp_path):
    plan, meta = fixture(tmp_path)
    result = N.validate(tmp_path, meta, plan)
    assert len(result["numeric_contract"]["outputs"]) == 2
    assert all(row["compare"] == "exact_int" for row in result["numeric_contract"]["outputs"])
    for changes in [
        dict(recipe_sha256="a" * 64),
        dict(opaque=1),
        dict(quantization_stats={"applied": True}),
        dict(output_abi=[dict(shape=[2], dtype="f32")]),
    ]:
        with pytest.raises(ValueError):
            N.validate(tmp_path, {**meta, **changes}, plan)
    (tmp_path / "golden.json").write_text("[[-2,3]]")
    with pytest.raises(ValueError, match="omitted"):
        N.validate(tmp_path, meta, plan)
    (tmp_path / "golden.json").write_text("[[-2.0,3.0],[-2,3]]")
    with pytest.raises(ValueError, match="signed output"):
        N.validate(tmp_path, meta, plan)


def test_floating_intermediates_and_verifier_mutation_do_not_pass_as_integer_source(tmp_path):
    plan, meta = fixture(tmp_path)
    (tmp_path / "model.mlir").write_text("""module { func.func @forward(%x: tensor<2xi8>) -> tensor<2xi8> {
      %c = arith.constant 0.0 : f32
      func.return %x : tensor<2xi8>
    } }""")
    with pytest.raises(ValueError, match="floating intermediate"):
        N.validate(tmp_path, meta, plan)
    with pytest.raises(ValueError, match="verifier source changed"):
        N.validate(tmp_path, meta, {**plan, "native_integer_verifier_sha256": "0" * 64})


def test_readonly_cache_hit_mounts_only_verified_bytes_without_source_writes(tmp_path, monkeypatch):
    store, destination = tmp_path / "store", tmp_path / "task"
    store.mkdir()
    destination.mkdir()
    original = tmp_path / "original"
    for relative in ["opt/capture-venv/bin/python", "opt/base/lib/test.py"]:
        path = original / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"selected")
    plan = dict(
        base="/opt/base",
        system_libs=[],
        selected_trees={
            "venv": M._snapshot_tree(original / "opt/capture-venv"),
            "base": M._snapshot_tree(original / "opt/base"),
        },
    )
    entry = store / STORE._identity(plan)
    original.rename(entry)
    marker = store / (entry.name + ".complete.json")
    marker.write_text(json.dumps(dict(identity=entry.name, base=plan["base"], system_libs=[])))
    before = {
        str(p.relative_to(store)): (p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_nlink)
        for p in store.rglob("*")
    }
    monkeypatch.setattr(STORE, "_entry_lock", lambda *_a, **_kw: pytest.fail("source CAS lock was opened"))
    selected = R.select(plan, entry=entry, marker_sha256=_file_digest(marker), destination=destination)
    assert R.verify(plan, selected, destination) == selected
    runtime = destination / "runtime"
    runtime.mkdir()
    R.stage(selected, runtime)
    assert not [p for p in runtime.rglob("*") if p.is_file()]
    argv = M._sandbox_argv(
        Path("/bwrap"), runtime, destination, destination, ("/python",), Path("/capture-out"), runtime_reuse=selected
    )
    assert argv[argv.index(str(entry / "opt")) - 1] == "--ro-bind"
    assert before == {
        str(p.relative_to(store)): (p.stat().st_mtime_ns, p.stat().st_ctime_ns, p.stat().st_nlink)
        for p in store.rglob("*")
    }
    bad = copy.deepcopy(selected)
    bad["inventory_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity changed"):
        R.verify(plan, bad, destination)
    with pytest.raises(ValueError, match="pinned selected cache hit"):
        R.select(plan, entry=entry, marker_sha256="0" * 64, destination=destination)


def test_absent_runtime_leaf_needs_exact_versioned_readonly_composition(tmp_path):
    root = tmp_path / "cas"
    root.mkdir()
    entry = root / "initial"
    for name in ["opt/capture-venv/bin/cache/python", "opt/capture-venv/bin/tool", "opt/base/bin/python"]:
        path = entry / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"original selected bytes")
    plan = dict(
        base="/opt/base",
        system_libs=[],
        selected_trees={
            "venv": M._snapshot_tree(entry / "opt/capture-venv"),
            "base": M._snapshot_tree(entry / "opt/base"),
        },
    )
    renamed = root / STORE._identity(plan)
    entry.rename(renamed)
    entry = renamed
    marker = root / (entry.name + ".complete.json")
    marker.write_text(json.dumps(dict(identity=entry.name, base=plan["base"], system_libs=[])))
    leaf = tmp_path / "selected-cache/python"
    leaf.parent.mkdir()
    (entry / "opt/capture-venv/bin/cache/python").rename(leaf)
    (entry / "opt/capture-venv/bin/cache").rmdir()
    with pytest.raises(ValueError, match="CAS bytes differ"):
        R.select(plan, entry=entry, marker_sha256=_file_digest(marker), destination=tmp_path)
    original_identity = entry.name
    other = root / ("a" * 64)
    entry.rename(other)
    marker.unlink()
    entry = other
    marker = root / (entry.name + ".complete.json")
    marker.write_text(json.dumps(dict(identity=entry.name, base=plan["base"], system_libs=[])))
    with pytest.raises(ValueError, match="pinned selected cache hit"):
        R.select(plan, entry=entry, marker_sha256=_file_digest(marker), destination=tmp_path)
    for limits in [dict(max_leaves=65, max_total_bytes=100), dict(max_leaves=64, max_total_bytes=1)]:
        with pytest.raises(ValueError, match="limit|budget"):
            R.select(
                plan,
                entry=entry,
                marker_sha256=_file_digest(marker),
                destination=tmp_path,
                overlays={"opt/capture-venv/bin/cache/python": leaf},
                overlay_limits=limits,
            )
    selected = R.select(
        plan,
        entry=entry,
        marker_sha256=_file_digest(marker),
        destination=tmp_path,
        overlays={"opt/capture-venv/bin/cache/python": leaf},
        overlay_limits=dict(max_leaves=64, max_total_bytes=1024),
    )
    assert selected["schema"] == R.OVERLAY_SCHEMA
    assert selected["source_cas_identity"] == entry.name
    assert selected["effective_runtime_identity"] == original_identity
    assert selected["overlay_bytes"] == leaf.stat().st_size
    assert R.verify(plan, selected, tmp_path) == selected
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    R.stage(selected, runtime)
    assert all(p.stat().st_size == 0 for p in runtime.rglob("*") if p.is_file())
    mounts = R.mounts(selected, runtime)
    assert mounts.count("--ro-bind") == 5  # CAS root, rebuilt ancestor+tool, absent flat directory+file
    assert not (entry / "opt/capture-venv/bin/cache").exists()
    leaf.write_bytes(b"changed interpreter bytes")
    with pytest.raises(ValueError, match="original selected inventory"):
        R.verify(plan, selected, tmp_path)


def test_actual_native_integer_worker_retains_original_frontend_and_complete_outputs(tmp_path):
    from merlin_experiments.phase0 import development_envelopes as E
    from merlin_experiments.phase0 import development_reference as REF
    from merlin_experiments.phase0 import development_sources as D

    from merlin.common.cpu_pool import shared_compute_pool
    from merlin.common.paths import module_source_path
    from merlin.targetgen import capsule_source

    source = capsule_source.PytorchRefSource()
    if not source.available():
        pytest.skip("selected original Model2MLIR capture runtime unavailable")
    declared = E.declaration(
        "conv_transition_nhwc", dict(H=3, W=5, CI=3, CO=5, OUT=7, KH=3, KW=3, SH=2, SW=1, PH=1, PW=1), ordinal=0
    )
    loader = tmp_path / "loader.py"
    loader.write_text(D.render(declared))
    output = tmp_path / "capture"
    output.mkdir()
    worker = module_source_path("merlin").parent / "targetgen/_m2m_capture_worker.py"
    argv = [
        str(source.python),
        "-B",
        str(worker),
        "--m2m-dir",
        str(source.m2m_dir),
        "--loader",
        str(loader),
        "--dtype",
        "int8",
        "--already-quantized",
        "--native-integer-source",
        "--materialize-bundle",
        "--out",
        str(output),
    ]
    with shared_compute_pool(timeout_s=60):
        result = subprocess.run(argv, env=dict(os.environ), capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr.decode()[-4000:]
    meta = json.loads((output / "meta.json").read_bytes())
    plan = dict(
        dtype="int8",
        recipe=None,
        worker_options={"source_mode": N.MODE},
        native_integer_verifier_sha256=_file_digest(Path(N.__file__)),
    )
    validated = N.validate(output, meta, plan)
    assert len(validated["numeric_contract"]["outputs"]) == 3
    inputs = REF.inputs(declared)
    expected = REF.evaluate(declared, inputs)
    with np.load(output / "inputs.npz", allow_pickle=False) as actual:
        for i, value in enumerate(inputs):
            np.testing.assert_array_equal(actual["in" + str(i)], value)
    golden = json.loads((output / "golden.json").read_bytes())
    for actual, wanted in zip(golden, expected, strict=True):
        np.testing.assert_array_equal(actual, wanted)
    assert (output / "capture_receipt.json").is_file()
    assert not (output / "sealed_m2m_pending.json").exists()  # functional worker proof, never a sealed receipt


@pytest.mark.parametrize("overlay_version", [2, 3])
def test_readonly_overlay_executes_in_actual_isolated_namespace(tmp_path, overlay_version):
    bwrap, shell = os.environ.get("MERLIN_TEST_BWRAP") or shutil.which("bwrap"), shutil.which("sh")
    if bwrap is None or shell is None or shutil.which("ldd") is None:
        pytest.skip("selected host namespace smoke tools unavailable")
    try:
        M._probe_sandbox(Path(bwrap))
    except M.SealedM2MError as error:
        pytest.skip(str(error))
    dynamic = subprocess.run(["ldd", shell], capture_output=True, text=True, check=True)
    libraries = sorted({p for line in dynamic.stdout.splitlines() if (p := M._ldd_library_path(line)) is not None})
    store = tmp_path / "store"
    store.mkdir()
    entry = store / "initial"
    package = "cache" if overlay_version == 2 else "cache/metadata/nested"
    python = entry / f"opt/capture-venv/bin/{package}/python"
    python.parent.mkdir(parents=True)
    shutil.copy2(Path(shell).resolve(), python)
    (entry / "opt/capture-venv/bin/marker").write_text("selected\n")
    base = entry / "opt/base"
    base.mkdir()
    (base / "identity").write_text("selected tiny namespace fixture; no framework environment copy")
    for library in libraries:
        destination = entry / library.relative_to("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(library, destination)
    plan = dict(
        base="/opt/base",
        system_libs=[str(p) for p in libraries],
        selected_trees={"venv": M._snapshot_tree(entry / "opt/capture-venv"), "base": M._snapshot_tree(base)},
    )
    moved = store / STORE._identity(plan)
    entry.rename(moved)
    entry = moved
    marker = store / (entry.name + ".complete.json")
    marker.write_text(json.dumps(dict(identity=entry.name, base=plan["base"], system_libs=plan["system_libs"])))
    overlay_root = tmp_path / "selected-cache"
    (entry / "opt/capture-venv/bin/cache").rename(overlay_root)
    overlay = overlay_root / ("python" if overlay_version == 2 else "metadata/nested/python")
    selected = R.select(
        plan,
        entry=entry,
        marker_sha256=_file_digest(marker),
        destination=tmp_path,
        overlays={f"opt/capture-venv/bin/{package}/python": overlay},
        overlay_version=overlay_version,
    )
    runtime, source, output = (tmp_path / name for name in ("runtime", "source", "output"))
    for p in (runtime, source, output):
        p.mkdir()
    R.stage(selected, runtime)
    for name in ("source", "tmp", "dev", "capture-out"):
        (runtime / name).mkdir()
    (runtime / output.relative_to("/")).mkdir(parents=True)
    command = (
        f"/opt/capture-venv/bin/{package}/python",
        "-c",
        "IFS= read -r value < /opt/capture-venv/bin/marker; "
        '[ "$value" = selected ] && ! (printf changed > /opt/capture-venv/bin/marker)',
    )
    result = M._execute(Path(bwrap), runtime, source, output, command, output, runtime_reuse=selected)
    assert result["returncode"] == 0
    assert (entry / "opt/capture-venv/bin/marker").read_text() == "selected\n"
    assert R.verify(plan, selected, tmp_path) == selected


def test_native_session_requires_every_integer_program_and_complete_numeric_abi(tmp_path, monkeypatch):
    import yaml

    out = tmp_path / "capture"
    stage = out / "stages" / "first"
    stage.mkdir(parents=True)
    plan, meta = fixture(stage)
    (stage / "model.mlir").write_text("""module {
      func.func @forward(%x: tensor<2x3xi8>, %w: tensor<3x4xi8>) -> tensor<2x4xi32> {
        %c = arith.constant 0 : i32
        %t = tensor.empty() : tensor<2x4xi32>
        %z = linalg.fill ins(%c : i32) outs(%t : tensor<2x4xi32>) -> tensor<2x4xi32>
        %r = linalg.matmul ins(%x, %w : tensor<2x3xi8>, tensor<3x4xi8>) outs(%z : tensor<2x4xi32>) -> tensor<2x4xi32>
        func.return %r : tensor<2x4xi32>
      }
    }""")
    x, w = np.arange(6, dtype=np.int8).reshape(2, 3) - 3, np.arange(12, dtype=np.int8).reshape(3, 4) - 6
    np.savez(stage / "inputs.npz", in0=x, in1=w)
    (stage / "golden.json").write_text(json.dumps((x.astype(np.int32) @ w.astype(np.int32)).tolist()))
    meta.update(
        input_abi=[dict(shape=[2, 3], dtype="i8"), dict(shape=[3, 4], dtype="i8")],
        output_abi=[dict(shape=[2, 4], dtype="i32")],
    )
    (stage / "meta.json").write_text(json.dumps(meta))
    (stage / "capture_receipt.json").write_text(json.dumps(dict(materialized_abi=dict(complete=True))))
    contract = dict(version=2, stages=["first"], programs=[dict(name="first", bundle="stages/first")])
    (out / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    report = dict(
        schema="merlin.model_session_capture.v1",
        source_state_unchanged=True,
        session_contract_sha256=_file_digest(out / "session_contract.yaml"),
        programs=[
            dict(
                name="first",
                precision_selection=N.MODE,
                materialized_abi=dict(complete=True),
                receipt_sha256=_file_digest(stage / "capture_receipt.json"),
            )
        ],
    )
    (out / "session-receipt.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        M, "_materialized", lambda p, *args, **kwargs: dict(receipt_sha256=_file_digest(p / "capture_receipt.json"))
    )
    checked = M._materialized_v3(out, tmp_path, out, plan)
    assert checked["integer_contractions"] == 1
    assert checked["programs"][0]["precision_selection"] == N.MODE
    report["programs"][0]["precision_selection"] = "no_recipe_work"
    (out / "session-receipt.json").write_text(json.dumps(report))
    with pytest.raises(M.SealedM2MError, match="precision selection"):
        M._materialized_v3(out, tmp_path, out, plan)
    report["programs"][0]["precision_selection"] = "untransformed"
    (out / "session-receipt.json").write_text(json.dumps(report))
    with pytest.raises(M.SealedM2MError, match="precision selection"):
        M._materialized_v3(out, tmp_path, out, plan)
    report["programs"][0]["precision_selection"] = N.MODE
    (out / "session-receipt.json").write_text(json.dumps(report))
    meta["output_abi"][0]["dtype"] = "f32"
    (stage / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(M.SealedM2MError, match="non-signed"):
        M._materialized_v3(out, tmp_path, out, plan)
    (out / "stages" / "undeclared").mkdir()
    with pytest.raises(M.SealedM2MError, match="membership"):
        M._materialized_v3(out, tmp_path, out, plan)
