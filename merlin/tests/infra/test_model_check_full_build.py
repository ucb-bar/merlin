"""Whole-model reuse binds content and recurring inputs retain actual link evidence."""

from __future__ import annotations

import json
import shutil
import struct
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower import c_runtime
from merlin.llvmlower.compilation_recipe import CompilationRecipe, verify_completed_recipe
from merlin.model_check import full_build as FB
from merlin.model_check import full_build_identity as ID
from merlin.model_check import full_build_native as NB


@pytest.mark.parametrize(
    "common_policy, own_policy, expected",
    [
        ({}, {}, "native"),
        ({"host_math_policy": "portable_trig"}, {}, "portable_trig"),
        ({"host_math_policy": "expf_via_double"}, {"host_math_policy": "portable_trig"}, "portable_trig"),
        ({"host_math_policy": "portable_trig"}, {"host_math_policy": "native"}, "native"),
    ],
)
def test_build_settings_uses_validated_common_and_model_host_math_selection(
    tmp_path,
    common_policy,
    own_policy,
    expected,
):
    models = SimpleNamespace(
        full={
            "board": "board",
            "board_catalog": str(tmp_path / "boards"),
            "host_dts": str(tmp_path / "host.dts"),
            "datapath": "scalar",
            **common_policy,
        }
    )
    model = SimpleNamespace(name="model", extra={"full": {"arena_mb": 1, **own_policy}})
    settings = FB.BuildSettings.of(models, model, tmp_path / "host")
    assert settings.host_math_policy == settings.record()["host_math_policy"] == expected


def test_direct_build_settings_refuses_unknown_math_policy_before_build_io(tmp_path):
    with pytest.raises(FB.FullBuildError, match="host_math_policy"):
        FB.BuildSettings(
            "board",
            tmp_path / "boards",
            tmp_path / "host.dts",
            "scalar",
            1,
            tmp_path / "host",
            host_math_policy="unknown",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "common,own,expected", [(False, None, False), (True, None, True), (True, False, False), (False, True, True)]
)
def test_model_host_fusion_selection_is_explicit_and_model_overrides_common(tmp_path, common, own, expected):
    models = SimpleNamespace(
        full={
            "board": "b",
            "board_catalog": "boards",
            "host_dts": "dts",
            "datapath": "scalar",
            "host_elementwise_fusion": common,
        }
    )
    model = SimpleNamespace(name="m", extra={"full": {"arena_mb": 1}})
    if own is not None:
        model.extra["full"]["host_elementwise_fusion"] = own
    settings = FB.BuildSettings.of(models, model, tmp_path / "host")
    assert settings.host_elementwise_fusion is expected
    assert settings.record()["host_elementwise_fusion"] is expected


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_host_fusion_setting_refuses_non_boolean_before_build_io(tmp_path, value):
    with pytest.raises(FB.FullBuildError, match="host elementwise fusion"):
        FB.BuildSettings(
            "b", tmp_path / "boards", tmp_path / "dts", "scalar", 1, tmp_path / "host", host_elementwise_fusion=value
        )
    assert not list(tmp_path.iterdir())


@pytest.fixture
def built(tmp_path, monkeypatch):
    compiler = Path(shutil.which("cc") or "")
    if not compiler.is_file():
        pytest.skip("a host C compiler is required")
    compiler = compiler.resolve()
    stage = tmp_path / "capture"
    stage.mkdir()
    (stage / "model.mlir").write_text("""module {
      func.func @forward(%x: tensor<2xf32>, %w: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> }
    }""")
    (stage / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "x"}, "1": {"kind": "param", "weight": "w"}})
    )
    header = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}).encode()
    (stage / "weights.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + np.array([5, 6], np.float32).tobytes()
    )
    (stage / "input_order.json").write_text(json.dumps({"x": 0}))
    np.savez(stage / "inputs.npz", in0=np.array([1, 2], np.float32))
    source = tmp_path / "builder.py"
    source.write_text("original compiler source\n")
    calls = []
    # Real host build/admission transport with bounded neutral metadata.
    monkeypatch.setattr(FB, "_lowering_sources", lambda: {"fixture.compiler": source})

    def material(stage, target, label, submission, submission_sha, settings, offload):
        from merlin.compile.model_execution_inputs import strict_tree_sha256

        return {
            "stage": str(stage.resolve()),
            "capture": strict_tree_sha256(stage),
            "source": FB.sha256(source),
            "target": target,
            "offload": offload,
            "host_math": ID.host_math_identity(settings.host_math_policy),
        }

    monkeypatch.setattr(ID, "build_material", material)
    monkeypatch.setattr(NB, "build_material", material)

    def compile_model(**kwargs):
        calls.append(kwargs)
        directory = kwargs["output"] / "build"
        directory.mkdir(parents=True)
        cgen = directory / "cgen"
        c_runtime.generate(stage, cgen, stage / "inputs.npz", prepared_dir=directory, max_session_steps=1)
        model = directory / "model.c"
        model.write_text("int shared_value(void) { return 5; }\n")
        driver = directory / "model_main.c"
        driver.write_text(
            '#include <stdio.h>\n#include "model_gen.h"\n#include "model_io.h"\n'
            'extern int shared_value(void);\nint main(void) { printf("%.0f %.0f %d\\n", '
            "merlin_in_0[0], merlin_in_0[1], shared_value()); return 0; }\n"
        )
        recipe = CompilationRecipe(directory, producer=source)
        from merlin.llvmlower.repeated_pointwise import prepare_repeated_pointwise, selected

        if selected():
            prepare_repeated_pointwise(stage / "model.mlir", directory / "repeated_pointwise", bundle_dir=stage)
            recipe.bind_preparation("repeated_pointwise", directory / "repeated_pointwise/report.json")
        from merlin.llvmlower.host_bounded_rne import bind_preparation
        from merlin.llvmlower.host_bounded_rne import selected as bounded_rne_selected

        if bounded_rne_selected():
            from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform
            from merlin.llvmlower.toolchain import clang
            from merlin.tests.ir.test_late_quant_rne_intrinsic import source as bounded_source

            original = directory / "bounded-source.ll"
            original.write_text(bounded_source())
            selected_ir = merlin_host_llvm_transform(
                Path(clang()).resolve().parent, host_isa="rv64gc", combine_clamp=True
            )(original, directory / "host_llvm")
            bind_preparation(recipe, directory / "host_llvm", {
                "source_sha256": FB.sha256(original), "selected_sha256": FB.sha256(selected_ir),
            })

        def run(argv, **kwargs):
            return subprocess.run(argv, capture_output=True, text=True, check=True, **kwargs)

        for name, path in (("model", model), ("model_main", driver)):
            obj = directory / f"{name}.o"
            recipe.run(
                [compiler, "-I", cgen, "-I", runtime_dir() / "c", "-c", path, "-o", obj],
                runner=run,
                inputs=[path],
                output=obj,
            )
        elf = directory / "model.elf"
        objects = [directory / "model.o", directory / "model_main.o"]
        linked = recipe.run(
            [compiler, *objects, "-Wl,--trace-symbol=shared_value", "-lm", "-o", elf],
            runner=run,
            inputs=objects,
            output=elf,
        )
        recipe.record_link_suppliers({"shared_value": directory / "model.o"}, linked)
        recipe.completed(elf)
        return {
            "output": {
                "elf": str(elf),
                "elf_sha256": FB.sha256(elf),
                "build_hash": "fixture-build",
                "compilation_recipe": {"path": str(recipe.path), "sha256": FB.sha256(recipe.path)},
            }
        }

    from merlin.compile import baremetal_model

    monkeypatch.setattr(baremetal_model, "compile_saved_model", compile_model)
    directory = tmp_path / "models"
    for name in ("boards", "host.dts"):
        (tmp_path / name).write_text("neutral selected build input\n")
    for name in ("host", "submission"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "manifest.yaml").write_text("neutral fixture package\n")
    settings = FB.BuildSettings("board", tmp_path / "boards", tmp_path / "host.dts", "scalar", 1, tmp_path / "host")

    def build(*, host_math_policy="native"):
        return FB.build(
            stage,
            target="t",
            label="model",
            submission=tmp_path / "submission",
            submission_sha256="a" * 64,
            settings=replace(settings, host_math_policy=host_math_policy),
            directory=directory,
            admission=_admission_service(),
            offload=False,
        )

    return build, stage, directory, source, calls


def test_build_reuse_checks_recipe_elf_bytes_and_current_capture_and_source(built):
    build, stage, directory, source, calls = built
    first = build()
    assert not first["reused"] and build()["reused"] and len(calls) == 1
    elf = Path(first["elf"])
    data = bytearray(elf.read_bytes())
    data[-1] ^= 1  # Same length; a size-only cache would accept this image.
    elf.write_bytes(data)
    assert not build()["reused"] and len(calls) == 2
    (Path(first["build_dir"]) / "model.c").write_text("changed producer input\n")
    assert not build()["reused"] and len(calls) == 3
    np.savez(stage / "inputs.npz", in0=np.array([3, 4], np.float32))
    assert not build()["reused"] and len(calls) == 4
    source.write_text("changed compiler source\n")
    assert not build()["reused"] and len(calls) == 5
    record_path = directory / "build.json"
    legacy = json.loads(record_path.read_text())
    legacy["schema"] = "merlin.model_check.full_build.v0"
    record_path.write_text(json.dumps(legacy))
    assert not build()["reused"] and len(calls) == 6


def test_full_build_forwards_only_selected_host_math_policy_and_separates_reuse(built):
    build, _stage, _directory, _source, calls = built
    native = build()
    assert "host_math_policy" not in calls[-1]
    selected = build(host_math_policy="portable_trig")
    assert calls[-1]["host_math_policy"] == "portable_trig"
    assert selected["key"]["producer_material"]["host_math"] != native["key"]["producer_material"]["host_math"]
    assert build(host_math_policy="portable_trig")["reused"]
    assert len(calls) == 2


def test_native_adoption_reuses_observed_image_and_relinks_real_inputs(built, tmp_path):
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.model_check.full_build_native import adopt

    build, stage, _directory, _source, calls = built
    original = build()
    settings = FB.BuildSettings("board", tmp_path / "boards", tmp_path / "host.dts", "scalar", 1, tmp_path / "host")
    for path in (settings.board_catalog, settings.host_dts):
        path.write_text("fixture")
    settings.host_package.mkdir(exist_ok=True)
    (settings.host_package / "manifest.yaml").write_text("host fixture")
    submission = tmp_path / "submission"
    submission.mkdir(exist_ok=True)
    (submission / "manifest.yaml").write_text("compiler fixture")
    receipt = {
        "schema": "merlin.baremetal-saved-model.v1",
        "status": "compiled",
        "execution_route": "device_requested_dispatch_unverified",
        "inputs": {
            "capture": str(stage.resolve()),
            "target": "t",
            "board": "board",
            "board_catalog_sha256": FB.sha256(settings.board_catalog),
            "dts_sha256": FB.sha256(settings.host_dts),
            "arena_mb": 1,
            "readback": "full",
            "host_math_policy": "native",
            "respect_captured_quantization_scope": True,
            "host_llvm_bitcode": True,
            "blob_weights": True,
            "capture_tree": strict_tree_sha256(stage),
            "package_tree": strict_tree_sha256(settings.host_package),
            "device": {"package_tree": strict_tree_sha256(submission)},
        },
        "output": {key: original[key] for key in ("elf", "elf_sha256", "build_hash", "compilation_recipe")},
    }
    kwargs = dict(
        stage=stage,
        target="t",
        label="toy:forward",
        submission=submission,
        submission_sha256="a" * 64,
        settings=settings,
        qualification={"producer": "observed"},
        admission=_admission_service(),
    )
    imported = adopt(receipt, **kwargs)
    assert imported["elf"] == original["elf"] and len(calls) == 1
    assert imported["key"]["native_qualification"] == kwargs["qualification"]
    with pytest.raises(FB.FullBuildError, match="selected full-model build"):
        adopt(receipt, **{**kwargs, "settings": replace(settings, host_elementwise_fusion=True)})
    receipt["inputs"]["host_elementwise_fusion"] = True
    with pytest.raises(FB.FullBuildError, match="selected full-model build"):
        adopt(receipt, **kwargs)
    receipt["inputs"].pop("host_elementwise_fusion")
    receipt["inputs"]["host_bounded_rne"] = {"host_isa": "rv64gc", "combine_clamp": True}
    with pytest.raises(FB.FullBuildError, match="host bounded RNE declaration"):
        adopt(receipt, **kwargs)
    receipt["inputs"].pop("host_bounded_rne")
    elf = FB.relink(imported, stage, {"in0": np.array([8, 9], np.float32)}, tmp_path / "adopted-step")
    assert subprocess.check_output([elf], text=True).strip() == "8 9 5"
    recipe = json.loads((elf.parent / "compilation_recipe.json").read_text())
    assert recipe["link_suppliers"]["symbols"]["shared_value"]["selected_input"]["path"].endswith("/model.o")
    for field, value in (
        ("host_math_policy", "portable_trig"),
        ("readback", "prefix"),
        ("capture_tree", {"sha256": "b" * 64, "n_files": 1}),
    ):
        saved = receipt["inputs"][field]
        receipt["inputs"][field] = value
        with pytest.raises(FB.FullBuildError, match="selected full-model build"):
            adopt(receipt, **kwargs)
        receipt["inputs"][field] = saved
    with pytest.raises(FB.FullBuildError, match="qualification binding"):
        adopt(receipt, **{**kwargs, "qualification": {}})
    Path(original["elf"]).unlink()
    with pytest.raises(FB.FullBuildError, match="ELF is absent"):
        adopt(receipt, **kwargs)


def _relink_admission(*, elf, evidence_root):
    directory = Path(evidence_root)
    assert not directory.exists(), "the independent native auditor requires a fresh evidence destination"
    directory.mkdir()
    report = directory / "selected-policy.json"
    result = {"status": "accepted", "elf_sha256": FB.sha256(elf)}
    report.write_text(json.dumps(result))
    return {**result, "report_path": str(report), "report_sha256": FB.sha256(report)}


def _ordinary_admission(*, elf, evidence_root):
    # Cache rechecks reopen the current policy in the existing evidence directory.
    directory = Path(evidence_root)
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "selected-policy.json"
    result = {"status": "accepted", "elf_sha256": FB.sha256(elf)}
    report.write_text(json.dumps(result))
    return {**result, "report_path": str(report), "report_sha256": FB.sha256(report)}


def _adopt_checked_record(record, stage, tmp_path, *, qualification=None, include_bounded_declaration=True):
    from merlin.compile.model_execution_inputs import strict_tree_sha256
    from merlin.model_check.full_build_native import adopt

    settings = FB.BuildSettings(
        **{
            key: Path(value) if key in ("host_package", "host_dts", "board_catalog") else value
            for key, value in record["key"]["settings"].items()
        }
    )
    submission = tmp_path / "submission"
    receipt = {
        "schema": "merlin.baremetal-saved-model.v1",
        "status": "compiled",
        "execution_route": "device_requested_dispatch_unverified",
        "inputs": {
            "capture": str(stage.resolve()),
            "target": "t",
            "board": settings.board,
            "board_catalog_sha256": FB.sha256(settings.board_catalog),
            "dts_sha256": FB.sha256(settings.host_dts),
            "arena_mb": settings.arena_mb,
            "readback": "full",
            "host_math_policy": settings.host_math_policy,
            "respect_captured_quantization_scope": True,
            "host_llvm_bitcode": True,
            "blob_weights": True,
            "capture_tree": strict_tree_sha256(stage),
            "package_tree": strict_tree_sha256(settings.host_package),
            "device": {"package_tree": strict_tree_sha256(submission)},
        },
        "output": {key: record[key] for key in ("elf", "elf_sha256", "build_hash", "compilation_recipe")},
    }
    recipe = json.loads(Path(record["compilation_recipe"]["path"]).read_bytes())
    if include_bounded_declaration and "host_bounded_rne" in recipe.get("preparation", {}):
        receipt["inputs"]["host_bounded_rne"] = {"host_isa": "rv64gc", "combine_clamp": True}
    return adopt(
        receipt,
        stage=stage,
        target="t",
        label="toy:forward",
        submission=submission,
        submission_sha256="a" * 64,
        settings=settings,
        qualification=qualification or {"producer": "neutral observed host-C fixture"},
        admission=_admission_service(),
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_native_adoption_and_relink_require_original_repeated_pointwise_arm(built, tmp_path, enabled):
    from merlin.llvmlower.optional_passes import Selection, applied

    build, stage, _directory, _source, calls = built
    selected = "prepare-repeated-pointwise"
    original_choice = selected if enabled else "-" + selected
    with applied(Selection.parse(original_choice)):
        original = build()
        imported = _adopt_checked_record(original, stage, tmp_path)
        assert type(imported) is NB.NativeBuild
        live = imported.admission
        elf = FB.relink(imported, stage, {"in0": np.array([8, 9], np.float32)}, tmp_path / "matched-step")
        assert subprocess.check_output([elf], text=True).strip() == "8 9 5"
        assert imported.admission is live and len(calls) == 1
    with applied(Selection.parse("-" + selected if enabled else selected)):
        with pytest.raises(FB.FullBuildError, match="repeated-pointwise selection"):
            _adopt_checked_record(original, stage, tmp_path)
        with pytest.raises(FB.FullBuildError, match="repeated-pointwise selection"):
            FB.relink(imported, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "opposite-step")
        assert not (tmp_path / "opposite-step").exists()


def test_repeated_pointwise_choices_cannot_alias_a_cached_whole_model_build(built):
    from merlin.llvmlower.optional_passes import Selection, applied

    build, _stage, _directory, _source, calls = built
    with applied(Selection.parse("-prepare-repeated-pointwise")):
        off = build()
        assert not off["reused"] and build()["reused"]
    with applied(Selection.parse("prepare-repeated-pointwise")):
        on = build()
        assert not on["reused"] and build()["reused"]
    assert off["key"]["lowering"]["project_controls_sha256"] != on["key"]["lowering"]["project_controls_sha256"]
    assert len(calls) == 2


def test_native_repeated_preparation_reopens_exact_report_bytes(built, tmp_path):
    from merlin.llvmlower.optional_passes import Selection, applied

    build, stage, _directory, _source, _calls = built
    with applied(Selection.parse("prepare-repeated-pointwise")):
        original = build()
        imported = _adopt_checked_record(original, stage, tmp_path)
        report = Path(original["build_dir"]) / "repeated_pointwise/report.json"
        report.write_bytes(report.read_bytes() + b" ")
        with pytest.raises(ValueError, match="preparation.*changed|preparation.*differs"):
            _adopt_checked_record(original, stage, tmp_path)
        with pytest.raises(ValueError, match="preparation.*changed|preparation.*differs"):
            FB.relink(imported, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "changed-report-step")
        assert not (tmp_path / "changed-report-step").exists()


@pytest.mark.parametrize("choice", ["", "-host-bounded-rne", "host-bounded-rne"])
def test_native_bounded_rne_adoption_and_relink_preserve_the_original_arm(built, tmp_path, choice):
    from merlin.llvmlower.optional_passes import Selection, applied

    build, stage, _directory, _source, calls = built
    with applied(Selection.parse(choice)):
        original = build()
        imported = _adopt_checked_record(original, stage, tmp_path)
        assert type(imported) is NB.NativeBuild
        live = imported.admission
        elf = FB.relink(imported, stage, {"in0": np.array([8, 9], np.float32)}, tmp_path / "matched-rne-step")
        assert subprocess.check_output([elf], text=True).strip() == "8 9 5"
        assert imported.admission is live and len(calls) == 1
    opposite = "-host-bounded-rne" if choice == "host-bounded-rne" else "host-bounded-rne"
    with applied(Selection.parse(opposite)):
        with pytest.raises(FB.FullBuildError, match="host bounded RNE selection"):
            _adopt_checked_record(original, stage, tmp_path)
        with pytest.raises(FB.FullBuildError, match="host bounded RNE selection"):
            FB.relink(imported, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "opposite-rne-step")
        assert not (tmp_path / "opposite-rne-step").exists()


def test_default_off_and_selected_bounded_rne_cannot_alias_cached_builds(built):
    from merlin.llvmlower.optional_passes import Selection, applied

    build, _stage, _directory, _source, calls = built
    records = []
    for choice in ("", "-host-bounded-rne", "host-bounded-rne"):
        with applied(Selection.parse(choice)):
            record = build()
            assert not record["reused"] and build()["reused"]
            records.append(record)
    assert len({x["key"]["lowering"]["project_controls_sha256"] for x in records}) == 3
    assert len(calls) == 3


def test_native_bounded_rne_reopens_the_original_report(built, tmp_path):
    from merlin.llvmlower.optional_passes import Selection, applied

    build, stage, _directory, _source, _calls = built
    with applied(Selection.parse("host-bounded-rne")):
        original = build()
        imported = _adopt_checked_record(original, stage, tmp_path)
        report = Path(original["build_dir"]) / "host_llvm/receipt.json"
        report.write_bytes(report.read_bytes() + b" ")
        with pytest.raises(ValueError, match="preparation.*changed|preparation.*differs"):
            _adopt_checked_record(original, stage, tmp_path)
        with pytest.raises(ValueError, match="preparation.*changed|preparation.*differs"):
            FB.relink(imported, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "changed-rne-step")
        assert not (tmp_path / "changed-rne-step").exists()


def test_native_bounded_rne_marker_cannot_replace_the_original_saved_build_declaration(built, tmp_path):
    from merlin.llvmlower.optional_passes import Selection, applied

    build, stage, _directory, _source, _calls = built
    with applied(Selection.parse("host-bounded-rne")):
        original = build()
        with pytest.raises(FB.FullBuildError, match="host bounded RNE declaration"):
            _adopt_checked_record(original, stage, tmp_path, include_bounded_declaration=False)


def _admission_service():
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    owner = Path(__file__).resolve()
    return LinkedElfAdmissionService("t", _ordinary_admission, ((str(owner), FB.sha256(owner)),))


def _relink_refusal(*, elf, evidence_root):
    result = _relink_admission(elf=elf, evidence_root=evidence_root)
    result["status"] = "refused"
    Path(result["report_path"]).write_text(json.dumps({"status": "refused", "elf_sha256": result["elf_sha256"]}))
    result["report_sha256"] = FB.sha256(Path(result["report_path"]))
    return result


def test_input_specific_native_relink_uses_original_selected_instruction_service(built, tmp_path):
    from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService

    build, stage, _directory, _source, _calls = built
    record = _adopt_checked_record(build(), stage, tmp_path)
    record.admission = LinkedElfAdmissionService(
        "t", _relink_admission, ((str(Path(__file__).resolve()), FB.sha256(Path(__file__))),)
    )
    step = tmp_path / "policy-step"
    elf = FB.relink(record, stage, {"in0": np.array([3, 4], np.float32)}, step)
    policy = json.loads((step / "native_linked_elf.json").read_text())
    assert policy["status"] == "accepted" and policy["elf_sha256"] == FB.sha256(elf)
    assert policy["elf_sha256"] != record["elf_sha256"]
    record.admission = LinkedElfAdmissionService(
        "t", _relink_refusal, ((str(Path(__file__).resolve()), FB.sha256(Path(__file__))),)
    )
    with pytest.raises(FB.FullBuildError, match="failed selected instruction policy"):
        FB.relink(record, stage, {"in0": np.array([7, 8], np.float32)}, tmp_path / "refused-step")


def test_recurrent_relinks_generate_weights_once_and_execute_current_inputs(built, tmp_path, monkeypatch):
    build, stage, _directory, _source, _calls = built
    record = _adopt_checked_record(build(), stage, tmp_path)
    calls = []
    generate = c_runtime.generate

    def counted(*args, **kwargs):
        calls.append(args)
        return generate(*args, **kwargs)

    monkeypatch.setattr(c_runtime, "generate", counted)
    for index, expected in enumerate(([3, 4], [8, 9], [-7, 6])):
        directory = tmp_path / f"step{index}"
        elf = FB.relink(record, stage, {"in0": np.array(expected, np.float32)}, directory)
        assert subprocess.check_output([elf], text=True).strip() == f"{expected[0]} {expected[1]} 5"
        recipe = directory / "compilation_recipe.json"
        observed = verify_completed_recipe(recipe, executable=elf, expected_recipe_sha256=FB.sha256(recipe))
        assert any(row["path"].endswith("model_io.h") for row in observed["commands"][0]["inputs"])
    assert len(calls) == 1, "unchanged weights and static call artifacts are generated once"


def test_fresh_native_adoptions_share_products_without_aliasing_runtime_templates(built, tmp_path, monkeypatch):
    build, stage, _directory, _source, _calls = built
    original = build()
    first = _adopt_checked_record(original, stage, tmp_path, qualification={"producer": "original owner"})
    second = _adopt_checked_record(original, stage, tmp_path, qualification={"producer": "fresh verified owner"})
    assert first["build_dir"] == second["build_dir"]
    assert first["compilation_recipe"] == second["compilation_recipe"]
    assert first["key"] != second["key"]
    legacy = Path(first["build_dir"]) / "model_check_single_forward"
    legacy.mkdir()
    marker = legacy / "template.json"
    marker.write_bytes(b"preserved original owner's template\n")
    original_marker = marker.read_bytes()
    calls = []
    generate = c_runtime.generate

    def counted(*args, **kwargs):
        calls.append(args)
        return generate(*args, **kwargs)

    monkeypatch.setattr(c_runtime, "generate", counted)
    for index, (record, expected) in enumerate([(first, [3, 4]), (second, [8, 9]), (first, [-7, 6])]):
        directory = tmp_path / f"owner-step{index}"
        elf = FB.relink(record, stage, {"in0": np.array(expected, np.float32)}, directory)
        assert subprocess.check_output([elf], text=True).strip() == f"{expected[0]} {expected[1]} 5"
        verify_completed_recipe(directory / "compilation_recipe.json", executable=elf,
                                expected_recipe_sha256=FB.sha256(directory / "compilation_recipe.json"))
    assert len(calls) == 2, "each exact adoption stamp generates once; returning to the first reuses only its template"
    assert marker.read_bytes() == original_marker
    templates = list(Path(first["build_dir"]).glob("model_check_single_forward_*/template.json"))
    assert len(templates) == 2
    assert {json.loads(path.read_text())["stamp"]["key"] for path in templates} == {
        ID.content_digest_bytes(record["key"]) for record in (first, second)
    }


def test_relink_rejects_changed_static_objects_before_launching_the_link(built, tmp_path):
    build, stage, _directory, _source, _calls = built
    record = build()
    static = Path(record["build_dir"]) / "model.o"
    static.write_bytes(static.read_bytes() + b"changed")
    with pytest.raises(FB.FullBuildError, match="immutable original build input|recipe.*changed"):
        FB.relink(record, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "step")
    assert not (tmp_path / "step/model.elf").exists()


def test_relink_rejects_changed_input_abi_and_another_capture(built, tmp_path):
    build, stage, _directory, _source, _calls = built
    record = build()
    with pytest.raises(FB.FullBuildError, match="capture differs|another model capture"):
        FB.relink(record, tmp_path / "another", {}, tmp_path / "step")
    (stage / "input_order.json").write_text(json.dumps({"x": 1}))
    with pytest.raises(FB.FullBuildError, match="build inputs changed|input ABI changed"):
        FB.relink(record, stage, {"in0": np.array([3, 4], np.float32)}, tmp_path / "step")


def test_relink_refuses_modified_cached_template_and_input_ownership(built, tmp_path):
    build, stage, _directory, _source, _calls = built
    record = _adopt_checked_record(build(), stage, tmp_path)
    inputs = {"in0": np.array([3, 4], np.float32)}
    FB.relink(record, stage, inputs, tmp_path / "step0")
    stamp = {"key": ID.content_digest_bytes(record["key"]), "recipe": record["compilation_recipe"]}
    template = Path(record["build_dir"]) / ("model_check_single_forward_" + ID.content_digest_bytes(stamp))
    manifest_path = template / "template.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stamp"]["key"] = "changed owner"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FB.FullBuildError, match="another model build"):
        FB.relink(record, stage, inputs, tmp_path / "foreign-owner-step")
    manifest["stamp"] = stamp
    manifest["bindings"][0]["argument"] = 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FB.FullBuildError, match="input ownership"):
        FB.relink(record, stage, inputs, tmp_path / "step1")
    manifest["bindings"][0]["argument"] = 0
    manifest_path.write_text(json.dumps(manifest))
    (template / "model_gen.h").write_text("changed static ABI\n")
    with pytest.raises(FB.FullBuildError, match="template bytes changed"):
        FB.relink(record, stage, inputs, tmp_path / "step2")


def test_build_material_binds_direct_tools_sources_settings_and_input_bytes(tmp_path, monkeypatch):
    from merlin.common import paths
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends import spike

    source = tmp_path / "src/merlin"
    source.mkdir(parents=True)
    compiler_source = source / "compiler.py"
    compiler_source.write_text("compiler one\n")
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "driver.c").write_text("runtime one\n")
    host = tmp_path / "host"
    host.mkdir()
    (host / "manifest.yaml").write_text("host one\n")
    stage = tmp_path / "capture"
    stage.mkdir()
    weights = stage / "weights"
    weights.write_bytes(b"weights one")
    tool = tmp_path / "tool"
    tool.write_bytes(b"tool one")
    mlir_install = tmp_path / "llvm"
    mlir_optimizer = mlir_install / "bin/mlir-opt"
    mlir_optimizer.parent.mkdir(parents=True)
    mlir_optimizer.write_bytes(b"MLIR one")
    boards, dts = tmp_path / "boards.yaml", tmp_path / "cpu.dts"
    boards.write_text("boards one\n")
    dts.write_text("cpu one\n")
    monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(paths, "runtime_dir", lambda: runtime)
    for name in ("clang", "mlir_translate", "llvm_opt", "compiler_python", "objcopy", "nm", "readelf"):
        monkeypatch.setattr(toolchain, name, lambda: tool)
    monkeypatch.setattr(toolchain, "llvm_install", lambda: mlir_install)
    monkeypatch.setattr(spike, "gcc_path", lambda: tool)
    monkeypatch.setattr(ID, "target_inputs", lambda: {"facts": "pinned"})
    settings = FB.BuildSettings("board", boards, dts, "scalar", 1, host)

    def material():
        return ID.build_material(stage, "t", "program", tmp_path / "absent-submission", "a" * 64, settings, False)

    first = material()
    # A timestamp-preserving same-size tool replacement still invalidates reuse.
    import os

    previous = tool.stat()
    tool.write_bytes(b"tool two")
    os.utime(tool, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert material() != first
    first = material()
    previous = mlir_optimizer.stat()
    mlir_optimizer.write_bytes(b"MLIR two")
    os.utime(mlir_optimizer, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    changed = material()
    assert changed["tools"]["mlir_optimizer"]["sha256"] != first["tools"]["mlir_optimizer"]["sha256"]
    assert changed != first
    first = material()
    (tmp_path / ".env").write_text("MERLIN_COMPILER_DRIVER=stock-cli\n")
    assert material() != first, "dotenv compiler choices are actual build inputs, even without process-env changes"
    first = material()
    weights.write_bytes(b"weights two")
    assert material() != first
    first = material()
    compiler_source.write_text("compiler two\n")
    assert material() != first
    first = material()
    boards.write_text("boards two\n")
    assert material() != first
    first = material()
    (host / "manifest.yaml").write_text("host two\n")
    assert material() != first
    first = material()
    monkeypatch.setenv("MERLIN_SYNTHETIC_CODEGEN_FLAG", "changed")
    assert material() != first

    # The recipe, not a hardcoded filename, binds non-Python runtime C bytes.
    from merlin.runtime import host_math

    portable_source = tmp_path / "portable_trig.c"
    portable_source.write_text("portable runtime one\n")
    monkeypatch.setattr(host_math, "Path", lambda path: tmp_path / "selected_recipe.py")
    first = material()
    assert first["host_math"]["source_sha256"] is None
    portable_source.write_text("portable runtime two\n")
    assert material() == first, "native must not acquire a dependency on unselected portable C"
    settings = replace(settings, host_math_policy="portable_trig")
    first = material()
    previous = portable_source.stat()
    portable_source.write_text("portable runtime new\n")
    os.utime(portable_source, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    changed = material()
    assert changed != first
    assert changed["host_math"]["source_sha256"] != first["host_math"]["source_sha256"]
    assert changed["host_math"]["compile_flags"] == ["-fno-builtin", "-fno-fast-math", "-ffp-contract=off"]
    assert changed["host_math"]["link_flags"] == ["-Wl,--wrap=sinf", "-Wl,--wrap=cosf"]
    recipe = host_math.host_math_recipe
    monkeypatch.setattr(
        host_math,
        "host_math_recipe",
        lambda policy: replace(
            recipe(policy),
            compile_flags=(*recipe(policy).compile_flags, "-frounding-math"),
        ),
    )
    assert material() != changed, "selected runtime compile flags are part of build correspondence too"
