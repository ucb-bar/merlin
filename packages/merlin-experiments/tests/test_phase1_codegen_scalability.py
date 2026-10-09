"""Public geometry probes expose emitted-text growth, never compiler certification."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability as S

    from merlin.targetgen import capability_probes, corpora, corpus_spec, lowering_coverage

    monkeypatch.setattr(corpora, "experiment_for", lambda _target: object())
    monkeypatch.setattr(
        corpus_spec,
        "derive_binding",
        lambda _experiment, _profile: SimpleNamespace(
            operand_dtype="int8", accum_dtype="i32", mlir_dtype=lambda token: "i8" if token == "int8" else "i32"
        ),
    )
    monkeypatch.setattr(capability_probes, "tile_edge", lambda _target, **_kwargs: 32)
    seen = []

    def emit(_package, **kwargs):
        seen.append(kwargs)
        scale = kwargs["m"] // 32
        return "lowered", None, 20 * scale**3

    monkeypatch.setattr(lowering_coverage, "probe_shape", emit)
    return S, seen


def test_cases_are_derived_only_from_public_geometry_and_formats(probe):
    S, seen = probe
    result = S.run(
        "candidate", target="synthetic", contract="selected-contract", timeout=12, additional_forbidden=("extra",)
    )
    assert [(row["m"], row["k"], row["n"]) for row in seen] == [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
    ]
    assert all(row["operand_mlir"] == "i8" and row["accum_mlir"] == "i32" for row in seen)
    assert all(row["contract"] == "selected-contract" and row["additional_forbidden"] == ("extra",) for row in seen)
    assert all(row["timeout"] == 12 for row in seen)
    assert [row["contraction_volume_ratio"] for row in result["samples"]] == [1, 8, 64, 512]
    assert [row["line_ratio_vs_baseline"] for row in result["samples"]] == [1, 8, 64, 512]
    assert result["scope"] == "public_emit_only"
    assert result["observations_complete"] is True
    assert "all_pass" not in result and "all_covered" not in result
    assert "not compilation" in result["note"]


def test_constant_size_looped_code_is_not_rejected(probe, monkeypatch):
    from merlin.targetgen import lowering_coverage

    S, _seen = probe
    monkeypatch.setattr(lowering_coverage, "probe_shape", lambda *_args, **_kwargs: ("lowered", None, 17))
    result = S.run("candidate", target="synthetic")
    assert result["observations_complete"] is True
    assert all(row["line_ratio_vs_baseline"] == 1 for row in result["samples"])
    assert "verdict" not in result


def test_explicit_build_measurement_keeps_emission_and_grade_advisory(probe, monkeypatch, tmp_path):
    from merlin.targetgen import lowering_coverage

    S, _seen = probe
    observed = []

    class SelectedBuild:
        def verify(self, target):
            assert target == "synthetic"

    def emit(_package, **kwargs):
        callback = kwargs["on_lowered"]
        artifact = "module { llvm.func @kernel() { llvm.return } }"
        callback(artifact, tmp_path)
        observed.append(kwargs["m"])
        return "lowered", None, len(artifact.splitlines())

    monkeypatch.setattr(lowering_coverage, "probe_shape", emit)
    monkeypatch.setattr(S, "_selected_build_inputs", lambda *_args: {"selected": "fixture"})
    monkeypatch.setattr(S, "_compile_observation", lambda artifact, work, **kwargs: {
        "status": "compiled", "object_bytes": 12, "artifact_sha256": "a" * 64
    })
    result = S.run("candidate", target="synthetic", build_service=SelectedBuild(), build_timeout_s=2)
    assert observed == [32, 64, 128, 256]
    assert result["scope"] == "public_emit_and_build_only"
    assert all(row["build_only"]["status"] == "compiled" for row in result["samples"])
    assert "all_pass" not in result and "verdict" not in result


def test_build_failure_and_large_source_are_advisory_not_success(probe, monkeypatch, tmp_path):
    from merlin.targetgen import lowering_coverage

    S, _seen = probe
    monkeypatch.setattr(S, "_selected_build_inputs", lambda *_args: {"selected": "fixture"})
    monkeypatch.setattr(S, "_MAX_BUILD_PROBE_BYTES", 4)
    observed = []

    def emit(_package, **kwargs):
        kwargs["on_lowered"]("too large", tmp_path)
        observed.append(kwargs["m"])
        return "lowered", None, 1

    monkeypatch.setattr(lowering_coverage, "probe_shape", emit)
    result = S.run("candidate", target="synthetic", build_service=object())
    assert observed == [32, 64, 128, 256]
    assert result["observations_complete"] is True
    assert all(row["build_only"]["status"] == "source_exceeds_probe_budget" for row in result["samples"])
    assert "all_pass" not in result


def test_build_selection_change_cannot_report_compiled(monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import codegen_scalability as S

    from merlin.targetgen.contract import compile as compiler

    selections = iter(({"tool": "before"}, {"tool": "after"}))
    monkeypatch.setattr(S, "_selected_build_inputs", lambda *_args: next(selections))
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *_args, **_kwargs: tmp_path / "object.o")
    result = S._compile_observation("module {}", tmp_path, target="fixture", service=object(), budget_s=2)
    assert result["status"] == "selection_changed"
    assert "object_sha256" not in result


def test_smaller_looped_emission_does_not_revoke_a_passing_round(probe, monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import loop_grading as L

    from merlin.targetgen import lowering_coverage as LC

    monkeypatch.setattr(LC, "tile_edge", lambda _target: 32)
    monkeypatch.setattr(
        LC,
        "probe_shape",
        lambda _package, **kw: ("lowered", None, 20 if (kw["m"], kw["k"], kw["n"]) == (64, 32, 32) else 30),
    )
    verdict = {"all_pass": True}
    L._attach_shape_generalization(
        verdict, "candidate", tmp_path, 0, timeout=30, context=SimpleNamespace(target="synthetic", repo=tmp_path)
    )
    assert verdict["all_pass"] is True
    assert verdict["shape_coverage"]["scope"] == "public_emit_only"
    assert verdict["shape_coverage"]["smaller_emitted_artifacts"] == ["m_2tiles"]
    assert verdict["shape_coverage"]["multi_tile_axes_uncovered"] == []


def test_declined_and_failed_baseline_are_measurements_not_passes(probe, monkeypatch):
    from merlin.targetgen import lowering_coverage

    S, _seen = probe
    monkeypatch.setattr(lowering_coverage, "probe_shape", lambda *_args, **_kwargs: ("declined", "unsupported", 0))
    result = S.run("candidate", target="synthetic")
    assert result["observations_complete"] is False
    assert all(row["line_ratio_vs_baseline"] is None for row in result["samples"])
    assert all(row["outcome"] == "declined" for row in result["samples"])


def test_round_feedback_persists_public_probe_without_changing_grade(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability as S
    from merlin_experiments.phase1.feedback import loop_grading as L

    observation = {"scope": "public_emit_only", "samples": [{"extent_scale": 8, "artifact_nonempty_lines": 7982}]}
    seen = []

    def run(_candidate, **kwargs):
        seen.append(kwargs)
        return observation

    monkeypatch.setattr(S, "run", run)
    verdict = {"all_pass": False}
    L._attach_codegen_scalability(
        verdict, "candidate", tmp_path, 3, timeout=300, context=SimpleNamespace(target="synthetic"), contract="selected"
    )
    assert verdict["all_pass"] is False
    assert verdict["codegen_scalability"] == {"ran": True, **observation}
    assert seen[0]["timeout"] == 30
    assert seen[0]["contract"] == "selected"
    assert json.loads((tmp_path / "qa_history/codegen_scalability_round_03.json").read_text()) == observation


def test_missing_scalability_measurement_is_explicit_but_not_a_numerical_verdict(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability as S
    from merlin_experiments.phase1.feedback import loop_grading as L

    def fail(*_args, **_kwargs):
        raise RuntimeError("selected geometry unavailable")

    monkeypatch.setattr(S, "run", fail)
    verdict = {"all_pass": False}
    L._attach_codegen_scalability(
        verdict, "candidate", tmp_path, 0, timeout=60, context=SimpleNamespace(target="synthetic")
    )
    assert verdict["all_pass"] is False
    assert verdict["codegen_scalability"]["ran"] is False
    assert "unavailable" in verdict["codegen_scalability"]["error"]


@pytest.mark.skipif(
    os.environ.get("MERLIN_TEST_PUBLIC_COMPILE_SMOKE") != "1",
    reason="explicitly selected LLVM toolchain required for compile-only diagnostic",
)
def test_neutral_straight_line_and_bounded_loop_have_measured_objects(tmp_path):
    """A neutral generated artifact makes expansion visible without a size-based grade."""
    import hashlib

    from merlin_experiments.phase1.feedback import codegen_scalability as S

    from merlin.llvmlower import toolchain
    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
    from merlin.targetgen.contract.build_service import BuildOnlyService

    clang, translator = toolchain.clang(), toolchain.mlir_translate()
    compiler = tmp_path / "selected-compiler"
    compiler.symlink_to(clang)
    # The object-only service uses the recipe's ISA/ABI and stack policy, not its C compiler.
    recipe = HarnessBuildRecipe(
        compiler=compiler, include_roots=(), support_sources=(), link_script=tmp_path / "unused.ld",
        load_address=0, cflags=("-march=rv64gc", "-mabi=lp64d"),
        kernel_stack_frame=KernelStackFramePolicy("fixture_entry", 4096),
    )

    def source_artifact(kind: str) -> tuple[str, BuildOnlyService]:
        body = (
            "\n".join(f"  acc += {index}ULL;" for index in range(4096))
            if kind == "straight_line" else
            "  for (unsigned long long i = 0; i < 4096; ++i) acc += i;"
        )
        source = tmp_path / f"{kind}.c"
        source.write_text(
            "unsigned long long fixture_entry(unsigned long long x) {\n"
            "  unsigned long long acc = x;\n" + body + "\n  return acc;\n}\n"
        )
        llvm_ir = tmp_path / f"{kind}.ll"
        llvm_mlir = tmp_path / f"{kind}.mlir"
        subprocess.run(
            [str(clang), "--target=riscv64-unknown-elf", "-march=rv64gc", "-mabi=lp64d",
             "-O0", "-S", "-emit-llvm", str(source), "-o", str(llvm_ir)],
            check=True, capture_output=True, timeout=20,
        )
        subprocess.run(
            [str(translator), "--import-llvm", str(llvm_ir), "-o", str(llvm_mlir)],
            check=True, capture_output=True, timeout=20,
        )
        service = BuildOnlyService(
            "fixture", recipe, lambda *_args, **_kwargs: "",
            tuple(
                (str(path), hashlib.sha256(path.read_bytes()).hexdigest())
                for path in (source, Path(__file__).resolve())
            ),
        )
        # Clang's imported module-flag carrier is outside the selected xDSL
        # scanner's grammar. The recipe above binds ABI/ISA for this neutral
        # object-only fixture; candidate artifacts are never rewritten here.
        artifact = "\n".join(
            line for line in llvm_mlir.read_text().splitlines()
            if not line.lstrip().startswith("llvm.module_flags ")
        ) + "\n"
        return artifact, service

    measured = {}
    for kind in ("bounded_loop", "straight_line"):
        artifact, service = source_artifact(kind)
        measured[kind] = S._compile_observation(
            artifact, tmp_path / kind, target="fixture", service=service, budget_s=60
        )
        assert measured[kind]["status"] == "compiled", measured[kind].get("error")
        assert measured[kind]["object_sha256"] and measured[kind]["build_selection"]["clang"]["sha256"]
    assert measured["straight_line"]["artifact_bytes"] > measured["bounded_loop"]["artifact_bytes"] * 8
    assert measured["straight_line"]["object_bytes"] > measured["bounded_loop"]["object_bytes"] * 4
    print({kind: {key: row[key] for key in ("artifact_bytes", "object_bytes", "compile_wall_s", "object_sha256")}
           for kind, row in measured.items()})
