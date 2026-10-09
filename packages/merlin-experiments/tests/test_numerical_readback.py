"""Protected full readback reconstruction, with an actual native independent case."""

from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from merlin_experiments.phase2 import numerical_readback as N

from merlin.perf.phase2_portfolio import QualityBudget, QualityLimit
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.sandbox import bwrap as BW


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def _frame(name, shape, values, width=4, signed=False):
    rows, cols = shape
    raw = b"".join(int(value).to_bytes(width, "little", signed=signed) for value in values)
    return (
        f"OUT_B64_BEGIN v1 {name} {rows} {cols} {width} {'s' if signed else 'u'}\n"
        f"OUT_B64_CHUNK 00000000 {len(raw):04x} {base64.b64encode(raw).decode()}\n"
        "OUT_B64_END\n"
    )


HARNESS = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "out_b64.h"
extern void run(float *, int32_t *, uint16_t *);
static void sink(const char *s) { fputs(s, stdout); }
int main(void) {
  float f[10]; int32_t i[10]; uint16_t h[3];
  run(f, i, h);
  merlin_out_b64 b;
  puts("OUT_B64_BEGIN v1 F 2 5 4 u");
  merlin_out_b64_init(&b, 10, 4, 0, sink);
  for (unsigned n=0; n<10; ++n) {
    uint32_t w; memcpy(&w, &f[n], 4);
    if (!merlin_out_b64_word(&b, w)) return 2;
  }
  if (!merlin_out_b64_finish(&b)) return 3;
  puts("OUT_B64_END");
  puts("OUT_B64_BEGIN v1 I 2 5 4 s");
  merlin_out_b64_init(&b, 10, 4, 1, sink);
  for (unsigned n=0; n<10; ++n)
    if (!merlin_out_b64_word(&b, (uint64_t)i[n])) return 4;
  if (!merlin_out_b64_finish(&b)) return 5;
  puts("OUT_B64_END");
  puts("OUT_B64_BEGIN v1 H 1 3 2 u");
  merlin_out_b64_init(&b, 3, 2, 0, sink);
  for (unsigned n=0; n<3; ++n)
    if (!merlin_out_b64_word(&b, h[n])) return 6;
  if (!merlin_out_b64_finish(&b)) return 7;
  puts("OUT_B64_END"); puts("DONE"); return 0;
}
"""


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", "")
    compiler = shutil.which("cc")
    assert compiler, "this native qualification requires a host C compiler"
    repo = tmp_path / "original"
    private = repo / "private"
    private.mkdir(parents=True)
    link = private / "link.ld"
    link.write_text("SECTIONS { .readback_test : { BYTE(0) } } INSERT AFTER .text;\n")
    provider = private / "provider.py"
    provider.write_text("# independent native renderer source identity\n")
    recipe = HarnessBuildRecipe(
        compiler=Path(compiler).resolve(),
        include_roots=(),
        support_sources=(),
        link_script=link,
        load_address=0,
        cflags=("-march=x86-64", "-ffp-contract=off"),
    )
    service = BuildOnlyService(
        "independent-native",
        recipe,
        lambda *_args, **_kwargs: HARNESS,
        tuple(
            (str(path), RB.file_sha256(path)) for path in (provider, Path(compiler).resolve(), Path(__file__).resolve())
        ),
    )
    policy = RB.ReadbackPolicy(RB.FULL_VALUES_B64)
    recipe_record, pins = RB.selected_build_inputs(service.target, recipe, service)
    cb = {
        "kernel_abi": {"kind": "whole_program", "outputs": ["F", "I", "H"]},
        "tensors": {
            "F": {"shape": [2, 5], "dtype": "f32", "role": "output"},
            "I": {"shape": [2, 5], "dtype": "i32", "role": "output"},
            "H": {"shape": [1, 3], "dtype": "bf16", "role": "output"},
        },
        "commands": [],
    }
    arms = []
    for name in ("reference", "candidate"):
        arm = private / name
        arm.mkdir()
        source = arm / "kernel.c"
        # Two independent expressions, exact on these integral input values.
        expression = "2 * (n - 3)" if name == "reference" else "n + n - 6"
        source.write_text(
            "#include <stdint.h>\nvoid run(float *f, int32_t *i, uint16_t *h) {\n"
            f" for(int n=0;n<10;++n) {{ f[n]=(float)({expression}); i[n]=n*n-7; }}\n"
            " h[0]=0x8000; h[1]=0x3f80; h[2]=0xc000;\n}\n"
        )
        harness = arm / "harness.c"
        harness.write_text(HARNESS)
        RB.stage_codec_header(arm)
        for src, output in ((source, arm / "kernel.o"), (harness, arm / "harness.o")):
            subprocess.run(recipe.compile_command(source=src, output=output), check=True, capture_output=True)
        executable = arm / "program.elf"
        subprocess.run(
            recipe.link_command(objects=[arm / "kernel.o", arm / "harness.o"], output=executable, link_script=link),
            check=True,
            capture_output=True,
        )
        console = subprocess.run([executable], check=True, capture_output=True, text=True).stdout
        (arm / "console.txt").write_text(console)
        _json(arm / "command_buffer.json", cb)
        _json(
            arm / "readback_build.json",
            RB.build_receipt(
                policy=policy,
                cb=cb,
                target=service.target,
                recipe_record=recipe_record,
                source_pins=pins,
                object_path=arm / "kernel.o",
                harness_path=harness,
                elf_path=executable,
            ),
        )
        arms.append(
            N.ReadbackFiles(
                arm / "command_buffer.json",
                arm / "readback_build.json",
                arm / "kernel.o",
                harness,
                executable,
                arm / "console.txt",
            )
        )
    budget = QualityBudget.elementwise("protected original independent reference", atol=0.03125, rtol=0.02)
    _json(private / "budget.json", budget.to_dict())
    run = tmp_path / "run"
    ws = run / "authoring/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [], "host_inputs": [{"path": "private"}]}
    manifest = run / "input_bundle_manifest.yaml"
    manifest.write_text(yaml.safe_dump(bundle))

    def freeze(selected_budget=budget):
        BW.materialize_bundle_inputs(ws, bundle, repo=repo)
        snapshot = BW.snapshot_record(ws)
        environment = {
            "workspace_path": str(ws),
            "bundle_input_snapshot": snapshot,
            "bundle_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        }
        kwargs = dict(
            run_dir=run,
            environment=environment,
            original_budget=private / "budget.json",
            budget=selected_budget,
            reference=arms[0],
            candidate=arms[1],
            reference_build=service,
            candidate_build=service,
        )
        return kwargs

    try:
        yield SimpleNamespace(
            private=private,
            arms=arms,
            budget=budget,
            service=service,
            recipe=recipe,
            cb=cb,
            policy=policy,
            pins=pins,
            recipe_record=recipe_record,
            freeze=freeze,
            ws=ws,
        )
    finally:
        BW.remove_bundle_snapshot(ws)


def _rebind(case, arm_index=1):
    arm = case.arms[arm_index]
    cb = json.loads(arm.command_buffer.read_text())
    _json(
        arm.build_receipt,
        RB.build_receipt(
            policy=case.policy,
            cb=cb,
            target=case.service.target,
            recipe_record=case.recipe_record,
            source_pins=case.pins,
            object_path=arm.kernel_object,
            harness_path=arm.harness,
            elf_path=arm.executable,
        ),
    )


def test_actual_native_full_outputs_are_admitted_without_hardware_claim(native_case):
    case = native_case
    # Independent numpy evaluation verifies the actual reference computation.
    raw, _ = N.parse_console(case.arms[0].console.read_text())
    values = N.decode_float_readback(raw, {"F": "f32", "H": "bf16"})
    assert values["F"] == np.asarray(2 * (np.arange(10) - 3), np.float32).reshape(2, 5).tolist()
    assert values["I"] == ((np.arange(10) ** 2) - 7).reshape(2, 5).tolist()
    observation = N.admit_protected_numerical_readback(**case.freeze())
    assert observation.quality.value_map == {"elementwise_violation_count": 0}
    assert observation.quality.parameters == case.budget.parameters
    assert observation.quality.complete and observation.element_count == 23
    assert observation.outputs == ("F", "H", "I")
    assert observation.missing and not hasattr(observation, "hardware_verified")
    assert all(RB.file_sha256(Path(path)) == digest for path, digest in observation.evidence)


@pytest.mark.parametrize("profile,expected", [("exact", 1), ("elementwise", 0)])
def test_signed_zero_uses_raw_bits_for_exact_policy(native_case, profile, expected):
    case = native_case
    console = case.arms[1].console.read_text()
    case.arms[1].console.write_text(
        console[: console.index("OUT_B64_BEGIN v1 H")] + _frame("H", (1, 3), [0, 0x3F80, 0xC000], 2) + "DONE\n"
    )
    budget = QualityBudget.exact(case.budget.reference) if profile == "exact" else case.budget
    _json(case.private / "budget.json", budget.to_dict())
    result = N.admit_protected_numerical_readback(**case.freeze(budget))
    assert result.quality.value_map == {budget.limits[0].metric: expected}


@pytest.mark.parametrize("word", [0x7FC12345, 0x7F800000, 0xFF800000])
@pytest.mark.parametrize("profile", ["exact", "elementwise"])
def test_nonfinite_raw_exactness_and_elementwise_refusal_count(native_case, word, profile):
    case = native_case
    for arm in case.arms:
        console = arm.console.read_text()
        arm.console.write_text(_frame("F", (2, 5), [word] + [0] * 9) + console[console.index("OUT_B64_BEGIN v1 I") :])
    budget = QualityBudget.exact(case.budget.reference) if profile == "exact" else case.budget
    _json(case.private / "budget.json", budget.to_dict())
    with np.errstate(invalid="ignore", over="ignore"):
        result = N.admit_protected_numerical_readback(**case.freeze(budget))
    assert result.quality.value_map[budget.limits[0].metric] == (0 if profile == "exact" else 1)


@pytest.mark.parametrize("word", [0x7F800000, 0xFF800000])
@pytest.mark.parametrize("arm", ["reference", "candidate"])
def test_asymmetric_infinity_counts_as_elementwise_violation(native_case, word, arm):
    case = native_case
    selected = case.arms[0 if arm == "reference" else 1]
    console = selected.console.read_text()
    # Keep both arm shapes equal and all other words identical and finite.
    values = np.asarray(2 * (np.arange(10) - 3), np.float32).view(np.uint32).copy()
    values[0] = word
    selected.console.write_text(_frame("F", (2, 5), values) + console[console.index("OUT_B64_BEGIN v1 I") :])
    raw, _ = N.parse_console(selected.console.read_text())
    decoded = N.decode_float_readback(raw, {"F": "f32"})["F"]
    assert np.isinf(decoded[0][0]) and (decoded[0][0] > 0) == (word == 0x7F800000)
    result = N.admit_protected_numerical_readback(**case.freeze())
    assert result.quality.value_map == {"elementwise_violation_count": 1}
    assert result.quality.parameters == (("atol", 0.03125), ("rtol", 0.02))
    assert result.quality.complete and result.element_count == 23 and result.missing


@pytest.mark.parametrize("kind", ["missing", "extra", "partial", "done", "geometry", "digest"])
def test_incomplete_or_extra_console_cannot_become_complete(native_case, kind):
    case = native_case
    arm = case.arms[1]
    console = arm.console.read_text()
    if kind == "missing":
        console = console[console.index("OUT_B64_BEGIN v1 I") :]
    elif kind == "extra":
        console = console.replace("DONE", _frame("EXTRA", (1, 1), [0]) + "DONE")
    elif kind == "partial":
        console = console.replace("OUT_B64_END", "", 1)
    elif kind == "done":
        console = console.replace("DONE", "")
    elif kind == "geometry":
        console = console.replace("F 2 5", "F 1 10")
    else:
        console = "OUTSUM F 2 5 0\nDONE\n"
    arm.console.write_text(console)
    with pytest.raises((ValueError, RuntimeError)):
        N.admit_protected_numerical_readback(**case.freeze())


@pytest.mark.parametrize("role", ["command_buffer", "build_receipt", "kernel_object", "harness", "executable", "codec"])
def test_stale_build_artifact_is_not_numerically_admitted(native_case, role):
    case = native_case
    arm = case.arms[1]
    path = arm.harness.parent / "out_b64.h" if role == "codec" else getattr(arm, role)
    if role == "command_buffer":
        cb = dict(case.cb)
        cb["commands"] = [{"changed": True}]
        _json(path, cb)
    elif role == "build_receipt":
        receipt = json.loads(path.read_text())
        receipt["elf_sha256"] = "0" * 64
        _json(path, receipt)
    else:
        path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises((ValueError, RuntimeError)):
        N.admit_protected_numerical_readback(**case.freeze())


@pytest.mark.parametrize("kind", ["dtype", "shape", "unknown", "integer_overflow"])
def test_output_type_and_extent_are_checked_even_after_build_rebinding(native_case, kind):
    case = native_case
    cb = json.loads(case.arms[1].command_buffer.read_text())
    if kind == "dtype":
        cb["tensors"]["F"]["dtype"] = "i32"
    elif kind == "shape":
        cb["tensors"]["F"]["shape"] = [1, 10]
    elif kind == "unknown":
        cb["tensors"]["F"]["dtype"] = "opaque"
    else:
        cb["tensors"]["I"]["dtype"] = "i4"
    _json(case.arms[1].command_buffer, cb)
    _rebind(case)
    with pytest.raises((ValueError, RuntimeError)):
        N.admit_protected_numerical_readback(**case.freeze())


def test_selected_recipe_and_renderer_sources_are_revalidated(native_case):
    case = native_case
    kwargs = case.freeze()
    service = case.service
    kwargs["candidate_build"] = replace(service, recipe=replace(service.recipe, cflags=(*service.recipe.cflags, "-O3")))
    with pytest.raises(ValueError, match="receipt"):
        N.admit_protected_numerical_readback(**kwargs)
    kwargs["candidate_build"] = service
    (case.private / "provider.py").write_text("# changed selected renderer\n")
    with pytest.raises(ValueError, match="pin changed"):
        N.admit_protected_numerical_readback(**kwargs)


@pytest.mark.parametrize("kind", ["tolerance", "profile", "reference", "caller_observation"])
def test_fixed_original_quality_budget_cannot_drift(native_case, kind):
    case = native_case
    kwargs = case.freeze()
    if kind == "tolerance":
        kwargs["budget"] = QualityBudget.elementwise(case.budget.reference, atol=1.0, rtol=1.0)
    elif kind == "profile":
        kwargs["budget"] = QualityBudget.exact(case.budget.reference)
    elif kind == "reference":
        kwargs["budget"] = QualityBudget.elementwise("different reference", atol=0.03125, rtol=0.02)
    else:
        kwargs["budget"] = {"passed": True, "sha256": "0" * 64}
    with pytest.raises(N.StageGateError):
        N.admit_protected_numerical_readback(**kwargs)


def test_task_metric_is_not_replaced_by_elementwise_proxy(native_case):
    case = native_case
    budget = QualityBudget.task("fixed task reference", limits=(QualityLimit("accuracy", "at_least", 0.5),))
    _json(case.private / "budget.json", budget.to_dict())
    with pytest.raises(N.StageGateError, match="exact or elementwise"):
        N.admit_protected_numerical_readback(**case.freeze(budget))


@pytest.mark.parametrize("kind", ["snapshot_bytes", "manifest", "environment", "public_path"])
def test_original_protected_ownership_is_required(native_case, kind):
    case = native_case
    kwargs = case.freeze()
    if kind == "snapshot_bytes":
        path = BW.bundle_snapshot_root(case.ws) / "repo/private/candidate/console.txt"
        path.chmod(0o600)
        path.write_text("DONE\n")
    elif kind == "manifest":
        (kwargs["run_dir"] / "input_bundle_manifest.yaml").write_text("host_inputs: []\n")
    elif kind == "environment":
        kwargs["environment"] = {"bundle_input_snapshot": {"version": 4, "content_sha256": "0" * 64}}
    else:
        kwargs["original_budget"] = case.private.parent / "public_budget.json"
        kwargs["original_budget"].write_text(json.dumps(case.budget.to_dict()))
    with pytest.raises((ValueError, RuntimeError)):
        N.admit_protected_numerical_readback(**kwargs)


def test_raw_caller_capability_cannot_supply_build_admission(native_case):
    kwargs = native_case.freeze()
    kwargs["candidate_build"] = {"verified": True, "elf_sha256": "0" * 64}
    with pytest.raises(N.StageGateError, match="typed host"):
        N.admit_protected_numerical_readback(**kwargs)


@pytest.mark.parametrize("delta,expected", [(0.01, 0), (0.5, 1)])
def test_elementwise_recomputes_every_value_at_fixed_tolerance(native_case, delta, expected):
    case = native_case
    floats = np.asarray(2 * (np.arange(10) - 3), np.float32)
    floats[-1] += delta
    console = case.arms[1].console.read_text()
    case.arms[1].console.write_text(
        _frame("F", (2, 5), floats.view(np.uint32)) + console[console.index("OUT_B64_BEGIN v1 I") :]
    )
    result = N.admit_protected_numerical_readback(**case.freeze())
    assert result.quality.value_map == {"elementwise_violation_count": expected}
    assert result.element_count == 23


def test_exact_preserves_distinct_nan_payloads(native_case):
    case = native_case
    for index, arm in enumerate(case.arms):
        console = arm.console.read_text()
        arm.console.write_text(
            _frame("F", (2, 5), [0x7FC12345 + index] + [0] * 9) + console[console.index("OUT_B64_BEGIN v1 I") :]
        )
    budget = QualityBudget.exact(case.budget.reference)
    _json(case.private / "budget.json", budget.to_dict())
    result = N.admit_protected_numerical_readback(**case.freeze(budget))
    assert result.quality.value_map == {"bitwise_mismatch_count": 1}


def test_equivalent_signed_float_containers_preserve_bits(native_case):
    case = native_case
    console = case.arms[1].console.read_text()
    case.arms[1].console.write_text(
        console[: console.index("OUT_B64_BEGIN v1 H")]
        + _frame("H", (1, 3), [-32768, 0x3F80, -16384], 2, signed=True)
        + "DONE\n"
    )
    budget = QualityBudget.exact(case.budget.reference)
    _json(case.private / "budget.json", budget.to_dict())
    result = N.admit_protected_numerical_readback(**case.freeze(budget))
    assert result.quality.value_map == {"bitwise_mismatch_count": 0}


def test_elementwise_does_not_silently_round_large_integer_outputs(native_case):
    case = native_case
    for arm in case.arms:
        cb = json.loads(arm.command_buffer.read_text())
        cb["tensors"]["I"]["dtype"] = "i64"
        _json(arm.command_buffer, cb)
        console = arm.console.read_text()
        arm.console.write_text(
            console[: console.index("OUT_B64_BEGIN v1 I")]
            + _frame("I", (2, 5), [2**53 + 1] + [0] * 9, 8, signed=True)
            + console[console.index("OUT_B64_BEGIN v1 H") :]
        )
    _rebind(case, 0)
    _rebind(case, 1)
    with pytest.raises(N.StageGateError, match="cannot preserve this integer"):
        N.admit_protected_numerical_readback(**case.freeze())


def test_mid_observation_private_mutation_is_refused(native_case, monkeypatch):
    case = native_case
    kwargs = case.freeze()
    original = N.compare

    def compare_then_tamper(*args, **kw):
        result = original(*args, **kw)
        path = BW.bundle_snapshot_root(case.ws) / "repo/private/candidate/console.txt"
        path.chmod(0o600)
        path.write_text("DONE\n")
        return result

    monkeypatch.setattr(N, "compare", compare_then_tamper)
    with pytest.raises((ValueError, RuntimeError)):
        N.admit_protected_numerical_readback(**kwargs)


def test_original_budget_does_not_accept_boolean_as_zero_threshold(native_case):
    case = native_case
    budget = QualityBudget.exact(case.budget.reference)
    record = budget.to_dict()
    record["limits"][0]["threshold"] = False
    _json(case.private / "budget.json", record)
    with pytest.raises(N.StageGateError, match="original quality budget"):
        N.admit_protected_numerical_readback(**case.freeze(budget))
