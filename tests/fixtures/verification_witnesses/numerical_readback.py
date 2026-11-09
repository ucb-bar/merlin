from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from merlin.experiments.phase2 import numerical_readback as N
from merlin.perf.phase2_portfolio import QualityBudget, QualityLimit
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.sandbox import bwrap as BW


def _json(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def _frame(name, shape, values, width=4, signed=False):
    (rows, cols) = shape
    raw = b"".join((int(value).to_bytes(width, "little", signed=signed) for value in values))
    return f"OUT_B64_BEGIN v1 {name} {rows} {cols} {width} {('s' if signed else 'u')}\nOUT_B64_CHUNK 00000000 {len(raw):04x} {base64.b64encode(raw).decode()}\nOUT_B64_END\n"


HARNESS = '\n#include <stdint.h>\n#include <stdio.h>\n#include <string.h>\n#include "out_b64.h"\nextern void run(float *, int32_t *, uint16_t *);\nstatic void sink(const char *s) { fputs(s, stdout); }\nint main(void) {\n  float f[10]; int32_t i[10]; uint16_t h[3];\n  run(f, i, h);\n  merlin_out_b64 b;\n  puts("OUT_B64_BEGIN v1 F 2 5 4 u");\n  merlin_out_b64_init(&b, 10, 4, 0, sink);\n  for (unsigned n=0; n<10; ++n) {\n    uint32_t w; memcpy(&w, &f[n], 4);\n    if (!merlin_out_b64_word(&b, w)) return 2;\n  }\n  if (!merlin_out_b64_finish(&b)) return 3;\n  puts("OUT_B64_END");\n  puts("OUT_B64_BEGIN v1 I 2 5 4 s");\n  merlin_out_b64_init(&b, 10, 4, 1, sink);\n  for (unsigned n=0; n<10; ++n)\n    if (!merlin_out_b64_word(&b, (uint64_t)i[n])) return 4;\n  if (!merlin_out_b64_finish(&b)) return 5;\n  puts("OUT_B64_END");\n  puts("OUT_B64_BEGIN v1 H 1 3 2 u");\n  merlin_out_b64_init(&b, 3, 2, 0, sink);\n  for (unsigned n=0; n<3; ++n)\n    if (!merlin_out_b64_word(&b, h[n])) return 6;\n  if (!merlin_out_b64_finish(&b)) return 7;\n  puts("OUT_B64_END"); puts("DONE"); return 0;\n}\n'


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
            (
                (str(path), RB.file_sha256(path))
                for path in (provider, Path(compiler).resolve(), Path(__file__).resolve())
            )
        ),
    )
    policy = RB.ReadbackPolicy(RB.FULL_VALUES_B64)
    (recipe_record, pins) = RB.selected_build_inputs(service.target, recipe, service)
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
        expression = "2 * (n - 3)" if name == "reference" else "n + n - 6"
        source.write_text(
            f"#include <stdint.h>\nvoid run(float *f, int32_t *i, uint16_t *h) {{\n for(int n=0;n<10;++n) {{ f[n]=(float)({expression}); i[n]=n*n-7; }}\n h[0]=0x8000; h[1]=0x3f80; h[2]=0xc000;\n}}\n"
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


# Explicit fixture/reference reexports consumed by retained direct tests.
__all__ = [
    "BW",
    "BuildOnlyService",
    "HARNESS",
    "HarnessBuildRecipe",
    "N",
    "Path",
    "QualityBudget",
    "QualityLimit",
    "RB",
    "SimpleNamespace",
    "_frame",
    "_json",
    "_rebind",
    "base64",
    "hashlib",
    "json",
    "native_case",
    "np",
    "pytest",
    "shutil",
    "subprocess",
    "yaml",
]
