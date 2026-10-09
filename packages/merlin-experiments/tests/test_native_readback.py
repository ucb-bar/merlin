"""Selected readback revalidation keeps source and produced-byte mutation gates."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import runtime_dir
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService
from merlin.targetgen.native_model_execution import NativeModelExecutionError
from merlin.targetgen.native_readback import verify_build


@pytest.fixture(params=[RB.FULL_VALUES_B64, RB.FULL_VALUES_BIN])
def actual_build(request, tmp_path, monkeypatch):
    compiler = shutil.which("cc")
    assert compiler, "this independent native build gate requires a host compiler"
    # Own only copied codec resources, so mutation tests never edit core data.
    resources = tmp_path / "runtime" / "baremetal"
    resources.mkdir(parents=True)
    for name in ("out_b64.h", "out_bin.h"):
        shutil.copyfile(runtime_dir() / "baremetal" / name, resources / name)
    monkeypatch.setenv("MERLIN_RUNTIME_DIR", str(resources.parent))
    build = tmp_path / "build"
    build.mkdir()
    link = tmp_path / "link.ld"
    link.write_text("SECTIONS { .readback_test : { BYTE(0) } } INSERT AFTER .text;\n")
    provider = tmp_path / "provider.py"
    provider.write_text("# independent renderer pin\n")
    recipe = HarnessBuildRecipe(
        compiler=Path(compiler).resolve(), include_roots=(), support_sources=(),
        link_script=link, load_address=0, cflags=("-march=x86-64", "-ffp-contract=off"),
    )
    # The native adapter's ABI resolver is specifically RISC-V. Substitute
    # only that query for this independent host compiler; actual compilation,
    # selected input/codec reads, receipt hashes and revalidation remain real.
    monkeypatch.setattr(HarnessBuildRecipe, "with_effective_abi", lambda self: self)
    service = BuildOnlyService(
        "independent-native", recipe, lambda *_a, **_k: "synthetic renderer",
        tuple((str(path), RB.file_sha256(path)) for path in (provider, Path(__file__).resolve())),
    )
    policy = RB.ReadbackPolicy(request.param)
    selected = RB.selected_build_inputs(service.target, recipe.with_effective_abi(), service, policy=policy)
    RB.stage_codec_header(build, policy=policy)
    kernel = build / "kernel.c"
    kernel.write_text("int run(void) { return 0; }\n")
    harness = build / "harness.c"
    header = '#include "out_b64.h"\n' if policy.transport != RB.COHERENT_DUMP_V1 else ""
    harness.write_text(header + 'extern int run(void);\nint main(void) { return run(); }\n')
    for source, output in ((kernel, build / "kernel.o"), (harness, build / "harness.o")):
        subprocess.run(recipe.compile_command(source=source, output=output), check=True, capture_output=True)
    elf = build / "program.elf"
    subprocess.run(recipe.link_command(
        objects=[build / "kernel.o", build / "harness.o"], output=elf, link_script=link,
    ), check=True, capture_output=True)
    cb = {"kernel_abi": {"kind": "whole_program", "outputs": ["out"]}, "tensors": {
        "out": {"shape": [1, 1], "dtype": "u8", "role": "output"},
    }}
    receipt = RB.build_receipt(
        policy=policy, cb=cb, target=service.target, recipe_record=selected[0], source_pins=selected[1],
        object_path=build / "kernel.o", harness_path=harness, elf_path=elf,
    )
    import json

    (build / RB.BUILD_RECEIPT).write_text(json.dumps(receipt))
    return {
        "kwargs": {"target": service.target, "policy": policy, "service": service,
                   "output": tmp_path, "cb": cb, "elf": elf},
        "selected": selected, "receipt": receipt, "provider": provider, "resources": resources,
        "build": build,
    }


def test_initial_l2_l3_revalidation_keeps_actual_compiled_bytes(actual_build):
    case = actual_build
    record = verify_build(**case["kwargs"])
    assert record == (*case["selected"], case["receipt"])
    for stage in ("L2", "L3"):
        assert verify_build(
            **case["kwargs"], expected=case["selected"], expected_receipt=case["receipt"], stage=stage,
        ) == record
    # This native compilation binds bytes only; the harness has no value frames.
    assert record[2]["scope"].endswith("not complete toolchain closure or numerical correctness")


@pytest.mark.parametrize("mutation", ["provider", "selected_codec", "staged_codec", "kernel", "harness", "elf"])
def test_post_execution_source_and_produced_byte_mutations_refuse(actual_build, mutation):
    case = actual_build
    policy = case["kwargs"]["policy"]
    codec = "out_bin.h" if policy.transport == RB.FULL_VALUES_BIN else "out_b64.h"
    path = {
        "provider": case["provider"], "selected_codec": case["resources"] / codec,
        "staged_codec": case["build"] / codec, "kernel": case["build"] / "kernel.o",
        "harness": case["build"] / "harness.c", "elf": case["kwargs"]["elf"],
    }[mutation]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed|bind"):
        verify_build(**case["kwargs"], expected=case["selected"], stage="L2")


def test_exact_selected_inputs_refuse_a_replacement_with_stage_context(actual_build):
    case = actual_build
    replacement = (dict(case["selected"][0], changed="recipe"), case["selected"][1])
    with pytest.raises(NativeModelExecutionError, match="during L3 execution"):
        verify_build(**case["kwargs"], expected=replacement, stage="L3")


def test_saved_receipt_refuses_a_replacement_even_with_current_inputs(actual_build):
    case = actual_build
    replacement = dict(case["receipt"], changed="produced bytes")
    with pytest.raises(NativeModelExecutionError, match="receipt changed during L3"):
        verify_build(
            **case["kwargs"], expected=case["selected"], expected_receipt=replacement, stage="L3",
        )


@pytest.mark.parametrize("actual_build", [RB.COHERENT_DUMP_V1], indirect=True)
def test_coherent_build_uses_current_receipt_without_serial_codec(actual_build):
    case = actual_build
    assert not (case["build"] / "out_b64.h").exists()
    assert not (case["build"] / "out_bin.h").exists()
    record = verify_build(
        **case["kwargs"], expected=case["selected"], expected_receipt=case["receipt"], stage="memory pre-decode",
    )
    assert record[2]["schema"] == "merlin_readback_build_v2"
    assert record[0]["readback_transport"] == case["kwargs"]["policy"].record()
    for path in (case["build"] / "kernel.o", case["build"] / "harness.c", case["kwargs"]["elf"]):
        original = path.read_bytes()
        path.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="receipt"):
            verify_build(**case["kwargs"], expected=case["selected"], stage="memory pre-decode")
        path.write_bytes(original)


@pytest.mark.parametrize("policy", [None, {"transport": RB.FULL_VALUES_BIN}])
def test_readback_build_helper_requires_typed_explicit_choice(actual_build, policy):
    kwargs = dict(actual_build["kwargs"], policy=policy)
    with pytest.raises(ValueError, match="explicit"):
        verify_build(**kwargs)
