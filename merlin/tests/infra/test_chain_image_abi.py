"""A same-size member-layout drift cannot reinterpret retained binary tables."""

import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.model_check.chain_image_abi import designated, verify
from merlin.model_check.chain_image_objects import pin

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def abi_case(tmp_path, request):
    runtime = ROOT / "merlin/runtime/c"
    original = tmp_path / "original"
    (original / "cgen").mkdir(parents=True)
    generated = original / "cgen/model_gen.h"
    generated.write_text("""#include "merlin_model.h"
static const merlin_arg_t MERLIN_ARGS[3] = {
 {MERLIN_WEIGHT, 16L, 2, {2,3}, 4},
 {MERLIN_INPUT, 0L, 1, {2}, 4},
 {MERLIN_OUTPUT, 0L, 1, {2}, 4},
};
""")
    if getattr(request, "param", {}).get("scalar"):
        generated.write_text(generated.read_text().replace("16L, 2, {2,3}", "16L, 0, {0}"))
    source = original / "model_main.c"
    source.write_text('#include "model_gen.h"\nconst void *table(void) { return MERLIN_ARGS; }\n')
    obj = original / "model_main.o"
    compiler = str(Path(shutil.which("gcc")).resolve())
    argv = [compiler, "-O0", "-I" + str(runtime), "-I" + str(generated.parent), "-c", str(source), "-o", str(obj)]
    subprocess.run(argv, check=True, capture_output=True, timeout=10)
    program = {
        "index": 0,
        "n_args": 3,
        "production": {
            "root": original,
            "by_name": {"model_main.o": pin(obj)},
            "recipe": {"commands": [{"requested_output": str(obj), "argv": argv, "inputs": [pin(source)]}]},
        },
    }
    work = tmp_path / "work"
    work.mkdir()
    header = work / "selected_model.h"
    header.write_bytes((runtime / "merlin_model.h").read_bytes())
    return program, work, header


def check(case):
    program, work, header = case
    recipe = CompilationRecipe(work, producer=Path(__file__))
    return verify(
        program,
        header,
        recipe=recipe,
        directory=work,
        runner=lambda argv: subprocess.run(argv, check=True, capture_output=True, text=True, timeout=10),
    )


def test_observed_full_argument_table_layout(abi_case):
    proof = check(abi_case)
    assert proof["status"] == "observed_argument_layout_equal"
    assert proof["table_bytes"] == 3 * 96


@pytest.mark.parametrize("abi_case", [{"scalar": True}], indirect=True)
def test_scalar_zero_rank_uses_real_c_dimension_padding_and_full_object_equality(abi_case):
    proof = check(abi_case)
    assert proof["status"] == "observed_argument_layout_equal"
    assert proof["table_bytes"] == 3 * 96


@pytest.mark.parametrize("rank_dims", ["1, {0}", "0, {1}", "0, {0,0}"])
def test_empty_tensor_or_noncanonical_scalar_padding_is_not_admitted(abi_case, rank_dims):
    generated = abi_case[0]["production"]["root"] / "cgen/model_gen.h"
    generated.write_text(generated.read_text().replace("2, {2,3}", rank_dims))
    with pytest.raises(ValueError, match="shape/storage ABI"):
        designated(generated, 3)


def test_same_size_member_name_drift_refused(abi_case):
    header = abi_case[2]
    text = (
        header.read_text()
        .replace("int kind;", "int TEMP;")
        .replace("int rank;", "int kind;")
        .replace("int TEMP;", "int rank;")
    )
    header.write_text(text)
    with pytest.raises(ValueError, match="member layout"):
        check(abi_case)


def test_generated_source_is_not_an_executable_expression(abi_case):
    program, _, _ = abi_case
    generated = program["production"]["root"] / "cgen/model_gen.h"
    generated.write_text(generated.read_text().replace("16L", "(8L+8L)"))
    with pytest.raises(ValueError, match="unsupported expression"):
        designated(generated, 3)
