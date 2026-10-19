"""Real native object links certify explicit portable suppliers without target runs."""

import shutil
import subprocess
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from merlin_experiments.phase1.feedback import private_compilation_inputs as compilation
from merlin_experiments.phase1.feedback import private_linalg_support as linalg
from merlin_experiments.phase1.feedback import private_linkage_support as linkage

from merlin.common.digest import sha256_file
from merlin.llvmlower.compilation_recipe import CompilationRecipe
from merlin.llvmlower.link_supplier_trace import trace_symbol_flags
from merlin.runtime.host_math import build_host_math, host_math_recipe, record_math_suppliers, runtime_symbol_bindings
from merlin.targetgen.host_linkage_contract import (
    RUNTIME_SCHEMA,
    RUNTIME_SUPPLIER,
    runtime_recipe,
    validate_source_linkage_contract,
)


def contract(symbol="sinf", policies=("portable_exp_trig", "portable_neural_f32")):
    records = []
    for policy in policies:
        recipe = host_math_recipe(policy)
        records.append(
            {
                "policy": policy,
                "source_sha256": sha256(recipe.source.encode()).hexdigest(),
                "compile_flags": list(recipe.compile_flags),
                "link_flags": list(recipe.link_flags),
            }
        )
    return {"schema": RUNTIME_SCHEMA, "supplier": RUNTIME_SUPPLIER, "symbols": [symbol], "recipes": records}


def test_closed_runtime_contract_selects_exact_source_policy_and_flags():
    body = "merlin.static_f32_math_source_body.v1"
    selected = validate_source_linkage_contract(body, "math.sin", contract())
    for policy in ("portable_exp_trig", "portable_neural_f32"):
        assert runtime_recipe(selected, policy)["policy"] == policy
    for key, value in (
        ("source_sha256", "0" * 64),
        ("compile_flags", ["-ffast-math"]),
        ("link_flags", ["-Wl,--wrap=cosf"]),
    ):
        wrong = deepcopy(selected)
        wrong["recipes"][0][key] = value
        with pytest.raises(ValueError):
            runtime_recipe(wrong, "portable_exp_trig")
    with pytest.raises(ValueError):
        runtime_recipe(selected, "native")
    with pytest.raises(ValueError):
        validate_source_linkage_contract("merlin.static_mixed_pointwise_body.v1", "f64_math_sin", selected)
    with pytest.raises(ValueError):
        validate_source_linkage_contract(body, "math.cos", selected)
    with pytest.raises(ValueError):
        validate_source_linkage_contract(body, "math.sin", {**selected, "supplier": "selected_math_archive"})


def _source(monkeypatch):
    import test_private_linkage_support as fixture

    original = fixture._declaration

    def declaration(*args, **kwargs):
        row = original(*args, **kwargs)
        row["linkage_contract"] = contract()
        return row

    monkeypatch.setattr(fixture, "_declaration", declaration)
    admission, row, parsed = fixture._admission("a" * 64)
    assert admission["status"] == "admitted"
    raw = "d" * 64
    witness = linalg.begin(raw, raw, 1)
    linalg.record(witness, row, admission, parsed, {0: row})
    pending = linkage.begin(raw, raw, 1)
    linkage.record(pending, row, admission, {0: row}, witness)
    return {
        "source_sha256": raw,
        "normalized_source_sha256": raw,
        "n_source_operations": 1,
        "linalg_host_support": witness,
        linkage.FIELD: pending,
    }


def _native_build(tmp_path, policy):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler unavailable")
    cc = str(Path(cc).resolve())
    flags = ["-O2"]
    archive = Path(subprocess.check_output([cc, "-print-file-name=libm.a"], text=True).strip()).resolve()
    if not archive.is_file():
        pytest.skip("explicit native libm unavailable")
    recipe = CompilationRecipe(tmp_path, producer=Path(__file__))

    def run(argv):
        return subprocess.run(argv, capture_output=True, text=True, timeout=15)

    def compile_math(argv):
        return recipe.run(
            argv, runner=run, inputs=[tmp_path / "host_math/host_math.c"], output=tmp_path / "host_math/host_math.o"
        )

    objects, wrap_flags = build_host_math(policy, tmp_path / "host_math", cc, flags, compile_math)
    main, main_obj = tmp_path / "main.c", tmp_path / "main.o"
    main.write_text("extern float sinf(float); volatile float value=0.5f; int main(void){return sinf(value)>2;}\n")
    recipe.run([cc, *flags, "-c", main, "-o", main_obj], runner=run, inputs=[main], output=main_obj)
    elf = tmp_path / "model.elf"
    wrappers = runtime_symbol_bindings(policy, ["sinf"])
    inputs = [main_obj, *objects, archive]
    result = recipe.run(
        [cc, *flags, *inputs, *wrap_flags, *trace_symbol_flags(list(wrappers.values())), "-o", elf],
        runner=run,
        inputs=inputs,
        output=elf,
    )
    record_math_suppliers(recipe, archive, None, wrappers, objects, result)
    recipe.completed(elf)
    selected = {
        "gcc_requested": cc,
        "gcc": cc,
        "gcc_sha256": sha256_file(Path(cc)),
        "gcc_cflags": flags,
        "archive": str(archive),
        "archive_sha256": sha256_file(archive),
        "host_isa": "native_fixture",
        "simulator_isa": "native_fixture",
        "host_package": str(tmp_path),
        "host_package_tree_sha256": "a" * 64,
        "dts": str(tmp_path / "unused.dts"),
        "dts_sha256": "b" * 64,
        "vlen": None,
    }
    binding = compilation.verify(
        {"compilation_recipe": {"path": str(recipe.path), "sha256": sha256_file(recipe.path)}}, elf, required=True
    )
    entry = {
        "source_sha256": "d" * 64,
        "capture_tree_sha256": "c" * 64,
        "elf_sha256": sha256_file(elf),
        "compilation_recipe": binding,
    }
    return selected, entry, recipe


@pytest.mark.parametrize("policy", ["portable_exp_trig", "portable_neural_f32"])
def test_actual_runtime_object_link_joins_source_and_rejects_mutations(tmp_path, monkeypatch, policy):
    source = _source(monkeypatch)
    assert linkage.build_options(source, policy) == {"host_math_symbols": ["sinf"]}
    selected, entry, recipe = _native_build(tmp_path, policy)
    monkeypatch.setattr(linkage, "_selected_archive", lambda *args: deepcopy(selected))
    linked = {"capture_tree_sha256": "c" * 64, "candidate_tree_sha256": "e" * 64, "elf_sha256": entry["elf_sha256"]}
    with pytest.raises(ValueError, match="original selected source/reference policy"):
        linkage.link(
            source,
            entry,
            host_package=tmp_path,
            host_package_tree_sha256="a" * 64,
            vlen=None,
            dts=tmp_path / "unused",
            dts_sha256="b" * 64,
            host_isa="native_fixture",
            simulator_isa="native_fixture",
            linked_build=linked,
            host_math_policy="native",
        )
    linkage.link(
        source,
        entry,
        host_package=tmp_path,
        host_package_tree_sha256="a" * 64,
        vlen=None,
        dts=tmp_path / "unused",
        dts_sha256="b" * 64,
        host_isa="native_fixture",
        simulator_isa="native_fixture",
        linked_build=linked,
        host_math_policy=policy,
    )
    assert linkage.linked_complete(source, entry, "e" * 64)
    runtime = source[linkage.FIELD]["selection"]["runtime"]
    assert runtime["object"] == str(tmp_path / "host_math/host_math.o")
    assert recipe.record["link_suppliers"]["symbols"]["__wrap_sinf"]["definition"] == runtime["object"]
    for key, value in (
        ("object", str(tmp_path / "main.o")),
        ("compiler_sha256", "0" * 64),
        ("compile_flags", ["-ffast-math"]),
        ("policy", "native"),
    ):
        wrong = deepcopy(source)
        wrong[linkage.FIELD]["selection"]["runtime"][key] = value
        assert not linkage.linked_complete(wrong, entry, "e" * 64)
    changed = deepcopy(source)
    changed[linkage.FIELD]["host_math_policy"] = "native"
    assert not linkage.linked_complete(changed, entry, "e" * 64)
    path = tmp_path / "host_math/host_math.c"
    path.write_text(path.read_text() + "\n")
    assert not linkage.linked_complete(source, entry, "e" * 64)


def test_explicit_runtime_request_preserves_native_and_f64_archive_defaults(monkeypatch):
    source = _source(monkeypatch)
    with pytest.raises(ValueError):
        linkage.build_options(source, "native")
    source[linkage.FIELD]["occurrences"][0]["contract"] = {
        "schema": "merlin.host_linkage_contract.v1",
        "supplier": "selected_math_archive",
        "archive_sha256": "a" * 64,
        "symbols": ["sinf"],
    }
    with pytest.raises(ValueError, match="portable-provider"):
        linkage.build_options(source, "portable_exp_trig")
    assert linkage.build_options(source, "native") == {"math_archive_symbols": ["sinf"]}
