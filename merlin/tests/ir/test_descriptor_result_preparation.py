"""Actual ordinary hook consumption, bounded selections and late mutation."""

import json
from dataclasses import asdict, replace

import pytest
from test_descriptor_result_identity import selection, source

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower import lower as L
from merlin.llvmlower.descriptor_result_preparation import DescriptorResultPreparation


def identities():
    return (selection(parse_mlir_text(source())),)


def test_missing_effects_reports_fallback_and_preserves_exact_source_bytes(tmp_path):
    roster = (replace(identities()[0], borrowed_effects=None),)
    prepared = DescriptorResultPreparation(roster)
    original = source()
    assert prepared.prepare(original, tmp_path) == original
    record = prepared.record()
    assert record["rewrites"] == 0
    assert record["selection"][0]["borrowed_effects"] is None
    assert record["calls"][0]["reason"] == "original borrowed effects are unavailable"
    assert record["selected_source"]["sha256"] == record["original_source"]["sha256"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("qualification_identity", "changed-original-effect-owner"),
        ("no_retained_aliases", False),
        ("no_releases", False),
        ("complete_before_return", False),
    ],
)
def test_effect_owner_and_each_original_lifetime_premise_stay_frozen(tmp_path, field, value):
    roster = identities()
    expected = asdict(roster[0].borrowed_effects)
    prepared = DescriptorResultPreparation(roster)
    prepared.prepare(source(), tmp_path)
    report = prepared.record()
    assert report["selection"][0]["borrowed_effects"] == expected
    assert report["calls"][0]["borrowed_effects"] == expected
    assert any(row["path"].endswith("/borrowed_buffer_calls.py") for row in report["implementation_sources"])
    object.__setattr__(roster[0].borrowed_effects, field, value)
    with pytest.raises(ValueError, match="original descriptor identity|borrowed ABI"):
        prepared.revalidate()


def test_earlier_callback_cannot_relax_or_replace_original_effect_owner(tmp_path, monkeypatch):
    from merlin.llvmlower import automatic_rows

    roster = identities()
    original = automatic_rows.prepare_automatic_rows

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(roster[0].borrowed_effects, "qualification_identity", "unreviewed-owner")
        return result

    monkeypatch.setattr(automatic_rows, "prepare_automatic_rows", mutate)
    monkeypatch.setattr(L, "preprocess_text", lambda *a, **k: pytest.fail("stale effects reached preprocessing"))
    with pytest.raises(ValueError, match="original descriptor identity"):
        L.lower_model(
            source(), tmp_path, targets=(), descriptor_result_identities=roster, index_bits=64, data_layout="e-p:64:64"
        )


@pytest.mark.parametrize("value", [[], None, (object(),)])
def test_only_original_typed_immutable_rosters_are_accepted(tmp_path, value):
    with pytest.raises(ValueError, match="immutable|typed"):
        L.lower_model(source(), tmp_path / "no-build", targets=(), descriptor_result_identities=value)


@pytest.mark.parametrize(
    "options",
    [
        {"op_profile": True},
        {"textual": True},
        {"vectorize": True},
        {"parallel": True},
        {"index_bits": 32},
        {"shared_source_lowerings": (object(),)},
        {"borrowed_buffer_calls": object()},
    ],
)
def test_unsupported_compositions_refuse_before_any_tools(tmp_path, monkeypatch, options):
    monkeypatch.setattr(L, "lower_to_llvm_ir", lambda *a, **k: pytest.fail("unsupported route reached compiler"))
    with pytest.raises(ValueError, match="scalar serial 64-bit"):
        L.lower_model(source(), tmp_path, targets=(), descriptor_result_identities=identities(), **options)


@pytest.mark.parametrize("kind", ["identity", "selected", "report"])
def test_original_identity_and_prepared_products_stay_fresh(tmp_path, kind):
    roster = identities()
    prepared = DescriptorResultPreparation(roster)
    prepared.prepare(source(), tmp_path)
    if kind == "identity":
        object.__setattr__(roster[0], "borrowed_symbol", "different_original_callback")
    elif kind == "selected":
        (tmp_path / "descriptor-identity.selected.mlir").write_text("module {}")
    else:
        path = tmp_path / "descriptor-result-identity.json"
        doc = json.loads(path.read_text())
        doc["rewrites"] = 99
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="original descriptor identity"):
        prepared.revalidate()


def test_library_mutation_during_actual_parse_refuses_before_compiler(tmp_path, monkeypatch):
    from merlin.llvmlower import descriptor_result_preparation as P

    dependency = tmp_path / "original_selected_implementation.py"
    dependency.write_text("# original independently retained implementation\n")
    monkeypatch.setattr(P, "implementation_paths", lambda: (dependency,))
    roster = identities()
    original = P.linalg_mlir.parse_mlir_text

    def mutate(source):
        module = original(source)
        dependency.write_text("# changed during original parse callback\n")
        return module

    monkeypatch.setattr(P.linalg_mlir, "parse_mlir_text", mutate)
    prepared = DescriptorResultPreparation(roster)
    with pytest.raises(ValueError, match="original descriptor identity"):
        prepared.prepare(source(), tmp_path)


def test_identity_mutation_by_earlier_automatic_callback_never_reaches_preprocessor(tmp_path, monkeypatch):
    from merlin.llvmlower import automatic_rows

    roster = identities()
    original = automatic_rows.prepare_automatic_rows

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        object.__setattr__(roster[0], "result_argument", 1)
        return result

    monkeypatch.setattr(automatic_rows, "prepare_automatic_rows", mutate)
    monkeypatch.setattr(L, "preprocess_text", lambda *a, **k: pytest.fail("stale identity reached preprocessing"))
    with pytest.raises(ValueError, match="original descriptor identity"):
        L.lower_model(
            source(), tmp_path, targets=(), descriptor_result_identities=roster, index_bits=64, data_layout="e-p:64:64"
        )


def test_wrong_current_signature_refuses_instead_of_minting_a_wrapper(tmp_path, monkeypatch):
    roster = identities()
    wrong = replace(roster[0], argument_types=roster[0].argument_types[:2])
    monkeypatch.setattr(L, "preprocess_text", lambda *a, **k: pytest.fail("mismatched original ABI reached compiler"))
    with pytest.raises(ValueError, match="identity domain"):
        L.lower_model(
            source(),
            tmp_path,
            targets=(),
            descriptor_result_identities=(wrong,),
            index_bits=64,
            data_layout="e-p:64:64",
        )


def test_actual_destination_preparation_precedes_identity_and_reaches_upstream(tmp_path, monkeypatch):
    from pathlib import Path

    from merlin.common.paths import merlin_dir

    original = (merlin_dir() / "tests/ir/insert_destination_fixture.mlir").read_text()
    t = "tensor<4x5xi8>"
    declaration = f""""func.func"() <{{sym_name="writer",sym_visibility="private",
function_type=({t},{t},{t})->{t},arg_attrs=[{{bufferization.access="read"}},
{{bufferization.access="read"}},{{bufferization.access="write"}}]}}> ({{}}) {{llvm.emit_c_interface}} : ()->()"""
    original = (
        original.replace("module {", "module {" + declaration, 1)
        .replace("%input: tensor<2x3xf32>", f"%input: tensor<2x3xf32>,%weights:{t}")
        .replace(
            f"return %result : {t}",
            f"""%e = tensor.empty() : {t}
%c=linalg.fill ins(%zero:i8) outs(%e:{t})->{t}
%r=func.call @writer(%result,%weights,%c):({t},{t},{t})->{t}
return %r:{t}""",
        )
    )
    module = parse_mlir_text(original)
    roster = (selection(module),)
    path = tmp_path / "input.mlir"
    path.write_text(original)
    monkeypatch.setenv("MERLIN_AUTOMATIC_HOST_DESTINATIONS", "1")
    result = L.lower_model_file(
        path,
        tmp_path / "lower",
        targets=(),
        descriptor_result_identities=roster,
        index_bits=64,
        data_layout="e-p:64:64",
    )
    report = result.stats["descriptor_result_identity"]
    assert result.stats["automatic_host_destinations"]["rewrites"] == report["rewrites"] == 1
    assert report["original_source"]["sha256"] == result.stats["automatic_host_permutations"]["output_file_sha256"]
    consumed = parse_mlir_text(Path(report["original_source"]["path"]).read_text())
    generic = next(op for op in consumed.walk() if op.name == "linalg.generic")
    assert generic.outputs[0].owner.name == "tensor.extract_slice"


@pytest.mark.parametrize("role", ["upstream", "report"])
def test_late_tool_callback_cannot_rebind_prepared_source_or_report(tmp_path, monkeypatch, role):
    roster = identities()
    original = L.lower_to_llvm_ir

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        name = "model.upstream.mlir" if role == "upstream" else "descriptor-result-identity.json"
        path = tmp_path / name
        path.write_bytes(path.read_bytes() + b"changed after original upstream compilation")
        return result

    monkeypatch.setattr(L, "lower_to_llvm_ir", mutate)
    monkeypatch.setattr(L, "build_host_shared", lambda *a, **k: pytest.fail("stale preparation reached native codegen"))
    with pytest.raises(ValueError, match="original descriptor identity"):
        L.lower_model(
            source(),
            tmp_path,
            targets=("host",),
            descriptor_result_identities=roster,
            index_bits=64,
            data_layout="e-p:64:64",
        )


def test_registered_pipeline_hook_cannot_change_after_selection(tmp_path, monkeypatch):
    from merlin.llvmlower.descriptor_result_identity import FEATURE
    from merlin.llvmlower.impr_features import get

    prepared = DescriptorResultPreparation(identities())
    prepared.prepare(source(), tmp_path)
    prepared.selected_feature()
    feature = get(FEATURE)
    original = feature.edit_pipeline
    try:
        object.__setattr__(feature, "edit_pipeline", lambda passes: passes)
        with pytest.raises(ValueError, match="original descriptor identity"):
            prepared.revalidate()
    finally:
        object.__setattr__(feature, "edit_pipeline", original)


def test_ineligible_current_source_has_no_selected_pipeline_feature(tmp_path):
    module = parse_mlir_text(source(shared=True))
    result = L.lower_model(
        source(shared=True),
        tmp_path,
        targets=(),
        descriptor_result_identities=(selection(module),),
        index_bits=64,
        data_layout="e-p:64:64",
    )
    report = result.stats["descriptor_result_identity"]
    assert report["rewrites"] == 0
    assert report["selected_source"]["sha256"] == report["original_source"]["sha256"]
    assert "descriptor_result_identity" not in json.loads((tmp_path / "lowering_recipe.json").read_text())["features"]


@pytest.mark.parametrize(
    "method", ["prepare", "revalidate", "selected_feature", "retain_upstream", "bind_recipe", "record"]
)
def test_earlier_callback_cannot_replace_selected_preparation_methods(tmp_path, monkeypatch, method):
    from merlin.llvmlower import automatic_rows

    original = automatic_rows.prepare_automatic_rows

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(DescriptorResultPreparation, method, lambda *a, **k: pytest.fail("changed method executed"))
        return result

    monkeypatch.setattr(automatic_rows, "prepare_automatic_rows", mutate)
    monkeypatch.setattr(L, "preprocess_text", lambda *a, **k: pytest.fail("changed preparation reached preprocessing"))
    with pytest.raises(ValueError, match="original descriptor identity"):
        L.lower_model(
            source(),
            tmp_path,
            targets=(),
            descriptor_result_identities=identities(),
            index_bits=64,
            data_layout="e-p:64:64",
        )


@pytest.mark.parametrize("method", ["revalidate", "bind_recipe", "record"])
def test_tool_callback_cannot_replace_final_preparation_methods(tmp_path, monkeypatch, method):
    original = L.lower_to_llvm_ir

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        monkeypatch.setattr(
            DescriptorResultPreparation, method, lambda *a, **k: pytest.fail("changed final method executed")
        )
        return result

    monkeypatch.setattr(L, "lower_to_llvm_ir", mutate)
    monkeypatch.setattr(
        L, "build_host_shared", lambda *a, **k: pytest.fail("changed preparation reached native codegen")
    )
    with pytest.raises(ValueError, match="original descriptor identity"):
        L.lower_model(
            source(),
            tmp_path,
            targets=("host",),
            descriptor_result_identities=identities(),
            index_bits=64,
            data_layout="e-p:64:64",
        )
