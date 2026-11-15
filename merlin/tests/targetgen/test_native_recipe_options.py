"""Native recipe choices are explicit, validated and bound to compilation identity."""

from argparse import Namespace
from dataclasses import replace
from types import SimpleNamespace

import pytest

from merlin.semantic_compiler.codegen import (
    CodegenOptions,
    codegen_options,
    current_codegen_options,
    options_from_args,
)

RECIPE_DEFAULTS = {
    "native_pipeline": "mandatory",
    "adjacent_patterns": "",
    "exponent_page_cache": False,
    "staging_storage": "dram",
    "looped_operand_reuse": "ordered",
    "quantized_operand_reuse": "ordered",
    "exact_binary_lane_groups": False,
    "exact_binary_vector_preflight": False,
    "exact_cast_vector_preflight": False,
    "static_quantizer_sizing": False,
    "exact_binary_routes": False,
    "capture_postspill": False,
    "kernel_library": False,
}
HISTORICAL_DEFAULT_RECORD = {
    "reduction_unroll": "auto",
    "program_size_limit_bytes": None,
    "tuning_profile": None,
    "io_memory_bytes": None,
}
RECIPE_VARIANTS = [
    ("native_pipeline", "optimized"),
    ("adjacent_patterns", "cast,binary"),
    ("exponent_page_cache", True),
    ("staging_storage", "vmem"),
    ("staging_storage", "vmem-inputs"),
    ("looped_operand_reuse", "resident-a"),
    *[("quantized_operand_reuse", policy) for policy in (
        "row-cache", "prefix-cache", "wide-cache", "paired-cache", "panel-cache", "batch-panel-cache",
    )],
    ("exact_binary_lane_groups", True),
    ("exact_binary_vector_preflight", True),
    ("exact_cast_vector_preflight", True),
    ("static_quantizer_sizing", True),
    ("exact_binary_routes", True),
    ("capture_postspill", True),
    ("kernel_library", True),
]


def test_recipe_defaults_are_explicit_and_roundtrip_without_arguments():
    options = options_from_args(Namespace())
    assert options == CodegenOptions()
    assert {name: getattr(options, name) for name in RECIPE_DEFAULTS} == RECIPE_DEFAULTS
    assert options.identity_record() == HISTORICAL_DEFAULT_RECORD
    assert CodegenOptions(**options.identity_record()) == options


def test_all_recipe_variants_partition_actual_materialization_cache_keys():
    from merlin.semantic_compiler.materialization import MaterializationCache, materialization_key
    from merlin.semantic_compiler.model import KernelRequest, SemanticNode, TensorType

    tensor = TensorType((32,), "bf16", "raw-bit-pattern")
    source = KernelRequest(
        nodes=(SemanticNode("input", "input", (), tensor, effect="input"),),
        outputs=("input",), output_storages=("external",),
        input_storages=(("input", "external"),), target_identity="recipe-test-target",
    )
    options = [CodegenOptions(), *[replace(CodegenOptions(), **{name: value}) for name, value in RECIPE_VARIANTS]]
    keys = [materialization_key(
        source, target_choice={"provider": "recipe-test"}, options=selected,
        compilation_identity={"snapshot": "fixed", "toolchain": "fixed"},
    ) for selected in options]
    historical_key = materialization_key(
        source, target_choice={"provider": "recipe-test"},
        options=SimpleNamespace(identity_record=lambda: HISTORICAL_DEFAULT_RECORD),
        compilation_identity={"snapshot": "fixed", "toolchain": "fixed"},
    )
    assert keys[0] == historical_key
    assert len({key.digest for key in keys}) == len(keys)
    assert len({key.identity for key in keys}) == len(keys)
    cache = MaterializationCache("recipe-test")
    for index, key in enumerate(keys):
        assert cache.get(key) is None
        cache.put(key, index)
    assert [cache.get(key) for key in keys] == list(range(len(keys)))


@pytest.mark.parametrize(("name", "value"), RECIPE_VARIANTS)
def test_each_recipe_choice_changes_only_its_identity_entry_and_restores_context(name, value):
    original = current_codegen_options()
    baseline = CodegenOptions().identity_record()
    selected = replace(CodegenOptions(), **{name: value})
    assert selected.identity_record() == {**baseline, name: value}
    assert options_from_args(Namespace(**{name: value})) == selected
    assert CodegenOptions(**selected.identity_record()) == selected
    with codegen_options(selected):
        assert getattr(current_codegen_options(), name) == value
    assert current_codegen_options() is original


@pytest.mark.parametrize("name", [name for name, default in RECIPE_DEFAULTS.items() if type(default) is bool])
@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_recipe_booleans_reject_non_boolean_values(name, value):
    with pytest.raises(ValueError, match=name):
        CodegenOptions(**{name: value})


@pytest.mark.parametrize("name", [
    "native_pipeline", "staging_storage", "looped_operand_reuse", "quantized_operand_reuse",
])
@pytest.mark.parametrize("value", ["", "unknown", None, True, 1])
def test_recipe_enums_reject_unknown_values(name, value):
    with pytest.raises(ValueError, match=name):
        CodegenOptions(**{name: value})


@pytest.mark.parametrize("value", [None, True, 1, [], {}])
def test_adjacent_patterns_requires_a_string(value):
    with pytest.raises(ValueError, match="adjacent_patterns"):
        CodegenOptions(adjacent_patterns=value)


def invocation(command, application):
    if command == "native-profile":
        return [
            command, "--support", "synthetic", "--snapshot", "snapshot", "--request", "request",
            "--target-source", "target", "--runner-command", "runner.json", "--out", "output",
        ]
    if command == "native-probe-model-regions":
        return [
            command, "--engine", "merlin_native", "--support", "synthetic", "--snapshot", "snapshot",
            "--compile-inputs", "inputs", "--abi", "abi.json", "--target-source", "target", "--out", "output",
        ]
    flags = [
        command, "--engine", "merlin_native", "--support", "synthetic", "--snapshot", "snapshot",
        "--mode", "diagnostic", "--out", "output",
    ]
    flags += ["--compile-inputs", "inputs", "--application"] if application else ["--request", "request"]
    if command == "native-compile":
        flags += ["--target-source", "target", "--abi", "auto"]
    else:
        flags += ["--artifact", "artifact"]
    return flags


@pytest.fixture(scope="module")
def parser():
    from merlin.targetgen.cli import build_parser

    return build_parser()


@pytest.mark.parametrize("command", ["native-compile", "native-verify"])
@pytest.mark.parametrize("application", [False, True], ids=["region", "application"])
def test_cli_recipe_defaults(parser, command, application):
    assert options_from_args(parser.parse_args(invocation(command, application))) == CodegenOptions()


@pytest.mark.parametrize("command", ["native-compile", "native-verify"])
@pytest.mark.parametrize("application", [False, True], ids=["region", "application"])
@pytest.mark.parametrize(("name", "value"), RECIPE_VARIANTS)
def test_each_cli_recipe_choice_roundtrips(parser, command, application, name, value):
    flags = ["--" + name.replace("_", "-")]
    if type(value) is not bool:
        flags.append(value)
    selected = options_from_args(parser.parse_args([*invocation(command, application), *flags]))
    assert selected == replace(CodegenOptions(), **{name: value})
    assert selected.identity_record()[name] == value


@pytest.mark.parametrize("name", [
    "native_pipeline", "staging_storage", "looped_operand_reuse", "quantized_operand_reuse",
])
def test_cli_rejects_invalid_recipe_enums(parser, name):
    with pytest.raises(SystemExit) as failure:
        parser.parse_args([*invocation("native-compile", False), "--" + name.replace("_", "-"), "unknown"])
    assert failure.value.code == 2


@pytest.mark.parametrize("command", ["native-profile", "native-probe-model-regions"])
def test_diagnostic_cli_recipe_defaults(parser, command):
    assert options_from_args(parser.parse_args(invocation(command, False))) == CodegenOptions()


@pytest.mark.parametrize("command", ["native-profile", "native-probe-model-regions"])
@pytest.mark.parametrize(("name", "value"), RECIPE_VARIANTS)
def test_diagnostic_cli_recipe_choices(parser, command, name, value):
    flags = ["--" + name.replace("_", "-")]
    if type(value) is not bool:
        flags.append(value)
    selected = options_from_args(parser.parse_args([*invocation(command, False), *flags]))
    assert selected == replace(CodegenOptions(), **{name: value})


def test_profile_preserves_recipe_for_source_key_and_all_variants(parser, tmp_path, monkeypatch):
    import json
    from unittest.mock import Mock

    from merlin.targetgen import cli, native_profile

    selected = {name: value for name, value in RECIPE_VARIANTS}
    flags = []
    for name, value in selected.items():
        flags.append("--" + name.replace("_", "-"))
        if type(value) is not bool:
            flags.append(value)
    runner = tmp_path / "runner.json"
    runner.write_text('["unused-runner"]')
    args = parser.parse_args([
        *invocation("native-profile", False), *flags,
        "--runner-command", str(runner), "--out", str(tmp_path / "profile"),
        "--program-size-limit-bytes", "8192", "--io-memory-bytes", "1048576",
        "--execution-instruction-limit", "9000000", "--exact-float-unroll",
    ])
    expected = options_from_args(args)
    original = current_codegen_options()
    parse_options = Mock(wraps=options_from_args)
    monkeypatch.setattr(native_profile, "options_from_args", parse_options)
    target_profile = SimpleNamespace(target_identity="synthetic", digest=lambda: "profile")
    snapshot = SimpleNamespace(profile=target_profile)
    monkeypatch.setattr(native_profile, "open_native_snapshot", lambda _: snapshot)
    observed = []

    def record_options(stage):
        observed.append((stage, current_codegen_options()))

    def load_request(*_args, **_kwargs):
        record_options("source")
        return SimpleNamespace(record=lambda: {}), "request", None, None

    def tuning_key(*_args):
        record_options("key")
        return {"recipe": current_codegen_options().identity_record()}

    binding = SimpleNamespace(
        profile=lambda: target_profile, tuning_key=tuning_key,
        plan_boundary_abi=lambda _: {"fixed_inputs": {}, "fixed_outputs": []},
    )
    monkeypatch.setattr(native_profile, "load_native_target_binding", lambda _: binding)
    monkeypatch.setattr(cli, "_load_selected_native_request", load_request)

    def compile_variant(_args):
        record_options("variant")
        raise RuntimeError("variant declined")

    monkeypatch.setattr(cli, "_native_compile_report_with_options", compile_variant)
    report = native_profile.run_profile(args)
    assert report["failed"] == 4
    parse_options.assert_called_once_with(args)
    assert observed == [
        ("source", expected), ("key", expected),
        *[("variant", replace(expected, reduction_unroll=str(factor))) for factor in (1, 2, 4, 8)],
    ]
    assert json.loads((tmp_path / "profile/profile.json").read_text())["key"] == {
        "recipe": expected.identity_record(),
    }
    assert current_codegen_options() is original
