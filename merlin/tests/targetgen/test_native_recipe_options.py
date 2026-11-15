"""Native recipe choices are explicit, validated and bound to compilation identity."""

from argparse import Namespace
from dataclasses import replace

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
    assert {name: options.identity_record()[name] for name in RECIPE_DEFAULTS} == RECIPE_DEFAULTS
    assert CodegenOptions(**options.identity_record()) == options


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
