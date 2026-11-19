"""Optional provider hooks keep target choices outside the generic CLI and options."""
from argparse import Namespace
from dataclasses import dataclass

from merlin.semantic_compiler.codegen import CodegenOptions, options_from_args
from merlin.targetgen.cli import build_parser
from merlin.targetgen.native_codegen_cli import prepare_native_arguments


@dataclass(frozen=True)
class ProviderOptions(CodegenOptions):
    provider_toggle: bool = False


class Provider:
    @staticmethod
    def configure_codegen_parser(parser):
        parser.add_argument("--provider-toggle", action="store_true")
        parser.set_defaults(_codegen_options_type=ProviderOptions)

    @staticmethod
    def expand_codegen_arguments(args):
        if args.recipe == "example":
            args.provider_toggle = True


def test_options_factory_retains_generic_fields_and_provider_fields():
    args = Namespace(_codegen_options_type=ProviderOptions, provider_toggle=True,
                     execution_instruction_limit=123)
    assert options_from_args(args) == ProviderOptions(provider_toggle=True, execution_instruction_limit=123)
    assert options_from_args(Namespace()) == CodegenOptions()


def test_selected_provider_owns_argument_registration_and_expansion(monkeypatch):
    monkeypatch.setenv("MERLIN_NATIVE_SNAPSHOT", "installed-snapshot")
    provider = Provider()
    parser = build_parser(binding=provider)
    args = parser.parse_args([
        "native-compile", "--engine", "merlin_native", "--support", "example",
        "--target-source", "target", "--compile-inputs", "inputs", "--application",
        "--recipe", "example", "--out", "output",
    ])
    prepare_native_arguments(args, provider)
    assert options_from_args(args).provider_toggle
    assert args.snapshot == "installed-snapshot"
    assert args.abi == "auto"
    assert args.mode == "diagnostic"
    assert args.materialization_cache
    assert not args.region_materialization_cache
