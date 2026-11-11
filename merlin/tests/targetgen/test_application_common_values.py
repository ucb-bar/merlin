"""Application commoning is explicit and supported on compile and verify."""

import pytest

from merlin.targetgen.cli import _native_compile_report_with_options, build_parser


@pytest.mark.parametrize("command", ["native-compile", "native-verify"])
def test_commoning_selection_is_opt_in_on_both_entrypoints(command):
    parser = build_parser()
    flags = [
        command,
        "--engine",
        "merlin_native",
        "--mode",
        "diagnostic",
        "--support",
        "synthetic",
        "--snapshot",
        "snapshot",
        "--target-source",
        "target",
        "--out",
        "out",
        "--application",
        "--compile-inputs",
        "capture",
    ]
    if command == "native-verify":
        flags += ["--artifact", "artifact"]
    else:
        flags += ["--abi", "auto"]
    assert parser.parse_args(flags).application_common_values is False
    assert parser.parse_args([*flags, "--application-common-values"]).application_common_values is True


def test_nonapplication_compile_refuses_the_flag_before_loading_support(monkeypatch):
    parser = build_parser()
    args = parser.parse_args(
        [
            "native-compile",
            "--engine",
            "merlin_native",
            "--abi",
            "auto",
            "--mode",
            "diagnostic",
            "--support",
            "synthetic",
            "--snapshot",
            "snapshot",
            "--target-source",
            "target",
            "--out",
            "out",
            "--request",
            "request",
            "--application-common-values",
        ]
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("support must not be loaded for an invalid selection")

    monkeypatch.setattr("merlin.semantic_compiler.target_binding.load_native_target_binding", unexpected)
    report = _native_compile_report_with_options(args)
    assert "--application-common-values requires --application" in str(report)


@pytest.mark.parametrize("selected", [False, True])
def test_staged_application_preserves_a_public_sibling_region(monkeypatch, selected):
    from dataclasses import replace
    from types import SimpleNamespace

    from merlin.semantic_compiler.source_preparation import verify_source_preparation
    from merlin.targetgen import native_application
    from merlin.tests.ir.test_common_values import sibling_graph
    from merlin.tests.ir.test_semantic_regions import plan_for

    source, _ = sibling_graph()
    # Keep both copies live; publishing one sibling must retain the other too.
    source = replace(source, outputs=("a", "ae", "b", "out"), output_storages=("external",) * 4)
    staged = SimpleNamespace(
        precision_policy={}, original_source_sha256="source", argument_bindings=(), program="program", entry="forward"
    )
    adapter = SimpleNamespace(plan=lambda request: plan_for(request, ((0, 1), (2, 3), (4,))))
    binding = SimpleNamespace(application_adapter=lambda *args, **kwargs: adapter)
    monkeypatch.setattr(native_application, "load_compile_inputs", lambda path: staged)
    monkeypatch.setattr(native_application, "load_lifted_argument_bytes", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        native_application, "translate_linalg_text", lambda *args, **kwargs: SimpleNamespace(request=source)
    )
    args = build_parser().parse_args(
        [
            "native-compile",
            "--engine",
            "merlin_native",
            "--abi",
            "auto",
            "--mode",
            "diagnostic",
            "--support",
            "synthetic",
            "--snapshot",
            "snapshot",
            "--target-source",
            "target",
            "--out",
            "out",
            "--application",
            "--compile-inputs",
            "capture",
            *(["--application-common-values"] if selected else []),
        ]
    )
    _, result, _, _, receipt, _ = native_application._source(args, binding, "synthetic")
    assert result == source
    assert verify_source_preparation(result, receipt) == source
    if selected:
        assert receipt["common_values"]["groups"] == [["a", "ae"], ["b", "be"], ["out"]]
