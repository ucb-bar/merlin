"""Measured choices never bypass variant correctness or compilation identity."""

import pytest

from merlin.semantic_compiler.codegen import (
    CodegenOptions,
    choose_reduction_unroll,
    codegen_options,
    current_codegen_options,
    options_from_args,
)


def profile():
    return {
        "schema": "merlin.native_tuning_profile.v1",
        "key": {"source": "a"},
        "objective": "device_cycles",
        "variants": [
            {
                "factor": f,
                "status": "passed",
                "correctness": "passed",
                "bytes": 100 * f,
                "device_cycles": [100 // f] * 5,
            }
            for f in (1, 2, 4, 8)
        ],
    }


def test_profile_selection_rejects_failed_variants_and_obeys_identity():
    value = profile()
    value["variants"][-1] = {"factor": 8, "status": "numerical_mismatch"}
    with codegen_options(CodegenOptions(tuning_profile=value)):
        assert choose_reduction_unroll({"source": "a"})[0] == 4
        assert choose_reduction_unroll({"source": "b"})[0] == 1
    assert choose_reduction_unroll({"source": "a"})[0] == 1
    with codegen_options(CodegenOptions("2", tuning_profile=value)):
        assert choose_reduction_unroll({"source": "a"})[0] == 2


def test_profile_ties_prefer_smaller_program_and_options_restore_on_error():
    value = profile()
    for row in value["variants"]:
        row["device_cycles"] = [50] * 5
    with codegen_options(CodegenOptions(tuning_profile=value)):
        assert choose_reduction_unroll(value["key"])[0] == 1
    with pytest.raises(ValueError):
        with codegen_options(CodegenOptions("4")):
            raise ValueError("test failure")
    assert choose_reduction_unroll({})[0] == 1


def test_matching_all_failed_profile_cannot_publish_an_unchecked_default():
    value = profile()
    for row in value["variants"]:
        row["status"] = "search_timeout"
    with codegen_options(CodegenOptions(tuning_profile=value)):
        with pytest.raises(ValueError, match="no checked variant"):
            choose_reduction_unroll(value["key"])
        assert choose_reduction_unroll({"source": "different"})[0] == 1


def test_identical_images_do_not_turn_host_timing_noise_into_unrolling():
    value = profile()
    for row in value["variants"]:
        row["program_sha256"] = "a" * 64
    with codegen_options(CodegenOptions(tuning_profile=value)):
        assert choose_reduction_unroll(value["key"])[0] == 1


def test_io_window_options_restore_and_parse_on_every_native_entrypoint():
    from merlin.targetgen.cli import build_parser

    parser = build_parser()
    for command, extra in (
        ("native-build", ["--support", "synthetic", "--cargo-target-dir", "cargo", "--source-revision", "test"]),
        (
            "native-compile",
            [
                "--support",
                "synthetic",
                "--snapshot",
                "snapshot",
                "--request",
                "request",
                "--target-source",
                "target",
                "--abi",
                "auto",
                "--mode",
                "diagnostic",
            ],
        ),
        (
            "native-verify",
            [
                "--support",
                "synthetic",
                "--snapshot",
                "snapshot",
                "--request",
                "request",
                "--artifact",
                "artifact",
                "--mode",
                "diagnostic",
            ],
        ),
    ):
        args = parser.parse_args(
            [command, "--engine", "merlin_native", "--out", "output", "--io-memory-bytes", "1048576", *extra]
        )
        assert options_from_args(args).io_memory_bytes == 1048576
    assert current_codegen_options().io_memory_bytes is None
    with pytest.raises(RuntimeError):
        with codegen_options(CodegenOptions(io_memory_bytes=1048576)):
            assert current_codegen_options().io_memory_bytes == 1048576
            with codegen_options(CodegenOptions(io_memory_bytes=524288)):
                assert current_codegen_options().io_memory_bytes == 524288
            raise RuntimeError("test failure")
    assert current_codegen_options().io_memory_bytes is None


@pytest.mark.parametrize("size", [0, -1, True, 1.5])
def test_invalid_io_sizes_are_not_silently_ignored(size):
    with pytest.raises(ValueError, match="I/O"):
        CodegenOptions(io_memory_bytes=size)


IO_GEOMETRY = {
    "io_memory_base": 0,
    "io_memory_alignment": 64,
    "io_memory_capacity_base": 0,
    "io_memory_capacity_bytes": 2**24,
}


def _native_args(command, output="output"):
    common = ["--out", str(output)]
    if command == "native-profile":
        return [*common, "--support", "synthetic", "--snapshot", "snapshot", "--request", "request",
                "--target-source", "target", "--runner-command", "runner.json"]
    common += ["--engine", "merlin_native", "--support", "synthetic"]
    if command == "native-build":
        return [*common, "--cargo-target-dir", "cargo", "--source-revision", "test"]
    common += ["--snapshot", "snapshot", "--mode", "diagnostic", "--request", "request"]
    if command == "native-compile":
        return [*common, "--target-source", "target", "--abi", "auto"]
    if command == "native-verify":
        return [*common, "--artifact", "artifact"]
    return common


@pytest.mark.parametrize("command", ["native-build", "native-select", "native-compile", "native-verify", "native-profile"])
def test_io_geometry_parses_and_preserves_historical_default_identity(command):
    from merlin.targetgen.cli import build_parser

    historical = {
        "reduction_unroll": "auto", "program_size_limit_bytes": None,
        "tuning_profile": None, "io_memory_bytes": None,
    }
    parser = build_parser()
    assert options_from_args(parser.parse_args([command, *_native_args(command)])).identity_record() == historical
    flags = [value for key, number in IO_GEOMETRY.items() for value in ("--" + key.replace("_", "-"), hex(number))]
    selected = options_from_args(parser.parse_args([command, *_native_args(command), *flags]))
    assert selected.identity_record() == {**historical, **IO_GEOMETRY}
    with codegen_options(selected):
        assert current_codegen_options() == selected
    assert current_codegen_options().identity_record() == historical


@pytest.mark.parametrize("name", tuple(IO_GEOMETRY))
@pytest.mark.parametrize("value", [-1, True, 1.5, "64"])
def test_io_geometry_rejects_invalid_integer_fields(name, value):
    with pytest.raises(ValueError, match=name):
        CodegenOptions(**{name: value})


@pytest.mark.parametrize("name", ["io_memory_alignment", "io_memory_capacity_bytes"])
def test_io_geometry_rejects_zero_size_and_alignment(name):
    with pytest.raises(ValueError, match=name):
        CodegenOptions(**{name: 0})


@pytest.mark.parametrize("command", ["native-build", "native-select", "native-compile", "native-verify"])
@pytest.mark.parametrize("name", tuple(IO_GEOMETRY))
def test_native_entrypoints_refuse_unsupported_geometry(command, name, tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    from merlin.semantic_compiler import snapshot, target_binding
    from merlin.targetgen.cli import main

    profile = SimpleNamespace(digest=lambda: "profile", target_identity="synthetic")
    binding = SimpleNamespace(profile=lambda: profile, codegen_capabilities=())
    monkeypatch.setattr(target_binding, "load_native_target_binding", lambda _: binding)
    monkeypatch.setattr(snapshot, "open_native_snapshot", lambda _: SimpleNamespace(profile=profile))
    output = tmp_path / "status.json"
    flags = ["--" + name.replace("_", "-"), str(IO_GEOMETRY[name])]
    assert main([command, *_native_args(command, output), *flags]) == 2
    report = (json.loads(output.read_text()) if command in ("native-select", "native-verify")
              else json.loads(capsys.readouterr().out))
    assert report["status"] == "unsupported_semantics"
    assert name in report["reason"]


def test_application_compile_receives_geometry_and_both_budgets(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.semantic_compiler import snapshot, target_binding
    from merlin.semantic_compiler.codegen import IO_MEMORY_OPTIONS
    from merlin.targetgen import native_application
    from merlin.targetgen.cli import main

    profile = SimpleNamespace(digest=lambda: "profile", target_identity="synthetic")
    binding = SimpleNamespace(profile=lambda: profile, codegen_capabilities=(*IO_MEMORY_OPTIONS, "program_size_limit_bytes"))
    monkeypatch.setattr(target_binding, "load_native_target_binding", lambda _: binding)
    monkeypatch.setattr(snapshot, "open_native_snapshot", lambda _: SimpleNamespace(profile=profile))
    observed = []

    def compile_application(*args, **kwargs):
        observed.append(current_codegen_options())
        return {"status": "emitted"}

    monkeypatch.setattr(native_application, "compile_application_report", compile_application)
    flags = [value for key, number in IO_GEOMETRY.items() for value in ("--" + key.replace("_", "-"), str(number))]
    assert main(["native-compile", *_native_args("native-compile", tmp_path / "compiled"),
                 "--application", "--application-memory-bytes", "8192", "--program-size-limit-bytes", "4096",
                 "--io-memory-bytes", "1024", *flags]) == 0
    assert observed == [CodegenOptions(program_size_limit_bytes=4096, io_memory_bytes=1024, **IO_GEOMETRY)]


@pytest.mark.parametrize("name", tuple(IO_GEOMETRY))
def test_application_replay_refuses_unsupported_geometry(name, tmp_path, monkeypatch, capsys):
    import json
    from types import SimpleNamespace

    from merlin.semantic_compiler import target_binding
    from merlin.targetgen.cli import main

    monkeypatch.setattr(target_binding, "load_native_target_binding", lambda _: SimpleNamespace(codegen_capabilities=()))
    assert main(["native-verify", *_native_args("native-verify", tmp_path / "status.json"),
                 "--application", "--" + name.replace("_", "-"), str(IO_GEOMETRY[name])]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "unsupported_semantics"
    assert name in report["reason"]


def test_profile_scope_receives_geometry_and_program_budget(monkeypatch):
    from argparse import Namespace

    from merlin.targetgen import native_profile

    monkeypatch.setattr(native_profile, "_run_profile_with_options", lambda _: current_codegen_options())
    selected = native_profile.run_profile(Namespace(program_size_limit_bytes=4096, **IO_GEOMETRY))
    assert selected == CodegenOptions(program_size_limit_bytes=4096, **IO_GEOMETRY)


@pytest.mark.parametrize('application', [False, True])
def test_verify_replays_stored_program_budget_separately_from_execution_limit(application, monkeypatch):
    from merlin.targetgen import cli

    parser = cli.build_parser()
    flags = ['native-verify', *_native_args('native-verify')]
    if application:
        flags.append('--application')
    defaults = options_from_args(parser.parse_args(flags))
    assert defaults.program_size_limit_bytes is None
    observed = []

    def replay(args):
        observed.append(current_codegen_options())
        return 0

    monkeypatch.setattr(cli, '_native_verify_with_options', replay)
    selected = parser.parse_args([
        *flags, '--program-size-limit-bytes', '8192', '--execution-instruction-limit', '128000000',
    ])
    assert selected.func(selected) == 0
    assert observed == [CodegenOptions(program_size_limit_bytes=8192, execution_instruction_limit=128000000)]
    assert current_codegen_options().identity_record() == defaults.identity_record()


@pytest.mark.parametrize("changed", ("samples", "correctness", "factor", "missing"))
def test_malformed_profiles_cannot_select_a_variant(changed):
    value = profile()
    if changed == "samples":
        value["variants"][0]["device_cycles"] = [1]
    elif changed == "correctness":
        value["variants"][0]["correctness"] = "unknown"
    elif changed == "factor":
        value["variants"][0]["factor"] = True
    else:
        value["variants"].pop()
    with pytest.raises(ValueError):
        CodegenOptions(tuning_profile=value)


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_exact_float_unrolling_requires_a_boolean(value):
    with pytest.raises(ValueError, match="Boolean"):
        CodegenOptions(exact_float_unroll=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "100"])
def test_execution_instruction_limit_requires_positive_integer(value):
    with pytest.raises(ValueError, match="execution instruction limit"):
        CodegenOptions(execution_instruction_limit=value)


def test_execution_budget_is_separate_from_stored_program_budget():
    from argparse import Namespace

    original = current_codegen_options().identity_record()
    assert "execution_instruction_limit" not in original
    selected = options_from_args(Namespace(execution_instruction_limit=9_000_000, program_size_limit_bytes=8192))
    with codegen_options(selected):
        assert current_codegen_options().execution_instruction_limit == 9_000_000
        assert current_codegen_options().program_size_limit_bytes == 8192
        assert current_codegen_options().identity_record() != original
    assert current_codegen_options().identity_record() == original


def test_exact_float_unrolling_is_explicit_and_scoped():
    from argparse import Namespace

    from merlin.targetgen.cli import build_parser

    assert options_from_args(Namespace()).exact_float_unroll is False
    parser = build_parser()
    for command, flags in (
        (
            "native-compile",
            [
                "--engine",
                "merlin_native",
                "--support",
                "synthetic",
                "--snapshot",
                "snapshot",
                "--request",
                "request",
                "--target-source",
                "target",
                "--abi",
                "auto",
                "--mode",
                "diagnostic",
                "--out",
                "output",
            ],
        ),
        (
            "native-verify",
            [
                "--engine",
                "merlin_native",
                "--snapshot",
                "snapshot",
                "--request",
                "request",
                "--mode",
                "diagnostic",
                "--artifact",
                "artifact",
                "--out",
                "output",
            ],
        ),
        (
            "native-profile",
            [
                "--support",
                "synthetic",
                "--snapshot",
                "snapshot",
                "--target-source",
                "target",
                "--runner-command",
                "runner.json",
                "--request",
                "request",
                "--out",
                "output",
            ],
        ),
    ):
        assert options_from_args(parser.parse_args([command, *flags])).exact_float_unroll is False
        selected = options_from_args(parser.parse_args([command, *flags, "--exact-float-unroll"]))
        with codegen_options(selected):
            assert current_codegen_options().exact_float_unroll is True
        assert current_codegen_options().exact_float_unroll is False
        selected = options_from_args(parser.parse_args([command, *flags, "--execution-instruction-limit", "9000000"]))
        assert selected.execution_instruction_limit == 9_000_000


def test_application_verify_reads_budgets_from_artifact(tmp_path, monkeypatch):
    import json
    from argparse import Namespace
    from merlin.targetgen import cli

    artifact = tmp_path / 'application'
    artifact.mkdir()
    budgets = dict(program_size_limit_bytes=8192, execution_instruction_limit=128000000,
                   exact_float_unroll=True)
    (artifact / 'manifest.json').write_text(json.dumps(budgets))
    args = Namespace(application=True, artifact=str(artifact), snapshot=str(tmp_path / 'snapshot'), out=str(tmp_path / 'status.json'),
                     program_size_limit_bytes=None, execution_instruction_limit=None, exact_float_unroll=False)
    observed = []
    def verify(selected):
        observed.append({name: getattr(current_codegen_options(), name) for name in budgets})
        return 0
    monkeypatch.setattr(cli, '_native_verify_with_options', verify)
    assert cli._cmd_native_verify(args) == 0
    assert observed == [budgets]
    args.execution_instruction_limit = 256000000
    assert cli._cmd_native_verify(args) == 2
    assert len(observed) == 1
    assert 'selected budget differs' in json.loads((tmp_path / 'status.json').read_text())['reason']
    before = (artifact / 'manifest.json').read_bytes()
    args.out = str(artifact / 'manifest.json')
    assert cli._cmd_native_verify(args) == 2
    assert (artifact / 'manifest.json').read_bytes() == before
