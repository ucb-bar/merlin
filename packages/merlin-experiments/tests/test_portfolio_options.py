"""Pure portfolio CLI admission: no controller, processes or listeners."""

from __future__ import annotations

import argparse
import hashlib
import socket
import subprocess
from pathlib import Path

import pytest
from merlin_experiments.phase2 import portfolio_options as OPTIONS


@pytest.fixture(autouse=True)
def forbid_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CLI admission must not launch processes or listeners")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "bind", forbidden)


def arguments(tmp_path: Path) -> list[str]:
    return [
        "--campaign-config",
        str(tmp_path / "campaign.json"),
        "--candidate",
        str(tmp_path / "candidate"),
        "--output",
        str(tmp_path / "output"),
    ]


def test_defaults_are_inspectable_without_admitting_files(tmp_path):
    invocation = OPTIONS.parse_invocation(arguments(tmp_path))
    assert invocation.args.portfolio_analysis_workers == 4
    assert invocation.args.baseline_emission_cache == tmp_path / "_global_phase2_baseline_emission_cache_v1"
    assert invocation.total_authoring == 600
    assert invocation.fast_evaluation_configured is False
    assert invocation.resource_policy.minimum_memory_available_bytes == 16 * 1024**3
    assert invocation.resource_policy.maximum_swap_used_bytes == 2 * 1024**3
    assert not list(tmp_path.iterdir())


def test_help_and_hidden_worker_option():
    parser = OPTIONS.build_parser(description="explicit launch description")
    help_text = parser.format_help()
    assert "explicit launch description" in help_text
    assert "--functional-gate" in help_text
    assert "--source-worker" not in help_text


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--portfolio-analysis-workers", "0"], "host resource limits"),
        (["--resource-sample-seconds", "nan"], "host resource limits"),
        (["--resource-sample-seconds", "0.1"], "host resource limits"),
        (["--min-memory-available-gib", "-1"], "host resource limits"),
        (["--historical-reference", "missing"], "historical-reference requires"),
        (["--static-analysis-seed-checkpoint", "missing"], "static-analysis seed requires"),
        (["--mechanism-catalog", "missing"], "mechanism-catalog requires its exact"),
        (["--mechanism-work-order", "missing"], "mechanism-work-order requires its exact"),
        (["--optimization-baseline", "missing"], "optimization-baseline requires"),
        (["--external-objective", "missing"], "external-objective requires"),
        (["--portfolio-external-objective", "missing"], "ordered exact SHA-256"),
        (["--portfolio-capsule", "a", "--portfolio-capsule", "a"], "must be distinct"),
        (["--max-rounds", "0"], "authoring bounds"),
        (["--iteration-seconds", "0"], "host-only full-graph static-analysis ceiling"),
        (["--validation-only"], "requires exactly one comparison-candidate"),
        (["--comparison-candidate", "missing"], "requires exactly one comparison-candidate"),
        (["--compare-controlled-context"], "requires validation-only"),
        (["--semantic-only"], "semantic-only requires"),
        (["--probe-interface", "missing"], "must be supplied together"),
        (["--probe-profile", "occupancy"], "requires its interface"),
        (["--analysis-only", "--max-rounds", "2"], "analysis-only excludes"),
        (["--fast-evaluation-calibration", "missing"], "fast-evaluation"),
    ],
)
def test_ordered_refusals(tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit) as exc:
        OPTIONS.parse_invocation(arguments(tmp_path) + extra)
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_explicit_budget_and_validation_mode(tmp_path):
    invocation = OPTIONS.parse_invocation(
        arguments(tmp_path)
        + [
            "--max-rounds",
            "3",
            "--round-seconds",
            "120",
            "--total-authoring-seconds",
            "300",
        ]
    )
    assert invocation.total_authoring == 300
    validation = OPTIONS.parse_invocation(
        arguments(tmp_path)
        + [
            "--validation-only",
            "--comparison-candidate",
            str(tmp_path / "comparison"),
            "--semantic-only",
        ]
    )
    assert validation.args.semantic_only


def test_static_seed_hash_is_admitted_without_decoding(tmp_path, capsys):
    seed = tmp_path / "checkpoint.json"
    seed.write_bytes(b"explicit raw checkpoint admission")
    digest = hashlib.sha256(seed.read_bytes()).hexdigest()
    extra = ["--static-analysis-seed-checkpoint", str(seed), "--static-analysis-seed-sha256", digest]
    assert OPTIONS.parse_invocation(arguments(tmp_path) + extra).args.static_analysis_seed_checkpoint == seed
    seed.write_bytes(b"changed")
    with pytest.raises(SystemExit):
        OPTIONS.parse_invocation(arguments(tmp_path) + extra)
    assert "differs from its pin" in capsys.readouterr().err


def test_catalog_requires_readonly_exact_absolute_file(tmp_path, capsys):
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes(b"{}")
    digest = hashlib.sha256(catalog.read_bytes()).hexdigest()
    extra = [
        "--edit-contract",
        str(tmp_path / "edit.json"),
        "--mechanism-catalog",
        str(catalog),
        "--mechanism-catalog-sha256",
        digest,
    ]
    with pytest.raises(SystemExit):
        OPTIONS.parse_invocation(arguments(tmp_path) + extra)
    assert "exact immutable absolute file" in capsys.readouterr().err
    catalog.chmod(0o444)
    assert OPTIONS.parse_invocation(arguments(tmp_path) + extra).args.mechanism_catalog == catalog


def test_fast_admission_precedes_resource_limits(tmp_path, capsys):
    with pytest.raises(SystemExit):
        OPTIONS.parse_invocation(
            arguments(tmp_path)
            + [
                "--fast-evaluation-calibration",
                "missing",
                "--portfolio-analysis-workers",
                "0",
            ]
        )
    message = capsys.readouterr().err
    assert "fast-evaluation" in message
    assert "host resource limits" not in message


@pytest.mark.parametrize("objective_flag", ["--objective-capsule", "--external-objective"])
def test_worker_maximal_parser_roundtrip(tmp_path, monkeypatch, objective_flag):
    # Deliberately use parser-only admission: these fake pins authorize no execution.
    monkeypatch.chdir(tmp_path)
    parser = OPTIONS.build_parser()
    argv = [objective_flag, "objective"]
    for action in parser._actions:
        if action.dest in {"help", "objective_capsule", "external_objective"}:
            continue
        flag = action.option_strings[0]
        if isinstance(action, argparse._StoreTrueAction):
            if action.dest != "source_worker":
                argv.append(flag)
        elif isinstance(action, argparse._AppendAction):
            if action.dest == "fast_evaluation_held_out_corpus":
                argv.extend([flag, "member-b", "corpus-b", "digest-b"])
                argv.extend([flag, "member-a", "corpus-a", "digest-a"])
            else:
                argv.extend([flag, f"{action.dest}-b", flag, f"{action.dest}-a"])
        else:
            value = (
                str(action.choices[-1])
                if action.choices
                else "7"
                if action.type is int
                else "3.5"
                if action.type is float
                else f"selected-{action.dest}"
            )
            argv.extend([flag, value])
    parent = parser.parse_args(argv)
    before = vars(parent).copy()
    forwarded = OPTIONS.worker_arguments(parent)
    worker = parser.parse_args(forwarded)
    expected = before.copy()
    for action in parser._actions:
        if action.type is Path and expected[action.dest] is not None:
            value = expected[action.dest]
            expected[action.dest] = [item.resolve() for item in value] if isinstance(value, list) else value.resolve()
    expected["source_worker"] = True
    expected["fast_evaluation_held_out_corpus"] = [
        ["member-b", str(tmp_path / "corpus-b"), "digest-b"],
        ["member-a", str(tmp_path / "corpus-a"), "digest-a"],
    ]
    assert vars(worker) == expected
    assert vars(parent) == before
    assert worker.portfolio_capsule == ["portfolio_capsule-b", "portfolio_capsule-a"]
    assert isinstance(forwarded, tuple)


def test_worker_minimal_admitted_invocation_parity(tmp_path):
    parent = OPTIONS.parse_invocation(arguments(tmp_path))
    worker = OPTIONS.parse_invocation(OPTIONS.worker_arguments(parent.args))
    expected = vars(parent.args) | {"source_worker": True}
    assert vars(worker.args) == expected
    assert worker.resource_policy == parent.resource_policy
    assert worker.total_authoring == parent.total_authoring
    assert worker.fast_evaluation_configured == parent.fast_evaluation_configured


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_worker_refuses_namespace_shape_drift(tmp_path, mutation):
    args = OPTIONS.build_parser().parse_args(arguments(tmp_path))
    if mutation == "missing":
        del args.candidate
    else:
        args.unregistered_option = "not an admitted CLI field"
    with pytest.raises(ValueError):
        OPTIONS.worker_arguments(args)


def test_worker_refuses_unsupported_parser_action(tmp_path, monkeypatch):
    parser = OPTIONS.build_parser()
    parser.add_argument("--unserializable", action="count", default=0)
    args = parser.parse_args(arguments(tmp_path) + ["--unserializable"])
    monkeypatch.setattr(OPTIONS, "build_parser", lambda: parser)
    with pytest.raises(ValueError, match="unsupported portfolio worker argument action"):
        OPTIONS.worker_arguments(args)
