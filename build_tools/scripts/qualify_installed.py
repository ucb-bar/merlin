"""Qualify committed core+experiments distributions outside the source checkout.

Usage: python build_tools/scripts/qualify_installed.py --ref HEAD --suite phase1
phase1 replays controller, CLI and RTL-feedback tests with the declared core[xdsl] extra.
phase0-inputs replays explicit recipe loading and declaration resolution, not hardware derivation.
reviewed-corpus joins derivation, explicit review, installed Phase-1 authoring,
formal receipts and the Phase-2 checkpoint lifecycle against the same candidate
bytes. Agent transport, oracle results, measurement and OS isolation are synthetic;
this is not hardware qualification.
phase2-policy replays workflow, receipt, transcript and frozen-input contracts, not native engines.
phase2-lifecycle replays checkpoint execution/resume with synthetic external execution and admission.
source-snapshot checks explicit frozen source ownership and stdlib-only verifier loading.
revision-journal checks static revision publication and retained artifact history.
host-policy checks logical source ownership and portable frozen policy identities.
portfolio-checkpoint checks global candidate/authoring admission and scoped semantic evidence.
portfolio-resume checks archived verifier selection and cold synthetic checkpoint admission.
portfolio-evaluation checks frozen analytical evaluator configuration and serialized execution.
global-inputs checks staged frozen experiment admission and retained input integrity.
revision-session checks live revision admission and immutable checkpoint publication.
portfolio-analysis checks host-admitted portfolio analysis and exact revision reuse.
static-analysis-import checks pinned cross-run reuse without inheriting dynamic evidence.
portfolio-probes checks admitted probe lifecycle and accounting with synthetic execution.
global-experiment checks concrete installed experiment assembly with explicit input roots.
mechanism-program checks frozen host assignments and exact portfolio analysis bindings.
sandbox-inputs checks explicit sandbox configuration with mocked process execution.
qualification-policy checks frozen execution resources and private alias handling.
agent-sandbox-inputs checks captured agent/tool execution configuration without native launches.
portfolio-sandbox checks compiler policy assembly and exact cached-policy rebinding.
portfolio-authoring checks bounded authoring and checkpoint continuation with synthetic transport.
portfolio-launch-inputs checks installed launch admission and analytical-provider inputs without native execution.
portfolio-launch checks explicit deployment, lease/resource refusal and frozen worker supervision with fake processes.
portfolio-providers checks explicit backend capability assembly without executing runtime tools.
portfolio-worker checks installed worker admission, analysis-only results and terminal evidence without execution.
portfolio-cli checks installed deployment/catalog admission and frozen-launch input pins without execution.
performance-providers checks installed experiment-owned probe providers and private access identities.
initializer-admission checks bounded source-to-command initializer evidence, not numerical equivalence.
functional-qualification checks synthetic qualification execution/resume outside the checkout.
tensor-inspection checks dense payload storage, reconstruction and compact xDSL views without native compilers.
target-fetch checks published branch/tag acquisition using local Git repositories, not compiler execution.
lowering-inspection replays stage evidence and compact-view contracts; native compiler tests may skip.
Requires installed Merlin path helpers and uv. Dependency downloads may be needed; no network
services, hardware or paid agents are launched. This is packaging/functional regression evidence,
not numerical qualification, hardware certification or security isolation.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import uuid
from pathlib import Path

SUITES = {
    "portfolio-cli": {
        "guarded_tests": True,
        "tests": ("test_portfolio_cli.py", "test_portfolio_catalog.py", "test_portfolio_launch.py"),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_cli", "merlin_experiments.portfolio_catalog"),
        "required_modules": ("xdsl",),
    },
    "portfolio-worker": {
        "guarded_tests": True,
        "tests": ("test_portfolio_worker.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_worker",),
        "required_modules": ("xdsl",),
    },
    "portfolio-providers": {
        "guarded_tests": True,
        "tests": ("test_portfolio_providers.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_providers",),
        "required_modules": ("xdsl",),
    },
    "portfolio-launch": {
        "guarded_tests": True,
        "tests": ("test_portfolio_launch.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_launch",),
        "required_modules": ("xdsl",),
    },
    "portfolio-launch-inputs": {
        "guarded_tests": True,
        "tests": ("test_portfolio_options.py", "test_fast_evaluation_installation.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase2.portfolio_options",
            "merlin_experiments.phase2.fast_evaluation_installation",
        ),
        "required_modules": ("xdsl",),
    },
    "rtl-protocol": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_rtl_check_providers.py", "test_rocc_provider_semantics.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.targetgen.rtl_checks",
            "merlin.targetgen.rtl_check_compiler",
            "merlin.targetgen.rtl_check_runner",
            "merlin.targetgen.circt_gate",
        ),
        "required_modules": ("xdsl",),
    },
    "phase1-provider-sources": {
        "tests": ("test_phase1_provider_sources.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase1.source_inputs",),
        "required_modules": ("xdsl",),
    },
    "target-fetch": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_oot_fetch.py",),
        "core_extras": (),
        "probe_modules": ("merlin.targetgen.oot_fetch",),
        "required_modules": (),
    },
    "measured-launch": {
        "tests": (
            "test_measured_claims_adapter.py",
            "test_checkpoint_child_environment.py",
            "test_chia_envelope.py",
            "test_chia_launch.py",
        ),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.measured_launch",
            "merlin_experiments.phase2.chia_envelope",
            "merlin_experiments.phase2.checkpoint_cli",
        ),
        "required_modules": ("xdsl",),
    },
    "chia-envelope": {
        "tests": ("test_chia_envelope.py", "test_chia_launch.py"),
        "core_extras": (),
        "probe_modules": (
            "merlin_experiments.phase2.chia_envelope",
            "merlin_experiments.phase2.chia_envelope_cli",
        ),
        "required_modules": (),
    },
    "formal-handoff": {
        "tests": ("test_phase1_formal_handoff.py",),
        "support_files": ("phase1_feedback_fixtures.py",),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase1.feedback.formal",
            "merlin_experiments.phase2.campaign",
            "merlin_experiments.phase2.global_inputs",
        ),
        "required_modules": ("xdsl",),
    },
    "broker-http": {
        "tests": ("test_phase2_broker.py", "test_broker_http_deadlines.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.broker",),
        "required_modules": (),
    },
    "codegen-declaration": {
        "include_experiments": False,
        "tests": ("test_preflight_codegen_declaration.py",),
        "core_extras": (),
        "probe_modules": ("merlin.targetgen.target_experiment",),
        "required_modules": (),
    },
    "codegen-smoke": {
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_codegen_smoke_fails_closed.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.targetgen.capsule_runner",),
        "required_modules": ("xdsl",),
    },
    "rocc-support": {
        "include_experiments": False,
        "tests_root": "merlin/tests/targetgen",
        "tests": ("test_rocc_provider_semantics.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.targetgen.rocc.decode", "merlin.targetgen.rocc.asm"),
        "required_modules": ("xdsl",),
    },
    "phase0-inputs": {
        "tests": ("test_phase0_explicit_inputs.py", "test_phase0_declarations.py", "test_phase0_comparison.py"),
        "core_extras": (),
        "probe_modules": (
            "merlin_experiments.phase0.profiles",
            "merlin_experiments.phase0.generation",
            "merlin_experiments.phase0.declarations",
        ),
        "required_modules": (),
    },
    "reviewed-corpus": {
        "tests": (
            "test_reviewed_corpus_phase1_handoff.py",
            "test_phase1_formal_handoff.py",
            "test_checkpoint_lifecycle.py",
            "test_corpus_release.py",
        ),
        "support_files": ("reviewed_corpus_fixtures.py", "phase1_feedback_fixtures.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase0",
            "merlin_experiments.corpus.release",
            "merlin_experiments.phase1.corpus_inputs",
            "merlin_experiments.phase2.functional_inputs",
        ),
        "required_modules": ("xdsl",),
    },
    "phase1": {
        "tests": ("test_phase1_controller.py", "test_phase1_cli.py", "test_phase1_rtlchecks.py"),
        "core_extras": ("xdsl",),
        "probe_modules": tuple(
            "merlin_experiments.phase1." + tail
            for tail in (
                "authoring",
                "audit",
                "runtime_environment",
                "controller",
                "task_staging",
                "workspace_transport",
                "feedback.certification",
                "feedback.rtlchecks",
            )
        ),
        "required_modules": ("xdsl",),
    },
    "phase2-policy": {
        "tests": (
            "test_phase2_workflow_policy.py",
            "test_phase2_broker_evidence.py",
            "test_phase2_transcript_audit.py",
            "test_phase2_functional_inputs.py",
        ),
        "core_extras": (),
        "probe_modules": tuple(
            "merlin_experiments.phase2." + tail
            for tail in (
                "broker",
                "broker_policy",
                "broker_evidence",
                "corpus_feedback",
                "whole_model",
                "transcript_audit",
                "functional_inputs",
            )
        ),
        "required_modules": (),
    },
    "phase2-lifecycle": {
        "tests": ("test_checkpoint_lifecycle.py",),
        "core_extras": (),
        "probe_modules": tuple(
            "merlin_experiments.phase2." + tail
            for tail in (
                "checkpoint_admission",
                "checkpoint_controller",
                "checkpoint_cli",
                "paired_cli",
                "holdout_corpus",
                "chia_launch",
            )
        ),
        "required_modules": (),
    },
    "source-snapshot": {
        "tests": ("test_source_snapshot_ownership.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.source_snapshot", "merlin_experiments.frozen_python"),
        "required_modules": (),
    },
    "revision-journal": {
        "tests": ("test_revision_journal.py", "test_revision_artifacts.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.revision_journal",),
        "required_modules": (),
    },
    "static-reuse": {
        "tests": ("test_static_identity.py", "test_static_reuse.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.static_identity",),
        "required_modules": (),
    },
    "host-policy": {
        "tests": ("test_host_policy.py", "test_static_identity.py"),
        "support_files": ("host_policy_fixtures.py",),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin_experiments.phase2.host_policy",
            "merlin_experiments.phase2.static_identity",
        ),
        "required_modules": ("xdsl",),
    },
    "portfolio-checkpoint": {
        "tests": ("test_portfolio_checkpoint.py", "test_portfolio_checkpoint_semantics.py"),
        "support_files": ("portfolio_checkpoint_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_checkpoint",),
        "required_modules": (),
    },
    "portfolio-resume": {
        "tests": ("test_portfolio_resume.py", "test_portfolio_resume_frozen.py"),
        "support_files": ("portfolio_checkpoint_fixtures.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin_experiments.phase2.portfolio_resume",),
        "required_modules": ("xdsl",),
    },
    "portfolio-evaluation": {
        "tests": ("test_portfolio_evaluation.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_evaluation",),
        "required_modules": (),
    },
    "global-inputs": {
        "tests": ("test_global_inputs.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.global_inputs",),
        "required_modules": (),
    },
    "revision-session": {
        "tests": ("test_revision_session.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.revision_session",),
        "required_modules": (),
    },
    "portfolio-analysis": {
        "tests": ("test_portfolio_analysis.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_analysis",),
        "required_modules": (),
    },
    "static-analysis-import": {
        "tests": ("test_static_analysis_import.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.static_analysis_import",),
        "required_modules": (),
    },
    "portfolio-probes": {
        "tests": ("test_portfolio_probes.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_probes",),
        "required_modules": (),
    },
    "global-experiment": {
        "tests": ("test_global_experiment.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.global_experiment",),
        "required_modules": (),
    },
    "static-cache": {
        "tests": ("test_static_cache.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.static_cache",),
        "required_modules": (),
    },
    "edit-authority": {
        "tests": ("test_edit_authority.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.edit_authority",),
        "required_modules": (),
    },
    "mechanism-program": {
        "tests": ("test_mechanism_program.py", "test_mechanism_rounds.py"),
        "core_extras": (),
        "probe_modules": (
            "merlin_experiments.phase2.mechanism_program",
            "merlin_experiments.phase2.mechanism_evidence",
            "merlin_experiments.phase2.mechanism_rounds",
        ),
        "required_modules": (),
    },
    "sandbox-inputs": {
        "tests": ("test_package_sandbox_inputs.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.campaign", "merlin.targetgen.sandbox.toolchain"),
        "required_modules": (),
    },
    "agent-sandbox-inputs": {
        "tests": ("test_agent_sandbox_inputs.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.agent_workspace", "merlin_experiments.phase2.broker"),
        "required_modules": (),
    },
    "portfolio-sandbox": {
        "tests": ("test_portfolio_sandbox.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_sandbox",),
        "required_modules": (),
    },
    "portfolio-authoring": {
        "tests": ("test_portfolio_authoring.py",),
        "support_files": ("portfolio_analysis_fixtures.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.portfolio_authoring",),
        "required_modules": (),
    },
    "performance-providers": {
        "tests": ("test_performance_providers.py",),
        "core_extras": ("xdsl",),
        "probe_modules": tuple(
            "merlin.perf." + name
            for name in (
                "isolated_probe_provider",
                "controlled_context_provider",
                "paired_context_provider",
                "host_region_qualifier",
                "host_physical_transition_qualifier",
                "lane_migration_qualifier",
                "source_contraction_preparation",
                "source_convolution_preparation",
                "source_program_pair",
                "source_initializer_elision",
                "source_program_pair_provider",
            )
        ),
        "required_modules": ("xdsl",),
    },
    "initializer-admission": {
        "tests_root": "merlin/tests/dse",
        "tests": ("test_source_initializer_elision.py",),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.perf.source_initializer_elision",),
        "required_modules": ("xdsl",),
    },
    "qualification-policy": {
        "tests": ("test_qualification_policy.py", "test_package_sandbox_inputs.py", "test_functional_qualification.py"),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.qualification_policy",),
        "required_modules": (),
    },
    "functional-qualification": {
        "tests_root": "merlin/tests/infra",
        "tests": ("test_functional_gsim_qualification.py",),
        "core_extras": (),
        "probe_modules": ("merlin_experiments.phase2.functional_qualification",),
        "required_modules": (),
    },
    "tensor-inspection": {
        "include_experiments": False,
        "tests_root": "merlin/tests/ir",
        "tests": ("test_ir_audit_tensors.py", "test_inspection_tensor_payloads.py"),
        "core_extras": ("xdsl",),
        "probe_modules": ("merlin.common.ir_audit", "merlin.xdsl_dialects.ir_inspection"),
        "required_modules": ("xdsl",),
    },
    "lowering-inspection": {
        "tests_root": "merlin/tests/ir",
        "tests": ("test_ir_audit.py", "test_ir_inspection.py"),
        "core_extras": ("xdsl",),
        "probe_modules": (
            "merlin.common.ir_audit",
            "merlin.compile_core",
            "merlin.llvmlower.ir_inspection",
        ),
        "required_modules": ("xdsl",),
    },
}
LIMITATIONS = [
    "Tool source is assumed static while Python initially imports it; observed hashes are rechecked at completion.",
    "Packaging and selected functional regressions only; no numerical/hardware certification.",
    "External venv and import-origin checks are not a security isolation guarantee.",
    "Dependency resolution uses configured indexes; exact resulting freeze is retained, not a lockfile replay.",
    "External venv is deliberately retained on success or failure; this command never deletes evidence.",
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_environment():
    return {k: v for k, v in os.environ.items() if not k.startswith(("MERLIN", "PYTHON", "AET_", "CHIA_"))}


def resolve_ref(root, ref):
    return subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", ref + "^{commit}"],
        cwd=root,
        text=True,
        timeout=30,
    ).strip()


def reserve_output(base, label):
    if (
        not label
        or len(label) > 120
        or not label.isascii()
        or not label[0].isalnum()
        or any(not (character.isalnum() or character in "_.-") for character in label)
    ):
        raise ValueError("output label must be one safe path component")
    base = Path(base)
    if any(p.is_symlink() for p in (base, *base.parents)):
        raise ValueError("output ancestors may not be symlinks")
    base.mkdir(parents=True, exist_ok=True)
    output = base / label
    output.mkdir()  # Existing even-empty directories and symlinks are refused.
    return output


class QualificationFailed(RuntimeError):
    pass


class Recorder:
    def __init__(self, output, report, timeout):
        self.output, self.report, self.timeout = output, report, timeout
        self.environment = clean_environment()

    def save(self):
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n")

    def run(self, label, argv, cwd, *, stdout=None):
        argv = list(map(str, argv))
        log = self.output / (label + ".log")
        record = {
            "step": label,
            "argv": argv,
            "cwd": str(cwd),
            "timeout_s": self.timeout,
            "log": str(log),
            "returncode": None,
            "status": "running",
        }
        self.report["commands"].append(record)
        self.save()
        start = time.monotonic()
        try:
            with log.open("wb") as errors:
                stream = Path(stdout).open("wb") if stdout is not None else errors
                try:
                    with subprocess.Popen(
                        argv, cwd=cwd, env=self.environment, stdout=stream, stderr=errors, start_new_session=True
                    ) as child:
                        try:
                            record["returncode"] = child.wait(timeout=self.timeout)
                            record["status"] = "passed" if child.returncode == 0 else "failed"
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            record.update(status="timeout", returncode=child.returncode)
                        except KeyboardInterrupt:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                            record.update(status="interrupted", returncode=child.returncode)
                            raise
                finally:
                    if stream is not errors:
                        stream.close()
        except OSError as exc:
            record.update(status="launch_failed", error=str(exc))
        finally:
            record["elapsed_s"] = time.monotonic() - start
            self.save()
        if record["status"] != "passed":
            raise QualificationFailed(f"{label}: {record['status']}; see {log}")


def projects(snapshot, extras, *, include_experiments=True):
    result = []
    roots = (Path("."), Path("packages/merlin-experiments")) if include_experiments else (Path("."),)
    for relative in roots:
        directory = snapshot / relative
        project = tomllib.loads((directory / "pyproject.toml").read_text())["project"]
        selected = extras if relative == Path(".") else ()
        for extra in selected:
            if extra not in project.get("optional-dependencies", {}):
                raise QualificationFailed(f"{project['name']} does not declare extra {extra}")
        result.append(
            {"name": project["name"], "version": project["version"], "path": str(directory), "extras": list(selected)}
        )
    return result


def only_artifact(directory, pattern):
    files = list(directory.glob(pattern))
    if len(files) != 1:
        raise QualificationFailed(f"expected one {pattern} in {directory}; got {files}")
    return files[0]


def qualify(root, output, commit, suite, timeout, *, requested_ref=None, invocation=None):
    own = Path(__file__).resolve()
    helper = own.with_name("installed_qualification_probe.py")
    tests_root = Path(SUITES[suite].get("tests_root", "packages/merlin-experiments/tests"))
    support_files = SUITES[suite].get("support_files", ())
    test_files = (*SUITES[suite]["tests"], *support_files)
    report = {
        "schema": "merlin.installed_qualification.v1",
        "ref": commit,
        "requested_ref": requested_ref,
        "invocation": invocation,
        "suite": suite,
        "status": "running",
        "commands": [],
        "limitations": LIMITATIONS,
        "tooling_revision": resolve_ref(root, "HEAD"),
        "tool_sources": {str(p.relative_to(root)): digest(p) for p in (own, helper)},
        "tool_source_note": "Actual executing bytes; hashes may include working-tree edits not in tooling_revision.",
        "selected_tests": list(SUITES[suite]["tests"]),
        "support_files": list(support_files),
        "tests_root": tests_root.as_posix(),
        "core_extras": list(SUITES[suite]["core_extras"]),
        "probe_modules": list(SUITES[suite]["probe_modules"]),
        "required_modules": list(SUITES[suite]["required_modules"]),
        "test_process_policy": (
            "deny_processes_and_listeners" if SUITES[suite].get("guarded_tests") else "suite_defined"
        ),
    }
    runner = Recorder(output, report, timeout)
    runner.save()
    try:
        copied_helper = output / "installed_qualification_probe.py"
        shutil.copyfile(helper, copied_helper)
        if digest(copied_helper) != report["tool_sources"][str(helper.relative_to(root))]:
            raise QualificationFailed("qualification helper changed during capture")
        runner.run(
            "resource-manifest",
            ["git", "show", commit + ":build_tools/package_resources.json"],
            root,
            stdout=output / "resources.json",
        )
        resources = json.loads((output / "resources.json").read_text())["files"]
        selected = [
            "src",
            "pyproject.toml",
            "setup.py",
            "README.md",
            "LICENSE",
            "MANIFEST.in",
            "build_tools/package_resources.json",
            "build_tools/extension_setup.py",
            "build_tools/scripts/check_distribution_layout.py",
            "packages/merlin-experiments/src",
            "packages/merlin-experiments/setup.py",
            "packages/merlin-experiments/pyproject.toml",
            *[(tests_root / n).as_posix() for n in test_files],
            *resources,
        ]
        archive = output / "source.tar"
        runner.run("source-archive", ["git", "archive", "--format=tar", commit, "--", *selected], root, stdout=archive)
        snapshot = output / "snapshot"
        snapshot.mkdir()
        with tarfile.open(archive) as source:
            source.extractall(snapshot, filter="data")
        for name in test_files:
            if not (snapshot / tests_root / name).is_file():
                raise QualificationFailed(f"selected committed test/support file is missing: {name}")
        report["source_archive_sha256"] = digest(archive)
        report["source_files"] = {str(p.relative_to(snapshot)): digest(p) for p in snapshot.rglob("*") if p.is_file()}
        report["projects"] = projects(
            snapshot,
            SUITES[suite]["core_extras"],
            include_experiments=SUITES[suite].get("include_experiments", True),
        )
        runner.save()
        wheels, install = [], []
        for project in report["projects"]:
            directory = output / "dist" / project["name"]
            directory.mkdir(parents=True)
            runner.run(
                project["name"] + "-sdist", ["uv", "build", "--sdist", "--out-dir", directory, project["path"]], output
            )
            sdist = only_artifact(directory, "*.tar.gz")
            runner.run(project["name"] + "-wheel", ["uv", "build", "--wheel", "--out-dir", directory, sdist], output)
            wheel = only_artifact(directory, "*.whl")
            wheels.append(wheel)
            install.append(str(wheel) + ("[" + ",".join(project["extras"]) + "]" if project["extras"] else ""))
            report.setdefault("artifacts", {}).update({str(p.relative_to(output)): digest(p) for p in (sdist, wheel)})
            runner.save()
        runner.run(
            "layout",
            [
                sys.executable,
                snapshot / "build_tools/scripts/check_distribution_layout.py",
                "--source-root",
                snapshot,
                *wheels,
            ],
            output,
        )
        external = Path(tempfile.mkdtemp(prefix="merlin-qualified-venv-"))
        report["external_venv"] = str(external)
        runner.save()
        if external.resolve().is_relative_to(root.resolve()):
            raise QualificationFailed("temporary venv must be outside the checkout; configure TMPDIR")
        runner.run("venv", ["uv", "venv", "--python", sys.executable, external], output)
        python = external / "bin/python"
        runner.run("install", ["uv", "pip", "install", "--python", python, *install], external)
        runner.run("dependencies", ["uv", "pip", "check", "--python", python], external)
        probe_args = [python, "-I", copied_helper]
        for module in SUITES[suite]["probe_modules"]:
            probe_args.extend(("--module", module))
        for module in SUITES[suite]["required_modules"]:
            probe_args.extend(("--require-module", module))
        runner.run("payload-probe", [*probe_args, *wheels], external)
        runner.run("pytest-install", ["uv", "pip", "install", "--python", python, "pytest"], external)
        runner.run("freeze", ["uv", "pip", "freeze", "--python", python], external, stdout=output / "dependencies.txt")
        report["dependency_freeze_sha256"] = digest(output / "dependencies.txt")
        tests = external / "qualification-tests"
        tests.mkdir()
        for name in test_files:
            shutil.copyfile(snapshot / tests_root / name, tests / name)
        shutil.copyfile(copied_helper, tests / "conftest.py")
        runner.run(
            "tests",
            [
                python,
                "-I",
                *([copied_helper, "--guarded-tests"] if SUITES[suite].get("guarded_tests") else ["-m", "pytest"]),
                "-q",
                "-c",
                "/dev/null",
                "-p",
                "no:cacheprovider",
                "--import-mode=importlib",
                # This venv is fresh and unique; pytest must not clean shared user temp roots.
                "--basetemp",
                external / "test-tmp",
                tests,
            ],
            external,
        )
        if any(digest(root / name) != expected for name, expected in report["tool_sources"].items()):
            raise QualificationFailed("qualification tooling changed during execution")
        report["status"] = "passed"
    except KeyboardInterrupt:
        report.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        runner.save()
    return report["status"] == "passed"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", required=True, help="Git commit/ref to archive; never the dirty worktree")
    parser.add_argument("--suite", choices=SUITES, default="phase1")
    parser.add_argument("--label", help="New output directory name beneath build_dir()/python/qualified-installs")
    parser.add_argument("--timeout", type=int, default=300, help="Maximum seconds per child command")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    from merlin.common.paths import build_dir, repo_root

    root = repo_root()
    commit = resolve_ref(root, args.ref)
    label = args.label or (
        datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + commit[:12] + "-" + uuid.uuid4().hex[:8]
    )
    output = reserve_output(build_dir() / "python/qualified-installs", label)
    print(f"Qualification evidence: {output}", flush=True)
    return (
        0
        if qualify(
            root,
            output,
            commit,
            args.suite,
            args.timeout,
            requested_ref=args.ref,
            invocation=[sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)],
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
