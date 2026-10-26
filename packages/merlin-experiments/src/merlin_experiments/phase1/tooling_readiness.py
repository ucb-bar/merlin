"""Pre-spend proof that the selected assisted bundle's authoring tools work.

The probe workspace is a sibling of the candidate workspace. Both resolve the
same verified, immutable ``bundle_inputs`` tree, while the probe never changes a
candidate's staged task or resumed submission.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import yaml

from merlin.common.paths import module_source_path, python_source_dir
from merlin.targetgen import tool_registry as TR
from merlin.targetgen.sandbox import bwrap as BW


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _grant_checks(te, bundle: dict, tools: tuple[str, ...]) -> list[dict]:
    """Match the registry's promised files to this selected manifest, not a public sibling."""
    allowed = {
        row["path"] for row in bundle.get("allowed", ()) if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    # A registry path the selected bundle records as a denial (an author-withheld module, e.g. under
    # substrate v15) is withheld on purpose: the tool runs without it, so it is not a missing grant.
    denied = {
        row["path"] for row in bundle.get("denied", ()) if isinstance(row, dict) and isinstance(row.get("path"), str)
    }
    # A release may grant the source tree selected when it was frozen while the
    # controller now runs from an installed wheel. The common/ grant is the
    # required, target-independent anchor for every assisted tool set. Resolve
    # one selected package root from that exact manifest grant, then require
    # each remaining registry path under the same root.
    anchors = {
        Path(path.rstrip("/")).parent for path in allowed if path.startswith("/") and path.endswith("/merlin/common/")
    }
    if len(anchors) > 1:
        raise RuntimeError("selected bundle has multiple Merlin Python source roots")
    sources = {Path(python_source_dir()) / "merlin", *anchors}
    checks = []
    for name in tools:
        spec = TR.spec(name)
        for logical in (*spec.bundle_paths, *(str(getattr(te, attr)) for attr in spec.derived_paths)):
            candidates = {logical}
            prefix = "merlin/python/merlin/"
            selected_logical = logical
            if (
                bundle.get("compiler_tool_profile") is not None
                and name == "verify_seam"
                and logical == prefix + "verify/"
            ):
                # Fresh profiles retain the historical client as exact leaves;
                # a directory grant would also expose optional proof authoring.
                selected_logical = prefix + "verify/cli.py"
                candidates = {selected_logical}
            if selected_logical.startswith(prefix):
                for source in sources:
                    selected = str((source / selected_logical[len(prefix) :]).resolve())
                    candidates.add(selected + ("/" if selected_logical.endswith("/") else ""))
            if name == "rtl_facts" and bundle.get("selected_rtl_facts_file"):
                selected = Path(bundle["selected_rtl_facts_file"])
                candidates.add(str(selected.parent) + "/")
            if not allowed & candidates and denied & candidates:
                checks.append(
                    _check(
                        f"grant:{name}:{logical}", True, f"withheld by recorded denial={sorted(denied & candidates)}"
                    )
                )
                continue
            checks.append(
                _check(
                    f"grant:{name}:{logical}", bool(allowed & candidates), f"selected={sorted(allowed & candidates)}"
                )
            )
    return checks


def _frozen_rtl_checks(te, ws: Path, bundle: dict, public_root: Path, *, repo: Path) -> list[dict]:
    """Compile a check from the run's frozen facts and public capsule bytes."""
    from merlin.targetgen import rtl_check_runner as RUN
    from merlin.targetgen.sandbox import toolchain as TC

    selected = bundle.get("selected_rtl_facts_file")
    source = Path(selected) if selected is not None else BW.resolve_grant(str(te.rtl_facts_pin), repo=repo)
    [frozen] = BW.snapshot_input_paths(ws, bundle, [source], repo=repo)
    facts_file = frozen / "facts.json" if frozen.is_dir() else frozen
    facts = json.loads(facts_file.read_text(encoding="utf-8"))
    body = facts.get("facts") or {}
    if not isinstance(body, dict) or not body:
        raise RuntimeError("frozen RTL facts are empty")
    from merlin.targetgen import rtl_check_compiler as CC
    from merlin.targetgen.rtl import facts as FACTS

    capsules = sorted(public_root.rglob("capsule.yaml"))
    if not capsules:
        raise RuntimeError("frozen public corpus has no capsule for RTL check compilation")
    selected_capsule = None
    compiled = {}
    with FACTS.observed_facts(te.target, facts, facts_file):
        for capsule in capsules:
            candidate = CC.compile_checks(facts, yaml.safe_load(capsule.read_text(encoding="utf-8")), te.target)
            if candidate.get("kernel") or candidate.get("trace"):
                selected_capsule, compiled = capsule, candidate
                break
    usable = selected_capsule is not None
    filecheck = RUN.find_filecheck(
        (
            Path(TC.ToolchainPaths.from_checkout().llvm) / "bin/FileCheck",
            *(Path(directory) / "FileCheck" for directory in TC._sim(te).path_dirs),
        )
    )
    return [
        _check("frozen_rtl_facts", True, str(facts_file)),
        _check("FileCheck", bool(filecheck), str(filecheck)),
        _check(
            "rtl_checks_compile",
            usable,
            f"capsule={selected_capsule.parent.name if selected_capsule else 'none'}; "
            f"inspected={len(capsules)}; endpoint={compiled.get('endpoint_status')}",
        ),
    ]


def _asm_probe(context) -> str:
    from merlin.targetgen import capsule_runner as CR
    from merlin.targetgen.isa_model import isa_model_for_target

    from .brokers.isa_tools import is_rocc_endpoint

    if is_rocc_endpoint(CR._endpoint_of(context.target)[0]):
        return "FENCE"
    model = isa_model_for_target(context.target)
    if model.resolve("FENCE") is not None or "FENCE" in model.opcode_table:
        return "FENCE"
    names = sorted(model.opcode_table)
    if not names:
        raise RuntimeError("derived ISA model has no opcode to assemble")
    return names[0]


def _selected_library_probe() -> str:
    """Import every exact public leaf inside the selected execution namespace."""
    from merlin.targetgen.compiler_library import active_compiler_library
    from merlin_experiments.phase2.compiler_libraries import guest_root

    selection = active_compiler_library()
    if selection is None:
        return ""
    before = selection.record()
    root = guest_root(selection)
    members = {member.module: member for member in selection.contract.members}
    rows = tuple(
        (name, str(root / members[name].path), members[name].sha256) for name in selection.contract.public_modules
    )
    lines = [
        "import hashlib, importlib",
        "from pathlib import Path",
        f"_library_members = {rows!r}",
        "for _name, _source, _digest in _library_members:",
        "    _module = importlib.import_module(_name)",
        "    _file = getattr(_module, '__file__', None)",
        "    _origin = getattr(getattr(_module, '__spec__', None), 'origin', None)",
        "    assert isinstance(_file, str) and Path(_file).resolve() == Path(_source), (_name, _file)",
        "    assert isinstance(_origin, str) and Path(_origin).resolve() == Path(_source), (_name, _origin)",
        "    assert hashlib.sha256(Path(_source).read_bytes()).hexdigest() == _digest, _name",
        f"print('SELECTED_PUBLIC_LIBRARY_IMPORTS_OK', {selection.contract.sha256!r}, len(_library_members))",
    ]
    if selection.record() != before:
        raise ValueError("selected public library changed during probe construction")
    return "\n".join(lines)


def _cca_withheld(bundle: dict) -> bool:
    """Whether the selected bundle records the CCA contract module as a denial (author-withheld)."""
    for row in bundle.get("denied", ()):
        parts = Path(row.get("path", "")).parts if isinstance(row, dict) else ()
        if parts[-3:] == ("merlin", "kernels", "cca_contract.py"):
            return True
    return False


def _sandbox_probe(
    target: str,
    tools: tuple[str, ...],
    mnemonic: str,
    *,
    common_substrate: bool = False,
    common_profile_sha256: str | None = None,
    common_modules: tuple[str, ...] | None = None,
    withheld_cca: bool = False,
) -> str:
    from .feedback import snapshot_execution

    external = snapshot_execution.selected()
    has = set(tools)
    lines = ["import json, subprocess, sys"]
    if not external:
        lines += _selected_library_probe().splitlines()
    if common_substrate and not external:
        from merlin.targetgen.tool_profile import COMMON_PUBLIC_MODULES

        lines += ["import importlib"]
        modules = COMMON_PUBLIC_MODULES if common_modules is None else common_modules
        lines += [f"importlib.import_module({module!r})" for module in modules]
        source = """builtin.module {
  func.func @readiness(%x: i8) -> i8 {
    %zero = arith.constant 0 : i8
    %result = arith.addi %x, %zero : i8
    func.return %result : i8
  }
}"""
        lines += [
            "from pathlib import Path",
            "import hashlib",
            "from merlin.transforms.client import apply_transform",
            f"source_ir = {source!r}",
            "result = apply_transform(Path.cwd(), source_ir,",
            "    {'normalize_shapes': False, 'simplify_integers': True, 'reuse_buffers': False}, wait_seconds=120)",
            "assert result.get('status') == 'verified', result",
            "assert result.get('source_sha256') == hashlib.sha256(source_ir.encode()).hexdigest(), result",
            "assert result.get('transformed_ir') and result['transformed_ir'] != source_ir, result",
            "assert result.get('receipts'), result",
            "assert all(r.get('status') == 'verified' and r.get('method') == 'lean_kernel'",
            "    and r.get('query_sha256') and r.get('toolchain_sha256') for r in result['receipts']), result",
        ]
        if common_profile_sha256 is not None:
            lines += [f"assert result.get('compiler_tool_profile_sha256') == {common_profile_sha256!r}, result"]
    if "isa_dialect_generation" in has:
        lines += [
            "from merlin.targetgen.isa_spec import resolve_isa",
            "from merlin.targetgen.isa_dialect_gen import generate_package",
            "assert callable(resolve_isa) and callable(generate_package)",
        ]
    if "lean_proof_authoring" in has:
        lines += [
            "from merlin.verify import lean_cli, lean_export, lean_checker",
            "assert callable(lean_cli.main)",
        ]
    if "xdsl_kit" in has:
        lines += [
            "from merlin.targetgen.evidence.store import Evidence",
            "from merlin.targetgen import synthesize as S, generate as G",
            "from merlin.targetgen.generate import target_repo",
            "from merlin.targetgen.contract.interface_emit import emit_interface_mlir",
            "from merlin.runtime.commandbuffer import pool_params",
            "from merlin.runtime.tensor import pool_out_dims",
            f"e = Evidence(target={target!r}, sources={{}})",
            f"contract = S.synthesize_target_contract(e, {target!r})",
            "plan = S.synthesize_dialect_plan(e, contract)",
            "assert isinstance(plan, dict) and plan",
            f"assert target_repo.generate_skeleton({target!r})",
            "assert G.xdsl.generate(plan)",
            "assert G.mlir_scaffold.generate(plan)",
            "assert callable(emit_interface_mlir)",
            "pooled = {'pool_in_dims':[4,4], 'pool_size':[2,2],",
            "          'pool_stride':[2,2], 'pool_padding':[0,0,0,0]}",
            "assert pool_params(pooled, op='readiness')['pool_in_dims'] == (4,4)",
            "assert pool_out_dims(4,4,[2,2],[2,2],[0,0,0,0]) == (2,2)",
        ]
    if "cca_spine" in has and withheld_cca:
        # The CCA modules are withheld from the author (recorded bundle denials): the spine grant is
        # rtl_backend only, and bijection / escalation ladder go through the cca_tools broker probe.
        lines += [
            "from merlin.targetgen import rtl_backend as RB",
            f"profile = RB.target_profile({target!r})",
            "assert not profile.discovered_nothing",
            "assert not profile.has_mesh or RB.derived_levers(profile)",
        ]
    elif "cca_spine" in has:
        lines += [
            "from merlin.targetgen import rtl_backend as RB",
            "from merlin.kernels import cca_contract, action_catalog",
            f"profile = RB.target_profile({target!r})",
            "assert not profile.discovered_nothing",
            "axes = sorted(cca_contract.leverable_axes(profile.target))",
            "report = cca_contract.check_bijection(profile.target)",
            "assert not (report.orphan_fields or report.orphan_routes or report.ladder_errors), report",
            "assert report.unexpected().clean, report",
            "assert not profile.has_mesh or (RB.derived_levers(profile) and axes)",
            "assert all(action_catalog.escalation_ladder(axis, profile.target) for axis in axes)",
        ]
    if "rtl_generators" in has:
        lines += [
            "import shutil",
            "assert shutil.which('FileCheck'), 'FileCheck is absent inside the selected sandbox'",
            "from merlin.targetgen.rtl import facts, gen_numeric_facts, gen_isa_module, gen_rtl_digest",
            f"doc = facts.load_facts({target!r})",
            "assert isinstance(doc.get('facts'), dict) and doc['facts']",
            "assert gen_numeric_facts.generate(doc).strip()",
            "if any(x.get('name') == 'funct_decode_table' for x in doc['facts'].get('interfaces', [])):",
            "    try: assert gen_isa_module.generate(doc).strip()",
            "    except gen_isa_module.NotARoccTarget: pass",
            "    assert gen_rtl_digest.generate(doc).strip()",
        ]
    if "isa_tools" in has:
        lines += [
            f"r = subprocess.run([sys.executable, 'isa_tools.py', 'asm', {mnemonic!r}],",
            "                   capture_output=True, text=True, timeout=30)",
            "assert r.returncode == 0, (r.stdout, r.stderr)",
            "isa = json.loads(r.stdout)",
            "assert isa.get('n', 0) >= 1 and (isa.get('words') or str(isa.get('mlir', '')).strip()), isa",
        ]
    if "cca_tools" in has:
        lines += [
            "from cca_contract import check_bijection",
            "from action_catalog import escalation_ladder",
            f"bijection = check_bijection({target!r})",
            "assert not bijection.get('error') and (bijection.get('unexpected') or {}).get('clean') is True, bijection",
            "axis = axes[0] if 'axes' in globals() and axes else 'spatial.dataflow'",
            f"ladder = escalation_ladder(axis, {target!r})",
            "assert not ladder.get('error'), ladder",
            "assert 'profile' not in globals() or not profile.has_mesh or ladder.get('n', 0) >= 1, ladder",
        ]
    lines.append("print('AUTHORING_AND_BROKER_ROUNDTRIPS_OK')")
    return "\n".join(lines)


@contextmanager
def public_probe_session(
    context, ws: Path, bundle: dict, tools: tuple[str, ...], *, facts_workspace: Path | None = None
):
    """Run the existing selected public brokers for a tool probe, without oracle jobs."""
    from merlin_experiments.frozen_python import inherited_python_command

    from .feedback.lifecycle import stage_client

    processes, channels = [], []
    private = ExitStack()
    selected_facts = BW.frozen_selected_rtl_facts(
        ws if facts_workspace is None else facts_workspace, bundle, repo=context.repo
    )
    specs = TR.brokers_for(tools)
    try:
        for spec in specs:
            channel = ws / spec.channel
            channels.append(channel)
            channel.mkdir(exist_ok=True)
            if (channel / "STOP").exists():
                raise ValueError("public tool probe refuses a stopped broker channel")
            for module, staged_as in spec.shims:
                stage_client(ws, module_source_path(module), staged_as)
            argv = [sys.executable, "-m", spec.module, "--ws", str(ws)]
            if spec.module.endswith(".isa_tools"):
                argv += ["--descriptor", str(context.descriptor), "--repo", str(context.repo)]
            if selected_facts is not None and spec.module.endswith((".isa_tools", ".cca")):
                argv += ["--rtl-facts", str(selected_facts)]
            with (channel / spec.log).open("w", encoding="utf-8") as log:
                processes.append(subprocess.Popen(inherited_python_command(argv), stdout=log, stderr=subprocess.STDOUT))
        from .feedback import snapshot_execution

        if bundle.get("compiler_tool_profile") is not None and not snapshot_execution.selected():
            from .context import context_argv

            authority = Path(
                private.enter_context(tempfile.TemporaryDirectory(prefix="tooling-authority-", dir=ws.parent))
            )
            authority.chmod(0o700)
            manifest = authority / "input_bundle_manifest.yaml"
            manifest.write_text(yaml.safe_dump(bundle, sort_keys=False), encoding="utf-8")
            manifest.chmod(0o600)
            channel = ws / ".qa_channel"
            channel.mkdir(exist_ok=True)
            channels.append(channel)
            if (channel / "STOP").exists():
                raise ValueError("common tool probe refuses a stopped original broker channel")
            argv = [
                sys.executable,
                "-m",
                "merlin_experiments.phase1.brokers.selfcheck",
                "--ws",
                str(ws),
                *context_argv(context),
                "--compiler-bundle-manifest",
                str(manifest),
            ]
            if selected_facts is not None:
                argv += ["--rtl-facts", str(selected_facts)]
            with (channel / "broker.log").open("w", encoding="utf-8") as log:
                processes.append(subprocess.Popen(inherited_python_command(argv), stdout=log, stderr=subprocess.STDOUT))
        yield _sandbox_probe(
            context.target,
            tools,
            _asm_probe(context) if "isa_tools" in tools else "",
            common_substrate=bundle.get("compiler_tool_profile") is not None,
            common_profile_sha256=(bundle.get("compiler_tool_profile") or {}).get("sha256"),
            common_modules=tuple((bundle.get("compiler_tool_profile") or {}).get("common_public_modules", ())) or None,
            withheld_cca=_cca_withheld(bundle),
        )
    finally:
        for channel in channels:
            if channel.is_dir():
                (channel / "STOP").write_text("stop", encoding="utf-8")
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        private.close()


def _live_probe(context, te, ws: Path, bundle: dict, tools: tuple[str, ...]) -> dict:
    from merlin.targetgen.sandbox import toolchain as TC

    with tempfile.TemporaryDirectory(prefix="tooling-readiness-", dir=ws.parent) as directory:
        probe_ws = Path(directory)
        with public_probe_session(context, probe_ws, bundle, tools, facts_workspace=ws) as probe:
            argv = BW.full_argv(te, probe_ws, bundle)
            command = TC.sandbox_env(te, probe_ws, bundle=bundle) + " python3 -c " + shlex.quote(probe)
            result = subprocess.run(
                [*argv, "bash", "-c", command],
                cwd=probe_ws,
                capture_output=True,
                text=True,
                timeout=180,
            )
            ok = result.returncode == 0 and "AUTHORING_AND_BROKER_ROUNDTRIPS_OK" in result.stdout
            output = (result.stdout + "\n" + result.stderr).strip()[-1600:]
            return _check("selected_sandbox_authoring", ok, f"rc={result.returncode}; output={output}")


def assess(
    context,
    te,
    ws: Path,
    bundle: dict,
    tools: tuple[str, ...],
    public_root: Path,
    *,
    without_tools: tuple[str, ...] = (),
) -> dict:
    """Return a fail-closed, selected-bundle readiness verdict; no paid agent starts."""
    selected_arm = bundle.get("arm")
    selected_assisted = set(tools) - {"cpp_oot_generators"}
    profile = bundle.get("compiler_tool_profile")
    if profile is None and (
        (selected_arm in {"raw_baseline", "cpp_merlininfra"} and not selected_assisted)
        or (selected_arm is None and not str(bundle.get("bundle_id", "")).startswith("merlin_assisted"))
    ):
        return {"status": "skipped", "reason": f"{selected_arm} has no assisted authoring contract", "checks": []}
    checks = []
    try:
        if selected_arm not in TR.ARM_TOOLS:
            raise RuntimeError(f"unrecognized assisted arm {selected_arm!r}")
        selected = tuple(dict.fromkeys(tools))
        if not selected and profile is None:
            raise RuntimeError("assisted bundle resolved no tools")
        unknown = set(selected) - set(TR.known_tools())
        if unknown:
            raise RuntimeError(f"unknown selected tools: {sorted(unknown)}")
        for name in without_tools:
            if not TR.spec(name).ablatable:
                raise RuntimeError(f"selected tool {name!r} cannot be ablated alone")
        if selected_arm == "merlin_rtlchecks":
            arm3 = set(TR.ARM_TOOLS["merlin_assisted"])
            arm4 = set(TR.ARM_TOOLS["merlin_rtlchecks"])
            checks.append(_check("arm4_grant_superset", arm3 <= arm4, f"added={sorted(arm4 - arm3)}"))
        required = set(TR.ARM_TOOLS[selected_arm]) - set(without_tools)
        missing = required - set(selected)
        checks.append(
            _check(
                "selected_tool_set",
                not missing,
                f"selected={sorted(selected)}; required={sorted(required)}; missing={sorted(missing)}",
            )
        )
        checks += _grant_checks(te, bundle, selected)
        from .feedback import snapshot_execution

        if profile is not None and not snapshot_execution.selected():
            from merlin.targetgen.tool_profile import verify_tool_profile

            verified_profile = verify_tool_profile(bundle, tools=selected)
            toolchain = verified_profile.get("lean_toolchain") or {}
            checks.append(
                _check(
                    "automatic_formal_toolchain_selection",
                    toolchain.get("selection") == "explicit_manifest"
                    and toolchain.get("readiness") == "declared_unprobed",
                    "explicit frozen toolchain declaration; successful proof checking remains required "
                    "per covered transformation",
                )
            )
            checks.append(
                _check(
                    "common_host_index_width_selection",
                    type(verified_profile.get("host_index_bits")) is int,
                    "reviewed host ABI index width; missing width refuses common transformation lowering",
                )
            )
        if "rtl_facts" in selected:
            checks += _frozen_rtl_checks(te, ws, bundle, public_root, repo=context.repo)
        if all(check["ok"] for check in checks):
            checks.append(_live_probe(context, te, ws, bundle, selected))
    except Exception as exc:  # noqa: BLE001 -- admission must record every failure
        checks.append(_check("readiness_exception", False, f"{type(exc).__name__}: {str(exc)[:1200]}"))
    return {"status": "ready" if all(check["ok"] for check in checks) else "no_go", "checks": checks}


def run(
    context,
    te,
    ws: Path,
    bundle: dict,
    tools: tuple[str, ...],
    public_root: Path,
    run_dir: Path,
    snapshot: dict | None,
    manifest_sha256: str,
    *,
    without_tools: tuple[str, ...] = (),
) -> dict:
    """Persist a private receipt bound to the selected manifest and frozen snapshot."""
    if snapshot is None:
        assisted = bundle.get("arm") in {"merlin_assisted", "merlin_rtlchecks", "merlin_eqsat", "merlin_verify"}
        assisted = (
            assisted or bool(set(tools) - {"cpp_oot_generators"}) or bundle.get("compiler_tool_profile") is not None
        )
        result = {
            "status": "no_go" if assisted else "skipped",
            "reason": (
                "assisted authoring requires a verified frozen bwrap snapshot" if assisted else "unsandboxed diagnostic"
            ),
            "checks": [_check("frozen_snapshot", False, "no verified snapshot")] if assisted else [],
        }
    else:
        result = assess(context, te, ws, bundle, tools, public_root, without_tools=without_tools)
    receipt = {
        "schema": "phase1_tooling_readiness/v1",
        "target": context.target,
        "bundle_id": bundle.get("bundle_id"),
        "bundle_manifest_sha256": manifest_sha256,
        "snapshot_content_sha256": (snapshot or {}).get("content_sha256"),
        "resolved_tools": list(tools),
        "without_tools": list(without_tools),
        **result,
    }
    path = run_dir / "tooling_readiness.yaml"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        os.fchmod(stream.fileno(), 0o600)
        yaml.safe_dump(receipt, stream, sort_keys=False)
    return receipt
