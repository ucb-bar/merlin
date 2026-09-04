"""Zero-generation tooling-readiness gate — prove every tool an arm advertises actually produces REAL
output for a target, WITHOUT running a single agent round.

The A/B arms promise the agent a toolset (arm-3 = the CCA compiler-modification spine + xDSL kit;
arm-4 = all of that PLUS the RTL fact/FileCheck surface). If any of those tools silently returns nothing
for the target (as the CCA seam menu did for every non-RVV target before it was wired), a run burns
tokens producing an unusable result and we only find out after. This gate exercises each advertised tool
directly and asserts it yields real output — so a target's readiness is verified statically, before spend.

Fully TARGET-AGNOSTIC: the only input is the target name; every capability + sample is DERIVED (descriptor,
manifest, mlc facts, the target's own capsule corpus). No target literals, no stored facts. It also
enforces the arm contract structurally: arm-4's tool grants ⊇ arm-3's (the delta is exactly the RTL
surface). Run it for a target (``--target atlas``) or import ``readiness(target)``.
"""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def _ok(name: str, cond: bool, evidence: str) -> dict:
    return {"check": name, "ok": bool(cond), "evidence": evidence}


def _is_mesh_target(target: str) -> bool:
    """True iff the target is a systolic/RoCC MESH target (derives a mesh + RoCC funct/array facts). A SIMT
    target (e.g. radiance) legitimately derives NONE of these — its dataflow levers and RoCC facts are
    empty BY DESIGN (cf. test_cross_target: radiance's derived_levers/legal_opcodes are asserted empty). So
    the 'systolic facts non-empty' readiness checks are n/a for it, not failures. Best-effort; a profile
    that cannot be built (no RTL facts) reads as non-mesh (the systolic checks then degrade to n/a)."""
    try:
        from merlin.targetgen import rtl_backend as RB
        return bool(RB.target_profile(target).has_mesh)
    except Exception:  # noqa: BLE001 — no profile -> treat as non-mesh (systolic checks become n/a)
        return False


def _seam_menu_checks(target: str) -> list[dict]:
    """The CCA seam menu the assisted arms are told to call — must enumerate real modifiable sections."""
    from merlin.kernels import cca_contract as CC, action_catalog as AC
    out = []
    lax = sorted(CC.leverable_axes(target))
    if _is_mesh_target(target):
        out.append(_ok("cca.leverable_axes non-empty", bool(lax), f"axes={lax}"))
    else:
        out.append(_ok("cca.leverable_axes", True,
                       f"n/a (SIMT/non-mesh target: no systolic dataflow axes by design); axes={lax}"))
    for ax in lax:
        lad = AC.escalation_ladder(ax, target)
        out.append(_ok(f"cca.escalation_ladder[{ax}] non-empty", bool(lad),
                       f"{len(lad)} rows: {[r['action_class'] for r in lad]}"))
    b = CC.check_bijection(target)
    out.append(_ok("cca.check_bijection clean", not (b.orphan_fields or b.orphan_routes or b.ladder_errors),
                   f"orphan_fields={b.orphan_fields} orphan_routes={b.orphan_routes} errors={b.ladder_errors}"))
    return out


def _derived_lever_checks(target: str) -> list[dict]:
    from merlin.targetgen import rtl_backend as RB
    prof = RB.target_profile(target)
    lev = RB.derived_levers(prof)
    if not prof.has_mesh:                                    # SIMT/non-mesh: no systolic dataflow levers
        return [_ok("rtl_backend.derived_levers", True,
                    f"n/a (non-mesh target: no systolic dataflow levers by design) dim={prof.dim} levers={lev}")]
    return [_ok("rtl_backend.derived_levers non-empty", bool(lev),
                f"dim={prof.dim} has_mesh={prof.has_mesh} has_accumulator={prof.has_accumulator} levers={lev}")]


def _rtl_fact_checks(target: str) -> list[dict]:
    from merlin.targetgen import rtl_check_runner as RUN
    f = RUN.load_facts(target)
    facts = (f or {}).get("facts", f) or {}
    n = len(facts.get("interfaces") or []) + len(facts.get("arrays") or [])
    if not _is_mesh_target(target):                         # SIMT/non-mesh: no RoCC funct/mesh facts to derive
        return [_ok("rtl facts derivable", True,
                    f"n/a (non-mesh target: no RoCC funct/array facts by design) {n} interface/array facts")]
    return [_ok("rtl facts derivable", n > 0, f"{n} interface/array facts")]


def _rtl_check_checks(target: str) -> list[dict]:
    """The arm-4 RTL FileCheck surface: FileCheck present + endpoint-appropriate checks compile + (for a
    self-hosted target) the ISA-def decode signatures derive. Sample capsule from the target's OWN corpus."""
    import yaml
    from merlin.targetgen import rtl_check_runner as RUN, rtl_check_compiler as CCk
    from merlin.targetgen.target_experiment import load_target_experiment
    out = [_ok("FileCheck binary present", RUN.find_filecheck() is not None, str(RUN.find_filecheck()))]
    desc = C.EXP.parent / target / "target_experiment.yaml"
    corpus = load_target_experiment(desc).capsule_corpus if desc.is_file() else None
    cap_p = next(iter(sorted((corpus or Path("/nonexistent")).rglob("capsule.yaml"))), None) if corpus else None
    if cap_p is None:
        out.append(_ok("rtl checks compile", False, "no capsule in the target corpus"))
        return out
    cap = yaml.safe_load(cap_p.read_text())
    facts = RUN.load_facts(target)
    checks = CCk.compile_checks(facts, cap, target)
    endpoint_check = checks.get("kernel") or checks.get("trace")
    out.append(_ok("rtl checks compile (endpoint-appropriate)", bool(endpoint_check),
                   f"kernel={bool(checks.get('kernel'))} trace={bool(checks.get('trace'))} "
                   f"(capsule={cap_p.parent.name})"))
    return out


def _oracle_checks(target: str) -> list[dict]:
    from merlin.targetgen import capsule_runner as CR
    ad = CR.oracle_adapters(target)
    kinds = {k: getattr(v, "__qualname__", str(v)).split(".")[0] for k, v in ad.items()}
    return [_ok("oracle adapters resolve to the target's endpoint oracle", bool(ad), f"{kinds}")]


def _arm_superset_check(target: str) -> list[dict]:
    """Structural arm contract: arm-4's granted tools ⊇ arm-3's (the delta is exactly the RTL surface)."""
    import yaml
    b = C.EXP.parent / target / "input_bundles"
    def allow(arm):
        p = b / arm / "input_bundle_manifest.yaml"
        return {a["path"] for a in (yaml.safe_load(p.read_text()).get("allowed") or [])} if p.is_file() else None
    a3, a4 = allow("merlin_assisted_hwbringup_v0"), allow("merlin_assisted_rtlchecks_hwbringup_v0")
    if a3 is None or a4 is None:
        return [_ok("arm-4 ⊇ arm-3 tool grants", False, "a bundle manifest is absent")]
    missing = a3 - a4
    return [_ok("arm-4 ⊇ arm-3 tool grants", not missing,
                f"arm4 adds {sorted(a4 - a3)}; arm4 missing-from-arm3 {sorted(missing)}")]


def _registry_arm(arm: str) -> str:
    """Normalize driver/user spellings to the tool registry's arm name."""
    if "rtlchecks" in arm:
        return "merlin_rtlchecks"
    if "merlin_assisted" in arm:
        return "merlin_assisted"
    raise ValueError(f"no assisted authoring-tool contract for arm {arm!r}")


def _target_experiment(target: str):
    from merlin.targetgen.target_experiment import load_target_experiment
    descriptor = C.DESCRIPTOR if target == C.TARGET else C.EXP.parent / target / "target_experiment.yaml"
    if not descriptor.is_file():
        raise FileNotFoundError(f"target descriptor is absent: {descriptor}")
    return load_target_experiment(descriptor)


def _public_bundle(te, arm: str) -> tuple[Path, dict]:
    """Return the one public bundle that an actual functional launch serves for this arm."""
    import yaml
    registry_arm = _registry_arm(arm)
    candidates: list[tuple[Path, dict]] = []
    for path in sorted((te.path.parent / "input_bundles").glob("*/input_bundle_manifest.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if body.get("arm") == registry_arm and str(body.get("bundle_id", "")).endswith("_public_v0"):
            candidates.append((path, body))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one {registry_arm} public bundle, found "
            f"{[str(path.parent.name) for path, _ in candidates]}")
    return candidates[0]


def _promised_paths(te, arm: str) -> tuple[list[str], tuple]:
    """Exact file grants and brokers promised by one registry arm."""
    from merlin.targetgen import tool_registry as registry
    tools = registry.arm_tools(_registry_arm(arm))
    paths: list[str] = []
    for name in tools:
        spec = registry.spec(name)
        paths.extend(spec.bundle_paths)
        paths.extend(str(getattr(te, attr)) for attr in spec.derived_paths)
    return list(dict.fromkeys(paths)), registry.brokers_for(tools)


def _tool_only_bundle(te, arm: str, bundle: dict) -> dict:
    """Restrict the real public manifest to the promised authoring-tool closure.

    This gate must use exactly what the paid cell will mount, but copying the unrelated corpus, LLVM and
    host lane just to test Python imports would turn a preflight into a multi-gigabyte operation.  Exact
    registry paths must therefore exist as exact manifest grants; their relevant deny overlays remain in
    force.  A stale manifest fails closed before bwrap starts.
    """
    promised, _ = _promised_paths(te, arm)
    by_path = {str(entry.get("path")): entry for entry in bundle.get("allowed", [])
               if isinstance(entry, dict) and entry.get("path")}
    missing = sorted(set(promised) - set(by_path))
    if missing:
        raise RuntimeError(
            "public bundle is missing promised authoring grant(s): " + ", ".join(missing))
    return {
        "bundle_id": f"{bundle.get('bundle_id', 'bundle')}__tooling_readiness",
        "arm": bundle.get("arm"),
        "allowed": [dict(by_path[path]) for path in promised],
        # Preserve the launch manifest's complete deny set. Deny-wins overlays that do not intersect a
        # tool grant are harmless; the two that do (runtime_adapter and xdsl/lowering) are essential.
        "denied": [dict(entry) for entry in bundle.get("denied", [])],
    }


#: The instruction the ISA round-trip asks the broker to assemble when the target's own derived model
#: cannot name one. Kept as the previous behaviour for every endpoint that already answered it.
_ASM_PROBE_FALLBACK = "FENCE"


def _asm_probe_mnemonic(target: str) -> str:
    """An instruction name the ARM the broker will actually take can assemble for THIS target.

    The probe used to ask for one fixed RISC-V mnemonic. That is an assumed ISA constant, and it is not
    true of every endpoint: an ISA derived from a core's own RTL decoder names instruction CLASSES, so
    the round-trip asked for an instruction the target does not have and the gate reported the promised
    ISA tools BROKEN for correctly refusing to invent one.

    Routed exactly as ``isa_tools_broker`` routes the request, so the probe and the broker cannot
    disagree: a RoCC/``inline_asm_insn`` endpoint is answered by ``rocc_asm`` (which knows nothing of the
    IsaModel), so the fallback is kept for it unchanged; every other endpoint is answered from the derived
    :class:`~merlin.targetgen.isa_model.IsaModel`, so the mnemonic comes from that model — the fallback
    when it defines it, otherwise the model's own first opcode. Any failure to resolve leaves the
    fallback, so the broker (not this helper) reports a target whose tools are genuinely dead.
    """
    try:
        import isa_tools_broker as _IB
        from merlin.targetgen import capsule_runner as _CR
        if _IB.is_rocc_endpoint(_CR._endpoint_of(target)[0]):
            return _ASM_PROBE_FALLBACK
        from merlin.targetgen.isa_model import isa_model_for_target
        model = isa_model_for_target(target)
    except Exception:  # noqa: BLE001 -- unresolvable endpoint/model: the broker reports it, not this helper
        return _ASM_PROBE_FALLBACK
    if model.resolve(_ASM_PROBE_FALLBACK) is not None:
        return _ASM_PROBE_FALLBACK
    if model.is_fixed_format():
        if _ASM_PROBE_FALLBACK in model.opcode_table:
            return _ASM_PROBE_FALLBACK
        names = sorted(model.opcode_table)
        if names:
            return names[0]
    return _ASM_PROBE_FALLBACK


def _authoring_probe(target: str) -> str:
    """Python body run inside the real bwrap after immutable materialization."""
    asm_probe = _asm_probe_mnemonic(target)
    return textwrap.dedent(f"""
        from merlin.targetgen.evidence.store import Evidence
        from merlin.targetgen import rtl_backend as RB
        from merlin.targetgen import synthesize as S
        from merlin.targetgen import generate as G
        from merlin.targetgen.generate import target_repo
        from merlin.targetgen.contract.interface_emit import emit_interface_mlir
        from merlin.targetgen.rtl import facts, gen_numeric_facts, gen_isa_module, gen_rtl_digest
        from merlin.runtime.commandbuffer import pool_params
        from merlin.runtime.tensor import pool_out_dims
        from merlin.kernels import cca_contract, action_catalog
        import json
        import subprocess
        import sys
        import merlin.xdsl_dialects
        import merlin.targetgen.oot_starterkit
        import merlin.targetgen.contract.interface_emit
        import merlin.targetgen.contract.linalg_iface

        target = {target!r}
        evidence = Evidence(target=target, sources={{}})
        contract = S.synthesize_target_contract(evidence, target)
        dialect_plan = S.synthesize_dialect_plan(evidence, contract)
        plans = (
            dialect_plan,
            S.synthesize_runtime_adapter_plan(evidence, contract),
            S.synthesize_zephyr_plan(evidence, contract),
            S.synthesize_llvm_extension_plan(evidence, contract),
        )
        assert all(isinstance(plan, dict) and plan for plan in plans)
        assert target_repo.generate_skeleton(target)
        assert G.xdsl.generate(dialect_plan)
        assert G.mlir_scaffold.generate(dialect_plan)
        assert G.zephyr_module.generate(plans[2])
        assert G.llvm_plan.generate(plans[3])

        # ``interface_emit`` imports its runtime shape helpers only on the pooled-COMMIT path. Exercise
        # those exact transitive grants so a max-pool capsule cannot be the first place they fail.
        pooled = {{"pool_in_dims": [4, 4], "pool_size": [2, 2], "pool_stride": [2, 2],
                   "pool_padding": [0, 0, 0, 0]}}
        assert pool_params(pooled, op="readiness")["pool_in_dims"] == (4, 4)
        assert pool_out_dims(4, 4, [2, 2], [2, 2], [0, 0, 0, 0]) == (2, 2)
        assert callable(emit_interface_mlir)

        profile = RB.target_profile(target)
        assert not profile.discovered_nothing, profile
        levers = RB.derived_levers(profile)
        if profile.has_mesh:
            assert levers, profile

        doc = facts.load_facts(target)
        body = (doc or {{}}).get("facts") or {{}}
        assert body, "reviewed RTL facts are empty"
        assert gen_numeric_facts.generate(doc).strip()
        if any(item.get("name") == "funct_decode_table" for item in body.get("interfaces", [])):
            # A DECODE TABLE IS NOT A ROCC SLOT. A self-hosted-ISA device fetches and decodes its own
            # stream: it has a funct decode table and NO RISC-V custom opcode, and `gen_isa_module`
            # refuses to emit an encoder rather than guess one -- which is the behaviour the repo
            # demands. Asserting on the call turned that refusal into an uncaught exception inside the
            # sandbox, so the probe reported the target's authoring tools BROKEN for doing the right
            # thing. What must hold is that the tool RUNS and either emits or declines for a stated
            # reason; both are recorded so a real crash still fails here.
            try:
                assert gen_isa_module.generate(doc).strip()
                print("ISA_MODULE=emitted")
            except gen_isa_module.NotARoccTarget as exc:
                print("ISA_MODULE=declined:", str(exc)[:120])
            assert gen_rtl_digest.generate(doc).strip()

        report = cca_contract.check_bijection(target).unexpected()
        assert report.clean, report
        axes = sorted(cca_contract.leverable_axes(target))
        if profile.has_mesh:
            assert axes
            assert action_catalog.escalation_ladder(axes[0], target)
        print("AUTHORING_IMPORTS_AND_OUTPUTS_OK")

        isa = subprocess.run(
            [sys.executable, "isa_tools.py", "asm", {asm_probe!r}],
            capture_output=True, text=True, timeout=30)
        assert isa.returncode == 0, (isa.stdout, isa.stderr)
        isa_result = json.loads(isa.stdout)
        # External-ISA targets return encoded ``words``; a RoCC command-buffer target returns a
        # non-empty inline-asm MLIR module.  Both are real emitted artifacts, never an import-only pass.
        assert isa_result.get("n", 0) >= 1, isa_result
        assert isa_result.get("words") or str(isa_result.get("mlir", "")).strip(), isa_result

        from cca_contract import check_bijection
        from action_catalog import escalation_ladder
        broker_bijection = check_bijection(target)
        assert not broker_bijection.get("error"), broker_bijection
        assert (broker_bijection.get("unexpected") or {{}}).get("clean") is True, broker_bijection
        broker_ladder = escalation_ladder("spatial.dataflow", target)
        if profile.has_mesh:
            assert broker_ladder.get("n", 0) >= 1, broker_ladder
        print("BROKER_ROUNDTRIPS_OK")
    """)


def sandbox_authoring_readiness(target: str, arm: str = "merlin_assisted_rtlchecks") -> dict:
    """Exercise promised tools through the exact immutable-snapshot + bwrap + broker path.

    This is deliberately separate from :func:`readiness`, whose original contract is a cheap host-side
    zero-generation probe across the target roster.  The launch gate calls this stronger check for its
    selected target.  Any missing grant, failed import, empty generator, broker error or non-zero command
    is a hard failure.
    """
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox import toolchain as TC

    if not shutil.which("bwrap"):
        return _ok("assembled bwrap authoring tools", False, "bwrap executable is absent")
    broker_specs = ()
    processes: list[subprocess.Popen] = []
    logs = []
    root: Path | None = None
    ws: Path | None = None

    try:
        te = _target_experiment(target)
        manifest_path, public = _public_bundle(te, arm)
        tool_bundle = _tool_only_bundle(te, arm, public)
        _, broker_specs = _promised_paths(te, arm)
        qa_root = te.path.parent / "_qa_ws"
        qa_root.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="tooling-readiness-", dir=qa_root))
        ws = root / "workspace"
        snapshot = BW.materialize_bundle_inputs(ws, tool_bundle, repo=C.REPO)
        ws.mkdir(parents=True, exist_ok=True)
        (ws / "submission").mkdir(exist_ok=True)

        broker_env = os.environ.copy()
        broker_env["MERLIN_TARGET_EXPERIMENT"] = str(te.path.resolve())
        for spec in broker_specs:
            channel = ws / spec.channel
            channel.mkdir(parents=True, exist_ok=True)
            for shim, staged_as in spec.shims:
                shutil.copy2(C.HARNESS / shim, ws / staged_as)
            log = (channel / spec.log).open("w", encoding="utf-8")
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, str(C.HARNESS / spec.module), "--ws", str(ws)],
                cwd=str(C.REPO), env=broker_env, stdout=log, stderr=subprocess.STDOUT))

        # The shims themselves poll for replies, so an ordinary scheduling race is safe.  Still wait a
        # short bounded interval and reject a broker that dies during import/startup before entering bwrap.
        time.sleep(0.5)
        dead = [f"{spec.module}:rc={process.poll()}"
                for spec, process in zip(broker_specs, processes, strict=True)
                if process.poll() is not None]
        if dead:
            raise RuntimeError("promised broker failed during startup: " + ", ".join(dead))

        probe = _authoring_probe(target)
        argv = [*BW.full_argv(te, ws, tool_bundle), "bash", "-c",
                TC.sandbox_env(te, ws) + f"python3 -c {shlex.quote(probe)}"]
        run = subprocess.run(argv, cwd=str(C.REPO), capture_output=True, text=True, timeout=180)
        evidence = (run.stdout + "\n" + run.stderr).strip()
        ok = (run.returncode == 0
              and "AUTHORING_IMPORTS_AND_OUTPUTS_OK" in run.stdout
              and "BROKER_ROUNDTRIPS_OK" in run.stdout)
        detail = (
            f"bundle={manifest_path.parent.name}; snapshot={snapshot['content_sha256']} "
            f"({snapshot['n_files']} files/{snapshot['n_bytes']} bytes); rc={run.returncode}; "
            f"output={evidence[-1200:]}")
        return _ok("assembled bwrap authoring tools", ok, detail)
    except Exception as exc:  # noqa: BLE001 -- prelaunch gate must report and fail closed
        return _ok("assembled bwrap authoring tools", False,
                   f"{type(exc).__name__}: {str(exc)[:1200]}")
    finally:
        for spec in broker_specs:
            channel = (ws or Path("/nonexistent")) / spec.channel
            if channel.is_dir():
                (channel / "STOP").write_text("stop", encoding="utf-8")
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for log in logs:
            log.close()
        if ws is not None:
            BW.remove_bundle_snapshot(ws)
        if root is not None:
            shutil.rmtree(root, ignore_errors=False)


# arm -> the capability groups that arm advertises (arm-4 is arm-3 ∪ the RTL surface).
_ARM3 = ("seam_menu", "derived_levers")
_ARM4 = _ARM3 + ("rtl_facts", "rtl_checks", "oracle")
_GROUPS = {"seam_menu": _seam_menu_checks, "derived_levers": _derived_lever_checks,
           "rtl_facts": _rtl_fact_checks, "rtl_checks": _rtl_check_checks, "oracle": _oracle_checks}


def readiness(target: str, arm: str = "merlin_assisted_rtlchecks") -> dict:
    """Run the readiness checks for ``target`` at the given arm's tool level (default arm-4 = the full
    superset). Returns {target, arm, checks:[...], ok}. Zero generation."""
    groups = _ARM4 if "rtlchecks" in arm else _ARM3
    checks: list[dict] = []
    for g in groups:
        try:
            checks += _GROUPS[g](target)
        except Exception as e:  # noqa: BLE001 — a dead tool is a readiness FAIL, reported not raised
            checks.append(_ok(f"{g} runnable", False, f"{type(e).__name__}: {str(e)[:160]}"))
    checks += _arm_superset_check(target)
    return {"target": target, "arm": arm, "checks": checks, "ok": all(c["ok"] for c in checks)}


def submission_language_ok(submission_dir, arm: str) -> tuple[bool, str]:
    """ENFORCE the arm language contract on an EMITTED submission: a merlin arm (arm-3/arm-4) must build
    its dialect with the xDSL/Python kit — NOT a hand C++/TableGen backend. Returns (ok, reason). Reads
    the submission's manifest.yaml: for a merlin arm it fails if ``language: cpp`` or a build block that
    compiles a C++ tool (cmake / mlir-tblgen / a *-opt binary) is present. The C++ arms are exempt (that
    is their mandated method). Pure + target-agnostic — the grader/driver calls this to reject a
    non-compliant round with an actionable reason instead of grading a forbidden backend."""
    import yaml
    from pathlib import Path
    if "merlin_assisted" not in arm:               # only the merlin (xDSL) arms are constrained
        return True, "not a merlin arm (no xDSL mandate)"
    mpath = Path(submission_dir) / "mlir_oot" / "manifest.yaml"
    if not mpath.is_file():
        mpath = next(iter(sorted(Path(submission_dir).rglob("manifest.yaml"))), None)
    if not mpath or not mpath.is_file():
        return False, "no manifest.yaml in submission"
    m = yaml.safe_load(mpath.read_text()) or {}
    lang = str(m.get("language", "")).strip().lower()
    if lang in ("cpp", "c++", "cxx"):
        return False, f"merlin arm must use xDSL/Python, manifest declares language: {lang}"
    build = m.get("build") or {}
    blob = " ".join(str(v) for v in (build.values() if isinstance(build, dict) else [build])).lower()
    for marker in ("cmake", "mlir-tblgen", "tblgen", "clang++", "g++"):
        if marker in blob:
            return False, f"merlin arm must use xDSL/Python, build block invokes {marker!r} (C++ toolchain)"
    return True, f"xDSL/Python (language={lang or 'python'})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default=C.TARGET)
    ap.add_argument("--arm", default="merlin_assisted_rtlchecks")
    a = ap.parse_args(argv)
    rep = readiness(a.target, a.arm)
    live = sandbox_authoring_readiness(a.target, a.arm)
    rep["checks"].append(live)
    rep["ok"] = rep["ok"] and live["ok"]
    print(f"=== tooling readiness: target={rep['target']} arm={rep['arm']} ===")
    for c in rep["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['check']}: {c['evidence']}")
    print(f"\n  readiness: {'READY' if rep['ok'] else 'NOT READY'}")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
