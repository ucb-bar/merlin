"""Generate the 4-arm experiment bundle manifests for ANY target from its ``target_experiment.yaml``.

The 4-arm ladder (raw C++ scaffold → C++ + Merlin infra → xDSL + the CCA where/how spine → + CIRCT RTL
checks) is a fixed METHODOLOGY. Each arm's toolset is (a) a per-rung block of TARGET-AGNOSTIC merlin
module paths — identical literal strings for every target — plus (b) a small TARGET/EXPERIMENT-specific
block that comes from the descriptor (ISA headers, hwbringup set, corpus) or is DERIVED from the target
name (the rtl_facts pin, the irdl pin, the prior-backend deny surfaces). This generator emits (a)+(b), so
a new accelerator drops a descriptor + registers its RTL with mlc and gets the whole ladder — no
hand-authored, gemmini-overfit YAML.

Faithful to the hand-authored gemmini bundles (verified by ``test_generate_bundles`` — the generated
allow/deny path SETS match, and verify_no_cheat + the sandbox stay green).
"""
from __future__ import annotations

from typing import Any

from pathlib import Path

from . import tool_registry as TR
from .target_experiment import TargetExperiment

_PY = "merlin/python/merlin/"  # the agnostic merlin package prefix (identical for every target)

# --- per-arm DENY-ONLY blocks (the ALLOW side now comes from :mod:`tool_registry`) ------------------
# The tools each rung GRANTS are named in tool_registry.ARM_TOOLS, so a rung and an ablation cell are
# the same kind of object: a set of tool names. What stays here is the deny side, which is not a tool
# list — it is the wider surface an arm must not reach even though a shared grant would expose it.
_CPP_DENY_AGN = [f"{_PY}targetgen/rtl/{m}.py" for m in
                 ("gen_iface_irdl", "gen_isa_module", "gen_rtl_digest", "gen_numeric_facts")] + [
    f"{_PY}targetgen/oot_starterkit/", f"{_PY}targetgen/synthesize/", f"{_PY}xdsl_dialects/",
    f"{_PY}targetgen/generate/xdsl.py", f"{_PY}targetgen/generate/runtime_adapter.py",
    f"{_PY}runtime/reference.py", f"{_PY}runtime/simulator.py", f"{_PY}xdsl_dialects/lowering/"]
# oracle-callable routes denied in the xDSL/CIRCT arms (arm3/arm4).
_ORACLE_DENY = [f"{_PY}runtime/reference.py", f"{_PY}runtime/simulator.py",
                f"{_PY}targetgen/generate/runtime_adapter.py", f"{_PY}xdsl_dialects/lowering/"]


def _tool_allow(te: TargetExperiment, tools: tuple[str, ...]) -> list[dict]:
    """The allow entries the resolved tool set contributes, in ladder order.

    A tool reached through a broker contributes no path — it is staged into the workspace by the driver,
    not bound from the repo — so it appears here only by its absence from the grant list.
    """
    out: list[dict] = []
    for name in tools:
        t = TR.spec(name)
        paths = list(t.bundle_paths) + [getattr(te, attr) for attr in t.derived_paths]
        out += [{"path": p, "mode": "ro", "note": t.note} for p in paths]
    return out


def _tool_paths(te: TargetExperiment, name: str) -> list[str]:
    """Every repo path a single tool grants (literal + descriptor-derived)."""
    t = TR.spec(name)
    return list(t.bundle_paths) + [getattr(te, attr) for attr in t.derived_paths]


def _host_lane_grants(te: TargetExperiment) -> tuple[list[dict], list[dict]]:
    """``(allow, deny)`` for the frozen host lane: read-only on what it IS, denied on what would CHANGE it.

    The experiment measures the target dialect, the target transforms, the partitioning and the
    boundary/dispatch. It does not measure a second CPU backend, and a submission that wins by rewriting
    the host compiler has answered a different question under this one's name. So the lane is pinned as
    infrastructure: readable (an agent that cannot see what the host side costs cannot decide where the
    boundary goes) and unwritable.

    The two sides MUST name different paths. Deny wins in the sandbox binder (see
    ``merlin.targetgen.sandbox.bwrap.base_argv``), so a path on both lists is tmpfs-masked and the
    "read-only" grant silently becomes no grant at all. That is checked here rather than documented,
    because a grant that quietly evaporates is exactly the class of bug this repo has already shipped.
    """
    # UNION ACROSS EVERY DECLARED PROFILE, not just the default. An arm is graded on whatever dtype its
    # capsules declare, so it must be able to read the lane it will actually be compiled with -- granting
    # only the default lane would leave a second-precision capsule graded against a package the agent
    # was never shown, which is the same invisibility the pin exists to prevent. A no-op for a target
    # with one profile.
    if te.host_lanes is None:
        return [], []
    lanes = list(te.host_lanes.profiles.values())
    read_only, deny = [], []
    for lane in lanes:
        read_only += [p for p in lane.read_only if p not in read_only]
        deny += [p for p in lane.deny_modification if p not in deny]
    if not read_only and not deny:
        raise ValueError(f"{te.path}: `host_lane` declares neither `read_only` nor `deny_modification`; "
                         f"a lane that is pinned by nothing is not pinned")
    clash = sorted(set(read_only) & set(deny))
    if clash:
        raise ValueError(
            f"{te.path}: host_lane path(s) {clash} are listed BOTH read-only and denied. Deny wins in "
            f"the sandbox, so those grants would bind nothing and the arm would simply not see the "
            f"frozen lane. Deny the implementation surface, not the artifact you are granting.")
    allow = [{"path": p, "mode": "ro", "note": "frozen host lane (pinned infrastructure, read-only)"}
             for p in read_only]
    denied = [{"path": p, "reason": "frozen host lane: the experiment measures the target lane, not a "
                                    "second CPU backend"} for p in deny]
    return allow, denied


def _shared_allow(te: TargetExperiment) -> list[dict]:
    """The target/experiment-parameterized allow block present in EVERY arm."""
    exp = te.exp_name
    out = [{"path": "merlin/contract/", "mode": "ro", "note": "frozen ABI v0.1"},
           {"path": te.corpus_rel(), "mode": "ro", "note": "capsule corpus"}]
    out += [{"path": s, "mode": "ro"} for s in te.corpus_siblings()]
    out += [{"path": h, "mode": "ro", "note": "ISA header (shared hardware spec)"} for h in te.isa_headers]
    out += [{"path": f"experiments/{exp}/task/", "mode": "ro"},
            {"path": "third_party/llvm-install/", "mode": "ro", "note": "LLVM/MLIR 23 toolchain"}]
    if te.hwbringup_set:
        out.append({"path": te.hwbringup_set, "as": te.target, "mode": "ro",
                    "note": "shared hardware spec: RTL + ISA headers + README + example (ALL arms)"})
    out.append({"path": f"experiments/{exp}/scripts/agent_selfcheck.py", "as": "agent_selfcheck.py",
                "mode": "ro", "note": "redacted self-check"})
    out += _host_lane_grants(te)[0]
    return out


def _shared_deny(te: TargetExperiment) -> list[dict]:
    """The target/experiment-parameterized deny block present in EVERY arm (answer surfaces)."""
    exp = te.exp_name
    # answer surfaces live under the out/ generated root (the hand-authored bundles used a stale prefix
    # missing the out/ root; the real backends are under out/artifacts/, matching the launcher's lock).
    out = [{"path": f"out/artifacts/targets/{te.target}/{b}/",
            "reason": "prior backend / exemplar (answer surface)"} for b in te.prior_backends]
    if te.hidden_corpus():
        out.append({"path": te.hidden_corpus(), "reason": "hidden capsules + goldens"})
    out += [{"path": f"experiments/{exp}/input_bundles/grader_private_v0/", "reason": "grader-private"},
            {"path": f"experiments/{exp}/runs/", "reason": "prior submissions"}]
    out += _host_lane_grants(te)[1]
    return out


def _arm_manifest(te: TargetExperiment, arm: str, bundle_id: str, *,
                  add_tools: tuple[str, ...] = (), drop_tools: tuple[str, ...] = ()) -> dict[str, Any]:
    """Assemble one arm's manifest = shared target/exp block + the tools its rung carries.

    ``add_tools`` / ``drop_tools`` express an ABLATION CELL — this rung plus or minus named tools. With
    neither (the default) the output is exactly the ladder's, byte for byte.

    A dropped tool is moved to ``denied``, never merely omitted: the shared ``merlin/contract`` and
    toolchain grants can re-expose a path that is simply absent from the allow list, and a cell that
    still reaches the tool it claims to have removed is a silently void measurement.
    """
    tools = TR.arm_tools(arm, add=add_tools, drop=drop_tools)
    allow = _shared_allow(te)
    deny = _shared_deny(te)
    allow += _tool_allow(te, tools)
    if arm == "raw_baseline":
        deny = [{"path": "merlin/", "reason": "Merlin internals (no tools for the raw arm)"}] + deny
    elif arm == "cpp_merlininfra":
        deny = ([{"path": p, "reason": "denied tool (kept a strict subset of the xDSL arm)"} for p in _CPP_DENY_AGN]
                + [{"path": te.irdl_pin, "reason": "IRDL spec (xDSL arm only)"},
                   {"path": te.rtl_facts_pin, "reason": "RTL facts (CIRCT arm only)"}] + deny)
    elif arm in ("merlin_assisted", "merlin_eqsat"):
        # The eqsat arm shares the xDSL arm's denials on purpose: an arm that also gained the RTL facts
        # would differ in TWO ways and its result would not attribute to the seam.
        deny = ([{"path": f"{_PY}targetgen/rtl/", "reason": "CIRCT RTL generators (CIRCT arm only)"},
                 {"path": te.rtl_facts_pin, "reason": "RTL facts (CIRCT arm only)"}]
                + [{"path": p, "reason": "oracle-callable route"} for p in _ORACLE_DENY] + deny)
    elif arm == "merlin_rtlchecks":
        deny = [{"path": p, "reason": "oracle-callable route"} for p in _ORACLE_DENY] + deny
    else:
        raise ValueError(f"unknown arm {arm!r}")
    if add_tools or drop_tools:
        allow, deny = _apply_ablation(te, allow, deny, add_tools, drop_tools)
    return {"bundle_id": bundle_id, "arm": arm, "task": f"{te.target}-mlir-oot-capsule",
            "description": f"{arm} arm for the {te.target} target (generated from target_experiment.yaml)",
            "allowed": allow, "denied": deny, "integrity_required": True}


def _apply_ablation(te: TargetExperiment, allow: list[dict], deny: list[dict],
                    add_tools: tuple[str, ...], drop_tools: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    """Reconcile the deny side with the cell's added/dropped tools.

    An ADDED tool must lose the deny entry the base rung wrote for it (deny wins in the sandbox, so a
    grant left standing beside its own denial grants nothing). A DROPPED tool gains one.
    """
    added_paths = {p for n in add_tools for p in _tool_paths(te, n)}
    deny = [e for e in deny if e.get("path") not in added_paths]
    for name in drop_tools:
        deny = [{"path": p, "reason": f"ablated: {name} withheld from this arm"}
                for p in _tool_paths(te, name)] + deny
    return allow, deny


# arm -> the bundle-id stem (the launcher appends the variant suffix).
_ARMS = {"raw_baseline": "raw_baseline", "cpp_merlininfra": "cpp_merlininfra",
         "merlin_assisted": "merlin_assisted", "merlin_rtlchecks": "merlin_assisted_rtlchecks",
         # arm5's stem CONTAINS "merlin_assisted" on purpose: generate_prompt._is_assisted_arm is a
         # substring test, so the arm inherits the assisted seam menu with no prompt edit.
         "merlin_eqsat": "merlin_assisted_eqsat"}


def generate_bundles(te: TargetExperiment, *, variant: str = "hwbringup_v0",
                     add_tools: tuple[str, ...] = (), drop_tools: tuple[str, ...] = (),
                     arms: tuple[str, ...] = ()) -> dict[str, dict]:
    """The bundle manifests for ``te``, keyed by bundle_id. Target-agnostic: the same code emits them
    for any target from its descriptor + derived paths (no hand-authored YAML).

    With ``add_tools``/``drop_tools`` the emitted bundles are ABLATION CELLS and their ids carry a
    suffix naming the variation, so a run directory records which cell produced it. ``arms`` narrows
    generation to the rungs a cell actually needs (default: the whole ladder).
    """
    suffix = TR.cell_suffix(add_tools, drop_tools)
    wanted = arms or tuple(_ARMS)
    for a in wanted:
        if a not in _ARMS:
            raise KeyError(f"unknown arm {a!r}; known: {sorted(_ARMS)}")
    out = {}
    for arm in wanted:
        bid = f"{_ARMS[arm]}_{variant}{suffix}"
        out[bid] = _arm_manifest(te, arm, bid, add_tools=add_tools, drop_tools=drop_tools)
    return out


def _dump_manifest(manifest: dict[str, Any]) -> str:
    """Serialize one bundle manifest to YAML (a generated header + the manifest body). Key order is
    preserved so the file reads like the hand-authored ones; consumers ``yaml.safe_load`` it."""
    import yaml
    header = (f"# GENERATED by merlin.targetgen.generate_bundles for target {manifest.get('task')!r}.\n"
              f"# Do not hand-edit: regenerate from target_experiment.yaml (the 4-arm ladder is a fixed\n"
              f"# methodology; the target/experiment-specific paths come from the descriptor).\n")
    return header + yaml.safe_dump(manifest, sort_keys=False, default_flow_style=False, width=120)


def _grant_txt(manifest: dict[str, Any], key: str) -> str:
    """The bundle's allow/deny path list as a newline file (the ``allowed_files.txt`` / ``denied_files.txt``
    the sandbox + graders read) — DERIVED verbatim from the manifest's ``allowed``/``denied`` path set."""
    paths = [e["path"] for e in manifest.get(key, []) if isinstance(e, dict) and e.get("path")]
    return "\n".join(paths) + ("\n" if paths else "")


def _allowed_merlin_tools_doc(manifest: dict[str, Any]) -> str:
    """Human-readable tool policy derived from the exact bundle manifest.

    This document deliberately does not choose an isolation mode.  A bundle declares capabilities;
    the launcher chooses and records the sandbox for one run.  Mixing those two layers produced a
    committed document that continued to claim ``sandbox=none`` long after scored launches required
    bwrap.
    """
    def entries(key: str, detail: str) -> str:
        rows = []
        for entry in manifest.get(key, ()):
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            why = str(entry.get(detail) or "declared by the generated bundle manifest")
            rows.append(f"- `{entry['path']}` — {why}")
        return "\n".join(rows) if rows else "- (none)"

    bundle_id = manifest.get("bundle_id", "unknown")
    arm = manifest.get("arm", "unknown")
    return f"""# Allowed Merlin tooling — generated for `{bundle_id}`

This is the human-readable view of the **authoring** grants for arm `{arm}`. It is generated from
`input_bundle_manifest.yaml`; the manifest remains the machine-readable authority.

## Runtime isolation (launch-derived)

This bundle does **not** select a sandbox. Read `TASK.md` → **Runtime scope** (`Active sandbox`) and the
run's `environment.yaml` → `sandbox` for the mode actually used. Both fields are written from the
launcher's real `--sandbox` argument. A scored, trusted run requires deny-by-default `bwrap` plus its
frozen input snapshot. An explicit `none` run is diagnostic only and supports no trusted isolation claim.
If older static prose names another mode, those run artifacts win.

## Allowed authoring inputs

{entries("allowed", "note")}

## Denied inputs

{entries("denied", "reason")}

## Submission boundary

The grants above may help author and debug the package. The submitted package must remain self-contained
and integrity-clean: it is graded only through its declared CLI entrypoints and may not import Merlin,
call an oracle/reference implementation, copy a prior backend or kernel, or embed expected outputs.
Denied paths and answer surfaces remain masked even when a broader parent directory is allowed.
"""


def _materialize_prompt_and_grants(te: TargetExperiment, bdir, bundle_id: str, variant: str,
                                   manifest: dict[str, Any], cap, written: list) -> None:
    """Emit the derivable, non-manifest bundle files idempotently:
      * ``STARTER_PROMPT.md`` — the target-general task prompt (``generate_prompt.render_prompt``); passed
        the bundle STEM as the arm so the assisted/CIRCT arms get their seam menu (``_is_assisted_arm``
        keys on the ``merlin_assisted`` substring, which the stem — not the short arm key — carries).
        It is written only when absent so a curated prompt is preserved.
      * ``allowed_files.txt`` / ``denied_files.txt`` — the manifest's allow/deny path lists.
      * ``ALLOWED_MERLIN_TOOLS.md`` — for assisted arms, the exact human-readable grants and the
        launch-derived isolation contract. These three policy files are pure manifest derivations and
        are rewritten when stale.
    ``cap`` is the target's capability manifest (or ``None`` if it could not be loaded — then the prompt is
    skipped with the grants still written)."""
    from pathlib import Path
    bdir = Path(bdir)
    stem = bundle_id[: -(len(variant) + 1)] if bundle_id.endswith("_" + variant) else bundle_id
    # hwbringup bundles are the REALISTIC experiment's info set; anything else renders at full scope.
    experiment = "realistic" if variant.startswith("hwbringup") else "full"

    def _w_if_absent(name: str, text: str) -> None:
        p = bdir / name
        if not p.exists():
            p.write_text(text)
            written.append(p)

    def _w_always(name: str, text: str) -> None:
        """Rewrite a file that is a PURE DERIVATION of the manifest.

        The grant lists and generated tooling document are exactly that (see :func:`_grant_txt` and
        :func:`_allowed_merlin_tools_doc`). They had been written only-when-absent beside the curated
        prompt. That let them go stale while the manifest — which is always regenerated — moved on, and
        they are read by DIFFERENT consumers: the sandbox binds from the manifest while the agent reads
        the text files. Regenerating them is not a policy change; it restores each file to its definition.
        """
        p = bdir / name
        if not p.exists() or p.read_text() != text:
            p.write_text(text)
            written.append(p)

    if cap is not None:
        from .generate_prompt import render_prompt
        # Thread the bundle's OWN grant set, so the mandatory-workflow block names only the tools this
        # bundle actually binds. Without it the prompt falls back to a coarse match on the arm string
        # and an ablation cell would be handed its full rung's checklist -- telling the agent to use a
        # tool the cell deliberately withheld, which is the one thing a cell must never do.
        granted = {e["path"] for e in (manifest.get("allowed") or [])
                   if isinstance(e, dict) and str(e.get("path", "")).startswith(("merlin/", "experiments/"))}
        _w_if_absent("STARTER_PROMPT.md", render_prompt(te, cap, experiment, stem, granted_tools=granted))
    _w_always("allowed_files.txt", _grant_txt(manifest, "allowed"))
    _w_always("denied_files.txt", _grant_txt(manifest, "denied"))
    if manifest.get("arm") in {"merlin_assisted", "merlin_rtlchecks", "merlin_eqsat"}:
        _w_always("ALLOWED_MERLIN_TOOLS.md", _allowed_merlin_tools_doc(manifest))


def materialize_bundles(te: TargetExperiment, dest, *,
                        variants: tuple[str, ...] = ("hwbringup_v0",),
                        add_tools: tuple[str, ...] = (), drop_tools: tuple[str, ...] = (),
                        arms: tuple[str, ...] = ()) -> list["Path"]:
    """Write every generated bundle under ``dest/<bundle_id>/`` for each requested ``variant``:
    ``input_bundle_manifest.yaml`` (always, overwritten — the manifest is fully generated) plus the
    derivable non-manifest files (a preserved ``STARTER_PROMPT.md`` plus always-derived grant lists and
    assisted-arm ``ALLOWED_MERLIN_TOOLS.md``). Target-agnostic. Returns the written paths.

    ``dest`` is typically ``experiments/<exp>/input_bundles`` — the same tracked location the launcher and
    ``require_scaffolding`` read (bundles are curated inputs, not ``out/`` generated output)."""
    from pathlib import Path
    dest = Path(dest)
    # The capability manifest (needed to render STARTER_PROMPT.md) is target-level; load it once and
    # degrade honestly if unavailable (e.g. mlc absent) — manifests + grant files are still written.
    try:
        from .target_experiment import declared_vs_resolved_contract, load_capability_manifest
        # When the registry resolves nothing for this target, fall back to the contract the DESCRIPTOR
        # declares. Without this a descriptor could name its contract, have that file sit right there on
        # disk, and still render no prompt — which is how a target reached "bundles generated" with three
        # of its four STARTER_PROMPT.md missing and the anti-cheat gate failing on their absence.
        declared, resolved, verdict = declared_vs_resolved_contract(te)
        explicit = declared if verdict == "declared_only" else None
        if explicit:
            print(f"  note: registry resolves no contract for {te.target!r}; using the descriptor's "
                  f"declared {te.declared_contract}")
        cap = load_capability_manifest(te.target, contract_path=explicit)
    except Exception as e:  # noqa: BLE001 — no capability manifest -> skip prompt, keep the rest
        cap = None
        print(f"  note: capability manifest for {te.target!r} unavailable ({type(e).__name__}: {e}); "
              f"STARTER_PROMPT.md not rendered (manifests + grant files still written).")
    written: list[Path] = []
    for variant in variants:
        bundles = generate_bundles(te, variant=variant, add_tools=add_tools, drop_tools=drop_tools,
                                   arms=arms)
        for bundle_id, manifest in bundles.items():
            bdir = dest / bundle_id
            bdir.mkdir(parents=True, exist_ok=True)
            out = bdir / "input_bundle_manifest.yaml"
            out.write_text(_dump_manifest(manifest))
            written.append(out)
            # The resolved tool set, written beside the grants so the CELL IS SELF-DESCRIBING: the
            # driver reads it to decide which brokers to start, instead of re-deriving the arm from a
            # substring of its name. Fully generated, so it is rewritten like the manifest.
            tools = TR.arm_tools(manifest["arm"], add=add_tools, drop=drop_tools)
            tp = bdir / "tools.txt"
            tp.write_text("".join(f"{t}\n" for t in tools))
            written.append(tp)
            _materialize_prompt_and_grants(te, bdir, bundle_id, variant, manifest, cap, written)
    return written


def _main(argv: list[str] | None = None) -> int:
    """CLI: materialize the 4-arm bundle manifests for a target's descriptor.

    Usage: python -m merlin.targetgen.generate_bundles --descriptor <target_experiment.yaml> \\
               [--dest <input_bundles dir>] [--variants hwbringup_v0,hwbringup_nokernel_v0]

    ``--dest`` defaults to ``<descriptor dir>/input_bundles`` (beside the descriptor)."""
    import argparse
    from pathlib import Path

    from .target_experiment import load_target_experiment

    ap = argparse.ArgumentParser(description="Materialize the 4-arm bundle manifests from a descriptor.")
    ap.add_argument("--descriptor", help="path to a target_experiment.yaml (not needed for --list-tools)")
    ap.add_argument("--dest", default=None, help="output input_bundles dir (default: beside the descriptor)")
    ap.add_argument("--variants", default="hwbringup_v0",
                    help="comma-separated bundle-id variant suffixes (default: hwbringup_v0)")
    ap.add_argument("--with-tool", action="append", default=[], metavar="NAME",
                    help=f"ABLATION: grant this tool on top of the arm's rung (repeatable). "
                         f"Known: {', '.join(TR.ablatable_tools())}")
    ap.add_argument("--without-tool", action="append", default=[], metavar="NAME",
                    help="ABLATION: withhold this tool from the arm's rung (repeatable). The bundle id "
                         "gains a suffix naming the cell.")
    ap.add_argument("--arms", default="", help="comma-separated arms to emit (default: the whole ladder)")
    ap.add_argument("--list-tools", action="store_true", help="print the tool catalog and exit")
    a = ap.parse_args(argv)

    if a.list_tools:
        for name in TR.known_tools():
            t = TR.spec(name)
            rungs = [arm for arm, ts in TR.ARM_TOOLS.items() if name in ts]
            how = "broker" if t.broker else "grant"
            flag = "" if t.ablatable else "  [NOT independently ablatable]"
            print(f"{name:22} {how:7} {t.blurb}{flag}\n{'':22} rungs: {', '.join(rungs) or '-'}")
        return 0

    if not a.descriptor:
        ap.error("--descriptor is required (except with --list-tools)")
    te = load_target_experiment(a.descriptor)
    dest = Path(a.dest) if a.dest else Path(a.descriptor).parent / "input_bundles"
    variants = tuple(v.strip() for v in a.variants.split(",") if v.strip())
    written = materialize_bundles(te, dest, variants=variants,
                                  add_tools=tuple(a.with_tool), drop_tools=tuple(a.without_tool),
                                  arms=tuple(x.strip() for x in a.arms.split(",") if x.strip()))
    print(f"materialized {len(written)} bundle manifests under {dest} (target={te.target}):")
    for p in written:
        print(f"  {p.parent.name}/{p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
