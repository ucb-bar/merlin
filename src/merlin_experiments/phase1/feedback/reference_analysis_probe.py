"""One real reference-analysis lift before a run that enables reference guidance.

The broker serves reference analysis from the campaign environment. A missing
optional dependency there (``merlin.mining`` for the experiment owner) turned
every author call of a whole fresh10 run into ``analysis_dependency_unavailable``
while startup checks passed. This probe sends one ordinary lift through the
broker's own handler with the run's frozen facts and reference grant, so the
same environment refuses before an author starts.
"""

from __future__ import annotations

from pathlib import Path

from . import reference_analysis, reference_bank, reference_experiments

#: No compiler snapshot has this digest. The experiment owner therefore imports
#: its dependencies and checks the binding, but publishes no current-analysis
#: record into the selected experiment store for a probe.
_UNBOUND_COMPILER = "0" * 64
_MAX_CANDIDATES = 16


def _request(operation, path, **fields):
    return dict(action=reference_analysis.ACTION, reference_analysis=dict(operation=operation, path=path, **fields))


def _dependency_gap(response):
    experiment = (response.get("report") or {}).get("experiment") or {}
    return experiment.get("reason") == reference_experiments.DEPENDENCY_UNAVAILABLE


_DEPENDENCY = (
    "reference analysis lacks a dependency in this environment; "
    "check the campaign's Merlin installation (for example 'merlin[mining]')"
)


def _bank_probe(selected, pin):
    """A bank arm: one inventory, one unit lift that must be served, one refused raw lift."""
    listed = reference_analysis.handle(_request("inventory", "reference-kernels", limit=1), **selected)
    if listed.get("reason") in ("cca_bank_dependency_unavailable", "analysis_dependency_unavailable"):
        raise ValueError(_DEPENDENCY)
    if listed.get("status") != "analyzed":
        raise ValueError(f"CCA bank inventory is {listed.get('status')}: {listed.get('reason')}")
    units = listed["report"].get("units") or ()
    if not units:
        raise ValueError("the selected CCA bank lists no admitted unit")
    unit = units[0]["unit_id"]
    lifted = reference_analysis.handle(
        _request("lift", "", entry=unit, limit=1), compiler_sha256=_UNBOUND_COMPILER, **selected
    )
    if lifted.get("reason") in ("cca_bank_dependency_unavailable", "analysis_dependency_unavailable"):
        raise ValueError(_DEPENDENCY)
    if lifted.get("status") != "analyzed":
        raise ValueError(f"CCA bank unit lift is {lifted.get('status')}: {lifted.get('reason')}")
    root = Path(selected["workspace"]) / reference_bank.RAW
    raw = (
        next(
            (path.relative_to(selected["workspace"]).as_posix() for path in sorted(root.rglob("*")) if path.is_file()),
            reference_bank.RAW,
        )
        if root.is_dir()
        else reference_bank.RAW
    )
    refused = reference_analysis.handle(_request("lift", raw), compiler_sha256=_UNBOUND_COMPILER, **selected)
    if refused.get("reason") != reference_bank.OUTSIDE:
        raise ValueError("a raw reference lift was not refused in CCA bank mode: " + str(refused.get("reason")))
    return dict(
        status="analyzed",
        mode="cca_bank",
        operation="lift",
        unit=unit,
        bank_sha256=pin,
        raw_reference=raw,
        raw_reference_refusal=refused["reason"],
    )


def probe(*, workspace, facts=None, bundle_manifest=None, repo=None, target=None, descriptor=None):
    """Lift the smallest visible reference input; None when reference guidance is not selected."""
    selected = dict(
        workspace=Path(workspace),
        facts=facts,
        bundle_manifest=bundle_manifest,
        repo=repo,
        target=target,
        descriptor=descriptor,
    )
    pin = reference_bank.selected_pin()
    if pin is not None:
        return _bank_probe(selected, pin)
    if reference_experiments._current() is None:
        return None
    listed = reference_analysis.handle(_request("inventory", "reference-kernels", limit=_MAX_CANDIDATES), **selected)
    if listed.get("status") != "analyzed":
        raise ValueError(f"reference-analysis inventory is {listed.get('status')}: {listed.get('reason')}")
    files = sorted(
        (row for row in listed["report"].get("files", ()) if isinstance(row.get("path"), str)),
        key=lambda row: (row.get("size_bytes", 0), row["path"]),
    )
    if not files:
        raise ValueError("reference guidance is selected but no visible reference input can be lifted")
    refused = []
    for row in files:
        lifted = reference_analysis.handle(_request("lift", row["path"]), compiler_sha256=_UNBOUND_COMPILER, **selected)
        if lifted.get("reason") == "analysis_dependency_unavailable" or _dependency_gap(lifted):
            raise ValueError(_DEPENDENCY)
        if lifted.get("status") == "analyzed":
            return dict(status="analyzed", operation="lift", input=row["path"], sha256=row.get("sha256"))
        refused.append(f"{row['path']}: {lifted.get('reason')}")
    raise ValueError("no visible reference input could be lifted: " + "; ".join(refused[:4]))


def for_prepared(prepared):
    """The probe with the broker's own selection for this prepared run."""
    from merlin.targetgen.sandbox import bwrap

    if reference_experiments._current() is None and reference_bank.selected_pin() is None:
        return None
    request, bundle = prepared.request, prepared.bundle
    manifest = (
        prepared.run_dir / "input_bundle_manifest.yaml" if bundle.get("compiler_tool_profile") is not None else None
    )
    return probe(
        workspace=prepared.workspace,
        facts=bwrap.frozen_selected_rtl_facts(prepared.workspace, bundle, repo=request.context.repo),
        bundle_manifest=manifest,
        repo=request.context.repo,
        target=getattr(request.context, "target", None),
        descriptor=getattr(request.context, "descriptor", None),
    )
