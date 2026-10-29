"""Documentary evidence API for a future reviewed pure checking launch.

No launch, binary load, qualification or original-pin refresh occurs here. Even
complete-looking external observations require independent observer/enforcement
review; their strings/booleans are not used as proof of actual process behavior.
This API never returns an execution permission or a qualified status.
"""

from __future__ import annotations

from merlin.verify.lean_graph_regions_declared_loader_plan import (
    Pins,
    Unavailable,
    absolute_path,
    closed_environment,
    strict_ref,
)

POLICY_OBLIGATIONS = {
    "environment": "replace the entire environment with the recorded mapping",
    "filesystem": "hold all original byte identities and selectors stable during launch",
    "system_preload": "isolate /etc/ld.so.preload; current absence is insufficient",
    "loader_cache": "inhibit the loader cache using a reviewed supported option",
    "default_directories": "deny fallback outside the admitted original closure",
    "hwcaps": "disable or isolate CPU/hwcaps-dependent library alternatives",
    "late_loads": "observe and deny dlopen, audit, filter and plugin loads outside the closure",
    "imports": "bind every source, object, interface and metadata file actually read",
    "observer": "justify observation completeness and enforce no unobserved code loads",
    "platform": "record kernel/ABI/host premises and instruction-fetch/file association",
}

RETAINED_ARTIFACTS = (
    "launch_record",
    "file_access_trace",
    "loaded_mappings",
    "stdout",
    "stderr",
    "import_trace",
)


def evidence_request(plan):
    """A schema/requirements request, not a manufactured observation."""
    if plan.get("schema") != "merlin.graph-regions.offline-pure-loader-plan.v1":
        raise Unavailable("unsupported offline loader-plan schema")
    return {
        "schema": "merlin.graph-regions.pure-launch-evidence-request.v1",
        "status": "UNAVAILABLE",
        "execution_authorized": False,
        "offline_plan_launch_status": plan["launch_status"],
        "proposed_commands": {
            "loader-list": plan["proposed_diagnostic_argv"],
            "pure-lean": plan["proposed_pure_argv"],
        },
        "environment_replacement": plan["proposed_environment"],
        "required_retained_artifacts": list(RETAINED_ARTIFACTS),
        "required_independent_policy_evidence": POLICY_OBLIGATIONS,
        "original_runtime_refs": [node["pin"] for node in plan["static_nodes"].values()],
        "original_pin_refresh_permitted": False,
        "actual_loaded_closure_established": False,
        "missing_original_runtime_pin_means": "UNAVAILABLE; never select current bytes",
        "candidate_sources": "separate explicit source-only inventory, never runtime replacement pins",
        "held_or_failed_profiles": "remain immutable and UNQUALIFIED; no replay under this request",
    }


def inspect_observation(plan, pins: Pins, observation, candidate_source_refs=()):
    """Check an external form against ORIGINAL identities, without trusting it.

    Retained artifact refs are documentary claims, not runtime expected pins.
    This function does not read/execute those artifacts or claim the observer was
    complete. A well-formed form still has status UNAVAILABLE awaiting review.
    """
    request = evidence_request(plan)
    issues = []
    if type(observation) is not dict:
        raise Unavailable("external observation must be an explicit document")
    if observation.get("schema") != "merlin.graph-regions.pure-launch-observation.v1":
        raise Unavailable("unsupported pure-launch observation schema")
    launch = observation.get("launch")
    if type(launch) is not dict:
        raise Unavailable("exact external launch record is missing")
    mode = launch.get("mode")
    if mode not in ("loader-list", "pure-lean"):
        raise Unavailable("unknown external launch mode")
    proposed = request["proposed_commands"].get(mode)
    if proposed is None or launch.get("argv") != proposed:
        issues.append("external argv does not match the exact unexecuted proposal")
    if closed_environment(launch.get("environment")) != request["environment_replacement"]:
        issues.append("external environment differs from proposed replacement mapping")
    absolute_path(launch.get("cwd"))

    candidates = {}
    for source in candidate_source_refs:
        source = strict_ref(source)
        if source["path"] in pins.files:
            raise Unavailable("candidate source cannot replace an ORIGINAL runtime/import pin")
        if source["path"] in candidates and candidates[source["path"]] != source:
            raise Unavailable("conflicting candidate source refs")
        candidates[source["path"]] = source

    observed = {}
    for category in ("loaded_files", "imported_files"):
        values = observation.get(category)
        if type(values) is not list:
            raise Unavailable("complete explicit observation list required: " + category)
        seen = set()
        for reference in values:
            reference = strict_ref(reference)
            path = reference["path"]
            if path in seen:
                issues.append("duplicate observation in " + category + ": " + path)
            seen.add(path)
            # Executable/shared objects can ONLY use ORIGINAL runtime pins.
            expected = pins.files.get(path)
            if expected is None and category == "imported_files":
                expected = candidates.get(path)
            if reference != expected:
                issues.append("unpinned or substituted " + category + ": " + path)
        observed[category] = seen
    missing = set(plan["static_nodes"]) - observed["loaded_files"]
    if missing:
        issues.append("static runtime nodes missing from observation: " + ",".join(sorted(missing)))
    extra = observed["loaded_files"] - set(plan["static_nodes"])
    if extra:
        issues.append(
            "unexpected runtime loads require a separately reviewed original plan: " + ",".join(sorted(extra))
        )
    if plan["issues"]:
        issues.append("offline plan still contains unresolved original-pin/search issues")
    if mode == "pure-lean" and "<candidate-source.lean>" in (proposed or []):
        issues.append("pure invocation is a template until exact source/import/output bindings exist")
    if not set(candidates) <= observed["imported_files"]:
        issues.append("candidate source inventory not fully represented in observed imports")

    retained = observation.get("retained_artifacts")
    if type(retained) is not dict or set(retained) != set(RETAINED_ARTIFACTS):
        raise Unavailable("complete separate documentary artifact references required")
    for reference in retained.values():
        strict_ref(reference)
    policy = observation.get("policy_evidence")
    if type(policy) is not dict or set(policy) != set(POLICY_OBLIGATIONS):
        raise Unavailable("every open enforcement/observer/platform obligation requires evidence")
    for reference in policy.values():
        strict_ref(reference)

    return {
        "schema": "merlin.graph-regions.pure-observation-form-inspection.v1",
        "status": "UNAVAILABLE",
        "structural_form_status": "REFUSED" if issues else "FORM_ONLY_MATCHES; INDEPENDENT_REVIEW_REQUIRED",
        "issues": issues,
        "execution_authorized": False,
        "actual_loaded_closure_established": False,
        "artifact_contents_and_observer_completeness_verified": False,
        "native_or_lean_execution_performed": False,
        "original_expected_hashes_refreshed": False,
        "independent_review_obligations": POLICY_OBLIGATIONS,
    }
