"""The human-readable ``coverage/README.md`` beside a Phase 0 evidence export."""

from __future__ import annotations


def render(accounting: dict, quantization: dict) -> bytes:
    """Human navigation generated from the same JSON views, never a second census."""

    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")

    overall, universe = accounting["overall"], accounting["framework_universe"]
    operation_count = overall["n_mlir_operations"] if overall["n_mlir_operations"] is not None else "unavailable"
    registry_count = (
        universe["n_registered_aten_operators"]
        if universe["n_registered_aten_operators"] is not None
        else "unavailable"
    )
    inventory_identity = accounting.get("selected_inventory") or {}
    lines = [
        "# Phase 0 operation and quantization accounting",
        "",
        "Generated diagnostic views; no compiler lowering or TorchAO realization is certified.",
        "",
        f"Inventory status: **{accounting['status']}**. Normalized MLIR operations: **{operation_count}**.",
        f"Inventory binding: **{inventory_identity.get('status', 'not_available')}**; "
        f"declared roster matches: **{inventory_identity.get('declared_roster_matches', 'unknown')}**.",
        f"Selected framework catalog: **{universe['status']}**; registered ATen overloads: **{registry_count}**.",
        "",
        "Frontend counts are static captured call sites, not dynamic execution frequencies.",
        "Missing source traces remain unknown; repeated lowering provenance is never counted as a frontend call.",
        "",
        "## Combined normalized-IR split",
        "",
        "| Partition | Operations |",
        "| --- | ---: |",
    ]
    lines += [f"| {cell(name)} | {count} |" for name, count in overall["classification_counts"].items()]
    operation_breakdown: dict[tuple[str, str], int] = {}
    for application in accounting["applications"].values():
        for signature in application["signatures"]:
            key = (signature["classification"], signature["observed_signature"]["mlir_operation"])
            operation_breakdown[key] = operation_breakdown.get(key, 0) + signature["count"]
    lines += [
        "",
        "## Normalized-IR operations by partition",
        "",
        "These are static occurrences from the same digest-bound accounting, not new support claims.",
        "",
        "| Partition | MLIR operation | Occurrences |",
        "| --- | --- | ---: |",
    ]
    for (partition, operation), count in sorted(
        operation_breakdown.items(), key=lambda item: (item[0][0], -item[1], item[0][1])
    ):
        lines.append(f"| {cell(partition)} | {cell(operation)} | {count} |")
    lines += [
        "",
        "## Selected hardware declaration screen",
        "",
        "This separate split exposes hardware-declared admission even when SW review is unresolved.",
        "It is not executed compiler lowering or hardware qualification.",
        "",
        "| Admission | Operations |",
        "| --- | ---: |",
    ]
    lines += [f"| {cell(name)} | {count} |" for name, count in overall["hardware_admission_counts"].items()]
    lines += [
        "",
        "## Independent host and accelerator support",
        "",
        "These declaration screens are independent; unreviewed inputs remain unknown.",
        "Neither screen implies actual dispatch or execution.",
        "A host placement request without a matching pinned host capability remains unknown.",
        "",
        "| Support partition | Operations |",
        "| --- | ---: |",
    ]
    lines += [f"| {cell(name)} | {count} |" for name, count in overall.get("support_partition_counts", {}).items()]
    lines += [
        "",
        "## Per-application split",
        "",
        "| Application | Role / scope | Original calls | Quantized calls | Prepared calls | "
        "MLIR ops | Source correspondence |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, application in accounting["applications"].items():
        source = (application.get("completeness") or {}).get("source_trace") or {}
        identity = application.get("workload_identity") or {}
        counts = [source.get(f"{stage}_invocation_count") for stage in ("original", "quantized", "prepared")]
        rendered = " | ".join("unknown" if value is None else str(value) for value in counts)
        scope = f"{identity.get('workload_role', 'unknown')} / {identity.get('coverage_scope', 'unknown')}"
        lines.append(
            f"| {cell(name)} | {cell(scope)} | {rendered} | {application['n_mlir_operations']} | "
            f"{cell(source.get('status', 'unknown'))} |"
        )
    lines += [
        "",
        "## Precision and typed-edge obligations",
        "",
        "`operation-accounting.json` retains ordered storage, compute, accumulator and output types",
        "for each exact operation ordinal, plus independent host/accelerator signature decisions.",
        "`completeness.transfer_obligations` lists typed SSA edges with conditional lane crossings;",
        "no crossing is claimed until placement is selected and a reviewed transfer contract matches.",
        "Inspect [frontend/index.json](../software/frontend/index.json) for exact trace and SSA graph files.",
        "Unknown lineage, precision, capability, transfer or exact capsule coverage blocks verified",
        "whole-workload admission. Unit tests and representative subsets do not certify a headline model.",
    ]
    native = {
        label: app["native_baseline_observation"]
        for label, app in accounting["applications"].items()
        if "native_baseline_observation" in app
    }
    if native:
        lines += [
            "",
            "## Selected finite native baselines",
            "",
            "These complete-program CPU checks do not change RVV host or accelerator admission.",
            "",
            "| Application | Input precisions | Maximum absolute error | Target executed |",
            "| --- | --- | ---: | --- |",
        ]
        for label, observation in native.items():
            precision = ", ".join(sorted({item["dtype"] for item in observation["input_abi"]}))
            lines.append(f"| {cell(label)} | {cell(precision)} | {observation['max_absolute_error']} | false |")
        lines += [
            "",
            "Inspect [native-baseline-observations.json](../software/native-baseline-observations.json)",
            "for exact ABIs, compiler identities, policies and links to byte-identical frozen receipts and outputs.",
            "Agreement is scoped to the saved inputs; individual-operation execution is not independently traced.",
        ]
    lines += [
        "",
        "## Format and operation decisions",
        "",
        "| Format | Hardware status | Operation decisions |",
        "| --- | --- | --- |",
    ]
    for row in quantization["formats"]:
        decisions = ", ".join(f"{entry['operation_id']}: {entry['status']}" for entry in row["operation_eligibility"])
        lines.append(f"| {cell(row['id'])} | {cell(row['status'])} | {cell(decisions)} |")
    lines += [
        "",
        "Inspect [operation-accounting.json](operation-accounting.json) for exact signature ordinals,",
        "per-application provenance groups, registry sets and unobserved declarations.",
        "Inspect [performance-basis.json](performance-basis.json) for static capture-bound operation mass",
        "and explicit unknowns; it makes no timing or speedup claim.",
        "Inspect [quantization-contract.json](../software/quantization-contract.json) for parameters,",
        "format-specific recipes, conflicts and reasons each operation is unknown or ineligible.",
        "The selected detailed inventory is copied as `application-inventory.json` when available.",
        "",
    ]
    return "\n".join(lines).encode()
