"""Closed descriptive feature projection; unsupported terms remain None."""

FEATURE_KEYS = (
    "source_macs", "source_reduction_domain_points", "host_integer_arithmetic",
    "host_floating_arithmetic", "host_conversion", "host_load_payload_bytes",
    "host_store_payload_bytes", "host_static_allocation_payload_bytes",
    "representation_directives", "linked_executable_bytes", "linked_readonly_data_bytes",
)


def candidate_features(report):
    """Single-entry host payload is logical activity, never traffic or cycles."""
    features = dict.fromkeys(FEATURE_KEYS)
    source = report.get("source", {})
    if source.get("status") == "observed":
        features["source_macs"] = source.get("macs")
        features["source_reduction_domain_points"] = source.get("reduction_domain_points")
    rows = report.get("host", {}).get("functions", [])
    selected = [row for row in rows if row.get("selected_entry")]
    if not selected and len(rows) == 1:
        selected = rows
    if len(selected) == 1:
        activity = selected[0]["activity"]
        dynamic = activity.get("dynamic_operations")
        if activity.get("status") == "derived" and dynamic is not None:
            for category in ("integer_arithmetic", "floating_arithmetic", "conversion"):
                features["host_" + category] = dynamic.get(category, 0)
            for category in ("load_payload_bytes", "store_payload_bytes", "static_allocation_payload_bytes"):
                features["host_" + category] = activity.get(category)
    representations = report.get("representations", {})
    if "representation_directives" in representations:
        features["representation_directives"] = len(representations["representation_directives"])
    image = report.get("linked_image", {})
    if image.get("status") == "observed":
        features["linked_executable_bytes"] = image["executable_bytes"]
        features["linked_readonly_data_bytes"] = image["readonly_allocated_data_bytes"]
    return {"schema": "merlin.candidate_features.v1", "features": features,
            "scope": "source demand, constant CFG scalar payload, declarations and linked extents",
            "ranking_authority": "none", "physical_traffic": "UNKNOWN", "cycle_prices": "UNKNOWN"}
