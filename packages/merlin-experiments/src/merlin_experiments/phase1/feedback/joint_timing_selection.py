"""Select an immutable timing epoch; covered tuned models never fall back."""


def choose(selection, capsule_sha256):
    """Exact capsule coverage selects the newest declared tuned epoch.

    Dynamic feature coverage is checked after observation. A mismatch or failed
    observation stays UNKNOWN on the selected epoch; retrying an older or untuned
    model would silently change the performance objective.
    """
    epochs = selection.get("timing_epochs", [])
    if not isinstance(epochs, list) or len(epochs) > 64:
        raise ValueError("joint timing epochs need a bounded explicit roster")
    found = []
    seen = set()
    for row in epochs:
        number = row.get("epoch")
        coverage = row.get("coverage") or {}
        if (type(number) is not int or number < 1 or number in seen
                or row.get("mode") != "tuned" or not row.get("runtime")
                or not row.get("calibration_evidence")
                or not coverage.get("scope_sha256") or not coverage.get("memory_regime_sha256")
                or not isinstance(coverage.get("capsule_sha256"), list)
                or not isinstance(coverage.get("feature_ranges"), dict)):
            raise ValueError("tuned epoch lacks explicit runtime, evidence or coverage")
        seen.add(number)
        if capsule_sha256 in coverage["capsule_sha256"]:
            found.append(row)
    if found:
        return max(found, key=lambda row: row["epoch"])
    return dict(epoch=0, mode="untuned", runtime=selection.get("runtime"), coverage=None)


def within_coverage(epoch, report):
    if epoch["mode"] == "untuned":
        return True
    coverage = epoch["coverage"]
    if (report.get("scope_sha256") != coverage["scope_sha256"]
            or report.get("domain", {}).get("memory_regime_sha256") != coverage["memory_regime_sha256"]):
        return False
    counts = report.get("feature_counts") or {}
    for name, interval in coverage["feature_ranges"].items():
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(v) not in (int, float) for v in interval)
                or interval[0] < 0 or interval[1] < interval[0]):
            raise ValueError("tuned model feature coverage is malformed")
        value = counts.get(name)
        if type(value) not in (int, float) or not interval[0] <= value <= interval[1]:
            return False
    return True
