"""Read count completeness separately from native event timing qualification.

These are original observer claims under the exact pinned source/domain. This
reader checks their internal coverage and scope; it issues no hardware verdict.
Historical v1 reports retain their original timing-completeness interpretation.
"""

from . import contracts as C

REPORT_V1 = "merlin.annotated_spike_report.v1"
REPORT_V2 = "merlin.annotated_spike_report.v2"
COUNT_SCHEMA = "merlin.spike_feature_observation.v1"


def _counter(value, label):
    if type(value) is not int or not 0 <= value < 2**64:
        raise C.ContractError(label + " must be an unsigned 64-bit observation")
    return value


def feature_complete(report):
    """Validate a fresh count contract; never upgrade an old saved report."""
    if report.get("schema") == REPORT_V1:
        return report.get("timing_complete") is True
    if report.get("schema") != REPORT_V2:
        raise C.ContractError("unsupported native feature observation report")
    observation = C.closed(
        report.get("feature_observation"),
        {
            "schema",
            "complete",
            "scope_instructions",
            "counted_cpu_instructions",
            "scope_kind",
            "window_index",
            "primary_cpu_features",
            "unresolved",
            "qualification",
        },
        "native count observation",
    )
    if observation["schema"] != COUNT_SCHEMA or type(observation["complete"]) is not bool:
        raise C.ContractError("native count observation requires an explicit version and boolean")
    if observation["qualification"] != "not_established":
        raise C.ContractError("count observations cannot declare timing qualification")
    reasons = observation["unresolved"]
    if (
        type(reasons) is not list
        or len(reasons) > 512
        or any(type(s) is not str or not 0 < len(s) <= 2048 for s in reasons)
    ):
        raise C.ContractError("native count reasons exceed their explicit metadata budget")
    names, features = observation["primary_cpu_features"], report.get("feature_counts")
    if (
        type(names) is not list
        or not 0 < len(names) <= 256
        or any(type(name) is not str or not 0 < len(name) <= 256 for name in names)
        or len(set(names)) != len(names)
        or type(features) is not dict
        or not set(names) <= features.keys()
    ):
        raise C.ContractError("native count observation lacks its complete primary class roster")
    total = sum(_counter(features[name], "primary class count") for name in names)
    counted = _counter(observation["counted_cpu_instructions"], "counted instructions")
    scoped = _counter(observation["scope_instructions"], "scope instructions")
    whole = _counter(report.get("instructions"), "whole-trace instructions")
    if total != counted or counted > whole or scoped > whole:
        raise C.ContractError("native primary counts contradict their whole-trace observations")
    kind, index = observation["scope_kind"], observation["window_index"]
    if kind == "whole_program":
        if index is not None or scoped != whole:
            raise C.ContractError("whole-program count scope differs from its observed trace")
    elif kind in {"counter_pair", "counter_pc_pair"}:
        windows = report.get("counter_windows")
        if type(windows) is not list or len(windows) > 65536 or type(index) is not int or not 0 <= index < len(windows):
            raise C.ContractError("native count window was not observed")
        window = windows[index]
        if not isinstance(window, dict) or scoped != _counter(window.get("instructions"), "window instructions"):
            raise C.ContractError("native count scope differs from its original counter window")
        if kind == "counter_pc_pair":
            binding = report.get("scope_binding")
            if (
                not isinstance(binding, dict)
                or index != binding.get("occurrence")
                or any(window.get(key) != binding.get(key) for key in ("begin_pc", "end_pc"))
            ):
                raise C.ContractError("native count window differs from its exact reviewed PCs")
    else:
        raise C.ContractError("unsupported native count scope")
    complete = observation["complete"]
    if complete and (
        reasons
        or total != scoped
        or report.get("execution_completed") is not True
        or report.get("functional_status") != "PASS"
        or type(report.get("raw_functional_exit_code")) is not int
        or report["raw_functional_exit_code"] != 0
    ):
        raise C.ContractError("native count completeness contradicts its execution, coverage or reasons")
    return complete
