"""Read one complete numerical pass followed by continuous session repeats."""

from __future__ import annotations

from .chain_image_timing import _ticks


def decode_repeats(image, console):
    from .chain_image_readback import decode

    repeats = image.get("timing_repeats")
    if (
        image["transport"] != "verified_repeats_v1"
        or type(repeats) is not int
        or not 1 <= repeats <= 10
        or image.get("group_profile", {}).get("enabled", False)
    ):
        raise ValueError(
            "select bounded verified repeats without group profile transport"
        )
    if b"OUT_BIN_" in console or b"CHAIN_PROFILE_" in console:
        raise ValueError("verified repeats cannot mix output transports")
    lines = [
        line
        for line in console.decode("ascii").splitlines()
        if line.startswith("CHAIN_") or line == "DONE"
    ]
    ends = [i for i, line in enumerate(lines) if line.startswith("CHAIN_END ")]
    if len(ends) != 1:
        raise ValueError("verified repeats need exactly one complete numerical pass")
    first = ends[0]
    compact = {**image, "transport": "all_element_comparison_v1"}
    report = decode(compact, ("\n".join(lines[: first + 1]) + "\nDONE\n").encode())
    plan = image["plan"]
    selected = [
        max(i for i, step in enumerate(plan["schedule"]) if step["program"] == p)
        for p in range(len(plan["programs"]))
    ]
    final_schedule = [plan["schedule"][i] for i in selected]
    final_reference = {
        **image["reference"],
        "offsets": [image["reference"]["offsets"][i] for i in selected],
    }
    final_count = sum(
        plan["programs"][s["program"]]["n_outputs"] for s in final_schedule
    )
    final_image = {
        **compact,
        "plan": {**plan, "schedule": final_schedule},
        "reference": final_reference,
        "chain_steps": len(final_schedule),
    }
    at = first + 1
    measured = []
    for ordinal in range(repeats):
        if at >= len(lines):
            raise ValueError("truncated verified timing repeat")
        row = lines[at].split()
        at += 1
        if (
            len(row) != 6
            or row[:2] != ["CHAIN_REPEAT", "v1"]
            or _ticks(row[2]) != ordinal
        ):
            raise ValueError("missing, duplicate or reordered verified repeat")
        total, forwards, errors = map(_ticks, row[3:])
        if not total or forwards > total:
            raise ValueError("repeat interval must enclose all forward intervals")
        final_lines = []
        failures = 0
        for step in final_schedule:
            final_lines.append(f"CHAIN_STEP v1 {step['program']} {step['step']} 0 0")
            for _ in range(plan["programs"][step["program"]]["n_outputs"]):
                if at >= len(lines):
                    raise ValueError("missing timed repeat final-state comparison")
                line = lines[at]
                fields = line.split()
                if len(fields) < 13 or fields[:2] != ["CHAIN_ARRAY", "v1"]:
                    raise ValueError("expected timed repeat final-state comparison")
                failures += _ticks(fields[6])
                final_lines.append(line)
                at += 1
        if at >= len(lines) or lines[at].split() != [
            "CHAIN_REPEAT_END",
            "v1",
            str(ordinal),
            str(final_count),
            str(failures),
        ]:
            raise ValueError("repeat final-state terminal counts differ")
        at += 1
        final_lines += [
            f"CHAIN_END v1 {len(final_schedule)} {final_count} {failures}",
            "DONE",
        ]
        final_report = decode(final_image, ("\n".join(final_lines) + "\n").encode())
        measured.append(
            {
                "repeat": ordinal,
                "whole_chain": total,
                "forward_sum": forwards,
                "session_overhead": total - forwards,
                "runtime_errors": errors,
                "final_state_comparisons": final_report["comparisons"],
                "final_state_numerical_status": final_report["numerical_status"],
            }
        )
    if lines[at:] != ["DONE"]:
        raise ValueError("extra or missing verified-repeat terminal")
    report.update(
        transport=image["transport"],
        timing_repeats=measured,
        timing_scope="Continuous complete sessions including input/state copies, arena resets, descriptors, forwards and scalar counter bookkeeping. Comparisons and console transport excluded.",
        correctness_scope="Initial pass checks every declared output at each step. Every timed repeat checks the last declared outputs of every program after its timer stops. Intermediate repeat outputs are not independently captured.",
        physical_cycles_verified=False,
    )
    if any(row["runtime_errors"] for row in measured):
        report["runtime_status"] = "fail"
    if any(row["final_state_numerical_status"] != "pass" for row in measured):
        report["numerical_status"] = "fail"
    return report
