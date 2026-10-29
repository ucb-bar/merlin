"""Bound host lane/proof expansion before constructing repeated symbolic terms.

These are resource limits. They do not impose a separate semantic factor cap:
large factors remain usable when the actual selected lane count is small.
"""

from .loop_transform_intake import Unavailable

_MAX_WORK = 65536
_MAX_VALUE_CHARACTERS = 262144
_MAX_TOTAL_CHARACTERS = 4 * 1024 * 1024


def check_lanes(capture, body, *, repetitions, environment, carry, continuation=()):
    """Charge finite syntax, proof indentation, and carried expression growth."""
    from .lean_graph_state_scalar import primitive

    if type(repetitions) is not int or repetitions < 0:
        raise ValueError("nonnegative concrete lane count required")
    controls = {row["identity"]: row for row in capture["index_operations"]}
    base = 1 + len(body) + len(environment) + len(carry)
    if repetitions > _MAX_WORK // base:
        raise Unavailable("compiler loop lane expansion work resource budget exceeded")

    def event_sizes(records):
        return [
            len(row["loop"]["carries"]) if "loop" in row else len(row["outputs"])
            for row in records
            if "loop" in row or "effect" in row and row["effect"] not in controls
        ]

    events, suffix_events = event_sizes(body), event_sizes(continuation)
    event_count = repetitions * len(events) + len(suffix_events)
    maximum_arity = max([0, *(events if repetitions else []), *suffix_events])
    if event_count * (event_count + 1) * (1 + maximum_arity) > _MAX_WORK:
        raise Unavailable("compiler loop proof case expansion resource budget exceeded")
    outer = [max(1, len(value)) for value in environment]
    values = [max(1, len(value)) for value in carry]
    work = 0

    def simulate(records, frame):
        nonlocal work
        frame = list(frame)
        for row in records:
            if "ret" in row:
                result = [frame[index] for index in row["ret"]]
                work += sum(result)
                if work > _MAX_TOTAL_CHARACTERS:
                    raise Unavailable("compiler loop symbolic expression resource budget exceeded")
                return result
            if "loop" in row:
                work += 4 * sum(frame)
                frame = [16] * len(row["loop"]["carries"]) + frame
            else:
                arguments = [frame[index] for index in row["arguments"]]
                if "pure" in row:
                    size = 16 + len(primitive(row["pure"])) + sum(arguments) + len(arguments)
                    if size > _MAX_VALUE_CHARACTERS:
                        raise Unavailable("compiler loop carried expression resource budget exceeded")
                    frame.insert(0, size)
                    work += size
                elif row["effect"] in controls:
                    control = controls[row["effect"]]
                    size = (
                        12 + len(str(control["value"])) if control["kind"] == "index-constant" else 16 + sum(arguments)
                    )
                    if size > _MAX_VALUE_CHARACTERS:
                        raise Unavailable("compiler loop index expression resource budget exceeded")
                    frame.insert(0, size)
                    work += size
                else:
                    work += 4 * sum(arguments)
                    frame = [16] * len(row["outputs"]) + frame
            if work > _MAX_TOTAL_CHARACTERS:
                raise Unavailable("compiler loop symbolic expression resource budget exceeded")
        raise ValueError("complete source lane terminator required")

    # repetitions is already bounded above by finite actual-syntax work.
    for _lane in range(repetitions):
        values = simulate(body, [32, *values, *outer])
    if continuation:
        simulate(continuation, [*values, *outer])
