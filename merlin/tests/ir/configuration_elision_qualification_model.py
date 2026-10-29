"""Finite differential tests of one explicit synthetic configuration meaning.

This independent Python interpreter exercises retained observations and failure
worlds. It is neither a universal proof nor a native/hardware implementation.
It consumes the retained, separately codec-checked instruction projections.
"""


def execute(operations, initial, *, fail_observation=None, hidden_set_pulse=False):
    state = dict(initial)
    trace = []
    observations = 0
    for row in operations:
        name = row.get("instruction")
        if name is None:
            if row["operation"] in {"arith.constant", "func.return"}:
                continue
            # Conservatively exercise an arbitrary retained opaque transition.
            trace.append(("opaque", tuple(sorted(state.items()))))
            state = {key: value * 3 + 17 for key, value in state.items()}
            continue
        args = row["bindings"]
        field = "configuration:" + str(args["slot"])
        if name == "set":
            state[field] = args["a"]
        elif name == "pair":
            state["configuration:0"], state["configuration:1"] = args["a"], args["b"]
        elif name == "work":
            trace.append(("work", tuple(sorted(state.items()))))
            if observations == fail_observation:
                return ("error", "work-stop", tuple(sorted(state.items())), tuple(trace))
            observations += 1
        elif name == "reset":
            trace.append(("reset", field))
            state[field] = -1
        elif name == "pulse":
            trace.append(("pulse", tuple(sorted(state.items()))))
            state[field] = args["a"]
        elif name == "unordered":
            trace.append(("unordered", tuple(sorted(state.items()))))
            state[field] = args["a"]
        elif name == "unknown":
            trace.append(("unknown", tuple(sorted(state.items()))))
            state = {key: value * 5 + 23 for key, value in state.items()}
        else:
            raise ValueError("no declared synthetic instruction interpretation")
        if hidden_set_pulse and name in {"set", "pair"}:
            trace.append(("forbidden-hidden-set-pulse", name))
    return ("ok", tuple(sorted(state.items())), tuple(trace))
