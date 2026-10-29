"""Proof author for bounded concrete execution; every local step is kernel checked.

Python supplies literal trace data, never acceptance. The emitted equalities use
GraphStateScalarSteps against the full operational evaluator, including loop
guards, every source instruction and complete success/failure observations.
"""

from __future__ import annotations

import json

from .lean_graph_state_scalar import indices, integer, primitive, program, types
from .lean_graph_state_scalar_models import evaluate, ints


def outcome(result):
    status, values, world = result
    if status == "ok":
        return ".ok (" + ints(values) + "," + ints(world) + ")"
    if status == "error":
        return ".error (.external " + integer(values) + "," + ints(world) + ")"
    raise ValueError("actual typed execution trace contains an internal arity fault")


class Author:
    def __init__(self, registry, failure):
        self.registry, self.failure = registry, failure
        self.definitions, self.proofs, self.code_names = [], [], {}
        self.count = 0

    def fresh(self, label):
        self.count += 1
        if self.count > 2000:
            raise ValueError("bounded concrete execution witness budget exceeded")
        return label + str(self.count)

    def code(self, records):
        identity = json.dumps(records, sort_keys=True)
        if identity not in self.code_names:
            name = self.fresh("code")
            self.code_names[identity] = name
            self.definitions.append("def " + name + " : Program := " + program(records) + "\n")
        return self.code_names[identity]

    def theorem(self, name, statement, proof):
        self.proofs.append("theorem " + name + " : " + statement + " := by\n  " + proof + "\n")

    def execute(self, records, environment, world):
        name, code = self.fresh("execution"), self.code(records)
        expected = evaluate(records, environment, self.registry, world, self.failure)
        statement = (
            "evalProgram concrete " + code + " " + ints(environment) + " " + ints(world) + " = " + outcome(expected)
        )
        head = records[0]
        if "ret" in head:
            self.theorem(name, statement, "rfl")
            return name, expected
        tail = self.code(records[1:])
        if "pure" in head:
            _, values, current = evaluate([head, {"ret": [0]}], environment, self.registry, world, self.failure)
            child, _ = self.execute(records[1:], [values[0], *environment], current)
            step = (
                "pure_execution concrete "
                + primitive(head["pure"])
                + " "
                + indices(head["arguments"])
                + " "
                + tail
                + " "
                + ints(environment)
                + " "
                + ints(world)
                + " "
                + integer(values[0])
                + " (by rfl)"
            )
            self.theorem(name, statement, "exact (" + step + ").trans " + child)
        elif "effect" in head:
            count = len(head["outputs"])
            result = evaluate([head, {"ret": list(range(count))}], environment, self.registry, world, self.failure)
            prefix = (
                "concrete "
                + integer(head["effect"], natural=True)
                + " "
                + indices(head["arguments"])
                + " "
                + types(head["outputs"])
                + " "
                + tail
                + " "
                + ints(environment)
                + " "
                + ints(world)
            )
            if result[0] == "error":
                step = (
                    "effect_failed "
                    + prefix
                    + " (.external "
                    + integer(result[1])
                    + ","
                    + ints(result[2])
                    + ") (by rfl)"
                )
                self.theorem(name, statement, "exact " + step)
            else:
                child, _ = self.execute(records[1:], [*result[1], *environment], result[2])
                step = "effect_execution " + prefix + " " + ints(result[1]) + " " + ints(result[2]) + " (by rfl)"
                self.theorem(name, statement, "exact (" + step + ").trans " + child)
        else:
            loop = head["loop"]
            body_code, body_fn = self.code(loop["body"]), self.fresh("body")
            self.definitions.append(
                "def " + body_fn + " (index : Int) (carry current : List Int) : Outcome Int (List Int) Int :=\n"
                "  evalProgram concrete "
                + body_code
                + " (concrete.index index::carry++"
                + ints(environment)
                + ") current\n"
            )
            lb, ub, stride = [environment[index] for index in loop["bounds"]]
            iterations = list(range(lb, ub, stride)) if stride > 0 else []
            if len(iterations) > 8:
                raise ValueError(
                    "concrete trace domain supports at most eight loop iterations; generic theorem is unbounded"
                )
            initial = [environment[index] for index in loop["initial"]]
            folded, result = self.fold(loop, body_fn, iterations, initial, world, environment)
            prefix = (
                "concrete "
                + " ".join(integer(v, natural=True) for v in loop["bounds"])
                + " "
                + indices(loop["initial"])
                + " "
                + types(loop["carries"])
                + " "
                + body_code
                + " "
                + tail
                + " "
                + ints(environment)
                + " "
                + ints(world)
            )
            if result[0] == "error":
                step = (
                    "loop_failed "
                    + prefix
                    + " (.external "
                    + integer(result[1])
                    + ","
                    + ints(result[2])
                    + ") "
                    + folded
                )
                self.theorem(name, statement, "exact " + step)
            else:
                child, _ = self.execute(records[1:], [*result[1], *environment], result[2])
                step = "loop_execution " + prefix + " " + ints(result[1]) + " " + ints(result[2]) + " " + folded
                self.theorem(name, statement, "exact (" + step + ").trans " + child)
        return name, expected

    def fold(self, loop, body, iterations, values, world, environment):
        name = self.fresh("fold")
        prefix = "loopEval " + str(len(loop["carries"])) + " " + body + " "
        if not iterations:
            expected = "ok", values, world
            statement = prefix + "[] " + ints(values) + " " + ints(world) + " = " + outcome(expected)
            self.theorem(name, statement, "exact loop_done _ _ _ _ (by decide)")
            return name, expected
        index = iterations[0]
        body_proof, result = self.execute(loop["body"], [index, *values, *environment], world)
        statement = prefix + ints(iterations) + " " + ints(values) + " " + ints(world) + " = "
        common = (
            str(len(loop["carries"]))
            + " "
            + body
            + " "
            + integer(index)
            + " "
            + ints(iterations[1:])
            + " "
            + ints(values)
            + " "
            + ints(world)
        )
        if result[0] == "error":
            step = (
                "loop_step_failed "
                + common
                + " (.external "
                + integer(result[1])
                + ","
                + ints(result[2])
                + ") (by decide) "
                + body_proof
            )
            self.theorem(name, statement + outcome(result), "exact " + step)
            return name, result
        child, final = self.fold(loop, body, iterations[1:], result[1], result[2], environment)
        step = "loop_step " + common + " " + ints(result[1]) + " " + ints(result[2]) + " (by decide) " + body_proof
        self.theorem(name, statement + outcome(final), "exact (" + step + ").trans " + child)
        return name, final


def emit_execution(function, registry, arguments, failure):
    author = Author(registry, failure)
    proof, expected = author.execute(function["before"], arguments, [23])
    text = "namespace Execution\n" + "".join(author.definitions) + "".join(author.proofs) + "end Execution\n"
    text += (
        "theorem actual_original_execution : evalProgram concrete F0.before arguments [23] = "
        + outcome(expected)
        + " := by\n"
        "  have program_join : F0.before = Execution." + author.code(function["before"]) + " := rfl\n"
        "  rw [program_join]\n  exact Execution." + proof + "\n"
    )
    return text
