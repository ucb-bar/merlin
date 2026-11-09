from merlin.compiler.schedule.scf_commands import emit_scf_commands


class ExecuteIndices:
    """Small test-only control interpreter; no library tensor implementation."""

    def op(self, name, a, b):
        return a + b if name == "add" else a * b

    def cmp(self, name, a, b):
        assert name == "eq"
        return a == b

    def select(self, test, a, b):
        return a if test else b

    def loop(self, low, high, step, body):
        for index in range(low, high, step):
            body(index)


def record(module):
    calls = []
    emit_scf_commands(module, ExecuteIndices(), lambda name, args, meta: calls.append((name, args)))
    return calls
