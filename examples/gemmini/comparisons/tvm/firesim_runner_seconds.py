"""Use the supported Runner API to report seconds while retaining raw cycles.

The frequency belongs to the target cycle counter. FPGA host frequency and
wall-clock emulation throughput must not be substituted for it.
"""

import math

from tvm import meta_schedule as ms
from tvm.meta_schedule.utils import derived_object


def cycles_to_seconds(cycles, clock_hz):
    if not math.isfinite(clock_hz) or clock_hz <= 0:
        raise ValueError("Require the documented target cycle-counter frequency")
    return float(cycles) / float(clock_hz)


@derived_object
class SecondsFuture(ms.runner.PyRunnerFuture):
    def __init__(self, future, clock_hz):
        self.future = future
        self.clock_hz = clock_hz

    def done(self):
        return self.future.done()

    def result(self):
        result = self.future.result()
        values = [cycles_to_seconds(float(v), self.clock_hz) for v in result.run_secs] if result.run_secs else None
        return ms.runner.RunnerResult(values, result.error_msg)


@derived_object
class SecondsRunner(ms.runner.PyRunner):
    def __init__(self, runner, clock_hz):
        cycles_to_seconds(1, clock_hz)
        self.runner = runner
        self.clock_hz = clock_hz

    def run(self, inputs):
        return [SecondsFuture(future, self.clock_hz) for future in self.runner.run(inputs)]
