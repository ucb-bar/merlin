"""Read-only functional observations before the FP8 producer's clamp."""
from contextlib import contextmanager
import math


class _ObservedInstruction:
    def __init__(self, instruction, observe):
        self.instruction, self.observe = instruction, observe

    def __getattr__(self, name):
        return getattr(self.instruction, name)

    def exec(self, state):
        self.observe(state)
        return self.instruction.exec(state)


class FP8Stats:
    def __init__(self, scale, expected_values):
        if type(scale) is not int or not -126 <= scale <= 119 or type(expected_values) is not int or expected_values <= 0:
            raise ValueError('valid captured scale and physical FP8 value count required')
        self.scale, self.expected_values = scale, expected_values
        self.values = self.saturated = self.nonfinite = self.nan = 0
        self.maximum = 0.0

    def observe(self, state):
        import torch
        # Copies are used for arithmetic; no model register, numerical factory,
        # or primitive is changed by the observation.
        low = state.read_mrf_u8(2).view(torch.bfloat16).float()
        high = state.read_mrf_u8(3).view(torch.bfloat16).float()
        bound = state.read_mrf_u8(8).view(torch.bfloat16).float()
        limit = 448.0 * 2.0**self.scale
        if not bool(torch.all(bound == -limit)):
            raise ValueError('observed maximum is not the expected first FP8 clamp')
        values = torch.cat((low.reshape(-1), high.reshape(-1)))
        finite = torch.isfinite(values)
        self.values += values.numel()
        self.nonfinite += int((~finite).sum())
        self.nan += int(torch.isnan(values).sum())
        self.saturated += int((finite & (values.abs() > limit)).sum())
        if bool(finite.any()):
            self.maximum = max(self.maximum, float(values[finite].abs().max()))

    def report(self, output):
        if self.values != self.expected_values or not math.isfinite(self.maximum):
            raise ValueError('incomplete producer preclip observation')
        return dict(scale_exponent=self.scale, physical_values=self.values, preclip_saturation_count=self.saturated,
                    preclip_nonfinite_count=self.nonfinite, preclip_nan_count=self.nan, preclip_max_abs=self.maximum,
                    emitted_nan_count=sum(value in (0x7f, 0xff) for value in output),
                    method='read-only register observation before each first FP8 maximum clamp')


@contextmanager
def observe_fp8(unit, scale, expected_values):
    """Wrap only clamp instructions, preserving original assembled bytes and exec."""
    stats = FP8Stats(scale, expected_values)
    original = unit.assemble
    def assemble(source):
        program, digest = original(source)
        for index, insn in enumerate(program.instructions):
            if (insn.mnemonic == 'vmaximum.bf16' and int(insn.vd) == 2
                    and int(insn.vs1) == 2 and int(insn.vs2) == 8):
                program.instructions[index] = _ObservedInstruction(insn, stats.observe)
        return program, digest
    unit.assemble = assemble
    try:
        yield stats
    finally:
        unit.assemble = original
