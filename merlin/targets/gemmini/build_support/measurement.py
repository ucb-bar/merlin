"""Pure final assembly of the existing measurement fragments; no counter discovery."""
from .format import CodegenError


def assemble_measurement_fragments(warmup_work: str, *, requested: str = 'warm',
                                  include: str = '', prologue: str = '',
                                  epilogue: str = '') -> dict:
    """Preserve the original window. A predecessor is not proof of observed cache state.

    The pure build-only default is one unmeasured predecessor and one measured call,
    with cycle-only instrumentation. The legacy backend explicitly passes its current
    requested cache protocol and discovered counter fragments, preserving old defaults.
    """
    if requested not in ('cold', 'warm'):
        raise CodegenError(f"unsupported cache-state measurement condition {requested!r}")
    warmup = ''
    if requested == 'warm':
        warmup = warmup_work.rstrip() + '\n  // merlin: warmup completed outside the measured/counter window.\n'
    return {'include': include, 'prologue': prologue, 'epilogue': epilogue, 'warmup': warmup,
            'cache_state': 'unknown', 'cache_state_observed': False,
            'cache_protocol': ('one_unmeasured_predecessor' if requested == 'warm'
                               else 'fresh_elf_process'),
            'requested_cache_condition': requested}
