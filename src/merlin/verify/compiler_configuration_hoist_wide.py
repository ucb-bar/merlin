"""Current-only wide extent profile selected by the original checked C consumer.

Default and historical parser limits stay unchanged. The existing Focus theorem
quantifies over the complete count; no dynamic execution supplies authority.
"""

from merlin.compile.scheduling.configuration_c_consumer import ConfigurationCConsumer
from merlin.sched.ir import Kernel, parse_kernel

from .compiler_configuration_hoist_multi import project
from .compiler_configuration_kernel import tree
from .configuration_hoist_bridge import Unsupported


def project_current(source, final, *, isa, effects, c_consumer):
    if type(c_consumer) is not ConfigurationCConsumer:
        raise Unsupported("original typed index64 C consumer required for wide extents")
    c_consumer.revalidate()
    kernels = [parse_kernel(value) for value in (source, final)]
    for text, kernel in zip((source, final), kernels, strict=True):
        if type(kernel) is not Kernel or kernel.text() != text:
            raise Unsupported("exact actual Kernel source/output required")
        c_consumer.admit(kernel)
    if not any(getattr(statement, "extent", 0) > 1_000_000 for kernel in kernels for statement in kernel.body):
        raise Unsupported("wide profile requires an actual extent beyond the legacy guard")
    return project(
        *(tree(kernel) for kernel in kernels),
        isa=isa,
        effects=effects,
        allow_single=True,
        extent_limit=(1 << (c_consumer.index_bits - 1)) - 1,
    )
