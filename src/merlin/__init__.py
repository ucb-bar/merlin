"""merlin: compiler-centered HW/SW abstraction exploration.

Core provides compiler dialects/lowering, capture and workload contracts, target package
invocation, runtime/verification primitives, and shared infrastructure. Optional research
distributions contribute the historical DSE, mining, analysis and evaluation namespaces.
Experiment definitions live in the root catalog; generated state belongs under out/.
"""

# Optional distributions contribute research modules under their historical names.
# One extended package path preserves class/function identity and avoids duplicate
# implementations or process-global import hooks during the packaging migration.
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
