"""merlin: compiler-centered HW/SW abstraction exploration.

Merlin owns compiler dialects/lowering, capture and workload contracts, target package
invocation, runtime/verification primitives, and the research workflows. Optional dependency
extras enable DSE, mining, analysis and evaluation under their historical import names.
Experiment definitions live in the root catalog; generated state belongs under out/.
"""

# Retain the historical package-path behavior without importing optional engines.
# All active Merlin implementations now have one distribution owner.
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
