"""merlin: compiler-centered HW/SW abstraction exploration.

The reusable library: the interface/target dialects + parametric lowering (``xdsl_dialects``), the
target-independent command-buffer ABI + reference runtime (``runtime``), target generation +
certification (``targetgen``), design-space exploration (``dse``), design-pressure analysis
(``design_pressure``), kernel mining (``kernels``), and shared infra (``common``). Experiments and
generated products live outside this package (``merlin/experiments``, ``out/artifacts/``, ``out/runs/``).
"""
