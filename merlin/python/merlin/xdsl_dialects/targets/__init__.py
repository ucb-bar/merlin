"""In-tree reference target dialects (xDSL).

Only *reference* targets live in-tree (toy_npu). Serious target dialects are generated
into external repos by TargetGen; these modules exist so the core lowering pipeline has
a concrete target to lower into without depending on build/ artifacts.
"""
