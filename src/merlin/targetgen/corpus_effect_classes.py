"""Project reviewed instruction effect classes into operation-presence obligations.

Equivalent selectors may share a class. This projection assigns neither an issue
order nor a preferred selector, and does not qualify arithmetic or execution.
"""

from merlin.targetgen import semantic_families as sf


def source(encoding: dict):
    """Return a pure operation-to-class projection of an explicit declaration."""
    classes, families = encoding.get("corpus_required_classes"), encoding.get("class_family")
    declarations = [encoding.get(key, {}) for key in ("semantic_class", "config_subtype")]
    if (type(classes) is not list or not classes
        or any(type(label) is not str or not label.isidentifier() for label in classes)
        or len(set(classes)) != len(classes)):
        raise ValueError("encoding.corpus_required_classes must declare unique effect classes")
    if (type(families) is not dict or set(families) != set(classes)
        or any(type(family) is not str or family not in sf.PRIMITIVES for family in families.values())):
        raise ValueError("encoding.class_family must bind every required class to a primitive family")
    if any(type(mapping) is not dict or any(type(label) is not str for label in mapping.values())
           for mapping in declarations):
        raise ValueError("required effect classes need explicit instruction-class declarations")
    declared = {label for mapping in declarations for label in mapping.values()}
    if set(classes) - declared:
        raise ValueError("required effect class has no declared instruction selector")

    def classes_for(*, op="matmul", output_dtype=None, epilogue=(), movement=False):
        primitives = ("movement",) if movement else sf.primitives_of(sf.from_op(op) or "")
        required = [label for label in classes if families[label] in primitives]
        if not required:
            raise ValueError(f"no required effect class is declared for operation {op!r}")
        return required

    return classes_for
