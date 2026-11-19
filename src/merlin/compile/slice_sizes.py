"""Choose aligned slice sizes without increasing the number of invocations."""


def uniform_slice_elements(elements: int, capacity: int, alignment: int = 1) -> int:
    """Prefer one reusable body size when the minimum slice count divides evenly.

    The caller supplies a proved capacity and element alignment. A partial final
    slice remains valid when equal slices cannot satisfy those constraints. No
    arithmetic, buffer ownership or target resource admission is inferred here.
    """
    if any(type(value) is not int or value <= 0 for value in (elements, capacity, alignment)):
        raise ValueError("slice extents, capacity and alignment must be positive integers")
    capacity -= capacity % alignment
    if capacity == 0:
        raise ValueError("slice capacity cannot hold one aligned group")
    slices = (elements + capacity - 1) // capacity
    if slices == 1:
        return capacity
    equal, remainder = divmod(elements, slices)
    return equal if remainder == 0 and equal % alignment == 0 else capacity
