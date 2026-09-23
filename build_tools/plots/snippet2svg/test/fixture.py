def choose_witness(required_cells, candidates):
    """Greedy cover keeps the expensive certification set small."""
    uncovered = set(required_cells)
    chosen = []
    while uncovered:
        witness = max(candidates, key=lambda item: len(item.cells & uncovered))
        chosen.append(witness.name)
        uncovered -= witness.cells
    return chosen
