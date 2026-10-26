"""Required fusion regimes: composition × scheme × axes obligations of a software spec.

One definition shared by the composition corpus (blockers, sampling) and the fusion
campaign stratification, so the regimes a seal must cover are the regimes the campaign
reserves execution for.
"""

from __future__ import annotations


def selected_schemes(chain, schemes, axes):
    """The chain's declared primary schemes, then its justified cross-checks; nothing implicit."""
    selection = chain.get("quantization_schemes")
    if selection is None:
        raise ValueError(f"composition {chain['id']!r} declares no quantization scheme selection")
    by_id = {scheme["id"]: scheme for scheme in schemes}
    from merlin.targetgen.quantization_spec import select_scheme_axes

    rows = [(identity, {}) for identity in selection["primary"]]
    rows += [(row["scheme"], row.get("axes", {})) for row in selection.get("cross_checks", [])]
    missing = [identity for identity, _ in rows if identity not in by_id]
    if missing:
        raise ValueError(f"composition {chain['id']!r} selects undeclared schemes {missing}")
    return [select_scheme_axes(by_id[identity], axes, point) for identity, point in rows]


def required_regimes(software_spec):
    """Public chain selections define obligations independently of campaign outcomes."""
    block = (software_spec.get("quantization") or {}).get("schemes") or {}
    schemes = block.get("schemes", [])
    chains = (software_spec.get("compositions") or {}).get("chains", [])
    required = []
    for chain in chains:
        selected_schemes(chain, schemes, block.get("axes", {}))
        selection = chain["quantization_schemes"]
        required.extend((chain, identity, {}) for identity in selection["primary"])
        required.extend((chain, row["scheme"], row.get("axes", {})) for row in selection.get("cross_checks", []))
    if not chains:
        required.extend((None, scheme["id"], {}) for scheme in schemes)
    return required


def regime_identity(chain, scheme, axes):
    """The reported identity of one required regime."""
    return dict(composition=chain["id"] if chain else None, scheme=scheme, axes=axes)


def represents_required_regime(semantic, chain, scheme, axes):
    """Whether a candidate graph is a witness of one required regime."""
    nodes = semantic.get("nodes", [])
    if not nodes or not any(
        node["scheme"] == scheme and all(node.get("regime", {}).get(key) == value for key, value in axes.items())
        for node in nodes
    ):
        return False
    if chain is None:
        return True
    if bool(chain.get("multiplicity")) != (semantic["structure"] == "shared_input"):
        return False
    producers = chain["producer"] if isinstance(chain["producer"], list) else [chain["producer"]]
    return all(
        ("movement" if node["form"] == "window_movement" else node["form"]) in producers
        and node["readout"]["exit"] in chain.get("exits", [])
        and bool(node.get("residual")) == bool(chain.get("extension"))
        and (
            "movement" not in producers
            or node["form"] == "window_movement"
            or any(stage["kind"] == "pool" for stage in node["readout"]["stages"])
        )
        for node in nodes
    )
