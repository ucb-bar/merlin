"""Actual complete Kernel projection for nonempty constant pure setter prefixes.

The prefix is read from the actual source, never a producer's move report.
Canonical Lean checks derive its last-write-wins assignment and full outcome
preservation. Existing indexed data admission remains trusted host analysis.
"""

from merlin.common.paths import module_source_path

from .compiler_configuration_hoist_indexed import program_current
from .configuration_elision_bridge import file_ref
from .configuration_hoist_bridge import Unsupported
from .configuration_hoist_intake import proof_sources as single_proof_sources


def project(source, final, *, isa, effects):
    analyses = []
    before, (position, loop) = program_current(source, isa, effects, analyses=analyses)
    after, _ = program_current(final, isa, effects, analyses=analyses)
    setups, body = [], list(loop["body"])
    while body:
        atom = before["atoms"][body[0]]
        effect = atom["effect"]
        if effect is None or not effect["ordered"] or not effect["pureWrite"] or not atom["eraseAllowed"]:
            break
        setups.append(body.pop(0))
    if not setups:
        raise Unsupported("nonempty literal ordered pure prefix required")
    established = {}
    for ident in setups:
        established.update(before["atoms"][ident]["effect"]["writes"])
    protected = set(established)
    for ident in body:
        effect = before["atoms"][ident]["effect"]
        if effect is None or not effect["ordered"]:
            raise Unsupported("retained prefix-hoist body requires exact ordered effects/frame")
        if protected.intersection(effect["clobbers"]) or any(
            key in established and value != established[key] for key, value in effect["writes"]
        ):
            raise Unsupported("retained body must preserve last-write-wins prefix assignment")
    fields = sorted(
        {
            key
            for p in (before, after)
            for atom in p["atoms"]
            if atom["effect"] is not None
            for key in [
                *(a for a, _ in atom["effect"]["reads"]),
                *(a for a, _ in atom["effect"]["writes"]),
                *atom["effect"]["clobbers"],
            ]
        }
    )
    if len(fields) > 256:
        raise Unsupported("actual prefix field registry exceeds bounded domain")
    for p in (before, after):
        for atom in p["atoms"]:
            effect = atom["effect"]
            if effect is not None:
                for kind in ("reads", "writes"):
                    effect[kind] = [[fields.index(key), value] for key, value in effect[kind]]
                effect["clobbers"] = [fields.index(key) for key in effect["clobbers"]]
    move = {
        "earlier": [row["call"] for row in before["commands"][:position]],
        "later": [row["call"] for row in before["commands"][position + 1 :]],
        "setups": setups,
        "body": body,
        "var": loop["var"],
        "extent": loop["extent"],
        "metadata": loop["metadata"],
    }
    return {
        "schema": "merlin.configuration-hoist.prefix-projection.v1",
        "before": before,
        "after": after,
        "move": move,
        "fields": fields,
        "authority": False,
        "kernel_outcome": "UNOBSERVED",
        "Python_projection_trusted": True,
        "actual_ISA_effect_model_approved": False,
        "indexed_data_analysis": analyses,
        "indexed_data_analysis_is_trusted_host": True,
        "all_index_model_premise_retained": True,
        "native_or_encoder_fidelity_proved": False,
    }


def proof_sources():
    # Finite additive canonical closure; no caller source root/artifact fallback.
    root = "Merlin.GraphStateConfigurationPrefixHoist"
    path = module_source_path("merlin.verify").parent / "lean" / "Merlin" / "GraphStateConfigurationPrefixHoist.lean"
    if path.is_symlink():
        raise ValueError("canonical prefix source may not be a symlink")
    expected = "import Merlin.GraphStateConfigurationHoistChecker"
    imports = [line for line in path.read_text().splitlines() if line.startswith("import ")]
    if imports != [expected]:
        raise ValueError("canonical prefix proof import closure differs")
    return [*single_proof_sources(), {"module": root, **file_ref(path)}]
