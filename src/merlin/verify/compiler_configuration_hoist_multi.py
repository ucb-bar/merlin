"""Actual complete flat Kernel projection for multiple top-level prefix hoists.

Stages are untrusted command-list candidates derived from actual endpoints. The
closed Focus checker verifies each exact context and the complete terminal join;
no producer report, replay or saved approval is consulted.
"""

import copy

from merlin.common.paths import module_source_path

from .compiler_configuration_hoist_indexed import program_current
from .compiler_configuration_hoist_prefix import proof_sources as prefix_sources
from .configuration_elision_bridge import file_ref
from .configuration_elision_gate import Unavailable
from .configuration_hoist_bridge import Unsupported, canonical

_MAX_STAGE_WORK = 65536
_MAX_CAPTURE_BYTES = 32 * 1024 * 1024


def project(source, final, *, isa, effects, allow_single=False, extent_limit=1_000_000):
    if type(allow_single) is not bool:
        raise Unsupported("explicit single/multiple focused-loop profile required")
    analyses = []
    before, loops = program_current(
        source, isa, effects, analyses=analyses, allow_multiple=True, extent_limit=extent_limit
    )
    after, final_loops = program_current(
        final, isa, effects, analyses=analyses, allow_multiple=True, extent_limit=extent_limit
    )
    if len(loops) < (1 if allow_single else 2) or len(loops) != len(final_loops):
        raise Unsupported("multiple complete flat top-level loop roster required")
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
        raise Unsupported("actual multi-loop field registry exceeds bounded domain")
    for p in (before, after):
        for atom in p["atoms"]:
            effect = atom["effect"]
            if effect is not None:
                for kind in ("reads", "writes"):
                    effect[kind] = [[fields.index(key), value] for key, value in effect[kind]]
                effect["clobbers"] = [fields.index(key) for key in effect["clobbers"]]
    # Bound full-context copies before constructing each untrusted stage. These
    # are operational work limits, not a separate semantic loop-count cap.
    units = max(1, len(before["atoms"]) + len(before["commands"]))
    byte_units = 3 * (len(canonical(before["commands"]).encode()) + 32 * len(before["atoms"]))
    current, stages = copy.deepcopy(before["commands"]), []
    for ordinal, ((_, loop), (_, actual_final)) in enumerate(zip(loops, final_loops, strict=True)):
        if loop["extent"] == 0:
            if loop != actual_final:
                raise Unsupported("changed zero-trip loop supplies no positive motion coverage")
            continue
        if loop["body"] == actual_final["body"]:
            continue
        setups, body = [], list(loop["body"])
        while body:
            atom = before["atoms"][body[0]]
            effect = atom["effect"]
            if effect is None or not effect["ordered"] or not effect["pureWrite"] or not atom["eraseAllowed"]:
                break
            setups.append(body.pop(0))
        if not setups:
            raise Unsupported("observed changed loop requires a nonempty literal ordered pure prefix")
        if (len(stages) + 1) * units > _MAX_STAGE_WORK or (len(stages) + 1) * byte_units > _MAX_CAPTURE_BYTES:
            raise Unavailable("compiler multi-loop stage expansion resource budget exceeded")
        position = [i for i, command in enumerate(current) if "loop" in command][ordinal]
        earlier, later = copy.deepcopy(current[:position]), copy.deepcopy(current[position + 1 :])
        move = {
            "earlier": [],
            "later": [],
            "setups": setups,
            "body": body,
            "var": loop["var"],
            "extent": loop["extent"],
            "metadata": loop["metadata"],
        }
        current = [*earlier, *({"call": ident} for ident in setups), {"loop": {**loop, "body": body}}, *later]
        stages.append({"after": copy.deepcopy(current), "focus": {"earlier": earlier, "move": move, "later": later}})
    if not stages:
        raise Unsupported("at least one actual changing positive loop-prefix scope required")
    return {
        "schema": "merlin.configuration-hoist.multi-projection.v1",
        "before": before,
        "after": after,
        "stages": stages,
        "fields": fields,
        "authority": False,
        "kernel_outcome": "UNOBSERVED",
        "Python_projection_trusted": True,
        "actual_ISA_effect_model_approved": False,
        "indexed_data_analysis": analyses,
        "indexed_data_analysis_is_trusted_host": True,
        "all_index_model_premise_retained": True,
        "native_or_encoder_fidelity_proved": False,
        "actual_loop_count": len(loops),
        "focused_loop_count": len(stages),
        "actual_final_join": current == after["commands"],
    }


def proof_sources():
    name = "Merlin.GraphStateConfigurationHoistFocus"
    path = module_source_path("merlin.verify").parent / "lean" / "Merlin" / "GraphStateConfigurationHoistFocus.lean"
    if path.is_symlink():
        raise ValueError("canonical focused-hoist source may not be a symlink")
    if [line for line in path.read_text().splitlines() if line.startswith("import ")] != [
        "import Merlin.GraphStateConfigurationPrefixHoist"
    ]:
        raise ValueError("canonical focused-hoist import closure differs")
    return [*prefix_sources(), {"module": name, **file_ref(path)}]
