"""Version two emission for the common Context's separate compilation/audits.

The historical emitters and saved queries remain unchanged. This adapter removes
only their enumerated top-level axiom-print commands, requiring the complete
expected roster. It is not a Lean sanitizer or a proof authority: it consumes
our generated source, and Context separately checks exact roots and full types.
No Context, SDK or native process is opened by this module.
"""

from __future__ import annotations

import hashlib

from .lean_graph_loops_shape_after import prepare_pair as prepare_pair_v1
from .lean_graph_loops_shape_after_epoch import prepare_chain as prepare_chain_v1

FORMAT = "merlin.graph.shape.after.declarations.v2"
PAIR_ROOT = "GraphShapeAfterObligation.complete"
EPOCH_ROOT = "GraphShapeAfterEpochObligation.actual_source_epoch_blueprint"


def separate_audits(query, roots, *, final_root):
    """Split this generator's explicit audit commands from its declarations.

    Lean string data is escaped by the existing serializer; a line beginning
    with '#' is therefore a generated command. Unexpected or missing commands
    are emitter-contract errors, never silently dropped. This function does not
    establish that arbitrary caller-supplied Lean is safe or declaration-only.
    """
    roots = tuple(roots)
    expected = set(roots)
    if len(expected) != len(roots) or final_root not in expected:
        raise ValueError("distinct complete root roster and included final root required")
    declarations, seen = [], set()
    for line in query.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("#"):
            prefix = "#print axioms "
            if not line.startswith(prefix) or stripped[len(prefix) :] not in expected:
                raise ValueError("unexpected generated query command")
            seen.add(stripped[len(prefix) :])
        else:
            declarations.append(line)
    if seen != expected:
        raise ValueError("generated audit commands do not cover the exact expected roots")
    source = "".join(declarations)
    return dict(
        query=source,
        query_format=FORMAT,
        final_root=final_root,
        support_roots=[root for root in roots if root != final_root],
        expected_roots=list(roots),
        legacy_query_sha256=hashlib.sha256(query.encode()).hexdigest(),
        query_sha256=hashlib.sha256(source.encode()).hexdigest(),
        expected_compile_stdout="",
        audit_owner="common Context: separate exact-root axioms and full-type commands",
        lean_invoked=False,
        lean_accepted=False,
        qualified=False,
        pipeline_receipt_issued=False,
    )


def prepare_pair(*args, **kwargs):
    prepared = prepare_pair_v1(*args, **kwargs)
    return dict(prepared, **separate_audits(prepared["query"], prepared["expected_roots"], final_root=PAIR_ROOT))


def prepare_chain(*args, **kwargs):
    prepared = prepare_chain_v1(*args, **kwargs)
    return dict(prepared, **separate_audits(prepared["query"], prepared["expected_roots"], final_root=EPOCH_ROOT))
