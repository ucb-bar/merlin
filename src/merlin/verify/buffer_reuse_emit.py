"""Factor the complete generated buffer query without changing its declarations.

No Lean/SDK access. The complete canonical query remains part of the request;
every section and global line is accounted for before emitting the eight-module
import DAG. Actual kernel/import/type checks, not these strings, grant authority.
"""

from __future__ import annotations

import hashlib

BASE_MODULES = (
    "WholeShape",
    "WholeReuse",
    "WholeMemory",
    "WholeEpoch",
    "WholeMaterialization",
    "WholeExpansion",
    "WholeFinal",
    "RawChainEntries",
)
EPOCH_MODULES = ("EpochSyntax", "EpochCommands", "EpochPlan", "EpochKernel", "WholeEpoch")
MODULES = (*BASE_MODULES[:3], *EPOCH_MODULES, *BASE_MODULES[4:])
SPANS = (
    ("section ShapeBoundary", "end ShapeBoundary"),
    ("section OldReuse", "end OldReuse"),
    ("namespace GraphMemorySCFObligation", "end GraphMemorySCFObligation"),
    ("namespace GraphShapeEpochObligation", "end GraphShapeEpochObligation"),
    ("namespace GraphMemoryAfterMaterialization", "end GraphMemoryAfterMaterialization"),
    ("namespace GraphMemoryAfterExpansion", "end GraphMemoryAfterExpansion"),
    ("namespace GraphMemoryAfterWhole", "end GraphMemoryAfterWhole"),
    ("namespace GraphMemoryAfterEntries", "end GraphMemoryAfterEntries"),
)
GLOBAL_OPTIONS = (
    "set_option autoImplicit false\n"
    "set_option maxRecDepth 16384\n"
    "set_option maxHeartbeats 8000000\n"
    "open Merlin Merlin.GraphLoopsShape\n"
)
ENTRY_IMPORT = "Merlin.GraphLoopsMemorySCFAfterEntriesWitness"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def theorem_roster(text):
    """Inventory of controlled emitted declarations, not a Lean parser or proof."""
    scope, result = [], []
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] in ("section", "namespace"):
            if len(words) != 2:
                raise ValueError("named generated section/namespace required")
            scope.append((words[0], words[1]))
        elif words[0] == "end":
            if len(words) != 2 or not scope or scope[-1][1] != words[1]:
                raise ValueError("exact generated scope closure required")
            scope.pop()
        elif words[0] == "theorem":
            result.append(".".join([name for tag, name in scope if tag == "namespace"] + [words[1]]))
    if scope or len(result) != len(set(result)):
        raise ValueError("closed distinct generated theorem roster required")
    return tuple(result)


def shape_validity(query, *, count, bits):
    """Derive constructorwise finite typing goals from actual source metadata.

    Metadata is not a proof premise: the width equality and exhaustive actual
    function cases must themselves check in Lean. No limit or theorem change.
    """
    if type(count) is not int or count < 1 or type(bits) is not int or bits < 2:
        raise ValueError("actual nonempty function count and index width required")
    lines = query.splitlines(True)
    starts = [i for i, line in enumerate(lines) if line == "namespace GraphShapeObligation\n"]
    stops = [i for i, line in enumerate(lines) if line == "end GraphShapeObligation\n"]
    if len(starts) != 1 or len(stops) != 1 or starts[0] >= stops[0]:
        raise ValueError("complete unique actual Shape namespace required")
    valid = [i for i in range(starts[0], stops[0]) if lines[i] == "theorem valid : Model.Valid actual := by\n"]
    accepted = [i for i in range(starts[0], stops[0])
                if lines[i] == "theorem accepted : Model.check actual = true := by\n"]
    if len(valid) != 1 or len(accepted) != 1 or valid[0] >= accepted[0]:
        raise ValueError("exact existing Shape validity statement/check required")
    previous = ("  simp [Model.Valid, Module.Valid, Function.Valid, GraphLoopsShape.Valid, Scope,\n"
                "    CastFacts, DimensionFacts, ConstantFacts, AllBound, Plain, Fresh, names,\n"
                "    extendContext, extendAliases, resolve, ValidDimension, actual]\n"
                "  all_goals decide\n")
    proof = (f"  have width : actual.bits = {bits} := rfl\n"
             "  refine ⟨?_, ?_, ?_, ?_⟩\n"
             "  · refine ⟨by decide, by decide, ?_, by decide⟩\n"
             "    intro function member\n"
             "    simp only [actual, List.mem_cons, List.not_mem_nil, or_false] at member\n"
             "    rcases member with " + " | ".join(["rfl"]*count) + "\n"
             "    all_goals\n"
             "      repeat' first\n"
             "        | apply And.intro\n"
             "        | (simp [width, AllBound, Plain, Fresh, names,\n"
             "            extendContext, extendAliases, resolve] <;> decide)\n"
             "  · decide\n"
             "  · simp [actual]\n"
             "  · simp [actual]\n")
    existing = "".join(lines[valid[0]+1:accepted[0]])
    if existing not in (previous, proof):
        raise ValueError("unknown Shape proof construction; explicit successor review required")
    return "".join(lines[:valid[0]+1]) + proof + "".join(lines[accepted[0]:])


def epoch_parts(part):
    """Losslessly partition exact declaration groups; replicate original scope."""
    lines = part["source"].splitlines(True)
    markers = ("def actualNormalized :", "def plan :", "theorem plan_valid :",
               "def kernel :", "theorem complete_adapter ")
    starts = []
    for marker in markers:
        found = [i for i, line in enumerate(lines) if line.startswith(marker)]
        if len(found) != 1:
            raise ValueError("unique complete epoch declaration groups required")
        starts.append(found[0])
    ends = [i for i, line in enumerate(lines) if line == "end GraphShapeEpochObligation\n"]
    if len(ends) != 1 or starts != sorted(set(starts)) or starts[-1] >= ends[0]:
        raise ValueError("exact epoch namespace/declaration order required")
    header = "".join(lines[1:starts[0]])
    if header != ("namespace GraphShapeEpochObligation\nset_option autoImplicit false\n"
                  "set_option maxRecDepth 16384\nset_option maxHeartbeats 4000000\n"
                  "open Merlin Merlin.GraphLoopsShape\n"):
        raise ValueError("exact original epoch namespace/options required")
    if lines[0] != "import WholeMemory\n" or any(line.strip() for line in lines[ends[0]+1:]):
        raise ValueError("complete epoch import/trailing source required")
    stops = starts[1:] + [ends[0]]
    bodies = ["".join(lines[start:stop]) for start, stop in zip(starts, stops, strict=True)]
    if "".join(bodies) != "".join(lines[starts[0]:ends[0]]):
        raise ValueError("complete epoch body reconstruction failed")
    result = []
    for index, (name, start, stop, body) in enumerate(zip(EPOCH_MODULES, starts, stops, bodies, strict=True)):
        parent = "WholeMemory" if index == 0 else EPOCH_MODULES[index-1]
        rendered = body if index < len(EPOCH_MODULES)-1 else body.rstrip() + "\n"
        source = "import " + parent + "\n" + header + rendered + "\nend GraphShapeEpochObligation\n"
        result.append({"module": name, "source": source, "imports": [parent], "sha256": digest(source),
                       "canonical_span": [part["canonical_span"][0]+start-1,
                                          part["canonical_span"][0]+stop-1],
                       "canonical_span_sha256": digest(body),
                       "unfactored_epoch_sha256": part["sha256"]})
    if set(theorem_roster(part["source"])) != set(root for row in result for root in theorem_roster(row["source"])):
        raise ValueError("finer factoring lost an epoch declaration/root")
    return result


def factor(query, roots):
    """Reconstruct all declaration spans from a freshly generated full query.

    No slice is selected by line number, and no declaration is discarded. New
    globals/namespaces, changed scope options or incomplete root rosters refuse.
    """
    if type(query) is not str or not query.endswith("\n") or type(roots) is not tuple:
        raise ValueError("complete canonical query and exact tuple root roster required")
    local_roots = {root for root in roots if not root.startswith("Merlin.")}
    if len(roots) != len(set(roots)) or set(theorem_roster(query)) != local_roots:
        raise ValueError("every generated declaration/root must be included")
    lines = query.splitlines(True)
    spans, consumed = [], set()
    previous = -1
    for start, stop in SPANS:
        begin = [i for i, line in enumerate(lines) if line == start + "\n"]
        end = [i for i, line in enumerate(lines) if line == stop + "\n"]
        if len(begin) != 1 or len(end) != 1 or not previous < begin[0] <= end[0]:
            raise ValueError("complete ordered unique generated section boundaries required")
        spans.append((begin[0], end[0] + 1))
        consumed.update(range(begin[0], end[0] + 1))
        previous = end[0]
    import_lines = [line for line in lines[: spans[0][0]] if line.startswith("import ")]
    if any(len(line.split()) != 2 or line != "import " + line.split()[1] + "\n" for line in import_lines):
        raise ValueError("each generated import must name exactly one bound module")
    imports = [line.split()[1] for line in import_lines]
    if (
        not imports
        or imports[0] != ENTRY_IMPORT
        or len(imports) != len(set(imports))
        or any(not name.startswith("Merlin.") for name in imports)
    ):
        raise ValueError("explicit complete project import header required")
    header_end = spans[0][0]
    if any(line.strip() and not line.startswith("import ") for line in lines[:header_end]):
        raise ValueError("unexpected declaration in generated import header")
    consumed.update(range(header_end))
    gap = "".join(lines[spans[3][1] : spans[4][0]]).strip() + "\n"
    if gap != GLOBAL_OPTIONS:
        raise ValueError("exact original global options/scope required")
    consumed.update(range(spans[3][1], spans[4][0]))
    if any(line.strip() for i, line in enumerate(lines) if i not in consumed):
        raise ValueError("unaccounted generated declaration outside complete sections")
    # The spans plus literal gaps form a lossless partition of the canonical input.
    fragments, cursor = [], 0
    for begin, end in spans:
        fragments.extend(("".join(lines[cursor:begin]), "".join(lines[begin:end])))
        cursor = end
    fragments.append("".join(lines[cursor:]))
    if "".join(fragments) != query:
        raise ValueError("complete canonical source reconstruction failed")
    result = []
    for index, (name, (begin, end)) in enumerate(zip(BASE_MODULES, spans, strict=True)):
        if index == 0:
            dependencies = tuple(name for name in imports if name != ENTRY_IMPORT)
        elif index == 7:
            dependencies = (BASE_MODULES[index - 1], ENTRY_IMPORT)
        else:
            dependencies = (BASE_MODULES[index - 1],)
        body = "".join(lines[begin:end])
        prelude = GLOBAL_OPTIONS if index >= 4 else ""
        source = "".join("import " + dep + "\n" for dep in dependencies) + prelude + body
        result.append(
            {
                "module": name,
                "source": source,
                "imports": list(dependencies),
                "sha256": digest(source),
                "canonical_span": [begin, end],
                "canonical_span_sha256": digest(body),
            }
        )
    result = [*result[:3], *epoch_parts(result[3]), *result[4:]]
    if set(root for part in result for root in theorem_roster(part["source"])) != local_roots:
        raise ValueError("factoring lost a generated theorem")
    return result
