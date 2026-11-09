"""Capability-specific disclosure over the common fresh10 compiler contract.

This owns text/inventory selection only. It grants no sources or service and
does not substitute a task from the older EL1/2/3 construction study.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import fresh10_profiles as P

COMMON_SKILLS = frozenset((
    "cycle-model-and-gsim", "delegate-to-helpers", "feedback-files", "fix-failure-cluster",
    "inspect-artifacts", "inspect-ir", "inspect-kernel", "inspect-model",
    "performance-feedback-levels", "read-command-buffer", "roofline-and-utilization",
))
# These sections advertise implementations, catalogues or reference-analysis
# routes, rather than defining the common numerical/compiler/placement contract.
CAPABILITY_SECTIONS = frozenset((
    "Common compiler infrastructure and preservation policy", "Editable compiler source",
    "Performance-driven compiler exploration", "Construct capabilities for the whole public corpus",
    "Learn reusable schedule regions", "Scheduling integration and shared-pass development",
    "Capability-selected optimization suggestions", "Optimization principles by operation",
    "Use selected kernel reference evidence", "Reference map",
))
COMMON_CONTRACT_SECTIONS = (
    "One evolving compiler, parallel bounded work",
    "Model deployment and numerical boundaries",
)

def _word_char(char):
    return char.isalnum() or char == "_"


def _word_end(text, literal, *, start_boundary=False):
    position = 0
    while (position := text.find(literal, position)) >= 0:
        end = position + len(literal)
        if ((not start_boundary or position == 0 or not _word_char(text[position - 1]))
                and (end == len(text) or not _word_char(text[end]))):
            return True
        position += 1
    return False


def _library_text(text):
    if any(_word_end(text, name) for name in
           ("merlin.transforms", "merlin.sched", "merlin.compile.scheduling")):
        return True
    if any(name in text for name in (
            "ab-library-pass", "compose-pass-pipeline", "heavy-pass-recipes", "pass-library",
            "scheduling-api", "shape-schedule-selection", "parallel_schedule_recipe",
            "dense_schedule_text", "Kernel.text()")):
        return True
    return any(tail and ("A" <= tail[0] <= "Z" or tail[0] == "_")
               for tail in text.split("OPTIMIZATION_")[1:])


def _cca_text(text):
    return (_word_end(text, "CCA", start_boundary=True) or _word_end(text, "merlin.mining")
            or any(name in text for name in ("cca_tools", "cca_spine", "reference-analysis", "reference_analysis")))


def _reference_text(text):
    return any(name in text.lower() for name in
               ("reference-kernels", "reference_map", "selected kernel reference evidence"))


def _extra_advertisement(text):
    lower = text.lower()
    if any(name in lower for name in ("shape_policy", "evaluate_variants", "supplied library",
            "shared library", "applicable library", "library mechanism", "library's", "shared-pass guidance")):
        return True
    return any("tools" in line.partition("exported")[2] for line in lower.splitlines() if "exported" in line)


def _sentences(paragraph):
    """Split prose only at punctuation followed by whitespace and an uppercase sentence start."""
    start, position = 0, 0
    while position < len(paragraph):
        if paragraph[position] in ".!?" and position + 1 < len(paragraph) and paragraph[position + 1].isspace():
            end = position + 1
            while end < len(paragraph) and paragraph[end].isspace():
                end += 1
            if end < len(paragraph) and "A" <= paragraph[end] <= "Z":
                yield paragraph[start:position + 1]
                start = end
                position = end
                continue
        position += 1
    yield paragraph[start:]


def _sections(text):
    """Retain exact bytes while splitting level-two Markdown sections."""
    current = []
    for line in text.splitlines(keepends=True):
        if line.startswith("## ") and current:
            yield "".join(current)
            current = []
        current.append(line)
    if current:
        yield "".join(current)



def _coverage_origin():
    return (
        "## Selected fresh10 compiler capabilities\n\n"
        "Continue the supplied seed compiler in submission/ and preserve its working behavior. It is the "
        "frozen compiler of an earlier run under the common fresh10 contract; no clock, grade credit, "
        "conversation or provider state is inherited. submission/compiler_library/ holds the complete current "
        "reviewed public compiler sources with the seed's own edits preserved. AUTHOR_PROFILE.json, "
        "FULL_COMPILER_SOURCE_SELECTION.json and the selected additional-library contract state the exact "
        "source/API inventories. Seed records staged at the workspace root are listed in SEED_INDEX.md when "
        "present.\n\n"
        "A source file's presence does not qualify every callable or grant a missing private dependency. "
        "Use explicit selected public facts for the admitted fact readers/generators.\n\n"
    )


def _current_scratch_origin():
    return (
        "## Selected fresh10 compiler capabilities\n\n"
        "Start with an empty target backend and construct one general compiler under the common "
        "fresh10 contract. submission/compiler_library/ holds the complete current reviewed public "
        "compiler sources; no prior authored target backend, result or provider state is supplied. "
        "AUTHOR_PROFILE.json, FULL_COMPILER_SOURCE_SELECTION.json and the selected additional-library "
        "contract state the exact source/API inventories.\n\n"
        "A source file's presence does not qualify every callable or grant a missing private dependency. "
        "Use explicit selected public facts for the admitted fact readers/generators.\n\n"
    )


def capability_text(name, *, bank_raw=None):
    """``bank_raw``: a bank arm's granted raw original paths, from its verified pinned bank.json."""
    capabilities = P.resolve(name)
    if capabilities.coverage or (capabilities.raw_originals and capabilities.scheduling_semantics
                                 and capabilities.transform_library):
        # The current reviewed full compiler and library (seeded coverage or from-scratch arm).
        return _coverage_text(capabilities, bank_raw=bank_raw)
    text = (
        "## Selected fresh10 compiler capabilities\n\n"
        "Start with an empty target backend and construct one general compiler under the common "
        "fresh10 contract. The supplied integration, frontend, representation, ABI, runtime and "
        "inspection sources are an explicitly reviewed subset of that environment. "
        "AUTHOR_PROFILE.json, FULL_COMPILER_SOURCE_SELECTION.json and the selected additional-library "
        "contract state the exact source/API inventories. Import only those selected sources; "
        "no prior authored target backend, result or provider state is supplied.\n\n"
        "The source selection also records original unavailable dependencies and optional call paths. "
        "A source file's presence does not qualify every callable or grant a missing private dependency. "
        "Use explicit selected public facts for the admitted fact readers/generators; automatic "
        "extraction and descriptor fallbacks remain unavailable when their source owners are withheld.\n\n"
    )
    if capabilities.scheduling_semantics:
        text += (
            "Scheduling IR/dialect declarations, generic Kernel representation and emission, and "
            "static/configuration/placement/synchronization checking APIs are supplied. Their reports "
            "can be partial, unchecked or undecidable. Target instruction declarations, facts and "
            "renderers remain caller-supplied; generic emission is not a completed target backend. "
        )
    else:
        text += "The supplied source inventory omits Merlin scheduling abstractions and their checking/emission APIs. "
    if capabilities.transform_library:
        text += ("The reviewed general transformation/scheduling implementation library is supplied. "
                 "SELECTED_LIBRARY_GUIDANCE.md and the staged optimization catalogue describe its actual APIs. ")
    else:
        text += "Optimized schedulers/planners and the reusable transformation/pass implementation library are withheld. "
    text += (
        "CCA helpers, reference-analysis services and automatic system kernel learning are disabled. "
        "You may implement general compiler capabilities yourself using the admitted languages, "
        "source semantics and hardware facts. Ordinary numerical, ISA, ordering, complete-call and "
        "privacy checks remain independent of compiler-side checking.\n\n"
    )
    if capabilities.raw_originals:
        # Exactly the pinned originals: the coverage study's sentence, never map or operator wording.
        text += _ORIGINALS_TEXT
    elif capabilities.raw_kernel_references:
        text += (
            "The same reviewed read-only reference bank is supplied to this arm. Reference inspection "
            "does not grant execution, expected answers or timing authority. Any operator-built "
            "reference map is manual reference material; no CCA service is enabled by it.\n"
        )
    else:
        text += (
            "Reference-kernel sources, ELF/disassembly/listing material, reference maps and "
            "CCA-derived artifacts are withheld. General compiler transformations and scheduling "
            "implementations remain supplied.\n"
        )
    text += (
        "\nThe source-only command-buffer helpers can select the common public "
        "schemas/command_buffer.schema.json with schema_path= in CommandBufferBuilder or "
        "verify.validate. Copy identical public data into the compiler package when candidate "
        "execution needs it; workspace aliases are not candidate execution paths.\n"
    )
    if capabilities.bank_sha256 is not None:
        text = _bank_text(text, bank_raw)
    return text


#: The raw-originals grant, shared by the coverage arms and the from-scratch raw-originals arms.
_ORIGINALS_TEXT = (
    "The raw expert reference originals are supplied read-only under reference-kernels/ "
    "(reference-manifest.json lists every file): the three original whole-model executables "
    "(ResNet50, SmolVLA, TinyLlama), the upstream Exo accelerator apps (matmul and convolution "
    "schedules) with their generated public reference library, headers, platform sources and "
    "example ELFs, the supplied Exo LLM source, and their licenses. No CCA products, "
    "disassemblies, kernel indexes, reference maps or other derived analyses are supplied; any "
    "analysis of these raw files is your own work within this run's budget. Reference inspection "
    "does not grant execution, expected answers or timing authority.\n"
)


def _coverage_text(capabilities, *, bank_raw=None):
    text = (_coverage_origin() if capabilities.coverage else _current_scratch_origin()) + (
        "Scheduling IR/dialect declarations, generic Kernel representation and emission, and "
        "static/configuration/placement/synchronization checking APIs are supplied. Their reports "
        "can be partial, unchecked or undecidable. Target instruction declarations, facts and "
        "renderers remain caller-supplied; generic emission is not a completed target backend. "
        "The reviewed general transformation/scheduling implementation library is supplied. "
        "SELECTED_LIBRARY_GUIDANCE.md and the staged optimization catalogue describe its actual APIs. "
        "CCA helpers, reference-analysis services and automatic system kernel learning are disabled. "
        "You may implement general compiler capabilities yourself using the admitted languages, "
        "source semantics and hardware facts. Ordinary numerical, ISA, ordering, complete-call and "
        "privacy checks remain independent of compiler-side checking.\n\n"
        + _ORIGINALS_TEXT +
        "\nThe source-only command-buffer helpers can select the common public "
        "schemas/command_buffer.schema.json with schema_path= in CommandBufferBuilder or "
        "verify.validate. Copy identical public data into the compiler package when candidate "
        "execution needs it; workspace aliases are not candidate execution paths.\n"
    )
    if capabilities.bank_sha256 is not None:
        text = _bank_text(text, bank_raw)
    return text


_RAW_ORIGINALS = ("The raw expert reference originals are supplied read-only under reference-kernels/ ",
                  "and their licenses. ")


def _raw_originals(text):
    """The Raw arm's sentence naming every original (all models, Exo apps and sources), exactly once."""
    start, end = _RAW_ORIGINALS
    if text.count(start) != 1:
        raise ValueError("coverage disclosure changed; review the bank-arm disclosure")
    tail = text[text.index(start):]
    if end not in tail:
        raise ValueError("coverage disclosure changed; review the bank-arm disclosure")
    return tail[:tail.index(end) + len(end)]


def _bank_originals(paths):
    listed = " (corpus paths: " + ", ".join(sorted(paths)) + ")" if paths else ""
    return (
        "Only the raw expert reference originals this arm's CCA bank was prepared from are supplied "
        "read-only under reference-kernels/; reference-manifest.json and the raw list in "
        "cca-evidence/bank.json name each of them by path and SHA256" + listed + ". Original "
        "executables, sources and kernels outside that list (other models' originals) and every other "
        "CCA bank are not supplied. "
    )


_RAW_SERVICES = "CCA helpers, reference-analysis services and automatic system kernel learning are disabled. "
_RAW_DERIVED = ("No CCA products, disassemblies, kernel indexes, reference maps or other derived analyses are "
                "supplied; any analysis of these raw files is your own work within this run's budget. ")
_BANK_SERVICES = (
    "Automatic system kernel learning and the other CCA helpers are disabled. One read-only "
    "reference-analysis service is enabled for this arm. It compares your emitted kernels with expert "
    "units: roles (operand A, weight W, output), transfer width per role, operand buffering, load/compute "
    "issue overlap, prefetch distance, output panel extent and residency, side by side, with the library "
    "routes that change each difference. Each fact is marked observed or unknown with its reason. "
    "After each grade, qa/round_brief.md groups your emitted kernels into families (same decoded roles, "
    "configured fields, loop-nest depth and dataflow, at any shape), names the nearest bank units for each "
    "family with the fields that agree and differ, and lists the experts' facts next to each member's "
    "shape, the representatives' differences and the members that deviate. "
    "It serves precomputed bottom-up CCA for a listed set of reference units only, mounted read-only "
    "under cca-evidence/ (bank.json pins every file and lists the admitted units), derived ahead of time "
    "from the raw originals with the selected ISA/RTL facts. "
    "These records are static evidence, not optimization recipes or performance results, and they do not "
    "establish semantic equivalence or which unit corresponds to which model layer. "
)
_BANK_DERIVED = (
    "Apart from the selected CCA bank, no CCA products, disassemblies, kernel indexes, reference maps or "
    "other derived analyses are supplied; any other analysis of these raw files is your own work within "
    "this run's budget. "
)
_BANK_USAGE = (
    "\nCall the bank service with the stdlib client:\n"
    "  python agent_selfcheck.py --reference-analysis inventory [--reference-offset N] [--reference-limit 1..16]\n"
    "  python agent_selfcheck.py --reference-analysis lift --reference-entry UNIT_ID "
    "[--reference-region REGION_ID]\n"
    "  python agent_selfcheck.py --reference-analysis compare --reference-entry UNIT_ID "
    "--reference-region REGION_ID --candidate-path artifacts/FILE [--candidate-symbol SYMBOL] "
    "[--candidate-format candidate_build] [--reference-observation-field Instruction.field]\n"
    "  python agent_selfcheck.py --reference-analysis applicability --reference-entry UNIT_ID "
    "--reference-region REGION_ID --candidate-path artifacts/FILE [--candidate-symbol SYMBOL] "
    "[--candidate-shape m=M,k=K,n=N | --candidate-shape kh=KH,kw=KW,c=C,oc=OC]\n"
    "inventory lists every admitted unit (UNIT_ID), its occurrences (REGION_ID), file paths and card "
    "fields. --reference-path cca-evidence/<listed file> selects a unit or occurrence instead of "
    "--reference-entry. Large replies keep the complete record in a hash-pinned detail file named in the "
    "reply. Requests naming a raw reference-kernels/ file, region or symbol, or a unit outside the bank, "
    "are refused with reason outside_selected_bank; no CCA is computed on demand. inventory leads each unit "
    "with its mechanism summary. Every request first prints where its reply will be written "
    "(.qa_channel/resp_ID.json), so a reply that arrives after your wait is still there, and prints a "
    "one-line summary on stderr when it arrives. Analysis of your own "
    "submission/, variants/, selfcheck_out/ and artifacts/ files works as usual.\n"
    "  python agent_selfcheck.py --reference-analysis verify --before selfcheck_out/requests/OLD "
    "--after selfcheck_out/requests/NEW [--mechanism EFFECT] [--sweep selfcheck_out/requests/ID,...] "
    "[--verdicts artifacts/RESPONSE.json] reports, from your own selfcheck outputs, per-role traffic "
    "(static sites and proved executions), storage checks, whether the emitted stream changed "
    "(applied, optimization_did_not_apply_emitted_fallback or unknown) and numerical verdicts separately.\n"
    "With --baseline selfcheck_out/requests/SEED it also reports per-role transfer width, buffering and issue "
    "overlap against the before and seed builds, regression flags, and rows whose emitted code is identical "
    "to either build; facts that need a proved command stream stay unknown without one.\n"
    "applicability takes the compare selectors and the same refusals. It derives the candidate's decoded "
    "effects with the bank's decoder and lists static differences from the unit (operand and weight loads "
    "per compute, transfer extents, broadcast loads, configuration and sync sites, loop descriptors), each "
    "with an evidence pointer and status observed, inferred or unknown. For each difference it lists the "
    "scheduling-catalog capabilities whose declared effect moves the candidate toward the unit, with their "
    "symbols, usage and requirements, and evaluates the library's capacity predicate against your declared "
    "shape and the selected RTL facts: satisfied, violated (with the bound) or unknown (with the missing "
    "input). A difference no capability addresses is reported as missing_capability. Routes are listed in "
    "catalog order and are not ranked; their execution laws are not established, and semantic equivalence "
    "and performance remain unknown.\n"
)


def _bank_text(text, bank_raw=None):
    """A bank arm: the Raw arm's disclosure with only its bank's originals, the bank and its interface."""
    if text.count(_RAW_SERVICES) != 1 or text.count(_RAW_DERIVED) != 1:
        raise ValueError("coverage disclosure changed; review the bank-arm disclosure")
    text = (text.replace(_RAW_SERVICES, _BANK_SERVICES).replace(_RAW_DERIVED, _BANK_DERIVED)
            .replace(_raw_originals(text), _bank_originals(bank_raw)))
    marker = "\nThe source-only command-buffer helpers"
    head, separator, tail = text.partition(marker)
    if not separator:
        raise ValueError("coverage disclosure changed; review the bank-arm disclosure")
    return head + _BANK_USAGE + separator + tail


def filter_markdown(text, *, library, references):
    """Filter complete Markdown blocks, never half a fenced code example.

    New capability-disclosure documents record the precise removals. This helper
    only suppresses existing advice that requires a withheld service/source.
    """
    paragraphs, lines, fence = [], [], None
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[0]
            length = len(stripped) - len(stripped.lstrip(marker))
            if fence is None:
                fence = (marker, length)
            elif marker == fence[0] and length >= fence[1] and not stripped[length:].strip():
                fence = None
        if not line.strip() and fence is None:
            if lines:
                paragraphs.append("\n".join(lines))
                lines = []
        else:
            lines.append(line)
    if fence is not None:
        raise ValueError("selected guidance has an unterminated Markdown code block")
    if lines:
        paragraphs.append("\n".join(lines))
    chosen = []
    for paragraph in paragraphs:
        if _cca_text(paragraph):
            continue
        if not library and _library_text(paragraph):
            continue
        if not references and _reference_text(paragraph):
            continue
        chosen.append(paragraph)
    return "\n\n".join(chosen) + "\n"


_paragraph_filter = filter_markdown


def author_prompt(text, *, continuation=False):
    """Common fresh10 author instructions, with unavailable implementation ads removed.

    Apply the same common text to all three new arms; their separate TASK block
    supplies the capability inventory, without an optimization recipe.
    """
    # The provider prompt is prose (not Markdown code). Preserve its complete
    # objective/measurement/parallel-work advice while removing only sentences
    # that direct use of supplied implementations absent from one or more arms.
    chosen = "\n\n".join(
        " ".join(sentence for sentence in _sentences(paragraph)
                 if not (_cca_text(sentence) or _library_text(sentence)
                         or _reference_text(sentence) or _extra_advertisement(sentence)))
        for paragraph in text.strip().split("\n\n")
    ).strip() + "\n"
    chosen = chosen.replace("PUBLIC_CAPABILITIES.json", "PUBLIC_CAPABILITIES.json and TASK.md")
    origin = ("continuing the supplied seed compiler and preserving its working behavior"
              if continuation else "starting from an empty target backend")
    return (
        "Construct one general compiler under the selected fresh10 contract, " + origin + ". Read TASK.md, AUTHOR_PROFILE.json, applicable workspace instructions and "
        "qa/round_brief.md when present. The exact source/tool inventories and current resource "
        "contract determine available capabilities and required cases.\n\n" + chosen
    )


def _staged_bank_raw(workspace, name):
    """A bank arm's granted raw original paths, read from its pinned bank when it is staged."""
    pin = P.resolve(name).bank_sha256
    path = Path(workspace) / "cca-evidence" / "bank.json"
    if pin is None or not path.is_file():
        return None
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin:
        raise ValueError("staged CCA bank differs from the profile's pinned bank")
    return [row["path"] for row in json.loads(raw).get("raw", ())]


def stage_task(workspace, run_dir, name, options):
    """Retain the generated fresh10 contract core; isolate capability disclosure."""
    workspace, run_dir = Path(workspace), Path(run_dir)
    task = workspace / "TASK.md"
    original = task.read_text()
    # These original obligations were historically delivered alongside library
    # advice. Retain them in every fresh10 TASK even when that advice is withheld.
    # Read the original owner rather than maintaining a second numerical policy.
    from .joint_pass_guidance import prompt as shared_guidance

    common_sections = {}
    for section in _sections(shared_guidance()):
        first = section.splitlines()[0] if section.splitlines() else ""
        heading = first[3:].strip() if first.startswith("## ") else ""
        if heading in COMMON_CONTRACT_SECTIONS:
            if heading in common_sections:
                raise ValueError("common compiler contract section is duplicated")
            common_sections[heading] = section
    if set(common_sections) != set(COMMON_CONTRACT_SECTIONS):
        raise ValueError("original common compiler contract sections are unavailable")
    for heading in COMMON_CONTRACT_SECTIONS:
        count = sum(line == "## " + heading for line in original.splitlines())
        if count > 1:
            raise ValueError("native common compiler contract section is duplicated")
        if not count:
            original += "\n\n" + common_sections[heading].rstrip() + "\n"
    sections = _sections(original)
    kept, removed, reference_sections = [], [], []
    for section in sections:
        first = section.splitlines()[0] if section.splitlines() else ""
        heading = first[3:].strip() if first.startswith("## ") else ""
        if heading in CAPABILITY_SECTIONS:
            removed.append(heading)
            if heading == "Reference map":
                reference_sections.append(section)
            continue
        # The selected tool registry renders this section from the five actual
        # services. Other common sections retain their numerical/ABI obligations.
        kept.append(section)
    core = "".join(kept)
    core = core.replace("Use this readable data leaf with `read_scheduling_contracts`;", "Read this data leaf directly;")
    core = "".join(line for line in core.splitlines(keepends=True)
                   if not line.startswith(("- `cca_tools`:", "- `cca_spine`:")))
    core = core.replace("the xDSL, CCA and RTL-profile authoring tools", "the selected xDSL and RTL-profile authoring tools")
    # Remove obsolete corpus-size prose inherited by the legacy joint template;
    # the selected runtime and suite blocks already enumerate the actual rosters.
    obsolete = "All 250 selected cases remain eventual obligations:"
    while obsolete in core:
        before, _, remaining = core.partition(obsolete)
        _, stop, after = remaining.partition("Without `--sim`")
        if not stop:
            break
        core = before + "All cases in the selected public inventories remain eventual obligations. " + stop + after
    # Keep the objective, all six full-model programs, numerical/placement/ABI
    # rules and routing intact. Capability advice is separated by section rather
    # than removing paragraphs that happen to contain the word 'schedule'.
    record = dict(schema="merlin.fresh10_common_task.v1", source="generated fresh10 contract",
                  common_core_sha256=hashlib.sha256(core.encode()).hexdigest(),
                  common_core_bytes=len(core.encode()), removed_capability_sections=removed,
                  profile=name)
    disclosure = capability_text(name, bank_raw=_staged_bank_raw(workspace, name))
    if P.resolve(name).raw_kernel_references and not P.resolve(name).raw_originals:
        # A raw-originals arm never receives a reference map, so no native map section either.
        disclosure += "\n" + "".join(reference_sections)
    record["capability_disclosure_sha256"] = hashlib.sha256(disclosure.encode()).hexdigest()
    text = core.rstrip() + "\n\n" + disclosure
    for directory in (workspace, run_dir):
        (directory / "TASK.md").write_text(text)
        (directory / "FRESH10_TASK_SELECTION.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def skill_sources(skills, name):
    """Select source bytes explicitly; no optimization helper survives under a neutral label."""
    capabilities = P.resolve(name)
    names = set(skills) if capabilities.transform_library else COMMON_SKILLS
    selected = {}
    for skill in sorted(names):
        files = skills[skill]
        out = {}
        for relative, data in files.items():
            text = data.decode()
            forbidden = _cca_text(text) or (not capabilities.raw_kernel_references and _reference_text(text))
            if relative.endswith(".py"):
                if forbidden or (not capabilities.transform_library and _library_text(text)):
                    raise ValueError("skill script requires a capability-specific review: " + skill + "/" + relative)
                out[relative] = data
                continue
            if relative == "SKILL.md":
                # Preserve the native discovery frontmatter, filter only its body.
                head, separator, body = text[4:].partition("\n---\n")
                if not text.startswith("---\n") or not separator:
                    raise ValueError("fresh10 selected skill has malformed frontmatter")
                body = _paragraph_filter(body, library=capabilities.transform_library,
                                         references=capabilities.raw_kernel_references)
                text = "---\n" + head + "\n---\n" + body
            else:
                text = _paragraph_filter(text, library=capabilities.transform_library,
                                         references=capabilities.raw_kernel_references)
            out[relative] = text.encode()
        selected[skill] = out
    return selected
