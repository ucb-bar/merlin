"""Capability-specific disclosure over the common fresh10 compiler contract.

This owns text/inventory selection only. It grants no sources or service and
does not substitute a task from the older EL1/2/3 construction study.
"""
from __future__ import annotations

import hashlib
import json
import re
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
_LIBRARY_TEXT = re.compile(
    r"(?:merlin\.(?:transforms|sched|compile\.scheduling)\b|OPTIMIZATION_[A-Z_]+|"
    r"ab-library-pass|compose-pass-pipeline|heavy-pass-recipes|pass-library|scheduling-api|"
    r"shape-schedule-selection|parallel_schedule_recipe|dense_schedule_text|Kernel\.text\(\))"
)
_CCA_TEXT = re.compile(r"(?:\bCCA\b|cca_tools|cca_spine|reference-analysis|reference_analysis|merlin\.mining\b)")
_REFERENCE_TEXT = re.compile(r"(?:reference-kernels|REFERENCE_MAP|selected kernel reference evidence)", re.I)


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


def capability_text(name):
    capabilities = P.resolve(name)
    if capabilities.coverage:
        return _coverage_text(capabilities)
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
    if capabilities.raw_kernel_references:
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
    return text


def _coverage_text(capabilities):
    text = _coverage_origin() + (
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
        "Three original whole-model reference executables are supplied read-only under "
        "reference-kernels/ (reference-manifest.json lists them). No CCA products, disassemblies, "
        "reference maps or other derived analyses of them are supplied; any analysis of these raw "
        "files is your own work within this run's budget. Reference inspection does not grant "
        "execution, expected answers or timing authority.\n"
        "\nThe source-only command-buffer helpers can select the common public "
        "schemas/command_buffer.schema.json with schema_path= in CommandBufferBuilder or "
        "verify.validate. Copy identical public data into the compiler package when candidate "
        "execution needs it; workspace aliases are not candidate execution paths.\n"
    )
    return text


def filter_markdown(text, *, library, references):
    """Filter complete Markdown blocks, never half a fenced code example.

    New capability-disclosure documents record the precise removals. This helper
    only suppresses existing advice that requires a withheld service/source.
    """
    paragraphs, lines, fence = [], [], None
    for line in text.splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if match:
            marker = match.group(1)
            if fence is None:
                fence = marker[0]
            elif marker[0] == fence:
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
        if _CCA_TEXT.search(paragraph):
            continue
        if not library and _LIBRARY_TEXT.search(paragraph):
            continue
        if not references and _REFERENCE_TEXT.search(paragraph):
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
    extra_ads = re.compile(r"shape_policy|evaluate_variants|(?:supplied|shared|applicable) library|"
                           r"library mechanism|library's|shared-pass guidance|exported.*tools", re.I)
    chosen = "\n\n".join(
        " ".join(sentence for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", paragraph)
                 if not (_CCA_TEXT.search(sentence) or _LIBRARY_TEXT.search(sentence)
                         or _REFERENCE_TEXT.search(sentence) or extra_ads.search(sentence)))
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
    for section in re.split(r"(?m)^(?=## )", shared_guidance()):
        first = section.splitlines()[0] if section.splitlines() else ""
        heading = first[3:].strip() if first.startswith("## ") else ""
        if heading in COMMON_CONTRACT_SECTIONS:
            if heading in common_sections:
                raise ValueError("common compiler contract section is duplicated")
            common_sections[heading] = section
    if set(common_sections) != set(COMMON_CONTRACT_SECTIONS):
        raise ValueError("original common compiler contract sections are unavailable")
    for heading in COMMON_CONTRACT_SECTIONS:
        count = len(re.findall(r"(?m)^## " + re.escape(heading) + r"$", original))
        if count > 1:
            raise ValueError("native common compiler contract section is duplicated")
        if not count:
            original += "\n\n" + common_sections[heading].rstrip() + "\n"
    sections = re.split(r"(?m)^(?=## )", original)
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
    core = re.sub(r"(?m)^- `cca_(?:tools|spine)`:[^\n]*\n", "", core)
    core = core.replace("the xDSL, CCA and RTL-profile authoring tools", "the selected xDSL and RTL-profile authoring tools")
    # Remove obsolete corpus-size prose inherited by the legacy joint template;
    # the selected runtime and suite blocks already enumerate the actual rosters.
    core = re.sub(r"All 250 selected cases remain eventual obligations:.*?(?=Without `--sim`)",
                  "All cases in the selected public inventories remain eventual obligations. ", core,
                  flags=re.S)
    # Keep the objective, all six full-model programs, numerical/placement/ABI
    # rules and routing intact. Capability advice is separated by section rather
    # than removing paragraphs that happen to contain the word 'schedule'.
    record = dict(schema="merlin.fresh10_common_task.v1", source="generated fresh10 contract",
                  common_core_sha256=hashlib.sha256(core.encode()).hexdigest(),
                  common_core_bytes=len(core.encode()), removed_capability_sections=removed,
                  capability_disclosure_sha256=hashlib.sha256(capability_text(name).encode()).hexdigest(),
                  profile=name)
    disclosure = capability_text(name)
    if P.resolve(name).raw_kernel_references:
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
            forbidden = _CCA_TEXT.search(text) or (not capabilities.raw_kernel_references and _REFERENCE_TEXT.search(text))
            if relative.endswith(".py"):
                if forbidden or (not capabilities.transform_library and _LIBRARY_TEXT.search(text)):
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
