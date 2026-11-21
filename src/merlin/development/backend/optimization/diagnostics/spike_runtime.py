"""Explicit operator-selected count runtime; no event prices or compiler grants."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from merlin.development.backend.optimization.contracts import StageGateError, canonical_json, document_sha256
from merlin.development.backend.telemetry import annotated_spike_evidence as E
from merlin.runtime.views import native_simulator as F

# ---- archive_spike_runtime ---------------------------------------------------------------------

MAX_TOOL_BYTES = 256 << 20

MAX_METADATA_BYTES = 64 << 20

COUNT_OBSERVATION_120S_PROTOCOL = "merlin.archive_count_observation.resources.120s.v1"

COUNT_OBSERVATION_600S_PROTOCOL = "merlin.archive_count_observation.resources.600s.v1"

COUNT_OBSERVATION_900S_PROTOCOL = "merlin.archive_count_observation.resources.900s.v1"


def metadata(path, *, maximum=MAX_METADATA_BYTES):
    pin = F.pin(Path(path), max_bytes=maximum)
    value = json.loads(Path(path).read_bytes())
    if F.pin(Path(path), max_bytes=maximum) != pin:
        raise StageGateError("archive screening metadata changed during read")
    return value, pin


@dataclass(frozen=True, eq=False)
class ArchiveSpikeRuntime:
    """Frozen exact tools and original command, selected by the host before search.

    ``functional_argv`` is the original functional runtime command WITHOUT an
    ELF. No author document can select this owner. Readonly namespace admission
    and fresh complete original numerics remain separate requirements.
    """

    review_id: str
    functional_argv: tuple[str, ...]
    observer: Path
    disassembler: Path
    model_path: Path
    source_manifest: Path
    libraries: tuple[Path, ...]
    dependencies: tuple[Path, ...]
    entry_symbol: str
    completion_symbol: str
    timing_json: bytes
    files: tuple[tuple[str, str], ...]
    model_json: bytes
    observation_resource_protocol: str | None = None

    def record(self):
        return {
            "schema": (
                "merlin.phase2.archive_spike_runtime.v4"
                if self.observation_resource_protocol == COUNT_OBSERVATION_900S_PROTOCOL
                else "merlin.phase2.archive_spike_runtime.v3"
                if self.observation_resource_protocol == COUNT_OBSERVATION_600S_PROTOCOL
                else "merlin.phase2.archive_spike_runtime.v2"
                if self.observation_resource_protocol is not None
                else "merlin.phase2.archive_spike_runtime.v1"
            ),
            "review_id": self.review_id,
            "functional_argv": list(self.functional_argv),
            "observer": str(self.observer),
            "disassembler": str(self.disassembler),
            "model": str(self.model_path),
            "source_manifest": str(self.source_manifest),
            "libraries": [str(p) for p in self.libraries],
            "dependencies": [str(p) for p in self.dependencies],
            "entry_symbol": self.entry_symbol,
            "completion_symbol": self.completion_symbol,
            "timing_selection": json.loads(self.timing_json),
            "files": [{"path": p, "sha256": d} for p, d in self.files],
            "scope": "count-only public kernel-call observation; native event timing unqualified",
            **(
                {
                    "observation_resources": {
                        "protocol": self.observation_resource_protocol,
                        "per_command_wall_seconds": (
                            900
                            if self.observation_resource_protocol == COUNT_OBSERVATION_900S_PROTOCOL
                            else 600
                            if self.observation_resource_protocol == COUNT_OBSERVATION_600S_PROTOCOL
                            else 120
                        ),
                        "parent_deadline_preserved": True,
                    }
                }
                if self.observation_resource_protocol is not None
                else {}
            ),
        }

    @property
    def sha256(self):
        return document_sha256(self.record())

    def verify(self):
        if type(self) is not ArchiveSpikeRuntime:
            raise StageGateError("archive screening requires the exact reviewed count runtime")
        if self.observation_resource_protocol is not None and (
            type(self.observation_resource_protocol) is not str
            or self.observation_resource_protocol
            not in (COUNT_OBSERVATION_120S_PROTOCOL, COUNT_OBSERVATION_600S_PROTOCOL, COUNT_OBSERVATION_900S_PROTOCOL)
        ):
            raise StageGateError("archive count runtime requires its explicit bounded observation resource protocol")
        for path, digest in self.files:
            if F.pin(Path(path), max_bytes=MAX_TOOL_BYTES)["sha256"] != digest:
                raise StageGateError("archive screening runtime source/tool/model changed")
        if canonical_json(json.loads(self.model_path.read_bytes())) != self.model_json:
            raise StageGateError("archive screening model changed")
        return self.record()

    def observation_limits(self, remaining_s):
        """Resource-only opt-in; the same cap brackets both count commands."""
        from merlin.analysis.optimization.candidates.dynamic_census import (
            WALL_600S_PROTOCOL,
            WALL_900S_PROTOCOL,
            CensusLimits,
        )

        self.verify()
        protocol = {
            COUNT_OBSERVATION_600S_PROTOCOL: WALL_600S_PROTOCOL,
            COUNT_OBSERVATION_900S_PROTOCOL: WALL_900S_PROTOCOL,
        }.get(self.observation_resource_protocol)
        limits = CensusLimits.for_timeout(remaining_s, wall_resource_protocol=protocol)
        if self.observation_resource_protocol == COUNT_OBSERVATION_120S_PROTOCOL:
            limits = replace(limits, wall_s=min(120.0, remaining_s))
            limits.verify()
        return limits

    @property
    def execution_files(self):
        """Only the frozen runtime leaves needed inside the child namespace."""
        return tuple(
            dict.fromkeys(
                (
                    Path(self.functional_argv[0]),
                    self.observer,
                    self.disassembler,
                    self.model_path,
                    *self.libraries,
                    *self.dependencies,
                )
            )
        )

    def commands(self, elf, report, scope):
        """Preserve the original ISA/memory/extension command, adding observation only."""
        self.verify()
        argv = list(self.functional_argv)
        original = [*argv, str(elf)]
        argv[0] = str(self.observer)
        last_library = max(i for i, token in enumerate(argv) if token.startswith("--extlib="))
        argv.insert(last_library + 1, "--extlib=" + str(self.libraries[-1]))
        argv.extend(
            [
                "--timing-model=" + str(self.model_path),
                "--timing-report=" + str(report),
                "--timing-scope-binding=" + str(scope),
                "--timing-input-sha256=" + json.loads(scope.read_bytes())["input_sha256"],
                str(elf),
            ]
        )
        return original, argv


def freeze_runtime(
    *,
    review_id,
    functional_argv,
    observer,
    disassembler,
    model_path,
    source_manifest,
    libraries,
    dependencies,
    entry_symbol,
    completion_symbol,
    timing_selection,
    observation_resource_protocol=None,
):
    """Review explicit runtime bytes; this supplies no hardware correspondence certificate."""
    if not isinstance(review_id, str) or not review_id.strip():
        raise StageGateError("archive count runtime needs explicit operator review attribution")
    if (
        type(functional_argv) is not tuple
        or not functional_argv
        or any(type(v) is not str or not v or "\0" in v for v in functional_argv)
        or any(v in {"-l", "--log-commits"} or v.startswith("--timing-") for v in functional_argv)
    ):
        raise StageGateError("archive count runtime needs the original nontracing functional command without an ELF")
    if any(not isinstance(p, Path) for p in (observer, disassembler, model_path, source_manifest)):
        raise StageGateError("archive runtime tool selections require explicit paths")
    if (
        type(libraries) is not tuple
        or len(libraries) < 2
        or type(dependencies) is not tuple
        or any(not isinstance(p, Path) for p in (*libraries, *dependencies))
    ):
        raise StageGateError("archive count runtime needs ordered functional libraries then one timing provider")
    model, model_pin = metadata(model_path)
    manifest, _ = metadata(source_manifest)
    expected_scope = E.kernel_call_scope(timing_selection)
    if "owned_model_scope" in timing_selection:
        expected_scope = scope_digest(timing_selection)
        if entry_symbol != "merlin_run_multi" or completion_symbol != "return_after_synchronous_dispatch":
            raise StageGateError("owned runtime must select its real call and synchronous return contract")
    if (
        model.get("schema") != E.MODEL_SCHEMA
        or model.get("profile") != "phase1-public"
        or model.get("scope") != {"kind": "counter_pc_pair"}
        or model.get("domain", {}).get("feature_engine_sha256") != manifest.get("sha256")
        or model.get("domain", {}).get("scope_sha256") != expected_scope
    ):
        raise StageGateError("archive count runtime differs from reviewed public source/scope model")
    original_libraries = [v.partition("=")[2] for v in functional_argv if v.startswith("--extlib=")]
    if original_libraries != [str(p) for p in libraries[:-1]]:
        raise StageGateError("archive observer must preserve original ordered functional libraries")
    if [F.pin(p, max_bytes=MAX_TOOL_BYTES)["sha256"] for p in libraries] != model.get("libraries"):
        raise StageGateError("archive count runtime libraries differ from the selected model")
    if [v.partition("=")[2] for v in functional_argv if v.startswith("--isa=")] != [model["cpu"]["isa"]]:
        raise StageGateError("archive count runtime ISA differs from the selected model")
    for symbol in (entry_symbol, completion_symbol):
        if type(symbol) is not str or not symbol.isidentifier():
            raise StageGateError("archive count runtime requires reviewed plain entry/completion symbols")
    timing_pins = timing_selection.get("counter_semantics", {}).get("source_pins", [])
    if not timing_pins:
        raise StageGateError("archive counter selection lacks original source pins")
    for pin in timing_pins:
        if F.pin(Path(pin["path"]), max_bytes=MAX_TOOL_BYTES) != pin:
            raise StageGateError("archive counter source changed")
    paths = tuple(
        dict.fromkeys(
            (
                Path(functional_argv[0]),
                observer,
                disassembler,
                model_path,
                source_manifest,
                *libraries,
                *dependencies,
                *(Path(p["path"]) for p in timing_pins),
            )
        )
    )
    files = tuple((str(path), F.pin(path, max_bytes=MAX_TOOL_BYTES)["sha256"]) for path in paths)
    if dict(files)[str(model_path)] != model_pin["sha256"]:
        raise StageGateError("archive model changed during runtime selection")
    result = ArchiveSpikeRuntime(
        review_id,
        functional_argv,
        observer,
        disassembler,
        model_path,
        source_manifest,
        libraries,
        dependencies,
        entry_symbol,
        completion_symbol,
        canonical_json(timing_selection),
        files,
        canonical_json(model),
        observation_resource_protocol,
    )
    result.verify()
    return result


# ---- owned_model_observation -------------------------------------------------------------------
# Owned synchronous model calls through the existing one-ELF timing observer.
#
# This adapter changes neither linked code nor historical build evidence. It
# requires the original completed recipe and source-bound dispatch proof, and
# separately checks returning device paths before admitting a whole-call window.

SCHEMA = "merlin.owned_synchronous_model_scope.v1"

REVIEW_SCHEMA = "merlin.owned_model_counter_window.v1"

SCOPE_CONTRACT = {
    "schema": SCHEMA,
    "caller": "main",
    "invocation_symbol": "merlin_run_multi",
    "completion": "all_dispatched_device_paths_synchronous",
    "includes": "descriptor_setup_host_work_between_device_calls_and_completion",
}


def scope_digest(selection):
    if selection.get("owned_model_scope") != SCOPE_CONTRACT:
        raise ValueError("whole-model timing requires its explicit synchronous scope selection")
    return document_sha256({"schema": "merlin.owned_model_timing_scope.v1", "selection": selection})


def synchronous_window(rows, *, counter_asm):
    """The selected runtime's direct synchronous model invocation and setup.

    Completion is supplied by the separately checked owned model contract, not
    by renaming a kernel or assuming a function return drains an accelerator.
    """
    pattern = E._instruction_form(counter_asm)
    if not pattern[0] or pattern[1].count("%0") != 1:
        raise ValueError("owned model counter instruction is malformed")

    def counter(row):
        return (
            row["mnemonic"] == pattern[0]
            and len(row["operands"]) == len(pattern[1])
            and all(want == "%0" or have == want for have, want in zip(row["operands"], pattern[1], strict=True))
        )

    body = [row for row in rows if row["function"] == "main"]
    reads = [row for row in body if counter(row)]
    if len(reads) != 2 or sum(counter(row) for row in rows) != 2:
        raise ValueError("owned model needs exactly one whole-program counter pair")
    middle = [row for row in body if reads[0]["pc"] < row["pc"] < reads[1]["pc"]]
    calls = [row for row in middle if "merlin_run_multi" in row["symbols"]]
    if len(calls) != 1 or calls[0]["mnemonic"] not in {"jal", "call"}:
        raise ValueError("owned model counter pair lacks one direct full model call")
    direct = [row for row in middle if row["mnemonic"] in {"jal", "call"}]
    sequences = (("merlin_run_multi",), ("merlin_reset_session", "merlin_prepare_step", "merlin_run_multi"))
    admitted = next(
        (
            names
            for names in sequences
            if len(names) == len(direct)
            and all(name in row["symbols"] for name, row in zip(names, direct, strict=True))
        ),
        None,
    )
    if admitted is None:
        raise ValueError("owned model counter pair has unsupported runtime call order")
    # The original driver and dispatcher sources are separately checked by
    # _runtime_sources. Their ordinary reset/step setup belongs to this whole
    # runtime ROI. No other call, indirect control or branch is admitted.
    admitted_calls = {id(row) for row in direct}
    allowed = {
        "li",
        "lui",
        "auipc",
        "addi",
        "addiw",
        "add",
        "sub",
        "mv",
        "slli",
        "srli",
        "andi",
        "ori",
        "ld",
        "lw",
        "lwu",
        "sd",
        "sw",
        "nop",
    }
    if any(id(row) not in admitted_calls and row["mnemonic"] not in allowed for row in middle):
        raise ValueError("owned model counter window has unsupported control or work")
    return dict(
        caller="main",
        begin_pc=reads[0]["pc"],
        end_pc=reads[1]["pc"],
        occurrence=0,
        invocation_symbol="merlin_run_multi",
        invocation_call_pc=calls[0]["pc"],
        included_runtime_calls=[dict(symbol=name, pc=row["pc"]) for name, row in zip(admitted, direct, strict=True)],
        completion="all dispatched device functions return after selected completion fence",
        counter_reads=[dict(pc=r["pc"], mnemonic=r["mnemonic"], operands=list(r["operands"])) for r in reads],
        program_counter_reads=2,
    )


def _runtime_sources(recipe):
    """The driver and descriptor dispatcher must be the ordinary owned runtime."""
    from merlin.runtime.backends.baremetal.image_link import _c_runtime_dir, _harness_dir

    expected = {"model_main.o": _harness_dir() / "model_main.c", "merlin_model.o": _c_runtime_dir() / "merlin_model.c"}
    pins = []
    for obj, source in expected.items():
        commands = [r for r in recipe["commands"] if Path(r["output"]["path"]).name == obj]
        if len(commands) != 1 or not any(
            item["sha256"] == F.digest(source) and Path(item["path"]).name == source.name
            for item in commands[0]["inputs"]
        ):
            raise ValueError("owned model did not consume the selected synchronous runtime")
        pins.append(F.pin(source, max_bytes=1 << 20))
    wrappers = [r for r in recipe["commands"] if Path(r["output"]["path"]).name == "model_call.o"]
    if len(wrappers) != 1:
        raise ValueError("owned model lacks one compiled invocation wrapper")
    inputs = [x for x in wrappers[0]["inputs"] if Path(x["path"]).name == "model_call.c"]
    if len(inputs) != 1:
        raise ValueError("owned invocation wrapper source is ambiguous")
    path = F.canonical(Path(inputs[0]["path"]))
    # This tiny generated bridge has no branches, async launch, or alternate
    # callback. Reconstruct its complete source, including every descriptor.
    text = path.read_text()
    n = text.partition("_mlir_ciface_forward(")[2].partition(");")[0].count("void*")
    if not 1 <= n <= 4096:
        raise ValueError("owned invocation wrapper signature is unsupported")
    expected_text = (
        "/* Generated. */\nextern void _mlir_ciface_forward("
        + ",".join(["void*"] * n)
        + ");\nvoid merlin_invoke(void **d) { _mlir_ciface_forward("
        + ",".join(f"d[{i}]" for i in range(n))
        + "); }\n"
    )
    if text != expected_text:
        raise ValueError("owned invocation wrapper is not the ordinary synchronous bridge")
    pins.append(F.pin(path, max_bytes=1 << 20))
    return pins


def _completion_modules(dispatch):
    from xdsl.parser import Parser

    from merlin.compiler.ir.llvm_context import make_llvm_context
    from merlin.target.isa.rocc.emitted_commands import closing_fence_on_normal_returns

    artifacts = dispatch.get("device_artifacts")
    roster = (dispatch.get("host_calls") or {}).get("mandatory_call_roster")
    if not isinstance(artifacts, dict) or not artifacts or set(artifacts) != set(roster or {}):
        raise ValueError("whole-call completion needs every dispatched device group")
    host_pin = (dispatch.get("host_calls") or {}).get("generic_cfg")
    if not isinstance(host_pin, dict):
        raise ValueError("owned completion has no actual host CFG")
    host_path = F.canonical(Path(host_pin["path"]))
    if F.digest(host_path) != host_pin["sha256"]:
        raise ValueError("owned host CFG changed")
    host = Parser(make_llvm_context(), host_path.read_text()).parse_module()
    defined = {op.sym_name.data for op in host.body.block.ops if op.name == "llvm.func" and op.body.blocks}
    allowed = (
        set(roster)
        | {"_mlir_ciface_" + name for name in roster}
        | defined
        | {"malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset"}
    )
    for op in host.walk():
        if op.name in {"llvm.inline_asm", "llvm.invoke"}:
            raise ValueError("owned host has opaque completion effects")
        if op.name == "llvm.call":
            attr = op.properties.get("callee")
            callee = getattr(getattr(attr, "root_reference", None), "data", None)
            if callee not in allowed:
                raise ValueError("owned host has unsupported or indirect completion effects")
    pins = [F.pin(host_path, max_bytes=16 << 20)]
    for name, row in artifacts.items():
        if type(roster[name]) is not int or roster[name] < 1:
            raise ValueError("whole-call dispatch multiplicity is absent")
        selected = row.get("optimized_llvm_mlir") or row.get("llvm_mlir")
        if not isinstance(selected, dict):
            raise ValueError("whole-call device completion source is missing")
        path = F.canonical(Path(selected["path"]))
        pin = F.pin(path, max_bytes=16 << 20)
        if pin["sha256"] != selected["sha256"]:
            raise ValueError("whole-call completion source changed")
        module = Parser(make_llvm_context(), path.read_text()).parse_module()
        if closing_fence_on_normal_returns(module) is not True:
            raise ValueError("whole-call device completion is unknown on a returning path")
        pins.append(pin)
    return pins


@dataclass(frozen=True)
class OwnedModelScope:
    """Only constructed after fresh owned source/dispatch/recipe verification."""

    elf: Path
    recipe_path: Path
    recipe_sha256: str
    evidence_json: bytes
    files: tuple[tuple[str, str], ...]

    def producer(self, elf, limits):
        from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe

        if F.canonical(Path(elf)) != self.elf:
            raise ValueError("owned timing selected another producer executable")
        for path, digest in self.files:
            if F.digest(Path(path)) != digest:
                raise ValueError("owned timing completion/source evidence changed")
        recipe = verify_completed_recipe(
            self.recipe_path, executable=self.elf, expected_recipe_sha256=self.recipe_sha256
        )
        if recipe["schema"] != "merlin.compilation_recipe.v2":
            raise ValueError("owned timing requires original v2 link custody")
        if recipe["executable"]["sha256"] != json.loads(self.evidence_json)["elf"]["sha256"]:
            raise ValueError("owned timing ELF differs from completed dispatch")
        return F.pin(self.elf, max_bytes=limits.max_elf_bytes)

    def validate_console(self, console):
        from hashlib import sha256

        if sha256(console).hexdigest() != json.loads(self.evidence_json)["numerical_console"]["sha256"]:
            raise ValueError("owned timing console differs from original complete numerical execution")

    def window(self, rows, *, counter_asm):
        result = synchronous_window(rows, counter_asm=counter_asm)
        result["owned_completion_contract"] = json.loads(self.evidence_json)
        result["owned_completion_contract_sha256"] = document_sha256(result["owned_completion_contract"])
        return result, REVIEW_SCHEMA


def prepare(result, *, target, evidence_root=None):
    """Reopen the native owner's evidence; a boolean completion claim is not enough."""
    from merlin.compiler.lower.model.compilation_recipe import verify_completed_recipe
    from merlin.development.targetgen.execution.host.model_audit import _dispatch_proof
    from merlin.target.capabilities import manifest as capability_manifest
    from merlin.target.registry import identity as target_registry

    selected = target_registry.resolve(target)
    manifest = capability_manifest.load_capability_manifest(target)
    if (
        manifest.endpoint_kind != "inline_asm_insn"
        or manifest.contract.get("logical_harness", {}).get("completion_asm") != "fence"
    ):
        raise ValueError("owned completion requires a declared RoCC host-fence contract")
    completion_contract = F.pin(selected.contract_path, max_bytes=16 << 20)
    record = result.get("owned_host_native_execution")
    if (
        not isinstance(record, dict)
        or record.get("status") != "verified"
        or (record.get("tiers", {}).get("L2") or {}).get("status") != "pass"
    ):
        raise ValueError("owned timing requires completed native model evidence")
    built = record["build"]["output"]
    elf = F.canonical(Path(built["elf"]))
    selected = built["compilation_recipe"]
    recipe_path = F.canonical(Path(selected["path"]))
    recipe = verify_completed_recipe(recipe_path, executable=elf, expected_recipe_sha256=selected["sha256"])
    if recipe["schema"] != "merlin.compilation_recipe.v2":
        raise ValueError("owned timing requires original v2 compilation recipe")
    if recipe["executable"]["sha256"] != built["elf_sha256"]:
        raise ValueError("owned timing ELF differs from numerical producer")
    runtime_pins = _runtime_sources(recipe)
    facts = record["rtl_facts"]
    facts_path = F.canonical(Path(facts["path"]))
    if F.digest(facts_path) != facts["sha256"] or facts.get("target") != target:
        raise ValueError("owned timing facts differ from selected target")
    if evidence_root is None:
        raise ValueError("owned observation requires its own review output directory")
    evidence_root = F.canonical(Path(evidence_root), exists=False)
    evidence_root.mkdir(mode=0o700)
    # JSON preserves the group's integer values but necessarily stringifies
    # mapping keys. Restore only canonical nonnegative integer keys, without
    # changing the source/device join or its roster.
    source_join = record["source_device_join"]
    joined = {}
    for key, value in source_join["joined_groups"].items():
        if not ((type(key) is int and key >= 0) or (type(key) is str and key.isdecimal() and str(int(key)) == key)):
            raise ValueError("owned source group index is noncanonical")
        index = int(key)
        if index in joined:
            raise ValueError("owned source group index is duplicated")
        joined[index] = value
    current = dict(record, source_device_join=dict(source_join, joined_groups=joined))
    dispatch = _dispatch_proof(current, target=target, facts=facts, evidence_root=evidence_root)
    if dispatch.get("status") != "verified" or dispatch["linked_elf"]["sha256"] != built["elf_sha256"]:
        raise ValueError("owned timing has no complete source-bound dispatch")
    completion_pins = _completion_modules(dispatch)
    numeric_console = record["tiers"]["L2"]["console"]
    console_path = F.canonical(Path(numeric_console["path"]))
    if F.digest(console_path) != numeric_console["sha256"]:
        raise ValueError("owned numerical console changed")
    evidence = dict(
        schema=SCHEMA,
        elf=F.pin(elf, max_bytes=64 << 20),
        recipe=selected,
        dispatch=dispatch,
        runtime_sources=runtime_pins,
        completion_sources=completion_pins,
        completion_contract=completion_contract,
        numerical_console=F.pin(console_path, max_bytes=64 << 20),
        completion_scope="all returning dispatched device paths; synchronous owned host call",
        physical_timing_qualified=False,
    )
    pins = [*runtime_pins, *completion_pins, completion_contract, facts, numeric_console]
    # Include the dispatch's host/object/LLVM custodians, not just fence source.
    for row in dispatch["device_artifacts"].values():
        pins.extend(v for v in row.values() if isinstance(v, dict) and "path" in v and "sha256" in v)
    pins.extend(v for v in dispatch["host_calls"].values() if isinstance(v, dict) and "path" in v and "sha256" in v)
    return OwnedModelScope(
        elf,
        recipe_path,
        selected["sha256"],
        json.dumps(evidence, sort_keys=True).encode(),
        tuple(sorted({(p["path"], p["sha256"]) for p in pins})),
    )
