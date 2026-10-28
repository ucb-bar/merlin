"""Checked shared-RHS/full-K member schedules using existing dense libraries.

This is a conditional composition, not a latency policy. Selection is explicit;
unknown epoch laws preserve per-effect drains. Target encoders, selected facts,
source grouping and independent execution qualification remain caller-owned.
"""

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

from merlin.sched.ir import Call, Const, Kernel, Loop, Ptr, TensorArg, Var, check_structure
from merlin.transforms.dense_planning import DenseProblem, DenseResources, dense_blocks
from merlin.transforms.pipeline_state import PassSpec, PipelineContext, PipelineSession

from .block_schedule import Load, Preload
from .compact_dense import _digest, _edge, _emit_dense_tile_group, _emit_lhs_panel, _Group, _groups
from .compact_dense_group import _callback, _same_callback
from .compact_dense_group_check import _group_contract_record, _group_output_contracts
from .dense_panel_epochs import EPOCH_FLAGS, DensePanelEpochLaw
from .dense_panel_pipeline import DRAIN, complete_dense_retained_reduction
from .dense_rhs_residency import plan_dense_rhs_residency
from .dense_shared_rhs_check import DenseRHSReuseLaw, shared_rhs_views, verify_dense_shared_rhs
from .schedule_ir import dense_schedule_text
from .shared_rhs_schedule_ir import make_shared_rhs_schedule_plan, read_shared_rhs_schedule_plan


def _emit(problems, blocks, storages, encode_member, name, dtypes, readouts):
    serial = 0

    def variable(prefix):
        nonlocal serial
        serial += 1
        return prefix + str(serial)

    def nest(extents, action):
        names = tuple(variable("tile") for _ in extents)
        body = tuple(action(*(Var(v) for v in names)))
        for var, extent in reversed(tuple(zip(names, extents, strict=True))):
            body = (Loop(var, extent, body),)
        return body

    p = problems[0]
    kg, ng = _Group(None, 1, Const(0), p.tiles[1], True), _Group(None, 1, Const(0), p.tiles[2], True)

    def rhs(k, n):
        return (
            encode_member(
                None,
                Load(
                    "weight",
                    (k, n),
                    k * p.tile_k,
                    n * p.tile_n,
                    _edge(kg, k, p.k, p.tile_k),
                    _edge(ng, n, p.n, p.tile_n),
                    storages[0].rhs.base + (k * p.tiles[2] + n) * p.tile_k,
                    1,
                ),
            ),
        )

    sections = [nest((p.tiles[1], p.tiles[2]), rhs)]
    for member, (p, b, s, r) in enumerate(zip(problems, blocks, storages, readouts, strict=True)):
        body = []

        def call(op):
            if type(op) is Preload:
                k, n = op.weight_block
                op = replace(op, weight_row=s.rhs.base + (k * p.tiles[2] + n) * p.tile_k)
            return encode_member(member, op)

        # Existing emitter supplies source order, clipped tails and exact readout.
        # Its per-panel RHS loads are suppressed only here; the global prefix is
        # preserved in every independent source proof below.
        def without_rhs(items):
            kept = []
            for item in items:
                if type(item) is Loop:
                    nested = without_rhs(item.body)
                    if nested:
                        kept.append(replace(item, body=nested))
                elif item is not None:
                    kept.append(item)
            return tuple(kept)

        def member_call(op):
            return None if type(op) is Load and op.role == "weight" else call(op)

        for mg in _groups(p.tiles[0], b.m, variable("outer_m")):
            inner = list(_emit_lhs_panel(p, b, s, call, nest, mg, _Group(None, 1, Const(0), p.tiles[1], True)))
            for ng in _groups(p.tiles[2], b.n, variable("outer_n")):
                emitted = without_rhs(
                    _emit_dense_tile_group(
                        p,
                        b,
                        s,
                        member_call,
                        variable,
                        nest,
                        mg,
                        ng,
                        include_lhs=False,
                        output_readout=r,
                    )
                )
                inner.extend((Loop(ng.var, ng.trips, emitted),) if ng.var else emitted)
            body.extend((Loop(mg.var, mg.trips, tuple(inner)),) if mg.var else inner)
        sections.append(tuple(body))
    return Kernel(
        name,
        (
            TensorArg("rhs", (problems[0].k, problems[0].n), dtypes[1], "read"),
            *(TensorArg(f"lhs{i}", (p.m, p.k), dtypes[0], "read") for i, p in enumerate(problems)),
            *(
                TensorArg(
                    f"output{i}", (p.m, p.n), dtypes[2] if readouts[i] is None else readouts[i].output_dtype, "write"
                )
                for i, p in enumerate(problems)
            ),
        ),
        tuple(Loop(f"shared_rhs_section_{i}", 1, body) for i, body in enumerate(sections)),
    )


def _complete(kernel, problems, blocks, resources, storages, adapter, project_member, asynchronous):
    serial = 0

    def synchronous(body):
        result = []
        for item in body:
            if type(item) is Loop:
                result.append(replace(item, body=synchronous(item.body)))
            else:
                result.append(item)
                if type(project_member(item)[1]) is not Preload:
                    result.append(Call(DRAIN, ()))
        return tuple(result)

    def rename_epochs(body):
        nonlocal serial
        result = []
        for item in body:
            if type(item) is Loop:
                var = item.var
                if var.startswith("dense_retained_epoch_"):
                    serial += 1
                    var = f"shared_rhs_completed_epoch_{serial}"
                result.append(replace(item, var=var, body=rename_epochs(item.body)))
            else:
                result.append(item)
        return tuple(result)

    sections = []
    projected = replace(adapter, project=lambda call: project_member(call)[1])
    for i, section in enumerate(kernel.body):
        member = max(0, i - 1)
        if asynchronous:
            completed = complete_dense_retained_reduction(
                replace(kernel, body=section.body),
                problems[member],
                blocks[member],
                resources,
                storages[member],
                projected,
            ).body
        else:
            completed = synchronous(section.body)
        sections.append(replace(section, body=rename_epochs(completed)))
    return replace(kernel, body=tuple(sections))


@dataclass(frozen=True)
class DenseSharedRHSResult:
    kernel: Kernel
    blocks: tuple
    realization: dict
    plan_text: str
    checked_emission: object
    _guard: object

    def revalidate(self):
        self._guard(self)

    def emit_index_program(self, control, lower_effect, *, complete_epoch):
        """Lower checked scalar operands; target binds the verified pointer schemas.

        Pointer metadata is removed only after exact source-pointer checking.
        Actual member, source rows/columns and all physical indices survive.
        The receipt does not qualify the target's pointer reconstruction.
        """
        from .llvm_program import emit_index_effect_program

        self.revalidate()
        schemas = set()

        def body(items):
            result = []
            for item in items:
                if type(item) is Loop:
                    result.append(replace(item, body=body(item.body)))
                else:
                    schemas.update((item.instr, key, value.tensor) for key, value in item.args if type(value) is Ptr)
                    result.append(replace(item, args=tuple((k, v) for k, v in item.args if type(v) is not Ptr)))
            return tuple(result)

        lowered = replace(self.kernel, body=body(self.kernel.body))
        receipt = emit_index_effect_program(
            lowered, control, lower_effect, complete_epoch=complete_epoch, completion_instruction=DRAIN
        )
        self.revalidate()
        return {
            **receipt,
            "checked_kernel_sha256": self.kernel.digest(),
            "pointer_schemas": sorted(schemas),
            "target_pointer_binding": "caller_obligation",
        }


def schedule_dense_shared_rhs(
    problems,
    resources,
    storages,
    adapter,
    *,
    encode_member,
    project_member,
    source_identity,
    name,
    dtypes,
    static_budget,
    external_buffers_disjoint,
    retention_law,
    epoch_admission=None,
    epoch_law=None,
    schedule_plan=None,
    block_tiles=None,
    output_readouts=None,
    readout_qualification_identities=None,
    selected_contracts=None,
    verification_timeout_ms=5000,
    max_members=128,
):
    """Compose existing block, retention and epoch libraries, then check effects.

    This first binding admits zero-seeded, square primitive tiles and a complete
    common RHS. Unsupported capacity returns None for the original route. It
    makes no performance preference or independent target qualification claim.
    """
    if (
        type(problems) is not tuple
        or type(max_members) is not int
        or not 2 <= len(problems) <= max_members <= 128
        or any(type(p) is not DenseProblem for p in problems)
        or type(resources) is not DenseResources
        or type(static_budget) is not int
        or static_budget <= 0
        or type(verification_timeout_ms) is not int
        or verification_timeout_ms <= 0
        or type(dtypes) is not tuple
        or len(dtypes) != 3
    ):
        raise ValueError("shared RHS requires a bounded typed source group/resources/dtypes")
    if type(retention_law) is not DenseRHSReuseLaw or not retention_law.admits(adapter):
        return None
    first = problems[0]
    if (
        any(
            p.bias
            or (p.k, p.n, p.tile_m, p.tile_k, p.tile_n, p.lhs_bytes, p.rhs_bytes, p.accumulator_bytes)
            != (
                first.k,
                first.n,
                first.tile_m,
                first.tile_k,
                first.tile_n,
                first.lhs_bytes,
                first.rhs_bytes,
                first.accumulator_bytes,
            )
            for p in problems
        )
        or not first.tile_m == first.tile_k == first.tile_n
    ):
        return None
    shared_rhs_views(problems, storages, adapter)
    contracts, readouts, qualifications = _group_output_contracts(
        len(problems),
        selected_contracts,
        output_readouts,
        readout_qualification_identities,
        require_selected=False,
    )
    if epoch_law is not None and type(epoch_law) is not DensePanelEpochLaw:
        raise ValueError("shared RHS epochs require the existing selected DensePanelEpochLaw")
    unknown = tuple(EPOCH_FLAGS) if epoch_law is None else epoch_law.unknown()
    asynchronous = epoch_admission is not None and not unknown
    if asynchronous:
        epoch_law.validate_for(adapter, epoch_admission)
    rhs_rows = first.tiles[1] * first.tiles[2] * first.tile_k
    rhs = storages[0].rhs
    if (
        plan_dense_rhs_residency(
            first.m,
            first.k,
            first.n,
            block=first.tile_k,
            rhs_base=rhs.base,
            operand_rows=rhs.base + rhs.capacity,
            accumulator_rows=storages[0].accumulator.capacity,
            max_static_rhs_tiles=first.tiles[1] * first.tiles[2],
        )
        is None
    ):
        return None
    choices = []
    for p, s in zip(problems, storages, strict=True):
        candidates = [
            b
            for b in dense_blocks(p, resources, k_limit=lambda m, n: s.k_limit(p, resources, m, n))
            if b.k == p.tiles[1]
            and s.admits(p, b, resources)
            and b.m * b.k * p.tile_m * s.lhs.unit_bytes + rhs_rows * rhs.unit_bytes <= resources.operand_bytes
        ]
        if not candidates:
            return None
        # Deterministic bounded proposal; empirical policy chooses among plans.
        choices.append(max(candidates, key=lambda b: (b.m * b.n, b.n, b.m)))
    if block_tiles is None:
        block_tiles = tuple((b.m, b.k, b.n) for b in choices)

    def context():
        return _digest(
            (
                tuple(asdict(p) for p in problems),
                asdict(resources),
                tuple(asdict(s) for s in storages),
                source_identity,
                dtypes,
                static_budget,
                external_buffers_disjoint,
                retention_law.record(),
                None if epoch_law is None else asdict(epoch_law),
                None if epoch_admission is None else asdict(epoch_admission),
                tuple(None if r is None else asdict(r) for r in readouts),
                qualifications,
                _group_contract_record(selected_contracts),
                adapter.configuration_identity,
                adapter.numerical_policy_identity,
                adapter.qualification_identity,
                adapter.synchronous_completion,
                adapter.exact_accumulator_continuation,
                adapter.index_bits,
            )
        )

    original = context()
    if schedule_plan is None:
        schedule_plan = make_shared_rhs_schedule_plan(
            problems,
            resources,
            storages,
            adapter,
            source_identity=source_identity,
            context_identity=original,
            dtypes=dtypes,
            static_budget=static_budget,
            block_tiles=block_tiles,
            output_readouts=readouts,
            asynchronous=asynchronous,
        )
    decisions = read_shared_rhs_schedule_plan(
        schedule_plan,
        problems,
        resources,
        storages,
        adapter,
        source_identity=source_identity,
        context_identity=original,
        dtypes=dtypes,
        static_budget=static_budget,
        output_readouts=readouts,
        asynchronous=asynchronous,
    )
    blocks = tuple(d.block for d in decisions)
    if any(d.storage != s for d, s in zip(decisions, storages, strict=True)):
        raise ValueError("shared RHS schedule may not silently reassign original shared/private windows")
    if any(b.k != p.tiles[1] for b, p in zip(blocks, problems, strict=True)):
        return None
    import importlib

    from .library_sources import library_modules

    callbacks = tuple(_callback(f) for f in (encode_member, project_member, adapter.encode, adapter.project))
    paths = tuple(
        dict.fromkeys(
            (
                Path(__file__).resolve(),
                Path(__file__).with_name("dense_shared_rhs_check.py"),
                Path(__file__).with_name("shared_rhs_schedule_ir.py"),
                *(c[3] for c in callbacks),
                *(Path(importlib.import_module(m).__file__).resolve() for m in library_modules()),
            )
        )
    )
    pins = tuple(sha256(p.read_bytes()).hexdigest() for p in paths)

    def fresh():
        if (
            context() != original
            or pins != tuple(sha256(p.read_bytes()).hexdigest() for p in paths)
            or any(
                not _same_callback(f, c)
                for f, c in zip(
                    (encode_member, project_member, adapter.encode, adapter.project), callbacks, strict=True
                )
            )
        ):
            raise ValueError("shared RHS original inputs, implementation or callbacks changed; replan")
        for c in contracts:
            if c is not None:
                c.revalidate()

    fresh()
    kernel = _complete(
        _emit(problems, blocks, storages, encode_member, name, dtypes, readouts),
        problems,
        blocks,
        resources,
        storages,
        adapter,
        project_member,
        asynchronous,
    )
    state = {"kernel": kernel}
    session = PipelineSession(
        state,
        context=PipelineContext(
            hardware=adapter.configuration_identity,
            numerical_policy=adapter.numerical_policy_identity,
            resources=_digest((asdict(resources), tuple(asdict(s) for s in storages))),
            execution=original,
            library=_digest(pins),
        ),
        ir_level="shared_rhs",
        max_repairs=0,
        fingerprint=lambda current: current["kernel"].digest(),
        verify_ir=lambda current: (
            (_ for _ in ()).throw(ValueError("invalid shared RHS structure"))
            if check_structure(current["kernel"])
            else None
        ),
    )
    spec = PassSpec("shared_rhs_epochs", 0, produces=("checked_emission",), mutates_ir=False, ir_level="shared_rhs")

    def check(current):
        fresh()
        report = verify_dense_shared_rhs(
            current["kernel"],
            problems,
            blocks,
            resources,
            storages,
            adapter,
            project_member=project_member,
            retention_law=retention_law,
            dtypes=dtypes,
            static_budget=static_budget,
            external_buffers_disjoint=external_buffers_disjoint,
            epoch_admission=epoch_admission,
            epoch_law=epoch_law,
            output_readouts=readouts,
            readout_qualification_identities=qualifications,
            selected_contracts=selected_contracts,
            timeout_ms=verification_timeout_ms,
        )
        fresh()
        return report

    session.run(spec, lambda: {"checked_emission": check(state)})
    report = session.analysis("checked_emission")
    text = dense_schedule_text(schedule_plan) if not isinstance(schedule_plan, str) else schedule_plan
    digest = kernel.digest()
    realization = {
        **report,
        "source_sha256": source_identity,
        "kernel_sha256": digest,
        "schedule_sha256": sha256(text.encode()).hexdigest(),
        "schedule_consumed": True,
        "unknown_epoch_premises": unknown,
        "machine_emission": "not_established",
        "readout_qualification_identities": qualifications,
        "original_source_binding": "caller_obligation",
        "pipeline": {"passes": [asdict(record) for record in session.records]},
    }
    stamp = session.stamp()
    checked = _digest((tuple(asdict(b) for b in blocks), realization))

    def guard(result):
        fresh()
        if (
            result.kernel.digest() != digest
            or result.plan_text != text
            or result.checked_emission != stamp
            or _digest((tuple(asdict(b) for b in result.blocks), result.realization)) != checked
        ):
            raise ValueError("checked shared RHS program or schedule changed; recheck")

    return DenseSharedRHSResult(kernel, blocks, realization, text, stamp, guard)
