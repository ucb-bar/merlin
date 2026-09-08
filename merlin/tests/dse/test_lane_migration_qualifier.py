"""Target-neutral lane-transition selection and bounded source-runtime orchestration."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

from merlin.perf import lane_migration_qualifier as Q
from merlin.perf.host_physical_transition_qualifier import ChangedRegionQualifierDispatch


def contraction(*, batches, lhs_broadcast=False, suffix="0"):
    names = [f"b{i}" for i in range(len(batches))]
    domain = ",".join([*names, "m", "n", "k"])
    batch = "x".join(map(str, batches))
    lhs_shape = "4x6" if lhs_broadcast else f"{batch}x4x6"
    rhs_shape = f"{batch}x6x5"
    out_shape = f"{batch}x4x5"
    lhs_map = "m,k" if lhs_broadcast else ",".join([*names, "m", "k"])
    rhs_map = ",".join([*names, "k", "n"])
    out_map = ",".join([*names, "m", "n"])
    iterators = ",".join([*["\"parallel\""] * (len(batches) + 2), "\"reduction\""])
    return f'''%r{suffix}=linalg.generic {{indexing_maps=[affine_map<({domain})->({lhs_map})>,affine_map<({domain})->({rhs_map})>,affine_map<({domain})->({out_map})>],iterator_types=[{iterators}]}}
ins(%a{suffix},%b{suffix}:tensor<{lhs_shape}xi8>,tensor<{rhs_shape}xi8>) outs(%c{suffix}:tensor<{out_shape}xi32>) {{
^bb0(%x:i8,%y:i8,%acc:i32):
%xx{suffix}=arith.extsi %x:i8 to i32
%yy{suffix}=arith.extsi %y:i8 to i32
%p{suffix}=arith.muli %xx{suffix},%yy{suffix}:i32
%v{suffix}=arith.addi %p{suffix},%acc:i32
linalg.yield %v{suffix}:i32
}}->tensor<{out_shape}xi32>''', (lhs_shape, rhs_shape, out_shape)


def source():
    first, one = contraction(batches=(7,), suffix="0")
    second, two = contraction(batches=(2, 3), lhs_broadcast=True, suffix="1")
    args = []
    for suffix, shapes in (("0", one), ("1", two)):
        args.extend(f"%{name}{suffix}:tensor<{shape}xi{dtype}>" for name, shape, dtype in
                    zip(("a", "b", "c"), shapes, ("8", "8", "32"), strict=True))
    return ("module {func.func @work(" + ",".join(args) + ")->(tensor<7x4x5xi32>,tensor<2x3x4x5xi32>){\n"
            + first + "\n" + second
            + "\nfunc.return %r0,%r1:tensor<7x4x5xi32>,tensor<2x3x4x5xi32>\n}}")


def artifact(path, tasks):
    return {"interface": path, "lowered_text": "", "command_buffer": {
        "params": {"global_program_plan": {"tasks": tasks}}}}


def pair(tmp_path):
    path = tmp_path / "interface.mlir"
    path.write_text(source())
    before = artifact(path, [{"task_index": 7, "kind": "host", "source_op_indices": [0, 1]}])
    after = artifact(path, [
        {"task_index": 8, "kind": "contraction", "source_op_indices": [0]},
        {"task_index": 9, "kind": "contraction", "source_op_indices": [1]},
    ])
    return path, before, after


def test_iteration_rank_and_broadcast_classes_select_bounded_representatives(tmp_path):
    path, before, after = pair(tmp_path)
    assert Q.has_contraction_lane_migration(before, after)
    rows = Q.contraction_lane_migrations(path.read_text(), before, after)
    selected, coverage = Q.representative_migrations(rows)
    assert [row["iteration_rank"] for row in rows] == [4, 5]
    assert [row["operand_batching"]["lhs"] for row in rows] == ["batched", "broadcast"]
    assert len(selected) == 2
    assert coverage["qualified_iteration_ranks"] == [4, 5]
    assert coverage["qualified_operand_batching"] == ["batched/batched", "broadcast/batched"]


def test_qualifier_binds_selected_member_and_runs_both_arms_per_class(tmp_path, monkeypatch):
    path, before, after = pair(tmp_path)
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    selection = {"portfolio_index": 2, "capsule": "selected"}
    binding = lambda arm: {"arm": arm, "source_sha256": source_sha}
    selected = {"selection": selection,
        "previous": {"artifacts": before, "member_binding": binding("previous")},
        "current": {"artifacts": after, "member_binding": binding("current")}}
    selections = []
    def context(_candidate, supplied):
        assert supplied in (None, selection)
        selections.append(supplied)
        return selected
    experiment = SimpleNamespace(selected_changed_portfolio_context=context)
    rows = {row["source_op_index"]: row for row in
            Q.contraction_lane_migrations(path.read_text(), before, after)}
    prepared = []
    def prepare(**kwargs):
        assert kwargs["expected_route_transition"] == ("host", "contraction")
        assert kwargs["expected_short_route_transition"] == (None, "contraction")
        assert kwargs["portfolio_member"] == selection
        assert kwargs["max_batch_extent"] == 2 and kwargs["max_macs"] <= 96
        prepared.append(kwargs["source_op_index"])
        row = rows[kwargs["source_op_index"]]
        return {"schema": "source_contraction_preparation_v1", "status": "prepared",
                "source_op_index": kwargs["source_op_index"],
                "expected_route_transition": ["host", "contraction"],
                "extraction": {"source_iteration_rank": row["iteration_rank"],
                    "operand_batching": row["operand_batching"],
                    "source_batch_shape": row["source_batch_shape"],
                    "source_geometry_mkn": row["source_geometry_mkn"],
                    "probe_batch_shape": [min(value, 2) for value in row["source_batch_shape"]]},
                "arms": {"after": {"declared_route": "contraction"}}}
    monkeypatch.setattr(Q, "prepare_source_contraction", prepare)
    executions = []
    def runtime(**kwargs):
        executions.append(kwargs["prepared"]["source_op_index"])
        return {"status": "passed", "correct": True,
                "arms": {arm: {"correct": True} for arm in ("before", "after")}}
    qualifier = Q.LaneMigrationContractionQualifier(target="test", runtime_provider=runtime,
        abi_provenance={"host": "pinned"}, output=tmp_path / "receipts")
    receipt = qualifier(candidate=tmp_path, experiment=experiment, timeout_s=60,
                        portfolio_member=selection)
    assert receipt["status"] == "passed_reduced_lane_migration_witness", receipt
    assert prepared == executions == [0, 1]
    assert receipt["bounded_programs_executed"] == 4
    assert receipt["portfolio_member_binding"] == {
        "selection": selection, "previous": binding("previous"), "current": binding("current")}
    assert Path(receipt["detail_path"]).is_file() and len(receipt["detail_sha256"]) == 64
    assert selections == [selection, selection]


def test_dispatch_never_falls_back_from_lane_migration(tmp_path):
    _path, before, after = pair(tmp_path)
    calls = []
    class Provider:
        abi_provenance = {"host": "same"}
        def __init__(self, name): self.name = name
        def __call__(self, **_kwargs):
            calls.append(self.name)
            return {"status": "UNKNOWN"}
    dispatch = ChangedRegionQualifierDispatch(
        physical=Provider("physical"), lane_migration=Provider("lane"), legacy=Provider("legacy"))
    selected = {"selection": {"portfolio_index": 0},
                "previous": {"artifacts": before}, "current": {"artifacts": after}}
    experiment = SimpleNamespace(selected_changed_portfolio_context=lambda _c, _s: selected)
    dispatch(candidate=tmp_path, experiment=experiment, timeout_s=60)
    assert calls == ["lane"]


def test_ambiguous_or_unbound_transition_refuses_without_runtime(tmp_path):
    path, before, after = pair(tmp_path)
    after["command_buffer"]["params"]["global_program_plan"]["tasks"][0]["source_op_indices"] = [0, 99]
    try:
        Q.contraction_lane_migrations(path.read_text(), before, after)
    except ValueError as error:
        assert "single host-task migration" in str(error) or "outside" in str(error)
    else:
        raise AssertionError("malformed source ownership was accepted")
