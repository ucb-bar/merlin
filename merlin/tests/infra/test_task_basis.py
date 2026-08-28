"""The task basis must be derived, deterministic, and honest about what it left out.

Each rule here fails in a direction that would flatter the study if it regressed: a shape-keyed
signature turns the reuse ladder into a lookup table, an undetermined region counted either way
moves the coverage denominator, a summed tick share exceeds 100% of the model, and a lower-bound
weight quoted as exact turns a bound into a measurement.
"""
import json

import pytest

from merlin.benchharness import task_basis as TB


class _Row:
    """A census row, minimally. Mirrors kernels.census.CensusRow's read surface."""

    def __init__(self, index=0, op_class="matmul", family="contraction", role="",
                 parallel=(64, 64), reduction=(64,), dtypes=("f32", "f32", "f32"),
                 work=1000, work_complete=True, ticks=None, key="k"):
        self.index, self.op_class, self.family, self.role = index, op_class, family, role
        self.parallel, self.reduction, self.dtypes = parallel, reduction, dtypes
        self.work, self.work_complete, self.ticks, self.key = work, work_complete, ticks, key


class _Census:
    def __init__(self, rows, total_work=None, model_ticks=None, model="m", stage="s"):
        self.rows = tuple(rows)
        self.total_work = total_work if total_work is not None else sum(r.work for r in rows)
        self.model_ticks = model_ticks
        self.model, self.stage, self.source = model, stage, "src"

    def measured_share(self, rows=None):
        if not self.model_ticks:
            return None
        pool = self.rows if rows is None else rows
        # Deduplicated by tick bucket, exactly as the real Census does.
        buckets = {(r.key, r.role): (r.ticks or 0) for r in pool}
        return sum(buckets.values()) / float(self.model_ticks)


def _Cap(families=("contraction",)):
    """A real capability map, so the tests that do not stub eligibility exercise the real path.

    `_family_support` groups by family and reads dtypes/ranks off each entry, so a stand-in object
    would only prove the stub agrees with itself.
    """
    from merlin.targetgen.compute_units import SemanticCapability
    return {f: SemanticCapability(family=f, dtypes=("f32", "f16")) for f in families}


def _basis(rows, cap=None, **kw):
    return TB.derive_basis(_Census(rows), cap if cap is not None else _Cap(), **kw)


# --- the signature ------------------------------------------------------------------------------

def test_the_signature_excludes_exact_shape():
    """Two sizes of the same op are ONE task at two configs, not two tasks.

    A shape-keyed signature makes every shape its own task, which turns the kernel library into a
    lookup table and makes a specialization audit impossible.
    """
    a = TB._row_signature(_Row(parallel=(64, 64), reduction=(64,)), boundaries=())
    b = TB._row_signature(_Row(parallel=(72, 72), reduction=(64,)), boundaries=())
    assert a.key() == b.key()


def test_a_different_size_class_is_a_different_task():
    small = TB._row_signature(_Row(parallel=(4, 4), reduction=(4,)), boundaries=())
    huge = TB._row_signature(_Row(parallel=(4096, 4096), reduction=(4096,)), boundaries=())
    assert small.key() != huge.key()


def test_dtype_is_part_of_task_identity():
    a = TB._row_signature(_Row(dtypes=("f32", "f32", "f32")), boundaries=())
    b = TB._row_signature(_Row(dtypes=("f16", "f16", "f32")), boundaries=())
    assert a.key() != b.key()


def test_regime_boundaries_come_from_the_caller_not_the_module():
    """The boundaries are hardware facts; the module must not invent them."""
    ext = (32, 32)
    assert TB.shape_regime(ext, boundaries=(10, 100000)) == "b1"
    assert TB.shape_regime(ext, boundaries=(10, 100)) == "b2"


def test_the_fallback_regime_is_monotone_in_volume():
    seen = [TB.shape_regime((n,)) for n in (2, 32, 1024, 1 << 20)]
    assert len(set(seen)) == len(seen), "distinct magnitudes must not collapse into one regime"


# --- weighting ----------------------------------------------------------------------------------

def test_measured_ticks_are_preferred_over_static_work():
    rows = [_Row(index=0, ticks=50, key="a"), _Row(index=1, ticks=50, key="b")]
    c = _Census(rows, model_ticks=200)
    b = TB.derive_basis(c, _Cap())
    assert all(g.cost_source == "measured_ticks" for g in b.entries)


def test_a_tick_bucket_shared_by_two_rows_is_counted_once():
    """Summing per-row shares double-counts a bucket -- measured at 106% of a 100% model."""
    rows = [_Row(index=0, ticks=100, key="same"), _Row(index=1, ticks=100, key="same")]
    c = _Census(rows, model_ticks=100)
    b = TB.derive_basis(c, _Cap())
    assert sum(g.cost for g in b.entries) == pytest.approx(1.0)


def test_an_incomplete_work_row_marks_its_weight_a_lower_bound():
    rows = [_Row(index=0, work_complete=False)]
    b = _basis(rows)
    assert b.entries[0].weight_is_lower_bound is True
    assert b.certificate["cover_fraction_is_bounded"] is True


def test_a_complete_work_census_is_not_marked_bounded():
    assert _basis([_Row(index=0)]).certificate["cover_fraction_is_bounded"] is False


# --- eligibility --------------------------------------------------------------------------------

def test_an_undetermined_group_leaves_both_sides_of_the_ratio(monkeypatch):
    """Counting it either way moves the denominator; it must be excluded and reported."""
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=False, family="contraction", reason="evidence cannot decide", undetermined=True))
    b = _basis([_Row(index=0)])
    assert b.entries == ()
    assert b.certificate["denominator"] == 0
    assert len(b.certificate["excluded_undetermined"]) == 1
    assert not b.certificate["excluded_ineligible"]


def test_an_ineligible_group_is_excluded_with_its_reason(monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=False, family="contraction", reason="no unit supports it"))
    b = _basis([_Row(index=0)])
    assert b.entries == ()
    assert b.certificate["excluded_ineligible"][0]["reason"] == "no unit supports it"


def test_the_capability_map_cannot_shrink_the_measured_denominator(monkeypatch):
    """Eligibility decides what is IN SCOPE, never what the model costs."""
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=True, family="contraction", reason="ok"))
    rows = [_Row(index=0, ticks=80, key="a"), _Row(index=1, ticks=20, key="b")]
    b = TB.derive_basis(_Census(rows, model_ticks=100), _Cap())
    assert b.certificate["denominator"] == pytest.approx(1.0)


# --- the cover ----------------------------------------------------------------------------------

def test_the_cover_stops_once_the_target_is_reached(monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=True, family="contraction", reason="ok"))
    rows = [_Row(index=i, ticks=t, key=f"k{i}", parallel=(2 ** (i + 3), 4), reduction=(4,))
            for i, t in enumerate((90, 5, 3, 2))]
    b = TB.derive_basis(_Census(rows, model_ticks=100), _Cap(), cover_target=0.9,
                        family_floor=False)
    assert len(b.entries) == 1, "the 90% group alone reaches the target"
    assert b.certificate["cover_fraction"] >= 0.9


def test_the_basis_is_deterministic_across_runs(monkeypatch):
    """Same census twice must give the same basis, or nothing downstream is reproducible."""
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=True, family="contraction", reason="ok"))
    rows = [_Row(index=i, ticks=10, key=f"k{i}", parallel=(2 ** (i + 3), 4), reduction=(4,))
            for i in range(6)]
    keys = [TB.derive_basis(_Census(rows, model_ticks=100), _Cap()).signature_keys()
            for _ in range(2)]
    assert keys[0] == keys[1]


def test_ties_are_broken_by_signature_not_by_input_order(monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=True, family="contraction", reason="ok"))
    rows = [_Row(index=i, ticks=10, key=f"k{i}", parallel=(2 ** (i + 3), 4), reduction=(4,))
            for i in range(4)]
    a = TB.derive_basis(_Census(rows, model_ticks=100), _Cap()).signature_keys()
    b = TB.derive_basis(_Census(list(reversed(rows)), model_ticks=100), _Cap()).signature_keys()
    assert a == b


def test_an_out_of_range_cover_target_is_refused():
    with pytest.raises(ValueError):
        _basis([_Row()], cover_target=1.5)


# --- the family floor ---------------------------------------------------------------------------

def test_a_family_evidenced_but_cheap_still_gets_one_task(monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=True, family="f", reason="ok"))
    rows = [_Row(index=0, family="contraction", ticks=99, key="a"),
            _Row(index=1, family="normalization", ticks=1, key="b")]
    b = TB.derive_basis(_Census(rows, model_ticks=100), _Cap(("contraction", "normalization")),
                        cover_target=0.9)
    assert "normalization" in b.certificate["families_covered"]
    assert b.certificate["family_floor_added"] == ["normalization"]


def test_a_declared_but_unevidenced_family_gets_no_manufactured_task(monkeypatch):
    """Inventing a task the model never does, then reporting it covered, is the failure here."""
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        eligible=True, family="contraction", reason="ok"))
    b = TB.derive_basis(_Census([_Row(index=0, family="contraction", ticks=10, key="a")],
                                model_ticks=10),
                        _Cap(("contraction", "attention", "softmax")))
    assert b.certificate["families_declared_not_evidenced"] == ["attention", "softmax"]
    assert b.certificate["families_covered"] == ["contraction"]


# --- the certificate ----------------------------------------------------------------------------

def test_the_certificate_accounts_for_every_group(monkeypatch):
    """Every group must appear somewhere, or the 95% claim is unauditable."""
    calls = {"n": 0}

    def verdict(*_a, **_k):
        calls["n"] += 1
        if calls["n"] == 1:
            return TB.EL.EligibilityVerdict(True, "contraction", "ok")
        if calls["n"] == 2:
            return TB.EL.EligibilityVerdict(False, "movement", "unsupported")
        return TB.EL.EligibilityVerdict(False, "attention", "undecided", undetermined=True)

    monkeypatch.setattr(TB.EL, "is_eligible", verdict)
    rows = [_Row(index=0, family="contraction", parallel=(8, 8), ticks=10, key="a"),
            _Row(index=1, family="movement", parallel=(1024, 1024), ticks=5, key="b"),
            _Row(index=2, family="attention", parallel=(65536, 4), ticks=5, key="c")]
    b = TB.derive_basis(_Census(rows, model_ticks=20), _Cap())
    c = b.certificate
    accounted = (len(b.entries) + len(c["excluded_ineligible"])
                 + len(c["excluded_undetermined"]) + len(c["eligible_not_chosen"]))
    assert accounted == c["groups_total"] == 3


def test_the_certificate_names_the_census_stage(monkeypatch):
    """An int8 bundle's model.mlir is f32; a basis read from the wrong stage is wrong throughout."""
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        True, "contraction", "ok"))
    c = _Census([_Row(index=0)]); c.stage = "post-quantization"
    assert TB.derive_basis(c, _Cap()).certificate["census_stage"] == "post-quantization"


def test_the_certificate_says_whether_regimes_came_from_hardware(monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        True, "contraction", "ok"))
    assert _basis([_Row()]).certificate["regime_source"] == "log2_volume_fallback"
    b = TB.derive_basis(_Census([_Row()]), _Cap(), regime_boundaries=(1024,))
    assert b.certificate["regime_source"] == "target_facts"


def test_the_certificate_is_json_serialisable(tmp_path, monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        True, "contraction", "ok"))
    p = tmp_path / "basis_certificate.json"
    _basis([_Row()]).write_certificate(p)
    assert json.loads(p.read_text())["cover_target"] == TB.DEFAULT_COVER_TARGET


def test_groups_carry_their_shapes_so_a_config_ladder_can_be_built(monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        True, "contraction", "ok"))
    rows = [_Row(index=0, parallel=(64, 64), reduction=(64,), ticks=5, key="a"),
            _Row(index=1, parallel=(72, 72), reduction=(64,), ticks=5, key="b")]
    b = TB.derive_basis(_Census(rows, model_ticks=10), _Cap())
    assert b.entries[0].shapes == ((64, 64, 64), (72, 72, 64))


# --- family resolution ----------------------------------------------------------------------------

def test_an_op_class_in_the_family_field_is_resolved_to_a_semantic_family():
    """Measured on the seed model: every row said 'matmul' where capabilities say 'contraction'.

    Unresolved, every group is ineligible and the basis comes back EMPTY -- which reads from the
    outside like "the target supports nothing", the same shape a real answer would have.
    """
    assert TB._semantic_family(_Row(family="matmul", op_class="matmul")) == "contraction"
    assert TB._semantic_family(_Row(family="batch_matmul", op_class="batch_matmul")) == "contraction"


def test_a_real_semantic_family_is_left_alone():
    assert TB._semantic_family(_Row(family="normalization")) == "normalization"


def test_the_family_falls_back_to_the_op_class_when_absent():
    assert TB._semantic_family(_Row(family="", op_class="matmul")) == "contraction"


def test_resolution_reaches_the_eligibility_question_and_the_signature():
    rows = [_Row(index=0, family="matmul", op_class="matmul", ticks=10, key="a")]
    b = TB.derive_basis(_Census(rows, model_ticks=10), _Cap(("contraction",)))
    assert b.certificate["families_evidenced"] == ["contraction"]
    assert len(b.entries) == 1, "the group must be eligible once its family resolves"


# --- census scope ---------------------------------------------------------------------------------

def test_a_family_the_census_cannot_see_is_unsearched_not_absent(monkeypatch):
    """A contraction census over the seed model lists normalization as unevidenced.

    Reported flatly, that reads as "the model never normalizes" -- of a transformer that plainly
    does. The census simply does not enumerate it, and the two must not share a field.
    """
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        True, "contraction", "ok"))
    b = TB.derive_basis(_Census([_Row(index=0, family="matmul", ticks=1, key="a")], model_ticks=1),
                        _Cap(("contraction", "normalization", "softmax")),
                        census_enumerates=("contraction",))
    assert b.certificate["families_outside_census_scope"] == ["normalization", "softmax"]
    assert b.certificate["families_declared_not_evidenced"] == []
    assert b.certificate["census_scope_known"] is True


def test_an_unknown_census_scope_makes_no_claim_about_absence(monkeypatch):
    monkeypatch.setattr(TB.EL, "is_eligible", lambda *_a, **_k: TB.EL.EligibilityVerdict(
        True, "contraction", "ok"))
    b = TB.derive_basis(_Census([_Row(index=0, family="matmul", ticks=1, key="a")], model_ticks=1),
                        _Cap(("contraction", "normalization")))
    assert b.certificate["census_scope_known"] is False
    assert b.certificate["families_outside_census_scope"] == []
