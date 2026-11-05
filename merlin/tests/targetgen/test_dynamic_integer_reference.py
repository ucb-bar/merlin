"""The dynamic-only oracle executes signed products independently, with complete outputs."""
import copy

import pytest

from merlin.targetgen._dynamic_integer_reference import run_reference, validate_receipt

torch = pytest.importorskip("torch")


def _model():
    graph = torch.fx.Graph()
    a, b, c = [graph.placeholder(name) for name in ("a", "b", "c")]
    first = graph.call_function(torch.ops.aten._int_mm.default, (a, b))
    second = graph.call_function(torch.ops.aten._int_mm.default, (c, b))
    graph.output((first, second))
    return torch.fx.GraphModule({}, graph)


def _inputs():
    return (torch.tensor([[-128, 127, -1], [3, -4, 5]], dtype=torch.int8),
            torch.tensor([[-128, 0], [-127, 2], [3, -4]], dtype=torch.int8),
            torch.tensor([[127, -128, 1], [-3, 4, -5]], dtype=torch.int8))


def _receipt():
    _, receipt = run_reference(_model(), _inputs())
    receipt["exported_integer_mm_count"] = 2
    row = dict(finite=True, atol=0.0, rtol=0.0, max_abs=0.0, max_rel=0.0, within_tolerance=True)
    receipt["golden_agreement"] = dict(row, status="passed", reference="dynamic_integer", samples=1,
                                       outputs=[dict(row), dict(row)])
    return receipt


def test_signed_ascending_products_are_independent_and_preserve_source(monkeypatch):
    module, inputs = _model(), _inputs()
    before = str(module.graph)
    original = torch.fx.Interpreter.call_function
    def guarded(self, target, args, kwargs):
        if target == torch.ops.aten._int_mm.default:
            raise AssertionError("native contraction must not execute in the oracle")
        return original(self, target, args, kwargs)
    monkeypatch.setattr(torch.fx.Interpreter, "call_function", guarded)
    output, receipt = run_reference(module, inputs)
    for lhs, actual in zip((inputs[0], inputs[2]), output, strict=True):
        expected = [[sum(int(lhs[i, k]) * int(inputs[1][k, j]) for k in range(3))
                     for j in range(2)] for i in range(2)]
        assert actual.dtype == torch.int32 and actual.tolist() == expected
    assert str(module.graph) == before
    assert receipt["reference_integer_mm_count"] == 2
    assert all(row["reduction_order"] == "k_ascending" for row in receipt["calls"])


@pytest.mark.parametrize("change", ["dtype", "budget", "shape", "overflow"])
def test_reference_refuses_unproved_domain(change):
    args = list(_inputs())
    kwargs = {}
    if change == "dtype":
        args[0] = args[0].to(torch.int16)
    if change == "budget":
        kwargs["max_macs"] = 1
    if change == "shape":
        args[0] = args[0][:, :2]
    if change == "overflow":
        args = [torch.ones((1, 131072), dtype=torch.int8), torch.ones((131072, 1), dtype=torch.int8),
                torch.ones((1, 131072), dtype=torch.int8)]
    with pytest.raises(ValueError):
        run_reference(_model(), args, **kwargs)


def test_complete_receipt_and_mutations():
    receipt = _receipt()
    validate_receipt(receipt, annotated=0, dynamic_calls=2, exported=2)
    mutations = [lambda r:r.update(reference_integer_mm_count=1),
                 lambda r:r.update(integer_mm_emitted=False), lambda r:r.update(annotated_contractions=False),
                 lambda r:r["calls"].pop(),
                 lambda r:r["calls"][0].update(initial_sum=1),
                 lambda r:r["calls"][0]["rhs"].update(dtype="u8"),
                 lambda r:r["calls"][0].update(max_abs_partial_sum=1 << 31),
                 lambda r:r["golden_agreement"]["outputs"][1].update(max_abs=0.01),
                 lambda r:r.update(reference_macs=0)]
    for mutate in mutations:
        changed = copy.deepcopy(receipt)
        mutate(changed)
        with pytest.raises(ValueError):
            validate_receipt(changed, annotated=0, dynamic_calls=2, exported=2)
    with pytest.raises(ValueError):
        validate_receipt(receipt, annotated=1, dynamic_calls=2, exported=2)


def test_capture_emission_and_producer_binding_preserve_complete_outputs(tmp_path):
    import hashlib
    import json

    from merlin.targetgen import _dynamic_integer_reference as reference
    from merlin.targetgen._m2m_capture_worker import _output_abi, _to_native

    output, _ = run_reference(_model(), _inputs())
    leaves, abi = _output_abi(output)
    expected = (json.dumps(dict(schema='merlin.capture.integer_reference.v1', output_abi=abi,
                                outputs=[_to_native(x) for x in leaves]),
                           sort_keys=True, separators=(',', ':')) + '\n').encode()
    receipt = reference.capture_reference(_model(), _inputs(), tmp_path,
                                          output_abi=_output_abi, to_native=_to_native)
    assert (tmp_path / 'integer-reference.json').read_bytes() == expected
    agreement = receipt['golden_agreement']
    assert agreement['status'] == 'passed' and len(agreement['outputs']) == 2
    assert agreement['output']['sha256'] == hashlib.sha256(expected).hexdigest()
    source_sha = hashlib.sha256(open(reference.__file__, 'rb').read()).hexdigest()
    assert agreement['source']['sha256'] == source_sha
    producer = dict(tool={'reference_sources': {'existing.py': 'existing-pin'}}, retained='unchanged')
    (tmp_path / 'capture_receipt.json').write_text(json.dumps(producer))
    reference.bind_reference_source(tmp_path)
    bound = json.loads((tmp_path / 'capture_receipt.json').read_bytes())
    assert bound['retained'] == 'unchanged'
    assert bound['tool']['reference_sources'] == {
        'existing.py': 'existing-pin', 'merlin/targetgen/_dynamic_integer_reference.py': source_sha}


@pytest.mark.parametrize('change', ['value', 'cardinality', 'dtype'])
def test_worker_agreement_retains_exact_original_output_checks(change):
    from merlin.targetgen._m2m_capture_worker import _integerized_agreement

    output, _ = run_reference(_model(), _inputs())
    modified = list(output)
    if change == 'value':
        modified[1] = modified[1] + 1
    elif change == 'cardinality':
        modified.pop()
    else:
        modified[1] = modified[1].to(torch.float32)
    receipt = _integerized_agreement(output, tuple(modified), atol=0.0, rtol=0.0)
    assert receipt['status'] == 'failed'


def test_selected_reference_budget_is_finite_and_does_not_change_defaults():
    from merlin.targetgen._dynamic_integer_reference import reference_work_limit

    assert reference_work_limit(None) == reference_work_limit({}) == 64 << 20
    policy = dict(software_numerical_engine='integer_reference',
                  dynamic_integer_reference_limits=dict(schema='merlin.dynamic_integer_reference_limits.v1',
                                                        max_macs=128 << 20))
    assert reference_work_limit(policy) == 128 << 20
    for limit in (False, 0, -1, 1.5, 1 << 41):
        changed = copy.deepcopy(policy)
        changed['dynamic_integer_reference_limits']['max_macs'] = limit
        with pytest.raises(ValueError):
            reference_work_limit(changed)
    for change in ({'software_numerical_engine': 'portable'},
                   {'dynamic_integer_reference_limits': {'schema': 'unknown', 'max_macs': 1}}):
        with pytest.raises(ValueError):
            reference_work_limit({**policy, **change})
    receipt = _receipt()
    receipt['reference_work_limit'] = receipt['reference_macs'] - 1
    with pytest.raises(ValueError, match='work limit'):
        validate_receipt(receipt, annotated=0, dynamic_calls=2, exported=2)
