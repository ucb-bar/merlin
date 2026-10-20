"""Original scheduling inputs remain current across independently owned callbacks."""

from dataclasses import replace

import pytest
from test_copy_prefetch import QUALIFICATION, inputs

from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets, compile_schedule


def _compile(model, *, budgets, banks, external_windows):
    selected, kernel, objects, adapter, encoding, limits, *_ = model
    return compile_schedule(
        kernel,
        objects,
        banks,
        adapter,
        encoding,
        limits,
        selected_contracts=selected,
        external_windows=external_windows,
        private_ownership=True,
        prefetch_qualification_identity=QUALIFICATION,
        budgets=budgets,
    )


def _case(tmp_path):
    model = inputs(tmp_path)
    banks, _, external = model[0].storage.allocation_model(
        banks=None, reservations=(), external_windows=None, private_ownership=True
    )
    budgets = ScheduleBudgets(prefetch_distance=3, max_fusions=0, eliminate_copies=False)
    return model, budgets, banks, external


def test_unchanged_original_context_keeps_verified_prefetch(tmp_path):
    model, budgets, banks, external = _case(tmp_path)
    result = _compile(model, budgets=budgets, banks=banks, external_windows=external)
    assert result.report["storage"]["prefetch"]["moves"][0]["distance"] == 3
    assert result.report["resources"]["budgets"]["prefetch_distance"] == budgets.prefetch_distance


@pytest.mark.parametrize("mutation", ["budget", "bank", "object", "external_window", "program_window"])
def test_encoding_callback_cannot_retain_stale_resource_context(tmp_path, mutation):
    model, budgets, banks, external = _case(tmp_path)
    encode = model[4].encode

    def changed(kernel):
        image = encode(kernel)
        if mutation == "budget":
            object.__setattr__(budgets, "prefetch_distance", 0)
        elif mutation == "bank":
            object.__setattr__(banks[0], "capacity", banks[0].capacity - 1)
        elif mutation == "object":
            object.__setattr__(model[2][0], "alignment", 2)
        elif mutation == "external_window":
            object.__setattr__(external[0], "size", external[0].size - 1)
        else:
            object.__setattr__(model[5].windows[0], "size", model[5].windows[0].size - 1)
        return image

    model[4] = replace(model[4], encode=changed)
    with pytest.raises(ValueError, match="context|resources?|budgets?|changed"):
        _compile(model, budgets=budgets, banks=banks, external_windows=external)
