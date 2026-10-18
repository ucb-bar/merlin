"""Complete ordinary model profiling through the actual fresh-writer void ABI."""

import pytest
from test_dense_source_whole_model import (
    test_source_scheduling_runs_through_complete_ordinary_model as run_source_model,
)


@pytest.mark.parametrize("tile,width,shared", [((2, 3, 4), 1, False), ((3, 2, 5), 2, True)])
def test_profiled_source_dense_model_preserves_complete_outputs_and_state(
    tmp_path, monkeypatch, record_property, tile, width, shared
):
    run_source_model(tmp_path, monkeypatch, record_property, tile, width, shared, group_profile=True)
