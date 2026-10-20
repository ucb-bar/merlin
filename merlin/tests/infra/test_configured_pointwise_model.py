"""Ordinary whole-model setup cleanup with real configuration-dependent work."""

import pytest
from test_configured_pointwise import EXECUTOR, case
from test_pointwise_source_provider import Provider
from test_pointwise_source_whole_model import test_ordinary_recurrent_model_runs_source_bound_fusion as run_pair

from merlin.compile.scheduling.schedule_pipeline import ScheduleBudgets


class ConfiguredProvider(Provider):
    case_builder = staticmethod(case)
    executor_source = EXECUTOR
    observation_count = 8

    def schedule_budgets(self):
        return ScheduleBudgets(max_fusions=0, eliminate_configuration=self.enabled)

    def expected_instructions(self):
        setup = ["setup"] * (1 if self.enabled else 4)
        return (
            [*setup, *setup, "readout", "finish"]
            if self.enabled
            else [*setup, *setup, "readout", *setup, *setup, "finish"]
        )

    @staticmethod
    def _wrapper(symbol, m, n, image, separate):
        wrapper = Provider._wrapper(symbol, m, n, image, separate)
        return wrapper.replace("counts[3];", "counts[5];").replace(
            "totals[5]++;", "totals[5]++;totals[6]=counts[3];totals[7]=counts[4];"
        )


@pytest.mark.parametrize("shape,separate", [((3, 7, 5), False), ((2, 5, 11), True)])
def test_ordinary_recurrent_model_eliminates_configuration(tmp_path, record_property, shape, separate):
    run_pair(
        tmp_path, record_property, shape, separate, provider_class=ConfiguredProvider, optimization="configuration"
    )
