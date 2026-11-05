"""Source-prepared physical storage crosses the same strict ordinary contract."""

from copy import deepcopy

import pytest

from merlin.compile.scheduling.tensor_storage import column_panel_storage
from merlin.perf.storage_encoding import GroupedAxesStorage
from merlin.targetgen.contract.schemas import ContractViolation, validate_command_buffer


def command(storage):
    return dict(abi_version="0.1", target="neutral", commands=[], outputs=["Y"], tensors={
        "W": dict(shape=list(storage.physical_shape), dtype="i8", role="weight", storage_encoding=storage.to_dict()),
        "Y": dict(shape=[1], dtype="i8", role="output")})


@pytest.mark.parametrize("storage", [column_panel_storage(18,6,3,"i8"),
    GroupedAxesStorage((3,6),"i8",((1,),(0,)),(6,3),(3,1),18),
    GroupedAxesStorage((),"i8",(),(),(),1)])
def test_storage_map_is_complete_and_bound_to_physical_tensor(storage):
    cb = command(storage)
    validate_command_buffer(cb)
    for field, bad in (("shape",[18]),("dtype","i16")):
        broken = deepcopy(cb)
        broken["tensors"]["W"][field] = bad
        with pytest.raises(ContractViolation, match="physical tensor"):
            validate_command_buffer(broken)


@pytest.mark.parametrize("mutation", [
    lambda e: e.update(extra_authority=True),
    lambda e: e.update(schema="unreviewed-map"),
    lambda e: e.update(axis_factors=[[2,8],[2,3]]),
    lambda e: e.update(physical_axis_order=[0,0,1]),
    lambda e: e.update(physical_axis_order=[0,1,4]),
])
def test_malformed_or_nonbijective_storage_refuses(mutation):
    cb = command(column_panel_storage(18,6,3,"i8"))
    mutation(cb["tensors"]["W"]["storage_encoding"])
    with pytest.raises(ContractViolation):
        validate_command_buffer(cb)
