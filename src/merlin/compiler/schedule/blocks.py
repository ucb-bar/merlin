"""Public shared block-scheduling API; implementations have responsibility-specific owners."""

from .block import (
    _emit as _emit,
)
from .block import (
    _ordered as _ordered,
)
from .block import (
    _regions as _regions,
)
from .block import (
    _schedule_nest as _schedule_nest,
)
from .block import (
    _schedule_ops as _schedule_ops,
)
from .block import (
    schedule_contraction as schedule_contraction,
)
from .block import (
    schedule_convolution as schedule_convolution,
)
from .block import (
    with_column_bias as with_column_bias,
)
from .convolution import (
    ConvolutionPanel as ConvolutionPanel,
)
from .convolution import (
    TiledConvolutionSchedule as TiledConvolutionSchedule,
)
from .convolution import (
    schedule_tiled_convolution as schedule_tiled_convolution,
)
from .execute import (
    execute as execute,
)
from .interface import (
    _conv_command as _conv_command,
)
from .interface import (
    _shape as _shape,
)
from .interface import (
    _shape_nd as _shape_nd,
)
from .interface import (
    schedule_interface_program as schedule_interface_program,
)
from .model import (
    AXES as AXES,
)
from .model import (
    BANK_ALIGNED as BANK_ALIGNED,
)
from .model import (
    CONTIGUOUS as CONTIGUOUS,
)
from .model import (
    DEFAULT_PROJECTION_OPERATION_LIMIT as DEFAULT_PROJECTION_OPERATION_LIMIT,
)
from .model import (
    GROUPINGS as GROUPINGS,
)
from .model import (
    LHS as LHS,
)
from .model import (
    NEST as NEST,
)
from .model import (
    OPPOSITE_END as OPPOSITE_END,
)
from .model import (
    PLACEMENTS as PLACEMENTS,
)
from .model import (
    ROLE as ROLE,
)
from .model import (
    ROLES as ROLES,
)
from .model import (
    WEIGHT as WEIGHT,
)
from .model import (
    BlockSchedule as BlockSchedule,
)
from .model import (
    BlockScheduleError as BlockScheduleError,
)
from .model import (
    ColumnBias as ColumnBias,
)
from .model import (
    Compute as Compute,
)
from .model import (
    Contraction as Contraction,
)
from .model import (
    ConvContraction as ConvContraction,
)
from .model import (
    GatherRun as GatherRun,
)
from .model import (
    Geometry as Geometry,
)
from .model import (
    InitializeAccumulator as InitializeAccumulator,
)
from .model import (
    K as K,
)
from .model import (
    Knobs as Knobs,
)
from .model import (
    Load as Load,
)
from .model import (
    M as M,
)
from .model import (
    N as N,
)
from .model import (
    Op as Op,
)
from .model import (
    Preload as Preload,
)
from .model import (
    Store as Store,
)
from .model import (
    _ceil_div as _ceil_div,
)
from .model import (
    gather_runs as gather_runs,
)
from .readout import (
    SpatialMaxReadout as SpatialMaxReadout,
)
from .readout import (
    SpatialReadoutSchedule as SpatialReadoutSchedule,
)
from .readout import (
    SpatialStore as SpatialStore,
)
from .readout import (
    _spatial_stores as _spatial_stores,
)
from .readout import (
    check_spatial_max_readout as check_spatial_max_readout,
)
from .readout import (
    execute_spatial_max_readout as execute_spatial_max_readout,
)
from .readout import (
    plan_spatial_max_readout as plan_spatial_max_readout,
)
from .readout import (
    retain_accumulator_outputs as retain_accumulator_outputs,
)
from .residency import (
    _check_reduction_lifetimes as _check_reduction_lifetimes,
)
from .residency import (
    check_residency as check_residency,
)
