"""Source-derived integer convolution task for the opt-in model route."""
from dataclasses import dataclass

@dataclass
class Convolution:
    """Canonical NCHW/OIHW direct convolution owned by the generated target scheduler.

    The target contracts ``weight[Co,K] @ im2col_row[K,Wo]`` one output row at a time, so the
    result lands directly in NCHW storage and the im2col working set never scales with ``Ho``.
    """

    activation: str
    weight: str
    dst: str
    batch: int
    ci: int
    hi: int
    wi: int
    co: int
    kh: int
    kw: int
    ho: int
    wo: int
    stride_h: int
    stride_w: int
    dilation_h: int
    dilation_w: int
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int
    direct_dma: bool = False
    input_layout: str = "NCHW"
    weight_layout: str = "CoK"
    output_layout: str = "NCHW"
    output_dtype: str = "i32"
    # Native-aligned narrow epilogue.  Bias is already quantized into i32
    # accumulator units; LOOP_CONV LOAD3 transports it independently of the
    # ordinary execute-path D operand.  The store applies one scalar f32 scale
    # and optional ReLU before saturating to i8.
    bias: str | None = None
    acc_scale: float = 1.0
    relu: bool = False
