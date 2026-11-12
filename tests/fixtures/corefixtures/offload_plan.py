from dataclasses import replace

from corefixtures.dense_source import contraction
from merlin.compiler.frontend.linalg_mlir import parse_mlir_text
from merlin.compiler.lower.model.offload_plan import bind_source
from merlin.compiler.transforms.pipeline_state import PipelineContext


def example(*, widths=("i8", "i8", "i32"), shape=(3, 7, 5)):
    m, k, n = shape
    x, y, z = widths
    module = parse_mlir_text(
        f"module {{func.func @forward(%a:tensor<{m}x{k}x{x}>,%b:tensor<{k}x{n}x{y}>)\n      ->(tensor<{m}x{n}x{z}>,tensor<{m}x{n}x{z}>) {{\n      {contraction(shape, widths, output='first', tag='same')}\n      {contraction(shape, widths, output='second', tag='same')}\n      func.return %first,%second:tensor<{m}x{n}x{z}>,tensor<{m}x{n}x{z}>\n    }}}}"
    )
    context = PipelineContext(
        "original hardware",
        "unchanged numerical policy",
        "physical resources",
        "qualified caller effects",
        "same shared library",
    )
    bound = bind_source(module, entry="forward", context=context)
    contractions = [index for index, operation in enumerate(bound.operations) if operation.name == "linalg.generic"]
    device = tuple(
        (
            replace(
                bound.cpu_alternatives[index],
                id=f"device_{index}",
                implementation=f"original_kernel_{index}",
                placement="selected_endpoint",
            )
            for index in contractions
        )
    )
    return (bound, device, context)
