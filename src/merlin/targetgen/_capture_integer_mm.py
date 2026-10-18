"""Exact bounded eager evaluation of CPU int8 matrix products.

For the admitted shapes, every product and every possible partial sum fits i32,
so each is exactly representable in f64. Float64 GEMM followed by an i32 cast
therefore computes the same integer result. Other operators, tensors, layouts
and overflow domains retain the framework implementation. Use only around eager
forwards, never around export or graph construction.
"""

from contextlib import contextmanager


@contextmanager
def exact_integer_mm(torch):
    stats = {"policy": "cpu_int8_i32_exact_fp64_v1", "accelerated_calls": 0, "native_integer_calls": 0}
    functional = getattr(torch, "_is_functional_tensor", None)
    wrapped = getattr(getattr(getattr(torch, "_C", None), "_functorch", None), "is_functorch_wrapped_tensor", None)
    active = getattr(getattr(torch, "_C", None), "_are_functorch_transforms_active", None)
    if not callable(functional) or not callable(wrapped) or not callable(active):
        stats["status"] = "native_api_unavailable"
        yield stats
        return
    try:
        from torch.compiler import is_compiling
        from torch.fx._symbolic_trace import is_fx_symbolic_tracing
        from torch.fx.experimental import proxy_tensor
        from torch.utils._python_dispatch import TorchDispatchMode, _get_current_dispatch_mode_stack
    except ImportError:
        stats["status"] = "native_api_unavailable"
        yield stats
        return
    if not hasattr(proxy_tensor, "_CURRENT_MAKE_FX_TRACER"):
        stats["status"] = "native_tracing_api_unavailable"
        yield stats
        return

    def tracing():
        return (
            proxy_tensor._CURRENT_MAKE_FX_TRACER is not None
            or proxy_tensor.get_proxy_mode() is not None
            or is_fx_symbolic_tracing()
            or is_compiling()
            or active()
            or bool(_get_current_dispatch_mode_stack())
        )

    if tracing():
        stats["status"] = "native_transform_context"
        yield stats
        return
    i8, i32 = torch.iinfo(torch.int8), torch.iinfo(torch.int32)
    max_product = max(abs(i8.min), abs(i8.max)) ** 2
    if 2 / torch.finfo(torch.float64).eps <= i32.max:
        stats["status"] = "native_precision_unavailable"
        yield stats
        return
    stats["status"] = "enabled"

    def tensor(value):
        return (
            type(value) is torch.Tensor
            and value.dtype == torch.int8
            and value.device.type == "cpu"
            and value.layout == torch.strided
            and value.ndim == 2
            and not value.is_neg()
            and not value.is_conj()
            and not functional(value)
            and not wrapped(value)
        )

    def row_dense(value):
        return value.stride(1) == 1 and value.stride(0) >= value.shape[1]

    class Mode(TorchDispatchMode):
        supports_higher_order_operators = True

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten._int_mm.default:
                if not tracing() and len(args) == 2 and not kwargs and all(tensor(value) for value in args):
                    lhs, rhs = args
                    if (
                        all(int(d) > 0 for d in (*lhs.shape, *rhs.shape))
                        and lhs.shape[1] == rhs.shape[0]
                        and lhs.shape[1] * max_product <= i32.max
                        and row_dense(lhs)
                        and (row_dense(rhs) or row_dense(rhs.t()))
                    ):
                        stats["accelerated_calls"] += 1
                        return (lhs.to(torch.float64) @ rhs.to(torch.float64)).to(torch.int32)
                stats["native_integer_calls"] += 1
            return func(*args, **kwargs)

    with Mode():
        yield stats
