"""A Triton vector add, compiled by Merlin for whatever target you name.

Run it (the extent need not be a multiple of the block — the tail is the interesting part):

    merlin-compile-kernel examples/triton/vector_add.py:vector_add --target saturn \
        --arg 'x_ptr=*fp32:1025:read' --arg 'y_ptr=*fp32:1025:read' \
        --arg 'out_ptr=*fp32:1025:write' --arg 'n_elements=i32' \
        --assume n_elements=1025 --constexpr BLOCK_SIZE=256 --grid 5 \
        --emit all --verify

`--assume n_elements=1025` is not boilerplate. The extent arrives as a runtime scalar, so nothing in
the kernel says it equals the declared shape; without it the compiler cannot check that the mask
keeps the launch inside the tensor, and it refuses rather than guessing.

This kernel has no matmul, so it compiles as generic computation through the LLVM path even when the
target is an accelerator — the route is chosen by the payload, not by the target.
"""
import triton
import triton.language as tl


@triton.jit
def vector_add(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)
