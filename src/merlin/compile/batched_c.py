"""Render dense batch iteration around an existing checked matrix C call."""

from __future__ import annotations

from merlin.semantic_compiler.batched_contraction import BatchedContractionPlan, plan_batched_contraction


def render_batched_contraction_c(plan: BatchedContractionPlan, name: str, matrix_dispatch: str) -> str:
    """Use byte offsets; no tensor arithmetic, allocation or target instructions.

    The provider passes its context/scratch ranges to the private helper. Those
    ranges are checked against the *complete* batch buffers before any matrix
    call, so an early call cannot overwrite a later batch's inputs. An error
    stops iteration; the enclosing application owns atomic output publication
    and draining/resetting any unfinished device work.
    """
    if plan != plan_batched_contraction(plan.source):
        raise ValueError("batched C plan differs from its source")
    for identifier in (name, matrix_dispatch):
        if not identifier.isascii() or not identifier.isidentifier():
            raise ValueError("batch C symbols must be C identifiers")
    inputs = plan.input_ids
    input_counts = dict(zip(plan.operand_ids, plan.matrix_elements[:2], strict=True))
    sizes = tuple(plan.batch_count * input_counts[key] * 4 for key in inputs)
    result_stride = plan.matrix_elements[-1] * 4
    output_bytes = plan.batch_count * result_stride
    if max(*sizes, output_bytes) > 2**63 - 1:
        raise ValueError("batched C buffers exceed the supported host size domain")
    checks = []
    for i, size in enumerate(sizes):
        checks.extend(
            (
                f"!inputs[{i}] || (uintptr_t)inputs[{i}]>UINTPTR_MAX-{size}ULL",
                f"((uintptr_t)inputs[{i}]<out+{output_bytes}ULL && out<(uintptr_t)inputs[{i}]+{size}ULL)",
            )
        )
    protected_checks = [f"(p<out+{output_bytes}ULL && out<p+reserved_bytes[r])"]
    protected_checks.extend(
        f"(p<(uintptr_t)inputs[{i}]+{size}ULL && (uintptr_t)inputs[{i}]<p+reserved_bytes[r])"
        for i, size in enumerate(sizes)
    )
    arguments = ",".join(
        f"(const unsigned char *)inputs[{inputs.index(key)}]+batch*{elements * 4}ULL"
        for key, elements in zip(plan.operand_ids, plan.matrix_elements[:2], strict=True)
    )
    return f"""
_Static_assert(SIZE_MAX >= {max(*sizes, output_bytes)}ULL, "batch tensor exceeds host size_t");
static int {name}_batch_run(void *context,const void *const *inputs,void *const *outputs,
                          const void *const *reserved,const size_t *reserved_bytes,size_t reserved_count) {{
  if (!context || !inputs || !outputs || !outputs[0]) return -1;
  uintptr_t out=(uintptr_t)outputs[0];
  if (out>UINTPTR_MAX-{output_bytes}ULL || {" || ".join(checks)}) return -1;
  for (size_t r=0;r<reserved_count;++r) {{
    uintptr_t p=(uintptr_t)reserved[r];
    if (!reserved[r] || p>UINTPTR_MAX-reserved_bytes[r] || {" || ".join(protected_checks)}) return -1;
  }}
  for (size_t batch=0;batch<{plan.batch_count}ULL;++batch) {{
    const void *matrix_inputs[2]={{{arguments}}};
    void *matrix_outputs[1]={{(unsigned char *)outputs[0]+batch*{result_stride}ULL}};
    int status={matrix_dispatch}(context,matrix_inputs,matrix_outputs);
    if (status) return status;
  }}
  return 0;
}}
"""
