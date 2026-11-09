"""Emit a real outlined-chain driver with controller-only preparation/dispatch.

Arithmetic calls occur only in the compute callback on its assigned worker.
Persistent worker startup and completion joins are caller-owned platform hooks;
the existing staged runtime controls every fresh, monotonically sequenced team.
"""

from __future__ import annotations

import json
import math

from merlin.llvmlower.deployable_outline import DeployableChain
from merlin.runtime.staged_dispatch import StagedRoles


def _views(views, positions) -> list[str]:
    lines = []
    for view in views:
        if view["kind"] == "assert_flag":
            flag = positions[view["input"]]
            lines.append(f"    if(!c->buffers[{flag}] || ((const unsigned char*)c->buffers[{flag}])[0]!=1)return -131;")
            continue
        if view["kind"] == "pack_insert_slice":
            source, initial = (positions[bid] for bid in view["inputs"])
            dst = positions[view["output"]]
            lines.append(f"    memcpy(c->buffers[{dst}],c->buffers[{initial}],{view['initial_bytes']});")
            rem = "rem=span, " if view["copy_iteration_shape"] else ""
            lines.append(
                f"    for(size_t span=0;span<{view['copy_count']};++span) {{ "
                f"size_t {rem}destination={view['copy_base_elements']};"
            )
            for size, pitch in reversed(
                list(zip(view["copy_iteration_shape"], view["copy_destination_pitches"], strict=True))
            ):
                lines.append(f"      destination += (rem%{size})*{pitch}; rem/={size};")
            lines.append(
                f"      memcpy((unsigned char*)c->buffers[{dst}]+destination*{view['element_bytes']},"
                f"(const unsigned char*)c->buffers[{source}]+span*{view['copy_bytes']},{view['copy_bytes']});\n    }}"
            )
            continue
        if view["kind"] == "pack_concat":
            dst = positions[view["output"]]
            lines.append(f"    for(size_t block=0;block<{view['outer_blocks']};++block) {{")
            offset = 0
            for bid, size in zip(view["inputs"], view["input_block_bytes"], strict=True):
                src = positions[bid]
                lines.append(
                    f"      memcpy((unsigned char*)c->buffers[{dst}]+block*{view['output_block_bytes']}+{offset},"
                    f"(const unsigned char*)c->buffers[{src}]+block*{size},{size});"
                )
                offset += size
            lines.append("    }")
            continue
        src, dst = positions[view["input"]], positions[view["output"]]
        if view["kind"] == "alias":
            lines.append(f"    c->buffers[{dst}] = c->buffers[{src}];")
            continue
        if view["kind"] == "pack_permutation":
            policy = view.get("packing_policy", {})
            if policy.get("strategy") == "blocked_rank2":
                rows, columns = view["output_shape"]
                row_stride, column_stride = view["source_strides"]
                tr, tc = policy["effective_tile_shape"]
                width = view["element_bytes"]
                lines.append(f"    for(size_t r0=0; r0<{rows};) {{ size_t nr={rows}-r0; if(nr>{tr})nr={tr};")
                lines.append(f"      for(size_t c0=0; c0<{columns};) {{ size_t nc={columns}-c0; if(nc>{tc})nc={tc};")
                lines.append("        for(size_t r=0;r<nr;++r)for(size_t q=0;q<nc;++q) {")
                lines.append(
                    f"          memcpy((unsigned char*)c->buffers[{dst}]+((r0+r)*{columns}+c0+q)*{width},"
                    f"(const unsigned char*)c->buffers[{src}]+"
                    f"((r0+r)*{row_stride}+(c0+q)*{column_stride})*{width},{width});"
                )
                lines.append("        } c0+=nc;\n      } r0+=nr;\n    }")
                continue
            count = math.prod(view["output_shape"])
            lines.append(f"    for (size_t i=0; i<{count}; ++i) {{ size_t rem=i, source=0;")
            for size, stride in reversed(list(zip(view["output_shape"], view["source_strides"], strict=True))):
                lines.append(f"      source += (rem % {size}) * {stride}; rem /= {size};")
            width = view["element_bytes"]
            lines.append(
                f"      memcpy((unsigned char*)c->buffers[{dst}]+i*{width},"
                f"(const unsigned char*)c->buffers[{src}]+source*{width},{width});\n    }}"
            )
            continue
        count = math.prod(view["sizes"])
        lines.append(f"    for (size_t i=0; i<{count}; ++i) {{ size_t rem=i, source=0;")
        dense = 1
        for offset, size, stride, extent in reversed(
            list(zip(view["offsets"], view["sizes"], view["strides"], view["source_shape"], strict=True))
        ):
            lines.append(f"      source += ({offset} + (rem % {size}) * {stride}) * {dense}; rem /= {size};")
            dense *= extent
        width = view["element_bytes"]
        lines.append(
            f"      memcpy((unsigned char*)c->buffers[{dst}]+i*{width},"
            f"(const unsigned char*)c->buffers[{src}]+source*{width},{width});\n    }}"
        )
    return lines


def render_outlined_chain(
    plan: DeployableChain,
    kernel_entries: tuple[str, ...],
    *,
    roles: StagedRoles,
    prefix="merlin_chain",
    prepared_kernels=None,
    provider_state_layout=None,
) -> dict:
    """Render C source/header; target facts/threads/ISA and data stay explicit inputs."""
    if (
        not isinstance(prefix, str)
        or not prefix.isascii()
        or not prefix.isidentifier()
        or not isinstance(kernel_entries, tuple)
        or len(kernel_entries) != len(plan.dispatches)
        or any(
            not isinstance(entry, str) or not entry.isascii() or not entry.isidentifier() for entry in kernel_entries
        )
    ):
        raise ValueError("safe explicit prefix and one kernel entry per dispatch required")
    positions = {buffer.id: i for i, buffer in enumerate(plan.buffers)}
    specs = {buffer.id: buffer for buffer in plan.buffers}
    args = sorted(
        (buffer for buffer in plan.buffers if buffer.arg_index is not None), key=lambda buffer: buffer.arg_index
    )
    frozen = plan.frozen_inputs
    if frozen is not None:
        source_indices = frozen["external_input_source_indices"]
        args = [next(b for b in args if b.arg_index == i) for i in source_indices]
    input_slots = {b.arg_index: i for i, b in enumerate(args)}
    results = [specs[bid] for bid in plan.program.results]
    if not results or (
        frozen is None and (not args or [buffer.arg_index for buffer in args] != list(range(len(args))))
    ):
        raise ValueError("complete model argument/result roster required")
    if len(roles.compute_harts) != plan.worker_count:
        raise ValueError("compiled role policy must match the plan's worker count")
    prepared_kernels = {} if prepared_kernels is None else dict(prepared_kernels)
    provider_state_layout = {} if provider_state_layout is None else dict(provider_state_layout)
    from merlin.runtime.prepared_staged_kernel import verify_selection

    verify_selection(plan, prepared_kernels, provider_state_layout)
    signatures = {}
    for index, (entry, step) in enumerate(zip(kernel_entries, plan.dispatches, strict=True)):
        signature = tuple((specs[bid].shape, specs[bid].dtype) for bid in step["inputs"] + step["outputs"])
        if entry in signatures and signatures[entry] != signature:
            raise ValueError("reused kernel entry has a different typed ABI")
        if index in prepared_kernels and any(dtype == "i1" for _, dtype in signature):
            raise ValueError("prepared i1 kernels require a separately qualified canonical-byte contract")
        signatures[entry] = signature
    n, workers = len(plan.dispatches), plan.worker_count
    arity = max(len(step["inputs"]) + len(step["outputs"]) for step in plan.dispatches)
    if not 1 <= arity <= 4096:
        raise ValueError("kernel descriptor arity exceeds the runtime contract")
    p = prefix
    provider_fields = ""
    if prepared_kernels:
        provider_fields = f"""
  merlin_arg_t prepared_arguments[{arity}];
  void *prepared_data[{arity}];
  size_t prepared_capacities[{arity}];
  void *provider_states[{n}];
  merlin_staged_kernel_invocation invocation;
"""
    header = f"""/* Generated closed outlined-chain ABI. */
#ifndef {p.upper()}_H
#define {p.upper()}_H
#include "merlin_model.h"
#include "merlin_staged.h"
{'#include "merlin_staged_kernel.h"' if prepared_kernels else ""}
#define {p.upper()}_DISPATCHES {n}
#define {p.upper()}_WORKSPACE_BYTES {plan.workspace_bytes}u
typedef struct {{
  void *inputs[{max(1, len(args))}], *outputs[{len(results)}];
  size_t input_capacity[{max(1, len(args))}], output_capacity[{len(results)}];
  unsigned char *workspace;
  size_t workspace_capacity;
  atomic_uint running;
  unsigned poisoned, active_dispatch, completed_dispatches, prepared, initialized;
  uint64_t last_sequence;
  unsigned observed_count;
  int status;
  void *buffers[{len(plan.buffers)}];
  merlin_descriptor_t descriptors[{arity}];
  void *descriptor_ptrs[{arity}];
  size_t kernel_calls[{workers}], calls_before[{workers}];
  merlin_staged_team team;
  merlin_staged_report observations[{n}];
  merlin_staged_hooks platform_hooks;
  void *platform;
{provider_fields}
}} {p}_context;
typedef void (*{p}_launch_fn)(void *, merlin_staged_team *);
typedef int (*{p}_join_fn)(void *, merlin_staged_team *);
/* Zero-initialize the context before first use. Reuse only after successful
 * quiescence/join and with a strictly greater first sequence. Failures poison
 * this context. Unquiesced failures retain it and every backing buffer.
 * Launch wakes already initialized pinned workers; join confirms all returned
 * and no longer access team storage. Neither hook may invoke model arithmetic.
 */
int {p}_run({p}_context *, const merlin_staged_config *,
            const merlin_staged_hooks *, {p}_launch_fn, {p}_join_fn, void *platform);
#endif
"""
    source = [
        f'''/* Generated: controller metadata/copies; all arithmetic is in compute. */
#include "{p}.h"
#include <string.h>
#include <stdint.h>
static unsigned self_hart(void *x) {{ {p}_context*c=x; return c->platform_hooks.current_hart(c->platform); }}
static uint64_t global_tick(void*x) {{ {p}_context*c=x; return c->platform_hooks.global_ticks(c->platform); }}
static uint64_t cycles(void*x) {{ {p}_context*c=x; return c->platform_hooks.worker_cycles(c->platform); }}
static uint64_t retired(void*x) {{ {p}_context*c=x; return c->platform_hooks.worker_instret(c->platform); }}
static int checked_tick(void*x,uint64_t*v) {{ {p}_context*c=x;
  return c->platform_hooks.shared_window_ticks(c->platform,v); }}
static void relax(void*x) {{ {p}_context*c=x; if(c->platform_hooks.relax)c->platform_hooks.relax(c->platform); }}
static int controller({p}_context*c) {{ return self_hart(c)==c->team.config.controller_hart; }}
static int overlap(const void*a,size_t an,const void*b,size_t bn) {{
  uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;
  if(!x||!y||!an||!bn||x>UINTPTR_MAX-an||y>UINTPTR_MAX-bn)return 1;
  return x<y+bn&&y<x+an;
}}
'''
    ]
    if frozen is not None:
        source.append(f"extern const unsigned char {p}_frozen_blob[{frozen['published_blob_bytes']}];")
    for index, kernel in sorted(prepared_kernels.items()):
        source.append("#include " + json.dumps(str(kernel.header)))
        source.append(f'_Static_assert(sizeof({kernel.state_type})<={kernel.state_bytes},"prepared state budget");')
        source.append(
            f'_Static_assert(_Alignof({kernel.state_type})<={kernel.state_alignment},"prepared state alignment");'
        )
        for field in ("prepare", "compute", "finish"):
            source.append(
                f"_Static_assert(_Generic(&{getattr(kernel, field)},"
                f'merlin_staged_kernel_{field}_fn:1,default:0),"prepared callback ABI");'
            )
    for index, (entry, step) in enumerate(zip(kernel_entries, plan.dispatches, strict=True)):
        if index in prepared_kernels:
            continue
        count = len(step["inputs"]) + len(step["outputs"])
        source.append(f"extern void _mlir_ciface_{entry}({','.join(['void*'] * count)});")
    source.append(f"static int model_init(void*x) {{ {p}_context*c=x; if(!controller(c))return -101;")
    source.append(
        "  if(c->active_dispatch!=c->completed_dispatches || c->team.config.sequence<=c->last_sequence)return -102;"
    )
    source.append("  c->last_sequence=c->team.config.sequence;")
    source.append("  if(c->active_dispatch==0) {")
    source.append(
        f"    if(!c->workspace || c->workspace_capacity<{plan.workspace_bytes} || "
        f"(uintptr_t)c->workspace%{plan.alignment_bytes})return -103;"
    )
    for buffer in args:
        i, pos = input_slots[buffer.arg_index], positions[buffer.id]
        source.append(
            f"    if(!c->inputs[{i}] || c->input_capacity[{i}]<{buffer.bytes} || "
            f"(uintptr_t)c->inputs[{i}]%{buffer.bytes // math.prod(buffer.shape)})return -104;"
        )
        source.append(
            f"    if(overlap(c->inputs[{i}],{buffer.bytes},c,sizeof(*c)) || "
            f"overlap(c->inputs[{i}],{buffer.bytes},c->workspace,{plan.workspace_bytes}))return -105;"
        )
        if buffer.dtype == "i1":
            source.append(
                f"    for(size_t byte=0;byte<{buffer.bytes};++byte)"
                f"if(((const unsigned char*)c->inputs[{i}])[byte]>1)return -126;"
            )
        source.append(f"    c->buffers[{pos}]=c->inputs[{i}];")
    if frozen is not None:
        symbol = p + "_frozen_blob"
        count = frozen["published_blob_bytes"]
        source.append(
            f"    if(overlap({symbol},{count},c,sizeof(*c)) || "
            f"overlap({symbol},{count},c->workspace,{plan.workspace_bytes}))return -126;"
        )
        for bid, entry in frozen["readonly_buffers"].items():
            source.append(f"    c->buffers[{positions[bid]}]=(void*)({symbol}+{entry['offset']});")
    for i, buffer in enumerate(results):
        source.append(
            f"    if(!c->outputs[{i}] || c->output_capacity[{i}]<{buffer.bytes} || "
            f"(uintptr_t)c->outputs[{i}]%{buffer.bytes // math.prod(buffer.shape)})return -106;"
        )
        source.append(
            f"    if(overlap(c->outputs[{i}],{buffer.bytes},c,sizeof(*c)) || "
            f"overlap(c->outputs[{i}],{buffer.bytes},c->workspace,{plan.workspace_bytes}))return -107;"
        )
        for arg in args:
            source.append(
                f"    if(overlap(c->outputs[{i}],{buffer.bytes},"
                f"c->inputs[{input_slots[arg.arg_index]}],{arg.bytes}))return -108;"
            )
        if frozen is not None:
            source.append(
                f"    if(overlap(c->outputs[{i}],{buffer.bytes},{p}_frozen_blob,"
                f"{frozen['published_blob_bytes']}))return -127;"
            )
        for earlier in range(i):
            source.append(
                f"    if(overlap(c->outputs[{i}],{buffer.bytes},"
                f"c->outputs[{earlier}],{results[earlier].bytes}))return -109;"
            )
    source.append(f"    if(overlap(c->workspace,{plan.workspace_bytes},c,sizeof(*c)))return -110;")
    for i, buffer in enumerate(plan.buffers):
        if buffer.workspace_offset is not None:
            source.append(f"    c->buffers[{i}]=c->workspace+{buffer.workspace_offset};")
    for index, slot in sorted(provider_state_layout.items()):
        source.append(f"    c->provider_states[{index}]=c->workspace+{slot['offset']};")
    source.append(f"    for(unsigned i=0;i<{workers};i++)c->kernel_calls[i]=0;")
    source.append("    c->initialized=1;\n  }\n  if(!c->initialized || c->prepared)return -111;")
    source.append(f"  for(unsigned i=0;i<{workers};i++)c->calls_before[i]=c->kernel_calls[i];\n  return 0;\n}}")
    source.append(
        f"static int prepare(void*x) {{ {p}_context*c=x; if(!controller(c)||!c->initialized||c->prepared)return -112;"
    )
    source.append("  switch(c->active_dispatch) {")
    for index, step in enumerate(plan.dispatches):
        source.append(f"  case {index}: {{")
        source.extend(_views(step["views_before"], positions))
        ordered = step["inputs"] + step["outputs"]
        source.append("    const merlin_arg_t arguments[]={")
        for i, bid in enumerate(ordered):
            b = specs[bid]
            kind = "MERLIN_INPUT" if i < len(step["inputs"]) else "MERLIN_OUTPUT"
            dimensions = ",".join(map(str, b.shape)) or "0"
            source.append(f"      {{{kind},0,{len(b.shape)},{{{dimensions}}},{b.bytes // math.prod(b.shape)}}},")
        source.append("    };")
        source.append("    void *data[]={" + ",".join(f"c->buffers[{positions[bid]}]" for bid in ordered) + "};")
        source.append("    const size_t capacities[]={" + ",".join(str(specs[bid].bytes) for bid in ordered) + "};")
        source.append(
            f"    if(merlin_prepare_descriptors(arguments,{len(ordered)},data,capacities,"
            "c->descriptors,c->descriptor_ptrs))return -113;"
        )
        if index in prepared_kernels:
            kernel = prepared_kernels[index]
            mask = sum(1 << s for s in kernel.worker_slots)
            source.append(
                "    memcpy(c->prepared_arguments,arguments,sizeof arguments);"
                "memcpy(c->prepared_data,data,sizeof data);"
                "memcpy(c->prepared_capacities,capacities,sizeof capacities);"
            )
            source.append(
                "    c->invocation=(merlin_staged_kernel_invocation){MERLIN_STAGED_KERNEL_ABI_VERSION,"
                f"{len(step['inputs'])},{len(step['outputs'])},{len(ordered)},"
                "c->prepared_arguments,c->prepared_data,c->prepared_capacities,c->descriptor_ptrs,"
                f"c->team.config.sequence,{workers},{mask},c->team.config.worker_harts}};"
            )
            source.append(f"    memset(c->provider_states[{index}],0,{kernel.state_bytes});")
            source.append(
                f"    int result={kernel.prepare}(c->provider_states[{index}],&c->invocation);if(result)return result;"
            )
        source.append("    break;\n  }")
    source.append("  default:return -114;\n  }\n  c->prepared=1;return 0;\n}")
    masks = [
        sum(1 << s for s in prepared_kernels[i].worker_slots) if i in prepared_kernels else 1 << s["worker_slot"]
        for i, s in enumerate(plan.dispatches)
    ]
    source.append(f"static const unsigned assigned_masks[{n}]={{{','.join(map(str, masks))}}};")
    source.append(f"static int compute(void*x,unsigned slot) {{ {p}_context*c=x;")
    source.append(
        f"  if(slot>={workers} || self_hart(c)!=c->team.config.worker_harts[slot] || "
        f"c->active_dispatch>={n} || !c->prepared)return -115;"
    )
    source.append("  if(!(assigned_masks[c->active_dispatch]&(1u<<slot)))return 0;\n  switch(c->active_dispatch) {")
    for index, (entry, step) in enumerate(zip(kernel_entries, plan.dispatches, strict=True)):
        if index in prepared_kernels:
            kernel = prepared_kernels[index]
            source.append(
                f"  case {index}: {{int status={kernel.compute}(c->provider_states[{index}],&c->invocation,slot);"
                "if(status)return status;break;}"
            )
            continue
        count = len(step["inputs"]) + len(step["outputs"])
        source.append(
            f"  case {index}: _mlir_ciface_{entry}("
            + ",".join(f"c->descriptor_ptrs[{i}]" for i in range(count))
            + ");break;"
        )
    boolean_outputs = {
        index: [bid for bid in step["outputs"] if specs[bid].dtype == "i1"]
        for index, step in enumerate(plan.dispatches)
        if any(specs[bid].dtype == "i1" for bid in step["outputs"])
    }
    if boolean_outputs:
        source.append("  default:return -116;\n  }\n  c->kernel_calls[slot]++;\n  switch(c->active_dispatch) {")
        for index, bids in boolean_outputs.items():
            source.append(f"  case {index}:")
            for bid in bids:
                source.append(
                    f"    for(size_t byte=0;byte<{specs[bid].bytes};++byte)"
                    f"if(((const unsigned char*)c->buffers[{positions[bid]}])[byte]>1)return -127;"
                )
            source.append("    break;")
        source.append("  default:break;\n  }\n  return 0;\n}")
    else:
        source.append("  default:return -116;\n  }\n  c->kernel_calls[slot]++;return 0;\n}")
    source.append(f"static int finish(void*x) {{ {p}_context*c=x;")
    source.append(
        f"  if(!controller(c)||!c->prepared||c->active_dispatch>={n} || "
        f"atomic_load_explicit(&c->team.done,memory_order_acquire)!=((1u<<{workers})-1))return -117;"
    )
    source.append(
        f"  for(unsigned i=0;i<{workers};i++)if(c->kernel_calls[i]!=c->calls_before[i]+"
        "!!(assigned_masks[c->active_dispatch]&(1u<<i)))return -118;"
    )
    if prepared_kernels:
        source.append("  switch(c->active_dispatch) {")
        for index, kernel in sorted(prepared_kernels.items()):
            source.append(
                f"  case {index}: {{int status={kernel.finish}(c->provider_states[{index}],&c->invocation);"
                "if(status)return status;break;}"
            )
        source.append("  default:break;\n  }")
    source.append(f"  if(c->active_dispatch=={n - 1}) {{")
    source.extend(_views(plan.trailing_views, positions))
    for index, buffer in enumerate(results):
        source.append(f"    memcpy(c->outputs[{index}],c->buffers[{positions[buffer.id]}],{buffer.bytes});")
    source.append("  }\n  c->prepared=0;c->completed_dispatches++;return 0;\n}")
    source.append(f"""int {p}_run({p}_context*c,const merlin_staged_config*config,
            const merlin_staged_hooks*hooks,{p}_launch_fn launch,{p}_join_fn join,void*platform) {{
  if(!c||!config||!hooks||!launch||!join||!hooks->current_hart||!hooks->worker_cycles||
     !hooks->worker_instret||config->worker_count!={workers}||!config->sequence||
     config->sequence>UINT64_MAX-{n - 1})return -120;
  if(config->controller_hart!={roles.dispatch_hart} ||
     {" || ".join(f"config->worker_harts[{i}]!={hart}" for i, hart in enumerate(roles.compute_harts))})return -124;
  if(hooks->current_hart(platform)!=config->controller_hart)return -121;
  unsigned idle=0;
  if(!atomic_compare_exchange_strong_explicit(&c->running,&idle,1,
      memory_order_acq_rel,memory_order_relaxed))return -122;
  if(config->sequence<=c->last_sequence){{atomic_store_explicit(&c->running,0,memory_order_release);return -125;}}
  if(c->poisoned){{atomic_store_explicit(&c->running,0,memory_order_release);return -123;}}
  c->platform_hooks=*hooks;c->platform=platform;c->completed_dispatches=0;
  c->prepared=0;c->initialized=0;c->observed_count=0;c->status=0;
  merlin_staged_hooks wrapper={{.current_hart=self_hart,.global_ticks=hooks->global_ticks?global_tick:NULL,
    .worker_cycles=cycles,.worker_instret=retired,.relax=relax,
    .shared_window_ticks=hooks->shared_window_ticks?checked_tick:NULL}};
  const merlin_staged_callbacks callbacks={{model_init,prepare,compute,finish}};
  for(unsigned i=0;i<{n};i++){{
    c->active_dispatch=i;merlin_staged_config next=*config;next.sequence=config->sequence+i;
    c->status=merlin_staged_team_init(&c->team,&next,&wrapper,&callbacks,c);
    if(c->status)break;
    launch(platform,&c->team);
    c->status=merlin_staged_controller(&c->team);
    if(!c->team.report.quiescent){{c->status=MERLIN_STAGED_UNQUIESCED;break;}}
    int joined=join(platform,&c->team);
    if(joined){{c->status=joined;break;}}
    c->observations[i]=c->team.report;c->observed_count=i+1;
    if(c->status)break;
  }}
  if(c->status)c->poisoned=1;
  atomic_store_explicit(&c->running,0,memory_order_release);
  return c->status;
}}
""")
    metadata = {
        "prefix": prefix,
        "kernel_entries": list(kernel_entries),
        "roles": roles.to_dict(),
        "controller_math": False,
        "persistent_workers": "caller-created and pinned",
        "sequence_policy": "strictly increasing fresh quiescent team per dispatch",
        "arithmetic_sites": len(plan.dispatches),
        "workspace_bytes": plan.workspace_bytes,
        "prepared_staged_kernels": {str(i): k.record() for i, k in prepared_kernels.items()},
        "provider_state_layout": provider_state_layout,
        "effective_worker_masks": masks,
        "full_model_admission": False,
        "physical_busy": None,
    }
    if frozen is not None:
        metadata["external_input_source_indices"] = frozen["external_input_source_indices"]
        metadata["frozen_blob_bytes"] = frozen["published_blob_bytes"]
    return {
        "header": header,
        "source": "\n".join(source) + "\n",
        "metadata": metadata,
    }
