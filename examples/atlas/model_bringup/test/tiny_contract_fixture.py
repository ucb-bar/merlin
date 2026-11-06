"""Synthetic Tiny contracts shared by full-model admission controls."""
import copy

from atlas_program import dma_load_bytes
import tiny_full_contract as tiny


def tiny_contract_fixture(decode_count, scales, greedy_tokens, projection_calls, *, full_provenance=False, reuse=False):
    kernels = []
    for index, item in enumerate(tiny.plan(decode_count)):
        m, k, n = item["physical_mkn"]
        load = (m*k*(n//32)+k*n if k*n <= 262144 else
                m*k+k*n*(m//32) if m*k <= 262144 else (m//32)*(n//32)*(2*k//32*1024))
        if reuse:
            load = dma_load_bytes(m, k, n, traffic_reuse=True)
        kernels.append({**item, "index": index, "dram_base": 0x100000000,
            "output_offset": m*k+k*n+256, "total_dram_bytes": m*k+k*n+512+2*m*n,
            "requested_dma": {"load_bytes": load, "store_bytes": 2*m*n},
            "A_scale": scales[0], "B_scale": scales[1], "rescale": scales[2]})
    references = []
    for name, (shape, dtype) in tiny.reference_geometry(decode_count).items():
        size = 2 if dtype == "uint16" else 4
        for extent in shape:
            size *= extent
        references.append({"actual_symbol": name, "expected_symbol": name + "_expected",
                           "shape": shape, "dtype": dtype, "bytes": size})
    parameters = {name: {"shape": shape, "dtype": dtype, "symbol": f"tiny_full_param_{index:03d}"}
                  for index, (name, (shape, dtype)) in enumerate(sorted(tiny.parameter_geometry().items()))}
    data = {"schema": "atlas.tiny_full_contract.v1", "cycle_cap": tiny.CYCLE_CAP,
        "decode_count": decode_count, "kernel_count": len(kernels), "projection_calls": projection_calls,
        "kernels": kernels, "host_parameters": parameters, "reference_buffers": references,
        "greedy_tokens": list(greedy_tokens), "mailbox_symbol": "tiny_full_mailbox", "panel_symbol": "tiny_full_panels",
        "state_symbol": "tiny_full_state_global", "context_symbol": "tiny_full_projection_state"}
    if reuse:
        data.update(program_policy="traffic-reuse-v1", reuse=dict(captured_data_unchanged=True, instruction_execution=False))
    return data, copy.deepcopy(data if full_provenance else {"kernel_count": len(kernels), "kernels": kernels})
