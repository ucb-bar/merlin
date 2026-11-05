#!/usr/bin/env python3
"""Check declared development policy and optional disjoint DRAM reservations."""

import argparse
import json
import re
from pathlib import Path

import yaml


def unique_mapping(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


class UniqueLoader(yaml.SafeLoader):
    pass


def yaml_mapping(loader, node):
    return unique_mapping(loader.construct_pairs(node))


UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, yaml_mapping)


def integer(value, name, minimum=0):
    if type(value) is not int or not minimum <= value < 2**64:
        raise ValueError(f"{name} must be an integer in [{minimum}, 2**64)")
    return value


def interval(address, size, name):
    start = integer(address, f"{name}.address")
    end = start + integer(size, f"{name}.size_bytes", 1)
    if end > 2**64:
        raise ValueError(f"{name} exceeds the 64-bit address space")
    return start, end


def check_target(config, memory_plan=None):
    profile = config["development"]
    if config["target"] != "gemmini" or config["selection"]["lowering_route"] != "gemmini_c_library":
        raise ValueError("this checker supports the Gemmini C-library target")
    sources = profile["sources"]
    for key in ("gemmini_commit", "rocc_tests_commit", "libgemmini_commit", "params_sha256", "operators_sha256"):
        length = 64 if key.endswith("sha256") else 40
        if not isinstance(sources[key], str) or re.fullmatch(f"[0-9a-f]{{{length}}}", sources[key]) is None:
            raise ValueError(f"invalid source identity: {key}")
    arithmetic = profile["arithmetic"]
    if arithmetic["input_dtype"] != "int8" or arithmetic["output_dtype"] != "int32" or arithmetic["full_accumulator_readout"] is not True:
        raise ValueError("the first adapter requires signed int8 inputs and full int32 output")
    for key in ("dim", "scratchpad_bytes", "accumulator_bytes"):
        integer(arithmetic[key], key, 1)
    dim = arithmetic["dim"]
    pe_bits = integer(arithmetic["pe_output_bits"], "pe_output_bits", 1)
    if dim != 16 or pe_bits != 20:
        raise ValueError("the initial OS adapter requires the provisional DIM=16 and 20-bit PE output")
    for key, row_bytes, min_rows in (("scratchpad_bytes", dim, 2 * dim), ("accumulator_bytes", 4 * dim, dim)):
        size = arithmetic[key]
        if size % row_bytes or size // row_bytes < min_rows:
            raise ValueError(f"{key} must contain whole rows with room for one OS tile")
    policy = profile["instructions"]
    if policy["dataflow"] != "OS" or policy["hardware_loops"] != "forbid":
        raise ValueError("the baseline requires primitive OS calls and hardware_loops: forbid")
    if config["selection"]["loop_policy"] != "primitive_only" or config["selection"]["runtime"] != "baremetal":
        raise ValueError("confirmed policy and runtime must match the development profile")
    if type(policy["paper_confirmed"]) is not bool:
        raise ValueError("instructions.paper_confirmed must be boolean")
    runtime = profile["runtime"]
    if runtime != {"kind": "baremetal", "isa": "rv64gc", "abi": "lp64d"}:
        raise ValueError("only the provisional baremetal rv64gc/lp64d profile is supported")
    memory = profile["memory"]
    base, end = interval(memory["base_address"], memory["requested_size_bytes"], "requested DRAM")
    confirmed = memory["confirmed_size_bytes"]
    if confirmed is not None:
        _, confirmed_end = interval(base, confirmed, "declared confirmed DRAM")
        if end > confirmed_end:
            raise ValueError("requested DRAM exceeds declared confirmed capacity")
    reserved = None
    regions = []
    if memory_plan is not None:
        if set(memory_plan) != {"regions"} or not isinstance(memory_plan["regions"], list) or not memory_plan["regions"]:
            raise ValueError("memory plan must contain a nonempty regions list")
        names = set()
        for region in memory_plan["regions"]:
            if set(region) != {"name", "address", "size_bytes"}:
                raise ValueError("each reservation needs only name, address and size_bytes")
            name = region["name"]
            if not isinstance(name, str) or not name.strip() or name in names:
                raise ValueError("reservation names must be nonempty and unique")
            names.add(name)
            start, stop = interval(region["address"], region["size_bytes"], name)
            if start < base or stop > end:
                raise ValueError(f"{name} is outside the requested DRAM map")
            regions.append((start, stop, name))
        regions.sort()
        for left, right in zip(regions, regions[1:]):
            if left[1] > right[0]:
                raise ValueError(f"reservations overlap: {left[2]} and {right[2]}")
        reserved = sum(stop - start for start, stop, _ in regions)
    return {
        "configuration_valid": True,
        "scope": "declared configuration and supplied reservation intervals only",
        "paper_ready": False,
        "deployment_qualified": False,
        "declared_sources": sources,
        "compile_defines": ["TVM_GEMMINI_FORBID_HW_LOOPS=1", "TVM_GEMMINI_PE_OUTPUT_BITS=20"],
        "instruction_policy_confirmed": policy["paper_confirmed"],
        "memory": {
            "base_address": base,
            "requested_size_bytes": end - base,
            "declared_confirmed_size_bytes": confirmed,
            "reserved_bytes": reserved,
            "unreserved_bytes": None if reserved is None else end - base - reserved,
            "fits_requested_map": None if memory_plan is None else True,
            "deployment_fit_verified": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--memory-plan", type=Path, help="JSON with disjoint coarse reservations: code/static, weights, arena, stack, etc.")
    args = parser.parse_args()
    try:
        config = yaml.load(args.config.read_text(), Loader=UniqueLoader)
        plan = None if args.memory_plan is None else json.loads(args.memory_plan.read_text(), object_pairs_hook=unique_mapping)
        print(json.dumps(check_target(config, plan), indent=2))
    except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError) as error:
        parser.exit(1, f"target check failed: {error}\n")


if __name__ == "__main__":
    main()
