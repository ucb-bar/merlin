"""Source-bound native convolution descriptors, not an im2col/library dispatch.

This target edge emits the pinned header's actual LOOP_CONV_WS register program.
It intentionally supports only one complete, capacity-fitting NHWC integer tile.
It does not change a compiler's source semantics, insert completion instructions,
claim numerical certification, or estimate cycles. Unsupported cases raise.
"""
from __future__ import annotations

import ast
import hashlib
import json
import struct
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Mapping

from merlin.perf.hw_counters import _module_lines, _operand_refs
from merlin.targetgen.address_space import derive_address_space
from merlin.targetgen.capability_discovery import (
    _balanced_end, _split_top_level, parse_c_header,
)
from merlin.targetgen.target_experiment import derived_readout_bits


class UnsupportedNativeConv(ValueError):
    """No proven native descriptor for this source operation/target revision."""


def _require(condition, message):
    if not condition:
        raise UnsupportedNativeConv(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _integer(model, name):
    macro = model.macro(name)
    _require(macro is not None and macro.int_value is not None, f"UNKNOWN header integer {name}")
    return macro.int_value


def _expression(source, values):
    """Restricted C integer packing expression, never Python eval."""
    cleaned = source
    for cast in ("(uint64_t)", "(uint32_t)", "(acc_scale_t)"):
        cleaned = cleaned.replace(cast, "")
    node = ast.parse(cleaned, mode="eval").body

    def visit(item):
        if isinstance(item, ast.Constant) and type(item.value) is int:
            return item.value
        if isinstance(item, ast.Name) and item.id in values:
            return values[item.id]
        if (isinstance(item, ast.Call) and isinstance(item.func, ast.Name)
                and item.func.id == "acc_scale_t_to_acc_scale_t_bits" and len(item.args) == 1
                and not item.keywords):
            return int.from_bytes(struct.pack("<f", visit(item.args[0])), "little")
        if isinstance(item, ast.BinOp):
            a, b = visit(item.left), visit(item.right)
            if isinstance(item.op, ast.BitOr):
                return a | b
            if isinstance(item.op, ast.LShift):
                return a << b
            if isinstance(item.op, ast.Mult):
                return a * b
            if isinstance(item.op, ast.Div):
                _require(b > 0 and a % b == 0, "nonintegral capacity partition")
                return a // b
        raise UnsupportedNativeConv(f"UNKNOWN header expression {source}")

    # Reject fields spilling into their neighbor instead of silently truncating.
    def fields(item):
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.BitOr):
            return fields(item.left) + fields(item.right)
        if isinstance(item, ast.BinOp) and isinstance(item.op, ast.LShift):
            return [(visit(item.right), visit(item.left))]
        return [(0, visit(item))]

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        packed = sorted(fields(node))
        for index, (shift, value) in enumerate(packed):
            end = packed[index + 1][0] if index + 1 < len(packed) else 64
            _require(0 <= shift < end <= 64 and 0 <= value < 1 << (end - shift),
                     f"descriptor field overflow in {source}")
    return visit(node)


def _narrow_store_proof(hw_text, *, full_bit, store_funct):
    """Trace the actual store output mux and prove its read-full bit is zero."""
    lines, error = _module_lines(hw_text, "LoopConvSt")
    _require(lines is not None, error)
    definitions = dict(line.strip().split(" = ", 1) for line in lines if " = " in line
                       and line.strip().startswith("%"))
    # Bind by the module's output port order, not a guessed SSA name/namehint.
    header = lines[0]
    ports = [part.strip().split(":", 1)[0][4:].strip() for part in _split_top_level(
        header[header.index("(") + 1:header.rfind(")")], ",") if part.strip().startswith("out ")]
    output = next((line.strip() for line in lines if line.strip().startswith("hw.output ")), "")
    refs = output.removeprefix("hw.output ").split(" :", 1)[0].split(",")
    _require("io_cmd_bits_rs2" in ports and "io_cmd_bits_inst_funct" in ports,
             "UNKNOWN convolution store output ports")
    rs2 = refs[ports.index("io_cmd_bits_rs2")].strip()
    funct = refs[ports.index("io_cmd_bits_inst_funct")].strip()

    def constant(ref):
        body = definitions.get(ref, "")
        _require(body.startswith("hw.constant "), "UNKNOWN store predicate constant")
        return int(body.split()[1])

    def known_bit(ref, bit):
        body = definitions.get(ref, "")
        if body.startswith("hw.constant "):
            return (constant(ref) >> bit) & 1
        operands = _operand_refs(body, "comb.mux")
        if operands:
            left, right = known_bit(operands[1], bit), known_bit(operands[2], bit)
            return left if left == right else None
        operands = _operand_refs(body, "comb.concat")
        if operands:
            widths = [int(t.strip().removeprefix("i")) for t in body.rsplit(" : ", 1)[1].split(",")]
            for operand, width in reversed(list(zip(operands, widths, strict=True))):
                if bit < width:
                    return known_bit(operand, bit)
                bit -= width
        return None

    mux = _operand_refs(definitions.get(rs2, ""), "comb.mux")
    _require(mux is not None, "UNKNOWN convolution store output mux")
    predicate = definitions.get(mux[0], "")
    _require(predicate.startswith("comb.icmp bin eq "), "UNKNOWN store output predicate")
    compared = predicate.removeprefix("comb.icmp bin eq ").split(" :", 1)[0].split(",")
    compared = [ref.strip() for ref in compared]
    _require(funct in compared, "store predicate not bound to output funct")
    _require(constant(compared[1 - compared.index(funct)]) == store_funct,
             "store predicate does not match pinned opcode")
    _require(known_bit(mux[1], full_bit.bit_length() - 1) == 0,
             "UNKNOWN native convolution full-width readout semantics")
    return {"module": "LoopConvSt", "output_ssa": rs2, "store_branch_ssa": mux[1],
            "full_width_mask": full_bit, "full_width_bit": 0}


def _descriptor_decode_proof(hw_text, instructions, parameters):
    """Overapproximate decoder mux paths for these exact descriptor values.

    Every possible non-hold update must equal the requested value. Unknown
    control inputs explore both branches; unsupported data paths refuse. This
    proves descriptor register values conditional on command acceptance, not
    sequencer progress or convolution arithmetic. Pointer truncation becomes an
    explicit physical-address-width obligation, never presumed identity.
    """
    lines, error = _module_lines(hw_text, "LoopConv")
    _require(lines is not None, error)
    definitions = dict(line.strip().split(" = ", 1) for line in lines if " = " in line
                       and line.strip().startswith("%"))
    aliases = {"weights_dram_addr": "weights", "output_dram_addr": "output",
               "bias_dram_addr": "bias", "input_dram_addr": "input",
               "a_ex_spad_id": "a_spad_id", "b_ex_spad_id": "b_spad_id"}
    registers = []
    for ref, body in definitions.items():
        if not ref.startswith("%loops_") or not body.startswith("seq.firreg "):
            continue
        suffix = ref.removeprefix("%loops_").split("_", 1)[1]
        field = suffix.removeprefix("outer_bounds_").removeprefix("inner_bounds_")
        field = aliases.get(field, field)
        if field in parameters:
            width = int(body.rsplit(" : i", 1)[1])
            registers.append((ref, body.split()[1], field, width))
    _require(registers, "UNKNOWN convolution descriptor state registers")
    covered = set()
    address_widths = set()
    proof_rows = []
    for instruction in instructions:
        command = {"%cmd_q.io_deq_bits_cmd_inst_funct": instruction["funct"],
                   "%cmd_q.io_deq_bits_cmd_rs1": instruction["rs1"],
                   "%cmd_q.io_deq_bits_cmd_rs2": instruction["rs2"]}

        @lru_cache(None)
        def values(ref):
            if ref in command:
                return frozenset([command[ref]])
            body = definitions.get(ref, "")
            if body.startswith("hw.constant "):
                if body.split()[1] in ("true", "false"):
                    return frozenset([int(body.split()[1] == "true")])
                width = int(body.rsplit(" : i", 1)[1])
                return frozenset([int(body.split()[1]) % (1 << width)])
            if body.startswith("seq.firreg ") or not body:
                return frozenset([ref])
            mux = _operand_refs(body, "comb.mux")
            if mux:
                conditions = values(mux[0])
                choices = set()
                if any(value != 0 for value in conditions):
                    choices.update(values(mux[1]))
                if any(value != 1 for value in conditions):
                    choices.update(values(mux[2]))
                return frozenset(choices)
            if body.startswith("comb.extract "):
                head, tail = body.removeprefix("comb.extract ").split(" from ", 1)
                offset = int(tail.split()[0])
                width = int(body.rsplit("-> i", 1)[1])
                result = set()
                for value in values(head):
                    if type(value) is int:
                        result.add((value >> offset) & ((1 << width) - 1))
                    elif value in (parameters["input"], parameters["weights"], parameters["output"]):
                        _require(offset == 0, "unsupported pointer address transformation")
                        result.add(("pointer", value, width))
                    else:
                        result.add(ref)
                return frozenset(result)
            if body.startswith("comb.icmp "):
                tail = body.removeprefix("comb.icmp ").removeprefix("bin ")
                predicate, operands = tail.split(" ", 1)
                a, b = [s.strip() for s in operands.split(" :", 1)[0].split(",")]
                result = set()
                for left, right in product(values(a), values(b)):
                    if type(left) is int and type(right) is int and predicate in ("eq", "ne"):
                        result.add(int((left == right) == (predicate == "eq")))
                    else:
                        result.update((0, 1))
                return frozenset(result)
            for op in ("and", "or", "xor", "concat"):
                refs = _operand_refs(body, "comb." + op)
                if refs is None:
                    continue
                widths = [int(t.strip().removeprefix("i")) for t in body.rsplit(" : ", 1)[1].split(",")]
                result = set()
                for operands in product(*(values(item) for item in refs)):
                    if op == "and" and 0 in operands:
                        result.add(0)
                    elif op == "or" and widths == [1] and 1 in operands:
                        result.add(1)
                    elif all(type(value) is int for value in operands):
                        value = operands[0]
                        for at, operand in enumerate(operands[1:], 1):
                            if op == "and": value &= operand
                            elif op == "or": value |= operand
                            elif op == "xor": value ^= operand
                            else: value = (value << widths[at]) | operand
                        result.add(value)
                    elif widths == [1]:
                        result.update((0, 1))
                    else:
                        result.add(ref)
                return frozenset(result)
            return frozenset([ref])

        for ref, next_value, field, width in registers:
            possible = set(values(next_value)) - {ref}
            if not possible:
                continue
            expected = parameters[field]
            if isinstance(expected, str):
                expected = ("pointer", expected, width)
                address_widths.add(width)
            _require(possible == {expected},
                     f"pinned RTL descriptor mismatch: {instruction['name']} -> {ref}: {possible!r} != {expected!r}")
            covered.add(field)
            proof_rows.append({"instruction": instruction["name"], "register": ref,
                               "parameter": field, "register_width": width})
    required = {field for _, _, field, _ in registers}
    _require(covered == required, f"UNKNOWN descriptor fields {sorted(required - covered)}")
    # These software descriptor fields are absent from this RTL's register state
    # and are only admitted at zero; do not invent support for their nonzero form.
    absent = set(parameters) - covered
    _require(all(parameters[field] == 0 for field in absent), f"unbound nonzero fields {sorted(absent)}")
    return {"status": "verified_exact_descriptor_register_updates", "module": "LoopConv",
            "scope": "all_possible_non_hold_mux_paths_for_actual_descriptor_values",
            "conditional_on": "descriptor command accepted by sequencer",
            "register_updates": proof_rows, "zero_only_absent_fields": sorted(absent),
            "required_physical_address_bits": sorted(address_widths),
            "sequencer_progress_and_arithmetic": "UNPROVEN"}


class NativeConvContract:
    """Load explicit sources; verify hardware bytes against the promoted facts."""

    def __init__(self, *, header: Path, params: Path, facts: Path, core_hw: Path):
        self.header_text = header.read_text()
        self.header = parse_c_header(header)
        self.params = parse_c_header(params)
        record = json.loads(facts.read_text())
        hw_bytes = core_hw.read_bytes()
        self.hw_text = hw_bytes.decode()
        _require(_sha(hw_bytes) == record["inputs"]["core_hw_sha256"], "stale core hardware identity")
        self.provenance = {"header_sha256": _sha(header.read_bytes()),
                           "params_sha256": _sha(params.read_bytes()),
                           "facts_sha256": _sha(facts.read_bytes()), "core_hw_sha256": _sha(hw_bytes)}
        tables = [entry for entry in record["facts"]["interfaces"] if entry.get("name") == "funct_decode_table"]
        _require(len(tables) == 1, "UNKNOWN instruction decode table")
        self.opcodes = {name: int(code) for code, name in tables[0]["names"].items()}
        self.custom_opcode = tables[0]["custom_opcode"]
        self.funct3 = tables[0]["funct3"]
        space = derive_address_space("gemmini", facts=record)
        stores = {store.name: store for store in space.stores}
        self.spad, self.acc = stores["scratchpad"], stores["accumulator"]
        self.dim = _integer(self.params, "DIM")
        _require(self.dim == self.spad.row_elems == self.acc.row_elems, "header/RTL mesh mismatch")
        _require(_integer(self.params, "BANK_NUM") * _integer(self.params, "BANK_ROWS") == self.spad.total_rows
                 and _integer(self.params, "ACC_ROWS") == self.acc.total_rows, "header/RTL capacity mismatch")
        types = {alias: underlying for alias, underlying, _ in self.params.typedefs}
        _require(types.get("elem_t") == f"int{self.spad.element_bits}_t"
                 and types.get("acc_t") == f"int{self.acc.element_bits}_t", "unsupported signed element ABI")
        identity = self.params.macro("ACC_SCALE_IDENTITY")
        _require(identity is not None and identity.body.strip() in ("1.0", "1.0f"),
                 "UNKNOWN identity accumulator scale")
        _require(types.get("acc_scale_t") == "float" and types.get("acc_scale_t_bits") == "uint32_t",
                 "unsupported accumulator scale representation")
        converter = "static acc_scale_t_bits acc_scale_t_to_acc_scale_t_bits(acc_scale_t x)"
        _require(converter in self.header_text, "UNKNOWN scale conversion")
        start = self.header_text.index("{", self.header_text.index(converter))
        end = _balanced_end(self.header_text, start)
        actual = " ".join(self.header_text[start:end].split())
        expected = "{ union { acc_scale_t_bits b; acc_scale_t f; } un; un.f = x; return un.b; }"
        _require(actual == expected, "unsupported scale conversion implementation")
        self.readout = _narrow_store_proof(hw_bytes.decode(),
            full_bit=derived_readout_bits(_integer(self.params, "ADDR_LEN"))["full_c_bit"],
            store_funct=self.opcodes["STORE_CMD"])
        self.capacity = {}
        constants = {name: _integer(self.params, name) for name in ("BANK_NUM", "BANK_ROWS", "ACC_ROWS")}
        for name in ("max_spad_rows", "max_acc_rows"):
            assignments = [line.split("=", 1)[1].split(";", 1)[0].strip()
                           for line in self.header_text.splitlines()
                           if line.strip().startswith(f"const int {name} =")]
            # Same expressions may occur in other tilers; require agreement.
            values = {_expression(value, constants) for value in assignments}
            _require(len(values) == 1, f"UNKNOWN {name} partition")
            self.capacity[name] = values.pop()


def native_entry_instructions(contract, *, output_channels, activation):
    """Exact header expansion of the library's no-pool native-conv entry configs."""
    ex = dict(dataflow=_integer(contract.header, "WEIGHT_STATIONARY"), sys_act=0,
              sys_shift=0, sys_acc_scale=0, C_stride=1, A_stride=1, A_transpose=0,
              B_transpose=0, set_only_strides=0, act_mx_fmt=0, wgt_mx_fmt=0, out_mx_fmt=0, uselut=0)
    st = dict(stride=output_channels * contract.spad.element_bits // 8, acc_act=activation,
              acc_scale=1, pool_stride=0, pool_size=0, pool_out_dim=0, porows=0, pocols=0,
              orows=0, ocols=0, upad=0, lpad=0)
    rows = []
    for name, parameters in (("gemmini_extended2_config_st", st), ("gemmini_extended3_config_ex", ex)):
        macro = contract.header.macro(name)
        _require(macro is not None and set(macro.params) == set(parameters), "UNKNOWN native entry config parameters")
        values = dict(parameters, CONFIG_ST=_integer(contract.header, "CONFIG_ST"),
                      CONFIG_EX=_integer(contract.header, "CONFIG_EX"))
        marker = "ROCC_INSTRUCTION_RS1_RS2("
        _require(macro.body.count(marker) == 1, "UNKNOWN native entry config instruction")
        start = macro.body.index(marker) + len(marker) - 1
        end = _balanced_end(macro.body, start)
        args = _split_top_level(macro.body[start + 1:end - 1], ",")
        _require(len(args) == 4 and args[0].strip() == "XCUSTOM_ACC", "unsupported config encoding")
        funct = _integer(contract.header, args[3].strip())
        _require(funct == contract.opcodes["CONFIG_CMD"], "config opcode differs from RTL")
        rows.append({"name": name, "funct": funct, "rs1": _expression(args[1].strip(), values),
                     "rs2": _expression(args[2].strip(), values)})
    return rows


def emit_native_conv(command: Mapping, tensors: Mapping, *, contract: NativeConvContract,
                     pointers: Mapping[str, str], row_strides: Mapping[str, int]) -> dict:
    """Emit native C plus exact register descriptors for an eligible schema CONV2D.

    Physical pointers/row strides are supplied by the compiler ABI. Entry must be
    drained, operands disjoint, and no unrelated commands concurrent; completion
    and host visibility remain the caller's explicit target-bound obligations.
    """
    _require(command.get("opcode") == "CONV2D", "not schema CONV2D")
    a = command.get("attributes", {})
    allowed = {"kernel", "stride", "padding", "dilation", "layout", "epilogue", "output_dtype"}
    _require(set(a) <= allowed, f"unsupported convolution attributes {sorted(set(a) - allowed)}")
    _require(a.get("layout") == "nhwc" and a.get("stride") == [1, 1]
             and a.get("dilation") == [1, 1] and a.get("padding") == [0, 0, 0, 0],
             "unsupported layout/stride/dilation/padding")
    _require(a.get("epilogue", []) in ([], ["relu"]), "unsupported rich epilogue; cannot silently drop stages")
    operands = command.get("operands", {})
    _require(set(operands) == {"ifm", "weight", "dst"}, "unsupported bias/resident/extra operand")
    _require(all(name in tensors for name in operands.values()), "unresolved resident or tensor operand")
    x, w, y = [tensors[operands[name]] for name in ("ifm", "weight", "dst")]
    dtype = contract.spad.element_dtype
    _require(x.get("dtype") == w.get("dtype") == y.get("dtype") == a.get("output_dtype") == dtype,
             "native convolution requires signed narrow saturating output; full-width/modular/floating output unsupported")
    _require(len(x["shape"]) == 4 and len(a.get("kernel", [])) == 4, "unsupported tensor/kernel rank")
    n, h, width, ci = x["shape"]
    kh, kw, kci, co = a["kernel"]
    _require(all(type(v) is int and v > 0 for v in (n, h, width, ci, kh, kw, kci, co)), "nonpositive/static shape")
    oh, ow = h - kh + 1, width - kw + 1
    _require(kh == kw and ci == kci and oh > 0 and ow > 0, "unsupported kernel geometry")
    _require(w["shape"] == [kh * kw * ci, co] and y["shape"] == [n * oh * ow, co], "CONV2D physical shape mismatch")
    _require(kh * kw * ci * (1 << (2 * (contract.spad.element_bits - 1)))
             <= (1 << (contract.acc.element_bits - 1)) - 1, "possible accumulator overflow")
    _require(set(pointers) == set(row_strides) == set(operands), "physical ABI bindings missing")
    _require(all(isinstance(p, str) and p.isascii() and p.isidentifier() for p in pointers.values()), "pointer must be a C identifier")
    _require(len(set(pointers.values())) == len(pointers), "aliased ABI pointer identity")
    _require(row_strides == {"ifm": ci, "weight": co, "dst": co}, "first native route requires explicit dense channel strides")
    dim = contract.dim
    a_rows = ((ci + dim - 1) // dim) * n * h * width
    b_rows = ((co + dim - 1) // dim) * kh * kw * ci
    c_rows = ((co + dim - 1) // dim) * n * oh * ow
    _require(a_rows + b_rows <= contract.capacity["max_spad_rows"]
             and c_rows <= contract.capacity["max_acc_rows"], "single native tile exceeds derived double-buffer capacity")
    macro = contract.header.macro("gemmini_loop_conv_ws")
    _require(macro is not None, "missing native convolution macro")
    v = {name: 0 for name in macro.params}
    v.update(batch_size=n, in_row_dim=h, in_col_dim=width, in_channels=ci, out_channels=co,
             out_row_dim=oh, out_col_dim=ow, pool_out_row_dim=oh, pool_out_col_dim=ow,
             stride=1, kernel_dim=kh, kernel_dilation=1, pool_size=1, pool_stride=1,
             batches=n, porows=oh, pocols=ow, pochs=co, krows=kh, kcols=kw, kchs=ci,
             orows=oh, ocols=ow, weights=pointers["weight"], output=pointers["dst"], input=pointers["ifm"],
             no_bias=1, no_pool=1, max_pixels_per_row=1, in_stride=ci, weight_stride=co, out_stride=co,
             activation=_integer(contract.header, "RELU" if a.get("epilogue") else "NO_ACTIVATION"))
    # Unknown newly-added parameters must never silently default to zero.
    known_zero = {"padding", "pool_padding", "lpad", "rpad", "upad", "dpad", "plpad", "prpad", "pupad", "pdpad",
                  "bias", "downsample", "wrot180", "input_dilated", "trans_output_1203", "trans_weight_1203",
                  "trans_weight_0132", "trans_input_3120", "dw", "a_spad_id", "b_spad_id"}
    _require(set(macro.params) == {name for name, value in v.items() if value != 0} | known_zero | {"activation"},
             "UNKNOWN native macro parameter")
    instructions = []
    rest = macro.body
    marker = "ROCC_INSTRUCTION_RS1_RS2("
    while marker in rest:
        start = rest.index(marker) + len(marker) - 1
        end = _balanced_end(rest, start)
        args = _split_top_level(rest[start + 1:end - 1], ",")
        _require(len(args) == 4 and args[0].strip() == "XCUSTOM_ACC", "unsupported descriptor invocation")
        name = args[3].strip()
        funct = _integer(contract.header, name)
        _require(contract.opcodes.get(name.removeprefix("k_")) == funct, "header/RTL opcode mismatch")
        instructions.append({"funct": funct, "name": name.removeprefix("k_"),
                             "rs1": _expression(args[1].strip(), v), "rs2": _expression(args[2].strip(), v)})
        rest = rest[end:]
    _require(len(instructions) == 7 and instructions[-1]["name"] == "LOOP_CONV_WS", "incomplete convolution program")
    decode_proof = _descriptor_decode_proof(contract.hw_text, instructions, v)
    call = "gemmini_loop_conv_ws(" + ", ".join(str(v[name]) for name in macro.params) + ");"
    for name in ("gemmini_extended_config_st", "gemmini_extended3_config_ex"):
        _require(contract.header.macro(name) is not None, f"missing entry config {name}")
    code = (f"gemmini_extended_config_st({co} * sizeof(elem_t), {v['activation']}, ACC_SCALE_IDENTITY);\n"
            "gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, 0, 1, 1, false, false, false, 0, 0, 0, 0);\n" + call)
    return {"schema": "native_conv_descriptor_v1", "status": "emitted_not_runtime_qualified",
            "instructions": instructions, "c_source": code, "parameters": v,
            "source_command_sha256": _sha(json.dumps(command, sort_keys=True, separators=(",", ":")).encode()),
            "source_tensor_abi_sha256": _sha(json.dumps(
                {"tensors": {name: tensors[name] for name in operands.values()},
                 "pointers": dict(pointers), "row_strides": dict(row_strides)},
                sort_keys=True, separators=(",", ":")).encode()),
            "provenance": contract.provenance, "readout_proof": contract.readout,
            "capacity": {"input_rows": a_rows, "weight_rows": b_rows, "accumulator_rows": c_rows, **contract.capacity},
            "required_entry_contract": ["drained", "no_external_concurrent_commands", "nonaliasing_operand_allocations"],
            "required_exit_contract": ["target_completion", "host_visibility_before_output_use"],
            "descriptor_to_rtl_field_qualification": decode_proof,
            "numerical_runtime_qualification": "UNPROVEN", "cycles": "UNKNOWN"}
