# Penguin TPU (Atlas Accelerator) Instruction Green Card

Conventions used in the opcode tables:

- Bit positions use the RISC-V convention, with bit `31` the MSB and bit `0`
  the LSB.
- `Hex value` is written as `opcode_hex/funct3_hex/funct7_or_imm_hex` when that
  decomposition exists.
- `*` in the `Hex value` column means the field is operand-dependent rather than
  part of the opcode match.
- `pending` means the instruction family is architecturally allocated and its
  assembly-visible behavior is frozen, but the exact bit packing is not yet
  frozen in the encoding supplement.
- Verilog-style descriptions use architectural state names such as `x[...]`,
  `m[...]`, `e[...]`, `pc`, `VMEM[...]`, and `mxuN.acc`.
- whenever an instruction semantic uses `{m[r], m[r+1]}`, the encoded register
  field names the low register of the pair, and encoding `r = 63` is illegal.

## 1. Register Set

The baseline scalar procedure-call convention follows standard `RV32I` ABI
practice for the `x` register file. No procedure-call ABI has been frozen yet
for `m`, `e`, or MXU-local state; this card uses the conservative convention
`Caller` for those architected state elements.

| Register | ABI Name | Description | Saver |
|-------|---|---|---|
| `pc`  | `pc` | Architectural program counter, stored as an instruction-word index | N/A |
| `x0`  | `zero` | Constant zero | Fixed |
| `x1`  | `ra` | Return address | Caller |
| `x2`  | `sp` | Stack pointer | Callee |
| `x3`  | `gp` | Global pointer | Fixed |
| `x4`  | `tp` | Thread pointer | Fixed |
| `x5`  | `t0` | Temporary register 0 | Caller |
| `x6`  | `t1` | Temporary register 1 | Caller |
| `x7`  | `t2` | Temporary register 2 | Caller |
| `x8`  | `s0/fp` | Saved register 0 / frame pointer | Callee |
| `x9`  | `s1` | Saved register 1 | Callee |
| `x10` | `a0` | Argument / return value 0 | Caller |
| `x11` | `a1` | Argument / return value 1 | Caller |
| `x12` | `a2` | Argument 2 | Caller |
| `x13` | `a3` | Argument 3 | Caller |
| `x14` | `a4` | Argument 4 | Caller |
| `x15` | `a5` | Argument 5 | Caller |
| `x16` | `a6` | Argument 6 | Caller |
| `x17` | `a7` | Argument 7 | Caller |
| `x18` | `s2` | Saved register 2 | Callee |
| `x19` | `s3` | Saved register 3 | Callee |
| `x20` | `s4` | Saved register 4 | Callee |
| `x21` | `s5` | Saved register 5 | Callee |
| `x22` | `s6` | Saved register 6 | Callee |
| `x23` | `s7` | Saved register 7 | Callee |
| `x24` | `s8` | Saved register 8 | Callee |
| `x25` | `s9` | Saved register 9 | Callee |
| `x26` | `s10` | Saved register 10 | Callee |
| `x27` | `s11` | Saved register 11 | Callee |
| `x28` | `t3` | Temporary register 3 | Caller |
| `x29` | `t4` | Temporary register 4 | Caller |
| `x30` | `t5` | Temporary register 5 | Caller |
| `x31` | `t6` | Temporary register 6 | Caller |
| `m0-m63` | `-` | Flat tensor register file, `1024` bytes per register | Caller |
| `e0-e31` | `-` | Whole-tensor scale register file, one `FP8_E8M0` exponent per register | Caller |
| `mxu0.w0` | `-` | MXU0 FP8 weight slot 0 | Caller |
| `mxu0.w1` | `-` | MXU0 FP8 weight slot 1 | Caller |
| `mxu1.w0` | `-` | MXU1 FP8 weight slot 0 | Caller |
| `mxu1.w1` | `-` | MXU1 FP8 weight slot 1 | Caller |
| `mxu0.acc0` | `-` | MXU0 local `32 x 32 BF16` accumulation buffer 0 | Caller |
| `mxu0.acc1` | `-` | MXU0 local `32 x 32 BF16` accumulation buffer 1 | Caller |
| `mxu1.acc0` | `-` | MXU1 local `32 x 32 BF16` accumulation buffer 0 | Caller |
| `mxu1.acc1` | `-` | MXU1 local `32 x 32 BF16` accumulation buffer 1 | Caller |
| `dma.base` | `-` | DMA base address | Caller |

## 2. Core Instruction Formats

### 2.1 Scalar RV32I-Compatible Formats
The following ASCII layout diagrams mirror the standard RISC-V bit positions
(RV32, MSB is bit `31`, LSB is bit `0`). `SB` below corresponds to the RISC-V `B`
format (branches), and `UJ` corresponds to the RISC-V `J` format (jumps).

<style>
  .rv32i-format-table th, .rv32i-format-table td {
    border: 1px solid #999;
    padding: 2px 6px;
    text-align: center;
  }
  /* Remove outer "top border" and "left border" only */
  .rv32i-format-table thead th {
    border-top: none;
    border-left: none;
    border-right: none;
    font-weight: 200;
  }
  .rv32i-format-table thead th:first-child,
  .rv32i-format-table tbody td:first-child {
    border-left: none;
    border-top: none;
    border-bottom: none;
    text-align: right;
  }
</style>

<table class="rv32i-format-table" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr>
      <th></th>
      <th>
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>31</span>
          <span>25</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>24</span>
          <span>20</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>19</span>
          <span>15</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>14</span>
          <span>12</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>11</span>
          <span>7</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>6</span>
          <span>0</span>
        </div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>R</td>
      <td>funct7</td>
      <td>rs2</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>I</td>
      <td colspan=2>imm[11:0]</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>S</td>
      <td>imm[11:5]</td>
      <td>rs2</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>imm[4:0]</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>SB</td>
      <td>imm[12|10:5]</td>
      <td>rs2</td>
      <td>rs1</td>
      <td>funct3</td>
      <td>imm[4:1|11]</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>U</td>
      <td colspan="4" style="text-align:center;">imm[31:12]</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>UJ</td>
      <td colspan="4" style="text-align:center;">imm[20|10:1|11|19:12]</td>
      <td>rd</td>
      <td>opcode</td>
    </tr>
  </tbody>
</table>



### 2.5 Tensor operation

<table class="rv32i-format-table" style="border-collapse: collapse; width: 100%;">
  <thead>
    <tr>
      <th></th>
      <th>
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>31</span>
          <span>25</span>
        </div>
      </th>
      <th>
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>24</span>
          <span>20</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>19</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>18</span>
          <span>16</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>15</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>14</span>
          <span>13</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>12</span>
          <span>7</span>
        </div>
      </th>
      <th style="text-align:left; white-space:nowrap;">
        <div style="display:flex; justify-content:space-between; width:100%;">
          <span>6</span>
          <span>0</span>
        </div>
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VLS</td>
      <td colspan=2>imm[11:0]</td>
      <td colspan=3>rs1</td>
      <td>f2</td>
      <td>vd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>VR</td>
      <td>funct7</td>
      <td colspan=2>vs2</td>
      <td colspan=3>vs1</td>
      <td>vd</td>
      <td>opcode</td>
    </tr>
    <tr>
      <td>VI</td>
      <td colspan=4>imm[15:0]</td>
      <td colspan=2>f3</td>
      <td>vd</td>
      <td>opcode</td>
    </tr>
  </tbody>
</table>

Software may rely on the mnemonic repertoire and architectural semantics of
those instruction families, but not yet on a frozen bit-exact subformat.

## 3. Instructions

Rows are ordered by hex value.

| Mnemonic                 | Fmt      | Opcode    | Funct3 or Funct2        | Funct7 or Imm    | Hex Value  | Name                              | Description (in Verilog) |
|--------------------------|----------|-----------|-------------------------|------------------|------------|-----------------------------------|--------------------------|
| `lb`                     | `I`      | `0000011` | `000`                   |                  | `03/0`     | Load Byte                         | `x[rd] = {{24{VMEM[x[rs1] + imm][7]}}, VMEM[x[rs1] + imm]}` |
| `lh`                     | `I`      | `0000011` | `001`                   |                  | `03/1`     | Load Halfword                     | `x[rd] = {{16{VMEM[x[rs1] + imm][15]}}, VMEM[x[rs1] + imm]}` |
| `lw`                     | `I`      | `0000011` | `010`                   |                  | `03/2`     | Load Word                         | `x[rd] = VMEM[x[rs1] + imm]` |
| `lbu`                    | `I`      | `0000011` | `100`                   |                  | `03/4`     | Load Byte Unsigned                | `x[rd] = {24'b0, VMEM[x[rs1] + imm]}` |
| `lhu`                    | `I`      | `0000011` | `101`                   |                  | `03/5`     | Load Halfword Unsigned            | `x[rd] = {16'b0, VMEM[x[rs1] + imm]}` |
| `seld`                   | `I`      | `0000011` | `110`                   |                  | `03/6`     | Scale Factor Load                 | `e[rd] = VMEM[x[rs1] + imm];` |
| `seli`                   | `I`      | `0000011` | `111`                   |                  | `03/7`     | Scale Factor Load Immediate       | `e[rd] = imm;` |
| `vload`                  | `VLS`    | `0000111` | `00`                    |                  | `07/0`     | Tensor Load                       | `m[vd] = VMEM[x[rs1] + (imm12 << 5)];` |
| `vstore`                 | `VLS`    | `0000111` | `01`                    |                  | `07/1`     | Tensor Store                      | `VMEM[x[rs1] + (imm12 << 5)] = m[vd];` |
| `fence`                  | `I`      | `0001111` | `000`                   | `000000000000`   | `0F/0/000` | Fence                             | `np-op` |
| `addi`                   | `I`      | `0010011` | `000`                   |                  | `13/0`     | Add Immediate                     | `x[rd] = x[rs1] + imm` |
| `slli`                   | `I`      | `0010011` | `001`                   | `0000000_shamt`  | `13/1/00`  | Shift Left Logical Immediate      | `x[rd] = x[rs1] << shamt` |
| `slti`                   | `I`      | `0010011` | `010`                   |                  | `13/2`     | Set Less Than Immediate           | `x[rd] = ($signed(x[rs1]) < $signed(imm))` |
| `sltiu`                  | `I`      | `0010011` | `011`                   |                  | `13/3`     | Set Less Than Immediate Unsigned  | `x[rd] = (x[rs1] < $unsigned($signed(imm)))` |
| `xori`                   | `I`      | `0010011` | `100`                   |                  | `13/4`     | XOR Immediate                     | `x[rd] = x[rs1] ^ imm` |
| `srli`                   | `I`      | `0010011` | `101`                   | `0000000_shamt`  | `13/5/00`  | Shift Right Logical Immediate     | `x[rd] = x[rs1] >> shamt` |
| `srai`                   | `I`      | `0010011` | `101`                   | `0100000_shamt`  | `13/5/20`  | Shift Right Arithmetic Immediate  | `x[rd] = $signed(x[rs1]) >>> shamt` |
| `ori`                    | `I`      | `0010011` | `110`                   |                  | `13/6`     | OR Immediate                      | `x[rd] = x[rs1] \| imm` |
| `andi`                   | `I`      | `0010011` | `111`                   |                  | `13/7`     | AND Immediate                     | `x[rd] = x[rs1] & imm` |
| `auipc`                  | `U`      | `0010111` |                         |                  | `17`       | Add Upper Immediate to PC         | `x[rd] = pc + {imm[31:12], 12'b0}` |
| `sb`                     | `S`      | `0100011` | `000`                   |                  | `23/0`     | Store Byte                        | `VMEM[x[rs1] + imm] = x[rs2][7:0]` |
| `sh`                     | `S`      | `0100011` | `001`                   |                  | `23/1`     | Store Halfword                    | `VMEM[x[rs1] + imm] = x[rs2][15:0]` |
| `sw`                     | `S`      | `0100011` | `010`                   |                  | `23/2`     | Store Word                        | `VMEM[x[rs1] + imm] = x[rs2]` |
| `add`                    | `R`      | `0110011` | `000`                   | `0000000`        | `33/0/00`  | Add                               | `x[rd] = x[rs1] + x[rs2]` |
| `sub`                    | `R`      | `0110011` | `000`                   | `0100000`        | `33/0/20`  | Subtract                          | `x[rd] = x[rs1] - x[rs2]` |
| `sll`                    | `R`      | `0110011` | `001`                   | `0000000`        | `33/1/00`  | Shift Left Logical                | `x[rd] = x[rs1] << x[rs2][4:0]` |
| `slt`                    | `R`      | `0110011` | `010`                   | `0000000`        | `33/2/00`  | Set Less Than                     | `x[rd] = ($signed(x[rs1]) < $signed(x[rs2]))` |
| `sltu`                   | `R`      | `0110011` | `011`                   | `0000000`        | `33/3/00`  | Set Less Than Unsigned            | `x[rd] = (x[rs1] < x[rs2])` |
| `xor`                    | `R`      | `0110011` | `100`                   | `0000000`        | `33/4/00`  | XOR                               | `x[rd] = x[rs1] ^ x[rs2]` |
| `srl`                    | `R`      | `0110011` | `101`                   | `0000000`        | `33/5/00`  | Shift Right Logical               | `x[rd] = x[rs1] >> x[rs2][4:0]` |
| `sra`                    | `R`      | `0110011` | `101`                   | `0100000`        | `33/5/20`  | Shift Right Arithmetic            | `x[rd] = $signed(x[rs1]) >>> x[rs2][4:0]` |
| `or`                     | `R`      | `0110011` | `110`                   | `0000000`        | `33/6/00`  | OR                                | `x[rd] = x[rs1] \| x[rs2]` |
| `and`                    | `R`      | `0110011` | `111`                   | `0000000`        | `33/7/00`  | AND                               | `x[rd] = x[rs1] & x[rs2]` |
| `lui`                    | `U`      | `0110111` |                         |                  | `37`       | Load Upper Immediate              | `x[rd] = {imm[31:12], 12'b0}` |
| `vadd.bf16`              | `VR`     | `1010111` |                         | `0000000`        | `57/00`    | Vector Add                        | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]}.view(bf16) + {m[vs2], m[vs2+1]}.view(bf16);` |
| `vredsum.bf16`           | `VR`     | `1010111` |                         | `0000001`        | `57/01`    | Vector Sublane Reduction Sum      | `{m[vd], m[vd+1]}[i, j] = sum({m[vs1], m[vs1+1]}.view(bf16)[:, j]);` |
| `vsub.bf16`              | `VR`     | `1010111` |                         | `0000010`        | `57/02`    | Vector Subtract                   | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]}.view(bf16) - {m[vs2], m[vs2+1]}.view(bf16);` |
| `vmul.bf16`              | `VR`     | `1010111` |                         | `0000011`        | `57/03`    | Vector Multiply                   | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]}.view(bf16) * {m[vs2], m[vs2+1]}.view(bf16);` |
| `vminimum.bf16`          | `VR`     | `1010111` |                         | `0000100`        | `57/04`    | Vector Minimum                    | `{m[vd], m[vd+1]} = min({m[vs1], m[vs1+1]}.view(bf16), {m[vs2], m[vs2+1]}.view(bf16));` |
| `vredmin.bf16`           | `VR`     | `1010111` |                         | `0000101`        | `57/05`    | Vector Sublane Reduction Min      | `{m[vd], m[vd+1]}[i, j] = min({m[vs1], m[vs1+1]}.view(bf16)[:, j]);` |
| `vmaximum.bf16`          | `VR`     | `1010111` |                         | `0000110`        | `57/06`    | Vector Maximum                    | `{m[vd], m[vd+1]} = max({m[vs1], m[vs1+1]}.view(bf16), {m[vs2], m[vs2+1]}.view(bf16));` |
| `vredmax.bf16`           | `VR`     | `1010111` |                         | `0000111`        | `57/07`    | Vector Sublane Reduction Max      | `{m[vd], m[vd+1]}[i, j] = max({m[vs1], m[vs1+1]}.view(bf16)[:, j]);` |
| `vredsum.row.bf16`       | `VR`     | `1010111` |                         | `0100001`        | `57/21`    | Vector Lane Reduction Sum         | `{m[vd], m[vd+1]}[i, j] = sum({m[vs1], m[vs1+1]}.view(bf16)[i, :]);` |
| `vredmin.row.bf16`       | `VR`     | `1010111` |                         | `0100100`        | `57/24`    | Vector Lane Reduction Min         | `{m[vd], m[vd+1]}[i, j] = min({m[vs1], m[vs1+1]}.view(bf16)[i, :]);` |
| `vredmax.row.bf16`       | `VR`     | `1010111` |                         | `0100110`        | `57/26`    | Vector Lane Reduction Max         | `{m[vd], m[vd+1]}[i, j] = max({m[vs1], m[vs1+1]}.view(bf16)[i, :]);` |
| `vmov`                   | `VR`     | `1010111` |                         | `1000000`        | `57/40`    | Vector Move                       | `m[vd] = m[vs1];` |
| `vrecip.bf16`            | `VR`     | `1010111` |                         | `1000001`        | `57/41`    | Vector Reciprocal                 | `{m[vd], m[vd+1]} = 1.f / {m[vs1], m[vs1+1]};` |
| `vexp.bf16`              | `VR`     | `1010111` |                         | `1000010`        | `57/42`    | Vector Exponential                | `{m[vd], m[vd+1]} = bf16_exp({m[vs1], m[vs1+1]});` |
| `vexp2.bf16`             | `VR`     | `1010111` |                         | `1000011`        | `57/43`    | Vector Base 2 Exponential         | `{m[vd], m[vd+1]} = bf16_exp2({m[vs1], m[vs1+1]});` |
| `vpack.bf16.fp8`         | `VR`     | `1010111` |                         | `1000100`        | `57/44`    | Vector Packing                    | `m[vd] = quantize(m[vs2], m[vs2+1], e[es1]);` |
| `vunpack.fp8.bf16`       | `VR`     | `1010111` |                         | `1000101`        | `57/45`    | Vector Unpacking                  | `m[vd], m[vd+1] = dequantize(m[vs2], e[es1]);` |
| `vrelu.bf16`             | `VR`     | `1010111` |                         | `1001000`        | `57/48`    | Vector ReLU                       | `{m[vd], m[vd+1]} = bf16_relu({m[vs1], m[vs1+1]});` |
| `vsin.bf16`              | `VR`     | `1010111` |                         | `1001001`        | `57/49`    | Vector Sin                        | `{m[vd], m[vd+1]} = bf16_sin({m[vs1], m[vs1+1]});` |
| `vcos.bf16`              | `VR`     | `1010111` |                         | `1001010`        | `57/4A`    | Vector Cos                        | `{m[vd], m[vd+1]} = bf16_cos({m[vs1], m[vs1+1]});` |
| `vtanh.bf16`             | `VR`     | `1010111` |                         | `1001011`        | `57/4B`    | Vector Tanh                       | `{m[vd], m[vd+1]} = bf16_tanh({m[vs1], m[vs1+1]});` |
| `vlog2.bf16`             | `VR`     | `1010111` |                         | `1001100`        | `57/4C`    | Vector log2                       | `{m[vd], m[vd+1]} = bf16_log2({m[vs1], m[vs1+1]});` |
| `vsqrt.bf16`             | `VR`     | `1010111` |                         | `1001101`        | `57/4D`    | Vector sqrt                       | `{m[vd], m[vd+1]} = bf16_sqrt({m[vs1], m[vs1+1]});` |
| `vsquare.bf16`           | `VR`     | `1010111` |                         | `1001110`        | `57/4E`    | Vector Square                     | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]} * {m[vs1], m[vs1+1]};` |
| `vcube.bf16`             | `VR`     | `1010111` |                         | `1001111`        | `57/4F`    | Vector Cube                       | `{m[vd], m[vd+1]} = {m[vs1], m[vs1+1]} ** 3;` |
| `vli.all`                | `VI`     | `1011111` | `000`                   |                  | `5F/0`     | Vector Load Immediate             | `m[vd][:] = imm;` |
| `vli.row`                | `VI`     | `1011111` | `001`                   |                  | `5F/1`     | Vector Load Immediate             | `m[vd][0, :] = imm;` |
| `vli.col`                | `VI`     | `1011111` | `010`                   |                  | `5F/2`     | Vector Load Immediate             | `m[vd][:, 0] = imm;` |
| `vli.one`                | `VI`     | `1011111` | `011`                   |                  | `5F/3`     | Vector Load Immediate             | `m[vd][0, 0] = imm;` |
| `beq`                    | `B`      | `1100011` | `000`                   |                  | `63/0`     | Branch Equal                      | `if (x[rs1] == x[rs2]) pc = pc + imm after 2 delay slots` |
| `bne`                    | `B`      | `1100011` | `001`                   |                  | `63/1`     | Branch Not Equal                  | `if (x[rs1] != x[rs2]) pc = pc + imm after 2 delay slots` |
| `blt`                    | `B`      | `1100011` | `100`                   |                  | `63/4`     | Branch Less Than                  | `if ($signed(x[rs1]) < $signed(x[rs2])) pc = pc + imm after 2 delay slots` |
| `bge`                    | `B`      | `1100011` | `101`                   |                  | `63/5`     | Branch Greater Or Equal           | `if ($signed(x[rs1]) >= $signed(x[rs2])) pc = pc + imm after 2 delay slots` |
| `bltu`                   | `B`      | `1100011` | `110`                   |                  | `63/6`     | Branch Less Than Unsigned         | `if (x[rs1] < x[rs2]) pc = pc + imm after 2 delay slots` |
| `bgeu`                   | `B`      | `1100011` | `111`                   |                  | `63/7`     | Branch Greater Or Equal Unsigned  | `if (x[rs1] >= x[rs2]) pc = pc + imm after 2 delay slots` |
| `jalr`                   | `I`      | `1100111` | `000`                   |                  | `67/0`     | Jump And Link Register            | `next_pc = x[rs1] + imm; x[rd] = pc + 4; pc = next_pc after 2 delay slots` |
| `delay`                  | `I`      | `1100111` | `001`                   |                  | `67/1`     | Frontend Delay                    | `hold decode issue for imm cycles;` |
| `vtrpose.xlu`            | `VR`     | `1101011` |                         | `0000000`        | `6B/00`    | Matrix Transpose                  | `m[vd] = m[vs1].T;` |
| `jal`                    | `J`      | `1101111` |                         |                  | `6F`       | Jump And Link                     | `x[rd] = pc + 4; pc = pc + imm after 2 delay slots` |
| `ecall`                  | `I`      | `1110011` | `000`                   | `000000000000`   | `73/0/000` | Environment Call                  | `halt_reason = ECALL; halt = 1'b1;` |
| `ebreak`                 | `I`      | `1110011` | `000`                   | `000000000001`   | `73/0/001` | Breakpoint                        | `halt_reason = EBREAK; halt = 1'b1;` |
| `vmatpush.weight.mxu0`   | `VR`     | `1110111` |                         | `0000000`        | `77/00`    | Push Tensor To MXU0 Weight Slot   | `mxu0.w[vd] = m[vs];` |
| `vmatpush.weight.mxu1`   | `VR`     | `1110111` |                         | `0000001`        | `77/01`    | Push Tensor To MXU1 Weight Slot   | `mxu1.w[vd] = m[vs];` |
| `vmatpush.acc.fp8.mxu0`  | `VR`     | `1110111` |                         | `0000010`        | `77/02`    | Push Tensor To MXU0 Accumulator   | `mxu0.acc[vd[0]] = dequantize(m[vs]);` |
| `vmatpush.acc.fp8.mxu1`  | `VR`     | `1110111` |                         | `0000011`        | `77/03`    | Push Tensor To MXU1 Accumulator   | `mxu1.acc[vd[0]] = dequantize(m[vs]);` |
| `vmatpush.acc.bf16.mxu0` | `VR`     | `1110111` |                         | `0000100`        | `77/04`    | Push Tensor To MXU0 Accumulator   | `mxu0.acc[vd[0]] = {m[vs], m[vs+1]};` |
| `vmatpush.acc.bf16.mxu1` | `VR`     | `1110111` |                         | `0000101`        | `77/05`    | Push Tensor To MXU1 Accumulator   | `mxu1.acc[vd[0]] = {m[vs], m[vs+1]};` |
| `vmatpop.fp8.acc.mxu0`   | `VR`     | `1110111` |                         | `0000110`        | `77/06`    | Pop MXU0 FP8 Accumulator View     | `m[vd] = quantize_fp8(mxu0.acc[vs2[0]], e[es1]);` |
| `vmatpop.fp8.acc.mxu1`   | `VR`     | `1110111` |                         | `0000111`        | `77/07`    | Pop MXU1 FP8 Accumulator View     | `m[vd] = quantize_fp8(mxu1.acc[vs2[0]], e[es1]);` |
| `vmatpop.bf16.acc.mxu0`  | `VR`     | `1110111` |                         | `0001000`        | `77/08`    | Pop MXU0 BF16 Accumulator         | `{m[vd], m[vd + 1]} = mxu0.acc[vs2[0]];` |
| `vmatpop.bf16.acc.mxu1`  | `VR`     | `1110111` |                         | `0001001`        | `77/09`    | Pop MXU1 BF16 Accumulator         | `{m[vd], m[vd + 1]} = mxu1.acc[vs2[0]];` |
| `vmatmul.mxu0`           | `VR`     | `1110111` |                         | `0001010`        | `77/10`    | MXU0 Matmul                       | `mxu0.acc[vd[0]] = m[vs1] @ mxu0.w[vs2[0]];` |
| `vmatmul.mxu1`           | `VR`     | `1110111` |                         | `0001011`        | `77/11`    | MXU1 Matmul                       | `mxu1.acc[vd[0]] = m[vs1] @ mxu1.w[vs2[0]];` |
| `vmatmul.acc.mxu0`       | `VR`     | `1110111` |                         | `0001100`        | `77/12`    | MXU0 Matmul Accumulate            | `mxu0.acc[vd[0]] = mxu0.acc[vd[0]] + m[vs1] @ mxu0.w[vs2[0]];` |
| `vmatmul.acc.mxu1`       | `VR`     | `1110111` |                         | `0001101`        | `77/13`    | MXU1 Matmul Accumulate            | `mxu1.acc[vd[0]] = mxu1.acc[vd[0]] + m[vs1] @ mxu1.w[vs2[0]];` |
| `dma.load.ch<N>`         | `R`      | `1111011` | `000 ~ 111`             | `0000000`        | `7B/00`    | DMA Load                          | `issue_dma_load(channel=N, vmem_addr=x[rd], dram_addr={base, x[rs1]}, size=x[rs2]);` |
| `dma.store.ch<N>`        | `R`      | `1111011` | `000 ~ 111`             | `0000001`        | `7B/01`    | DMA Store                         | `issue_dma_store(channel=N, vmem_addr=x[rs1], dram_addr={base, x[rd]}, size=x[rs2]);` |
| `dma.config.ch<N>`       | `R`      | `1111111` | `000 ~ 111`             | `0000000`        | `7F/00`    | DMA Load                          | `dma.base = x[rs1]` |
| `dma.wait.ch<N>`         | `R`      | `1111111` | `000 ~ 111`             | `0000001`        | `7F/01`    | DMA Wait                          | `wait_until_dma_channel_idle(channel=N);` |


## 4. Architectural Design Parameters

| Item | Value |
|---|---:|
| Instruction width | `32` bits |
| Instruction alignment | `4` bytes |
| Control-flow delay slots | `2` |
| Scalar registers | `32` |
| Tensor registers | `64` |
| Scale registers | `32` |
| Tensor register storage | `32 rows x 32 bytes = 1024 bytes` |
| MXU count | `2` |
| MXU weight slots per MXU | `2` |
| MXU accumulator storage | `32 x 32 BF16` |
| DMA channels | `8` |
| `IMEM` | `0x0002_0000` ~ `0x0004_0000` |
| `VMEM` | `0x2000_0000` ~ `0x2004_0000` |
| `PERIPH` (non-cacheable) | `0x4000_0000` ~ `0x8000_0000` |
| `DRAM` (cacheable) | `0x8000_0000` ~ `0x8_8000_0000` |
| `DONT TELL PROF` (non-cacheable) | `0x8_8000_0000` ~ `0x10_0000_0000` |

`PERIPH`, `DRAM`, and `DONT TELL PROF` are the off-chip memory region that is mapped on serial TileLink interface.
