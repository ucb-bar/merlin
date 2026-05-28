//===- Transform.h - MLIR Dialect for RISC-V Gemmmini extension ---------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Original File sourced and modified from
// https://github.com/buddy-compiler/buddy-mlir
//
//===----------------------------------------------------------------------===//

#ifndef GEMMINI_TRANSLATE_H
#define GEMMINI_TRANSLATE_H

#include <cstdint>

#define NO_ACTIVATION 0
#define RELU 1
#define LAYERNORM 2
#define IGELU 3
#define SOFTMAX 4
#define CONFIG_LD 1
#define CONFIG_ST 2
#define CONFIG_EX 0
#define CONFIG_BERT 3

#define GARBAGE_ADDR ((uint32_t)(-1))
#define OUTPUT_STATIONARY 0
#define WEIGHT_STATIONARY 1

#define MVIN_SCALE_IDENTITY 1.0
#define ACC_SCALE_IDENTITY 1.0
#define BANK_NUM 4
#define MAX_BYTES 64
#define HAS_FIRST_LAYER_OPTIMIZATIONS

typedef uint32_t acc_scale_t_bits;
typedef float acc_scale_t;
typedef uint32_t scale_t_bits;
typedef float scale_t;
typedef int32_t scale_acc_t;

namespace mlir {

class LLVMConversionTarget;
class LLVMTypeConverter;
class RewritePatternSet;
using OwningRewritePatternList = RewritePatternSet;

// `mxFormat` selects the mxGemmini format encoding (per
// `MxParameters.scala:124-130`): -1=Disabled (vanilla Gemmini, all MX bits
// zero in CONFIG_EX), 0=Fp4, 1=Fp6_0, 2=Fp8_0, 3=Fp6_1, 4=Fp8_1.
//
// `clampSingleBlockMvin` forces the tile-matmul lowering to issue one MVIN
// per j-tile (cols == dim). Required for the mxGemmini-MMIO target
// (RadianceGemminiOnlyConfig) whose generated LoadController only allocates
// a 6-bit MvinRs2.num_cols field — the default `blocks * dim` MVINs (with
// blocks up to MAX_BYTES/dim = 4) overflow that field and trip the
// "A single mvin instruction must load more than 0 bytes" assertion.
// Default false preserves Phase 1-4 RoCC/Spike behavior byte-identically.
//
// `useLoopWs` (Phase 8) replaces the per-tile MVIN/PRELOAD/COMPUTE/MVOUT
// expansion with a single LOOP_WS sequence (~11 commands per matmul:
// CONFIG_EX + CONFIG_ST + 3×CONFIG_LD + 5×LOOP_WS_CONFIG_* + LOOP_WS +
// FLUSH). The hardware then loops over the I/J/K tiles internally without
// flooding the MMIO command queue. Default false keeps the Phase 1-7
// per-tile lowering for the RoCC/Spike path. Required for the MMIO path
// (RadianceGemminiOnlyConfig) to clear the GemminiTile.scala:446 backpressure
// assertion that fires when ~56 commands per matmul are pushed faster
// than gemmini's queue can drain.
//
// `dispatchDebug` opt-in (default false): emit volatile stores of binding
// pointers + matmul-operand pointers to fixed DRAM trace regions
// (MERLIN_DEBUG_BINDING_TRACE_ADDR / MERLIN_DEBUG_MATMUL_TRACE_ADDR) so
// the runtime debug probes (built with -DMERLIN_DISPATCH_DEBUG=ON) can
// read them back. Off in production; adds a handful of stores per matmul
// dispatch when on.
void populateGemminiLegalizeForLLVMExportPatterns(LLVMTypeConverter &converter,
	RewritePatternSet &patterns, int64_t dim, int64_t addrLen, int64_t accRows,
	int64_t bankRows, size_t sizeOfElemT, size_t sizeOfAccT,
	int64_t mxFormat = -1, bool clampSingleBlockMvin = false,
	bool useLoopWs = false, bool dispatchDebug = false);
void configureGemminiLegalizeForExportTarget(LLVMConversionTarget &target);

} // namespace mlir

#endif // GEMMINI_TRANSLATE_H
