#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <stdarg.h> // <--- ADDED THIS
#include <riscv_vector.h>

// ==========================================================================
// BME Custom Instruction Definitions (Inlined from bme.h for standalone use)
// ==========================================================================

// Register aliases for inline assembly
#define m0 "x0"
#define m1 "x1"
#define m2 "x2"
#define m3 "x3"
#define v0 "x0"
#define v16 "x16"
#define v18 "x18"

// Initialize accumulator: Copies vector vs2 to accumulator md
#define OPMVINBCAST(md, vs2) \
  asm volatile(".insn r 0x57, 0x6, 0x59, " md ", x0, " vs2);

// Accumulate: md += vs1 * vs2
#define VOPACC(md, vs2, vs1) \
  asm volatile(".insn r 0x57, 0x2, 0x51, " md ", " vs1 ", " vs2);

// Move Vector from Row: Extract row rs1 from accumulator ms2 into vector vd
#define VMV_VR(vd, rs1, ms2) \
  asm volatile(".insn r 0x57, 0x6, 0x5d, " vd ", %0, " ms2 : : "r"(rs1));

// ==========================================================================
// Test Infrastructure
// ==========================================================================

typedef struct {
    const char* name;
    bool passed;
    char msg[256];
} TestResult;

void record_pass(TestResult* res, const char* name) {
    res->name = name;
    res->passed = true;
    snprintf(res->msg, sizeof(res->msg), "OK");
    printf("[PASS] %s\n", name);
}

void record_fail(TestResult* res, const char* name, const char* format, ...) {
    res->name = name;
    res->passed = false;
    va_list args;
    va_start(args, format);
    vsnprintf(res->msg, sizeof(res->msg), format, args);
    va_end(args);
    printf("[FAIL] %s: %s\n", name, res->msg);
}

// ==========================================================================
// Tests
// ==========================================================================

void test_vlen(TestResult* res) {
    size_t vlenb;
    asm volatile("csrr %0, vlenb" : "=r"(vlenb));
    
    if (vlenb == 0) {
        record_fail(res, "VLEN", "VLENB is 0");
        return;
    }
    
    size_t avl = 32;
    size_t vl = __riscv_vsetvl_e32m1(avl);
    
    if (vl == 0) {
        record_fail(res, "VLEN", "vsetvl returned 0");
        return;
    }
    record_pass(res, "VLEN");
}

void test_int_add(TestResult* res) {
    const int n = 16;
    int32_t a[16], b[16], c[16];
    for (int i = 0; i < n; i++) { a[i] = i; b[i] = 10; }

    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vint32m1_t va = __riscv_vle32_v_i32m1(&a[i], vl);
        vint32m1_t vb = __riscv_vle32_v_i32m1(&b[i], vl);
        vint32m1_t vc = __riscv_vadd_vv_i32m1(va, vb, vl);
        __riscv_vse32_v_i32m1(&c[i], vc, vl);
    }

    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            record_fail(res, "Int Add", "Mismatch at %d: %d != %d", i, c[i], a[i]+b[i]);
            return;
        }
    }
    record_pass(res, "Int Add");
}

void test_fp64(TestResult* res) {
#if defined(__riscv_zve64d) || defined(__riscv_v)
    const int n = 8;
    double a[8], b[8], c[8];
    for (int i = 0; i < n; i++) { a[i] = i * 1.5; b[i] = 2.5; }

    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e64m1(n - i);
        vfloat64m1_t va = __riscv_vle64_v_f64m1(&a[i], vl);
        vfloat64m1_t vb = __riscv_vle64_v_f64m1(&b[i], vl);
        vfloat64m1_t vc = __riscv_vfadd_vv_f64m1(va, vb, vl);
        __riscv_vse64_v_f64m1(&c[i], vc, vl);
    }

    for (int i = 0; i < n; i++) {
        if (fabs((a[i] + b[i]) - c[i]) > 1e-6) {
             record_fail(res, "FP64", "Mismatch at %d", i);
             return;
        }
    }
    record_pass(res, "FP64");
#else
    record_pass(res, "FP64 (SKIPPED)");
#endif
}

void test_fp16(TestResult* res) {
#if defined(__riscv_zvfh)
    const int n = 8;
    _Float16 a[8], b[8], c[8];
    for (int i = 0; i < n; i++) { a[i] = (_Float16)i; b[i] = (_Float16)0.5; }

    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e16m1(n - i);
        vfloat16m1_t va = __riscv_vle16_v_f16m1(&a[i], vl);
        vfloat16m1_t vb = __riscv_vle16_v_f16m1(&b[i], vl);
        vfloat16m1_t vc = __riscv_vfadd_vv_f16m1(va, vb, vl);
        __riscv_vse16_v_f16m1(&c[i], vc, vl);
    }

    for (int i = 0; i < n; i++) {
        if (fabs((float)a[i] + (float)b[i] - (float)c[i]) > 1e-3) {
             record_fail(res, "FP16", "Mismatch at %d", i);
             return;
        }
    }
    record_pass(res, "FP16");
#else
    record_pass(res, "FP16 (SKIPPED)");
#endif
}

void test_zvbb(TestResult* res) {
#if defined(__riscv_zvbb)
    const int n = 16;
    uint32_t a[16], b[16], c[16];
    for (int i = 0; i < n; i++) { a[i] = 0xFFFFFFFF; b[i] = 0x0000FFFF; }

    size_t vl;
    for (int i = 0; i < n; i += vl) {
        vl = __riscv_vsetvl_e32m1(n - i);
        vuint32m1_t va = __riscv_vle32_v_u32m1(&a[i], vl);
        vuint32m1_t vb = __riscv_vle32_v_u32m1(&b[i], vl);
        vuint32m1_t vc = __riscv_vandn_vv_u32m1(va, vb, vl);
        __riscv_vse32_v_u32m1(&c[i], vc, vl);
    }

    for (int i = 0; i < n; i++) {
        uint32_t expected = a[i] & ~b[i];
        if (c[i] != expected) {
            record_fail(res, "Zvbb", "Mismatch at %d", i);
            return;
        }
    }
    record_pass(res, "Zvbb");
#else
    record_pass(res, "Zvbb (SKIPPED)");
#endif
}

// --- Custom BME VOPACC Test ---
void test_bme_vopacc(TestResult* res) {
    // We will test a small tile accumulation
    // M0=16 (implied by using m0/m8 logic), N0=vl
    
    size_t M0 = 16;
    size_t N0 = 16; 
    
    int8_t lhs[128]; 
    int8_t rhs[128]; 
    int32_t out[256]; 
    
    for(int i=0; i<128; ++i) { lhs[i] = 1; rhs[i] = 2; }
    memset(out, 0, sizeof(out));

    // 1. Setup VL
    size_t vl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(N0));
    
    // 2. Initialize Accumulator m0 to Zero
    asm volatile("vmv.v.i v0, 0");
    OPMVINBCAST(m0, v0);

    // 3. Load Inputs
    asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(M0));
    asm volatile("vle8.v v16, (%0)" : : "r"(lhs));
    
    asm volatile("vsetvli zero, %0, e8, m2, ta, ma" : : "r"(vl));
    asm volatile("vle8.v v18, (%0)" : : "r"(rhs));
    
    // 4. Execute VOPACC
    VOPACC(m0, v18, v16);

    // 5. Extract Results
    asm volatile("vsetvli zero, %0, e32, m8, ta, ma" : : "r"(vl));
    for (size_t r = 0; r < M0; r++) {
        VMV_VR(v0, r, m0);
        asm volatile("vse32.v v0, (%0)" : : "r"(&out[r * N0]));
    }

    // 6. Validate
    int errors = 0;
    for (size_t r = 0; r < M0; r++) {
        for (size_t c = 0; c < vl; c++) {
            int32_t val = out[r * N0 + c];
            if (val != 2) {
                if (errors < 5) {
                    printf("    Mismatch at [%zu][%zu]: Expected 2, Got %d\n", r, c, val);
                }
                errors++;
            }
        }
    }

    if (errors > 0) {
        record_fail(res, "VOPACC", "%d errors found", errors);
    } else {
        record_pass(res, "VOPACC");
    }
}

// ==========================================================================
// Main Runner
// ==========================================================================

int main() {
    printf("\n=== Saturn Vector Unit Feature Test Suite ===\n\n");
    
    TestResult results[7];
    int count = 0;

    test_vlen(&results[count++]);
    test_int_add(&results[count++]);
    test_fp64(&results[count++]);
    test_fp16(&results[count++]);
    test_zvbb(&results[count++]);
    test_bme_vopacc(&results[count++]);

    printf("\n=== Test Summary ===\n");
    int passed = 0;
    for(int i=0; i<count; ++i) {
        printf("%-15s: %s\n", results[i].name, results[i].passed ? "PASS" : "FAIL");
        if (results[i].passed) passed++;
    }
    
    printf("\nTotal: %d, Passed: %d, Failed: %d\n", count, passed, count - passed);
    
    return (passed == count) ? 0 : 1;
}
