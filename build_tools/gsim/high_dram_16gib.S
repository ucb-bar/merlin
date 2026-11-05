// Bare-metal address-span smoke for a 16 GiB DRAM window at 0x80000000.
// These words straddle the 32-bit boundary and touch the final mapped page.
// Assemble with riscv-tests env/p/riscv_test.h and test_macros.h.
#include "riscv_test.h"
#include "test_macros.h"

RVTEST_RV64U
RVTEST_CODE_BEGIN
  li TESTNUM, 1
  li t0, 0x100001000
  li t1, 0x123456789abcdef0
  sd t1, 0(t0)
  fence rw, rw
  ld t2, 0(t0)
  bne t1, t2, fail
  li t0, 0x47ffff000
  li t1, 0xfedcba9876543210
  sd t1, 0(t0)
  fence rw, rw
  ld t2, 0(t0)
  bne t1, t2, fail
  TEST_PASSFAIL
RVTEST_CODE_END

.data
RVTEST_DATA_BEGIN
  TEST_DATA
RVTEST_DATA_END
