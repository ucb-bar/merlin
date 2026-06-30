/* Trimmed fixture modeled on OpenBLAS kernel/riscv64/amax.c (scalar fallback — must be
   SKIPPED by the openblas ingest adapter). */
#include "common.h"
#include <math.h>

FLOAT CNAME(BLASLONG n, FLOAT *x, BLASLONG inc_x)
{
    BLASLONG i = 0;
    FLOAT maxf = 0.0;
    while (i < n) {
        if (fabs(x[i]) > maxf)
            maxf = fabs(x[i]);
        i += inc_x;
    }
    return maxf;
}
