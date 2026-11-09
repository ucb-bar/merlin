#ifndef MERLIN_SOURCE_F32_MATH_H
#define MERLIN_SOURCE_F32_MATH_H
#include <math.h>
#include <string.h>
/* Optional admitted compiler capability. Default is the original C library
 * operation. A provider may define this before all numerical headers only
 * with a source single-rounding/errno/nontrapping observation contract. */
#ifndef MERLIN_SOURCE_F32_FMA
#define MERLIN_SOURCE_F32_FMA(a,b,c) fmaf((a),(b),(c))
#endif
#ifndef MERLIN_SOURCE_BITCAST_COPY
#define MERLIN_SOURCE_BITCAST_COPY(d,s,n) memcpy((d),(s),(n))
#endif
#ifndef MERLIN_SOURCE_ISFINITE
#define MERLIN_SOURCE_ISFINITE(x) isfinite((x))
#endif
#ifndef MERLIN_SOURCE_F32_ABS
#define MERLIN_SOURCE_F32_ABS(x) fabsf((x))
#endif
#ifndef MERLIN_SOURCE_F64_ABS
#define MERLIN_SOURCE_F64_ABS(x) fabs((x))
#endif
#ifndef MERLIN_SOURCE_F32_MIN
#define MERLIN_SOURCE_F32_MIN(a,b) fminf((a),(b))
#endif
#ifndef MERLIN_SOURCE_F32_MAX
#define MERLIN_SOURCE_F32_MAX(a,b) fmaxf((a),(b))
#endif
#ifndef MERLIN_SOURCE_F64_MIN
#define MERLIN_SOURCE_F64_MIN(a,b) fmin((a),(b))
#endif
#ifndef MERLIN_SOURCE_F64_MAX
#define MERLIN_SOURCE_F64_MAX(a,b) fmax((a),(b))
#endif
#ifndef MERLIN_SOURCE_F32_FLOOR
#define MERLIN_SOURCE_F32_FLOOR(x) floorf((x))
#endif
#endif
