/* Correctly-rounded reciprocal square root of binary32 value.

Copyright (c) 2022-2026 Alexei Sibidanov and Paul Zimmermann

This file is part of the CORE-MATH project
(https://core-math.gitlabpages.inria.fr/).

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include <stdint.h>
#ifdef CORE_MATH_SUPPORT_ERRNO
#include <errno.h>
#endif
#include <fenv.h> // for feraiseexcept, FE_INVALID

// Warning: clang also defines __GNUC__
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#endif

#pragma STDC FENV_ACCESS ON

typedef union {float f; uint32_t u;} b32u32_u;

float cr_rsqrtf(float x){
  double xd = x;
  b32u32_u ix = {.f = x};
  if(__builtin_expect(ix.u >= 0xff<<23 || ix.u==0, 0)){
    if(!(ix.u << 1))
    {
#ifdef CORE_MATH_SUPPORT_ERRNO
      errno = ERANGE;
#endif
      return 1.0f/x; // +/-0
    }
    if(ix.u >> 31){
      ix.u &= ~0u>>1;
      if(ix.u > 0xff<<23) return x + x; // nan
#ifdef CORE_MATH_SUPPORT_ERRNO
      errno = EDOM;
#endif
      feraiseexcept (FE_INVALID);
      return __builtin_nanf("<0");
    }
    if(!(ix.u<<9)) return 0.0f;
    return x + x; // nan
  }
  double c = 1.0, sq = __builtin_sqrt(xd);
  unsigned m = ix.u<<8;
  if(__builtin_expect(ix.u == 0x2f7e2au || m == 0xbdf8a800u || m == 0x55b7bd00u, 0)){
    /* The expression 1/xd*sqrt(xd) fails to return the correct rounding for
       x=0x1.7bf15p-128*2^(2k) and x=0x1.ab6f7ap-125*2^(2k) with k>=0, for rndu,
       where the returned value is 1 ulp too large. We multiply by a correcting
       factor just below 1, which guarantees correct rounding in these cases. */
    c = 0x1.ffffffffffffep-1;
  }
  return sq*(c/xd);
}

#ifdef __INTEL_CLANG_COMPILER
// rsqrt is called invsqrt with icx
extern float invsqrtf (float);
float rsqrtf(float x){
  return invsqrtf (x);
}
#endif
