/* Stable binary32/binary64 ABI. RNE, gradual underflow, enabled FPU required.
 * NaN results are canonical 0x7fc00000. errno/FP flags are unobserved.
 * Binary64 NaNs are canonical 0x7ff8000000000000.
 * Generated C must call merlin_host_check_fp_env() once before host math,
 * fail unless it returns 1, and retain that environment on each calling thread.
 * A declaration does not imply supplier availability: check policy first.
 * No libm fallback. Missing implementations fail policy admission/linking.
 */
#ifndef MERLIN_HOST_MATH_H
#define MERLIN_HOST_MATH_H
#ifdef __cplusplus
extern "C" {
#endif
int merlin_host_check_fp_env(void);
double merlin_host_pow(double x, double y);
float merlin_host_powf(float x, float y);
double merlin_host_sin(double x);
double merlin_host_cos(double x);
double merlin_host_exp(double x);
double merlin_host_log(double x);
float merlin_host_expf(float x);
float merlin_host_logf(float x);
float merlin_host_sinf(float x);
float merlin_host_cosf(float x);
float merlin_host_rsqrtf(float x);
float merlin_host_sqrtf(float x);
float merlin_host_tanhf(float x);
float merlin_host_erff(float x);
#ifdef __cplusplus
}
#endif
#endif
