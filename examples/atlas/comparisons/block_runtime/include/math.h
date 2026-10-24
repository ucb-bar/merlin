#ifndef ATLAS_ISOLATED_MATH_H
#define ATLAS_ISOLATED_MATH_H
#define isfinite(x) __builtin_isfinite(x)
float expf(float);
float sqrtf(float);
float fmaf(float,float,float);
float nearbyintf(float);
float scalbnf(float,int);
double pow(double,double);
#endif
