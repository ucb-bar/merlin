/* Shared binary32 tanh and private expm1 for portable_neural_f32.
 * OpenLibm v0.8.7, commit9fbeafcd4f1b6ef6aa3946c1c8faead50f38a94d.
 * The Sun permission notices in the source below apply; see portable_tanh.LICENSE
 * and portable_math.provenance.json. Arithmetic/coefficient ordering is unchanged.
 * Support headers are replaced by local IEEE-word helpers; signed shifts use
 * unsigned words to preserve the same bits without C signed-shift undefined behavior.
 * Assumes binary32/binary64 and nearest ties-even, no fast-math/contraction.
 */
#include <float.h>
#include <stdint.h>
#define MERLIN_TANH_GET(i,d) do { union {float f;uint32_t word;} u={.f=(d)}; (i)=u.word; } while(0)
#define MERLIN_TANH_SET(d,i) do { union {uint32_t word;float f;} u={.word=(uint32_t)(i)}; (d)=u.f; } while(0)
#define MERLIN_TANH_ASSIGN(type,lval,rval) ((lval)=(rval))
static float merlin_tanh_abs(float x) { uint32_t bits; MERLIN_TANH_GET(bits,x); bits &= 0x7fffffff; MERLIN_TANH_SET(x,bits); return x; }

/* s_expm1f.c -- float version of s_expm1.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

//__FBSDID("$FreeBSD: src/lib/msun/src/s_expm1f.c,v 1.12 2011/10/21 06:26:38 das Exp $");



static const float
merlin_expm_one		= 1.0,
merlin_expm_huge		= 1.0e+30,
merlin_expm_tiny		= 1.0e-30,
merlin_expm_o_threshold	= 8.8721679688e+01,/* 0x42b17180 */
merlin_expm_ln2_hi		= 6.9313812256e-01,/* 0x3f317180 */
merlin_expm_ln2_lo		= 9.0580006145e-06,/* 0x3717f7d1 */
merlin_expm_invln2		= 1.4426950216e+00,/* 0x3fb8aa3b */
/*
 * Domain [-0.34568, 0.34568], range ~[-6.694e-10, 6.696e-10]:
 * |6 / x * (1 + 2 * (1 / (exp(x) - 1) - 1 / x)) - q(x)| < 2**-30.04
 * Scaled coefficients: Qn_here = 2**n * Qn_for_q (see s_expm1.c):
 */
merlin_expm_Q1 = -3.3333212137e-2,		/* -0x888868.0p-28 */
merlin_expm_Q2 =  1.5807170421e-3;		/*  0xcf3010.0p-33 */

static float
merlin_tanh_expm1f(float x)
{
        float y,hi,lo,c,t,e,hxs,hfx,r1,twopk;
        int32_t k,xsb;
        uint32_t hx;

        MERLIN_TANH_GET(hx,x);
        xsb = hx&0x80000000;		/* sign bit of x */
        hx &= 0x7fffffff;		/* high word of |x| */

    /* filter out merlin_expm_huge and non-finite argument */
        if(hx >= 0x4195b844) {			/* if |x|>=27*ln2 */
            if(hx >= 0x42b17218) {		/* if |x|>=88.721... */
                if(hx>0x7f800000)
                    return x+x; 	 /* NaN */
                if(hx==0x7f800000)
                    return (xsb==0)? x:-1.0;/* exp(+-inf)={inf,-1} */
                if(x > merlin_expm_o_threshold) return merlin_expm_huge*merlin_expm_huge; /* overflow */
            }
            if(xsb!=0) { /* x < -27*ln2, return -1.0 with inexact */
                if(x+merlin_expm_tiny<(float)0.0)	/* raise inexact */
                return merlin_expm_tiny-merlin_expm_one;	/* return -1 */
            }
        }

    /* argument reduction */
        if(hx > 0x3eb17218) {		/* if  |x| > 0.5 ln2 */
            if(hx < 0x3F851592) {	/* and |x| < 1.5 ln2 */
                if(xsb==0)
                    {hi = x - merlin_expm_ln2_hi; lo =  merlin_expm_ln2_lo;  k =  1;}
                else
                    {hi = x + merlin_expm_ln2_hi; lo = -merlin_expm_ln2_lo;  k = -1;}
            } else {
                k  = merlin_expm_invln2*x+((xsb==0)?(float)0.5:(float)-0.5);
                t  = k;
                hi = x - t*merlin_expm_ln2_hi;	/* t*merlin_expm_ln2_hi is exact here */
                lo = t*merlin_expm_ln2_lo;
            }
            MERLIN_TANH_ASSIGN(float, x, hi - lo);
            c  = (hi-x)-lo;
        }
        else if(hx < 0x33000000) {  	/* when |x|<2**-25, return x */
            t = merlin_expm_huge+x;	/* return x with inexact flags when x!=0 */
            return x - (t-(merlin_expm_huge+x));
        }
        else k = 0;

    /* x is now in primary range */
        hfx = (float)0.5*x;
        hxs = x*hfx;
        r1 = merlin_expm_one+hxs*(merlin_expm_Q1+hxs*merlin_expm_Q2);
        t  = (float)3.0-r1*hfx;
        e  = hxs*((r1-t)/((float)6.0 - x*t));
        if(k==0) return x - (x*e-hxs);		/* c is 0 */
        else {
            MERLIN_TANH_SET(twopk,0x3f800000+((uint32_t)k<<23));	/* 2^k */
            e  = (x*(e-c)-c);
            e -= hxs;
            if(k== -1) return (float)0.5*(x-e)-(float)0.5;
            if(k==1) {
                if(x < (float)-0.25) return -(float)2.0*(e-(x+(float)0.5));
                else 	      return  merlin_expm_one+(float)2.0*(x-e);
            }
            if (k <= -2 || k>56) {   /* suffice to return exp(x)-1 */
                y = merlin_expm_one-(e-x);
                if (k == 128) y = y*2.0F*0x1p127F;
                else y = y*twopk;
                return y-merlin_expm_one;
            }
            t = merlin_expm_one;
            if(k<23) {
                MERLIN_TANH_SET(t,0x3f800000 - (0x1000000>>k)); /* t=1-2^-k */
                y = t-(e-x);
                y = y*twopk;
           } else {
                MERLIN_TANH_SET(t,((0x7f-k)<<23));	/* 2^-k */
                y = x-(e+t);
                y += merlin_expm_one;
                y = y*twopk;
            }
        }
        return y;
}

/* s_tanhf.c -- float version of s_tanh.c.
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 */

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

//__FBSDID("$FreeBSD: src/lib/msun/src/s_tanhf.c,v 1.9 2008/02/22 02:30:36 das Exp $");



static const float merlin_tanh_one=1.0, merlin_tanh_two=2.0, merlin_tanh_tiny = 1.0e-30, merlin_tanh_huge = 1.0e30;
float
__wrap_tanhf(float x)
{
        float t,z;
        int32_t jx,ix;

        MERLIN_TANH_GET(jx,x);
        ix = jx&0x7fffffff;

    /* x is INF or NaN */
        if(ix>=0x7f800000) {
            if (jx>=0) return merlin_tanh_one/x+merlin_tanh_one;    /* tanh(+-inf)=+-1 */
            else       return merlin_tanh_one/x-merlin_tanh_one;    /* tanh(NaN) = NaN */
        }

    /* |x| < 9 */
        if (ix < 0x41100000) {		/* |x|<9 */
            if (ix<0x39800000) {	/* |x|<2**-12 */
                if(merlin_tanh_huge+x>merlin_tanh_one) return x; /* tanh(merlin_tanh_tiny) = merlin_tanh_tiny with inexact */
            }
            if (ix>=0x3f800000) {	/* |x|>=1  */
                t = merlin_tanh_expm1f(merlin_tanh_two*merlin_tanh_abs(x));
                z = merlin_tanh_one - merlin_tanh_two/(t+merlin_tanh_two);
            } else {
                t = merlin_tanh_expm1f(-merlin_tanh_two*merlin_tanh_abs(x));
                z= -t/(t+merlin_tanh_two);
            }
    /* |x| >= 9, return +-1 */
        } else {
            z = merlin_tanh_one - merlin_tanh_tiny;		/* raise inexact flag */
        }
        return (jx>=0)? z: -z;
}
