#ifndef SRF_COMMON_H
#define SRF_COMMON_H

#include <math.h>
#include <stdlib.h>
#include <stdbool.h>

#define MAX(a,b) (((a)>(b))?(a):(b))
#define MIN(a,b) (((a)<(b))?(a):(b))

/* Legacy precision truncation (optional, used to match Fortran behavior) */
static inline float srf_store(float x) { volatile float y = x; return y; }

/* Givens Rotation: Construct c, s such that the second component is zeroed */
static void srf_givens(float a, float b, float *c, float *s) {
    float aa = fabsf(a);
    float bb = fabsf(b);
    if (aa > bb) {
        float u = a + a;
        float v = b / u;
        float r = sqrtf(0.25f + v*v) * u;
        *c = a / r;
        *s = v * (*c + *c);
    } else if (b != 0.0f) {
        float u = b + b;
        float v = a / u;
        float r = sqrtf(0.25f + v*v) * u;
        *s = b / r;
        *c = v * (*s + *s);
    } else {
        *c = 1.0f;
        *s = 0.0f;
    }
}

/* Apply Givens rotation */
static void srf_rotate(int n, float c, float s, float *x, float *y) {
    for (int i = 0; i < n; i++) {
        float xi = x[i];
        float yi = y[i];
        x[i] = c * xi + s * yi;
        y[i] = -s * xi + c * yi;
    }
}

/* Hyperbolic approximations for tension splines */
static void srf_snhcsh(float x, float *sinhm, float *coshm, float *coshmm) {
    float ax = fabsf(x);
    float xs = ax * ax;
    if (ax <= 0.5f) {
        *sinhm = x * (1.0f + xs/6.0f + xs*xs/120.0f); // approx
        float x2 = 0.5f * xs;
        *coshmm = x2 * (1.0f + xs/12.0f);
        *coshm = *coshmm + x2;
    } else {
        float e = expf(ax);
        float inv_e = 1.0f / e;
        *sinhm = 0.5f * (e - inv_e) - ax;
        if (x < 0) *sinhm = -(*sinhm);
        *coshm = 0.5f * (e + inv_e) - 1.0f;
        *coshmm = *coshm - 0.5f * xs;
    }
}

#endif
