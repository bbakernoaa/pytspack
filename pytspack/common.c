#include <math.h>
#include "common.h"

void snhcsh(double x, double* sinhm, double* coshm, double* coshmm) {
    double ax = fabs(x);
    double xs = ax * ax;
    if (ax <= 0.5) {
        double xc = x * xs;
        double p1 = -3.51754964808151394800e5;
        double p2 = -1.15614435765005216044e4;
        double p3 = -1.63725857525983828727e2;
        double p4 = -7.89474443963537015605e-1;
        double q1 = -2.11052978884890840399e6;
        double q2 = 3.61578279834431989373e4;
        double q3 = -2.77711081420602794433e2;
        double q4 = 1.0;
        double p = ((p4 * xs + p3) * xs + p2) * xs + p1;
        double q = ((q4 * xs + q3) * xs + q2) * xs + q1;
        *sinhm = xc * (p / q);
        double xsd4 = 0.25 * xs;
        double xsd2 = xsd4 + xsd4;
        p = ((p4 * xsd4 + p3) * xsd4 + p2) * xsd4 + p1;
        q = ((q4 * xsd4 + q3) * xsd4 + q2) * xsd4 + q1;
        double f = xsd4 * (p / q);
        *coshmm = xsd2 * f * (f + 2.0);
        *coshm = *coshmm + xsd2;
    } else {
        double expx = exp(ax);
        *sinhm = -(((1.0 / expx + ax) + ax) - expx) / 2.0;
        if (x < 0.0) *sinhm = -(*sinhm);
        *coshm = ((1.0 / expx - 2.0) + expx) / 2.0;
        *coshmm = *coshm - xs / 2.0;
    }
}

double store(double x) { return x; }
