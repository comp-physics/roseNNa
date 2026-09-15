/* Self-contained: same shape as roseNNa's generated infer -- 5 dense layers
   2->20->30->30->40->1 over two 40-double locals, called from a target
   teams region over device-resident heap data. No roseNNa headers. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#pragma omp declare target
extern double w0[40], b0[20], w1[600], b1[30], w2[900], b2[30], w3[1200], b3[40], w4[40], b4[1];
#pragma omp end declare target
double w0[40], b0[20], w1[600], b1[30], w2[900], b2[30], w3[1200], b3[40], w4[40], b4[1];

#pragma omp declare target
static inline void infer(const double *restrict x, double *restrict y) {
    double t0[40], t1[40];
    for (int i = 0; i < 20; ++i) { double a = b0[i]; for (int j = 0; j < 2;  ++j) a += x[j]  * w0[i*2+j];  t0[i] = a; }
    for (int i = 0; i < 20; ++i) t1[i] = t0[i] < 0.0 ? 0.0 : t0[i];
    for (int i = 0; i < 30; ++i) { double a = b1[i]; for (int j = 0; j < 20; ++j) a += t1[j] * w1[i*20+j]; t0[i] = a; }
    for (int i = 0; i < 30; ++i) t1[i] = 1.0 / (1.0 + exp(-t0[i]));
    for (int i = 0; i < 30; ++i) { double a = b2[i]; for (int j = 0; j < 30; ++j) a += t1[j] * w2[i*30+j]; t0[i] = a; }
    for (int i = 0; i < 30; ++i) t1[i] = t0[i] < 0.0 ? 0.0 : t0[i];
    for (int i = 0; i < 40; ++i) { double a = b3[i]; for (int j = 0; j < 30; ++j) a += t1[j] * w3[i*30+j]; t0[i] = a; }
    for (int i = 0; i < 40; ++i) t1[i] = tanh(t0[i]);
    double a = b4[0]; for (int j = 0; j < 40; ++j) a += t1[j] * w4[j];
    y[0] = 1.0 / (1.0 + exp(-a));
}
#pragma omp end declare target

int main(void) {
    long n = 1000000;
    double *x = malloc(sizeof(double)*n*2), *y = malloc(sizeof(double)*n);
    for (long i = 0; i < n*2; ++i) x[i] = 0.5;
    for (int i = 0; i < 40; ++i) w0[i] = 0.01;
    for (int i = 0; i < 600; ++i) w1[i] = 0.01;
    for (int i = 0; i < 900; ++i) w2[i] = 0.01;
    for (int i = 0; i < 1200; ++i) w3[i] = 0.01;
    for (int i = 0; i < 40; ++i) w4[i] = 0.01;
#pragma omp target update to(w0, b0, w1, b1, w2, b2, w3, b3, w4, b4)
#pragma omp target enter data map(to: x[0:n*2]) map(alloc: y[0:n])
/* Swap this line for the distribute form to reproduce the trap:
   #pragma omp target teams distribute parallel for */
#pragma omp target teams loop
    for (long p = 0; p < n; ++p) infer(x + p*2, y + p);
#pragma omp target exit data map(from: y[0:n]) map(delete: x[0:n])
    printf("OK y[0]=%.6f\n", y[0]);
    return 0;
}
