/* Calling a generated roseNNa model from C.
 *
 * Built by run_basic.sh, which generates gemm_small first. gemm_small is well
 * under the embedding threshold, so its weights are compiled into the
 * generated source and there is no gemm_small_init to call; a larger model
 * would load its weights once with
 *     if (gemm_small_init("gemm_small.rwt") != 0) return 1;
 * before the first infer, and never again. */
#include <stdio.h>
#include "gemm_small.h"

int main(void) {
    /* One point: n_in = 2 values in, n_out = 3 values out. Plain row-major. */
    const float x[2] = {1.0f, 1.0f};
    float y[3];

    gemm_small_infer(x, y);

    for (int i = 0; i < 3; ++i) printf("%f ", (double)y[i]);
    printf("\n");
    return 0;
}
