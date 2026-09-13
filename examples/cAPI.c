#include <stdio.h>

void use_model(double * i0, double * o0);
void initialize(const char * model_file, const char * weights_file);

int main(void) {

    /* roseNNa expects column-major (Fortran) ordering. */
    double a[2] = {1, 1};
    double b[3];

    initialize("onnxModel.txt", "onnxWeights.txt");
    use_model(a, b);

    for (int i = 0; i < 3; i++) {
        printf("%f ", b[i]);
    }
    printf("\n");
    return 0;
}
