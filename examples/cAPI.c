#include <stdio.h>

void use_model(double * i0, double * o0);
void initialize(char * model_file, char * weights_file);

int main(void) {

    double a[1][2] = {1,1};
    double b[1][3];
    initialize("onnxModel.txt","onnxWeights.txt");
    use_model(a, b);

    for (int i = 0; i < 3; i++) {
        printf("%f ",b[0][i]);
    }
}
