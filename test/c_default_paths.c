#include <stddef.h>

void initialize(const char * model_file, const char * weights_file);

/* No arguments: default paths. Run in an empty directory, so it must fail on onnxModel.txt. */
int main(void) {
    initialize(NULL, NULL);
    return 0;
}
