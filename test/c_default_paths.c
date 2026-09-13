#include <stddef.h>

void initialize(const char * model_file, const char * weights_file);

/* Both arguments absent: initialize must fall back to the default paths.
   Run in an empty directory, it must fail to open onnxModel.txt by name. */
int main(void) {
    initialize(NULL, NULL);
    return 0;
}
