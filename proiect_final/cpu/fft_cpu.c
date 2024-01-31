// #include <inttypes.h>
#include <stdio.h>
#include <malloc.h>

#include "fft_cpu.h"
// #include "threaded_fft_cpu.h"

int main() {
    uint32_t rows, cols;
    uint8_t *image_r, *image_g, *image_b;
    float *image_rf, *image_gf, *image_bf;

    if (read_shape(&rows, &cols) != 0) {
        fprintf(stderr, "Unable to read image shape.");
        return -1;
    }

    printf("Allocating memory.\n");
    image_r = malloc(rows*cols*sizeof(*image_r));
    if (image_r == NULL) {
        return -1;
    }
    image_g = malloc(rows*cols*sizeof(*image_g));
    if (image_g == NULL) {
        return -1;
    }
    image_b = malloc(rows*cols*sizeof(*image_b));
    if (image_b == NULL) {
        return -1;
    }

    if (get_color(image_r, rows, cols, IMG_R_FILENAME) != 0 ||
            get_color(image_g, rows, cols, IMG_G_FILENAME) != 0 ||
            get_color(image_b, rows, cols, IMG_B_FILENAME) != 0) {
        fprintf(stderr, "Unable to read image colors.");
        return -1;
    }

    image_rf = malloc(rows*cols*sizeof(*image_rf));
    if (image_rf == NULL) {
        return -1;
    }
    image_gf = malloc(rows*cols*sizeof(*image_gf));
    if (image_gf == NULL) {
        return -1;
    }
    image_bf = malloc(rows*cols*sizeof(*image_bf));
    if (image_bf == NULL) {
        return -1;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            image_rf[i*cols+j] = image_r[i*cols+j];
            image_gf[i*cols+j] = image_g[i*cols+j];
            image_bf[i*cols+j] = image_b[i*cols+j];
        }
    }

    printf("CPU sequential FFT:\n");
    if (matrix_fft(image_rf, rows, cols) != 0 ||
            matrix_fft(image_gf, rows, cols) != 0 ||
            matrix_fft(image_bf, rows, cols) != 0) {
        fprintf(stderr, "Error calculating fft.\n");
        return -1;
    }
    printf("\n");

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         image_rf[i*cols+j] = image_r[i*cols+j];
    //         image_gf[i*cols+j] = image_g[i*cols+j];
    //         image_bf[i*cols+j] = image_b[i*cols+j];
    //     }
    // }

    // printf("CPU Threaded FFT:\n");
    // printf("Address of matrix: %p\n", image_rf);
    // if (threaded_matrix_fft(image_rf, rows, cols) != 0 ||
    //         threaded_matrix_fft(image_gf, rows, cols) != 0 ||
    //         threaded_matrix_fft(image_bf, rows, cols) != 0) {
    //     fprintf(stderr, "Error calculating fft.\n");
    //     return -1;
    // }

    printf("writing.\n");
    write_fft(image_rf, rows, cols, FFT_R_FILENAME);
    write_fft(image_gf, rows, cols, FFT_G_FILENAME);
    write_fft(image_bf, rows, cols, FFT_B_FILENAME);

    free(image_r);
    free(image_g);
    free(image_b);

    free(image_rf);
    free(image_gf);
    free(image_bf);

    return 0;
}