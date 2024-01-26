#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "shared_defs.h"

int read_shape(uint32_t *rows, uint32_t *cols) {
    FILE *fileptr;
    size_t objects_read;
    
    fileptr = fopen(SHAPE_FILENAME, "rb");
    if (fileptr == NULL) {
        fprintf(stderr, "unable to open file.");
        return -1;
    }

    objects_read = fread(rows, sizeof(*rows), 1, fileptr);
    if (objects_read != 1) {
        fclose(fileptr);
        return -1;
    }
    objects_read = fread(cols, sizeof(*cols), 1, fileptr);
    if (objects_read != 1) {
        fclose(fileptr);
        return -1;
    }

    if (fgetc(fileptr), !feof(fileptr)) {
        return -1;
    }
    if (fclose(fileptr) != 0) {
        return -1;
    }
    return 0;
}

int get_color(uint8_t *image_color, const uint32_t rows, const uint32_t cols, const char *color_file) {
    FILE *fileptr;
    size_t objects_read;

    fileptr = fopen(color_file, "rb");
    if (fileptr == NULL) {
        fprintf(stderr, "Unable to open %s.\n", color_file);
        return -1;
    }

    objects_read = fread(image_color, sizeof(*image_color), rows*cols, fileptr);
    if (objects_read != rows*cols) {
        fprintf(stderr, "Could not read all bytes from %s. Bytes read: %d, Expected: %d\n",
                color_file, objects_read, rows*cols);
        return -1;
    }

    if (fgetc(fileptr), !feof(fileptr)) {
        fprintf(stderr, "Read operation for %s did not reach EOF.\n", color_file);
        return -1;
    }
    if (fclose(fileptr) != 0) {
        fprintf(stderr, "Error closing %s.\n", color_file);
        return -1;
    }
    return 0;
}

int write_fft(float *matrix, const uint32_t rows, const uint32_t cols, const char *matrix_file) {
    FILE *fileptr;
    size_t objects_written;

    fileptr = fopen(matrix_file, "wb");
    if (fileptr == NULL) {
        fprintf(stderr, "Unable to open %s.\n", matrix_file);
        return -1;
    }

    objects_written = fwrite(matrix, sizeof(*matrix), rows*cols, fileptr);
    if (objects_written != rows*cols) {
        fprintf(stderr, "Could not read all bytes from %s. Bytes read: %d, Expected: %d\n",
                matrix_file, objects_written, rows*cols);
        return -1;
    }

    if (fclose(fileptr) != 0) {
        fprintf(stderr, "Error closing %s.\n", matrix_file);
        return -1;
    }
    return 0;
}

int is_power_of_2(uint32_t num) {
    int popcount = 0;
    for (int i = 0; i < 32; i++) {
        popcount += num&1;
        num >>= 1;
    }
    return popcount == 1;
}

void swap_float(float *a, float *b) {
    char *a_bytes, *b_bytes;
    if (*a == *b) {
        return;
    }
    a_bytes = (char *)a;
    b_bytes = (char *)b;

    for (int i = 0; i < sizeof(*a); i++) {
        a_bytes[i] ^= b_bytes[i];
        b_bytes[i] ^= a_bytes[i];
        a_bytes[i] ^= b_bytes[i];
    }
}

void swap_int(uint32_t *a, uint32_t *b) {
    if (*a == *b) {
        return;
    }
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

uint32_t log_of_pow_of_2(uint32_t num) {
    uint32_t l;
    for (l = 0; l < 32; l++) {
        if (num&1) {
            return l;
        }
        num >>= 1;
    }
}

void scale(float *row, const uint32_t N, uint32_t index_multiplier, const float scalar) {
    for (int i = 0; i < N; i++) {
        row[i * index_multiplier] *= scalar;
    }
}

int complex_fft(float *real, float *imag, const uint32_t N, const uint32_t index_multiplier) {
    float *a_real, *a_imag;
    uint32_t u, pow_2_i, iterations;
    float sin_u, cos_u;

    a_real = (float *)malloc(N*sizeof(*a_real));
    if (a_real == NULL) {
        return -1;
    }
    a_imag = (float *)malloc(N*sizeof(*a_imag));
    if (a_imag == NULL) {
        return -1;
    }

    iterations = log_of_pow_of_2(N);
    for (int i = 1; i <= iterations; i++) {
        pow_2_i = pow(2, i);
        for (int n = 0; n < N; n++) {
            u = n / (N/pow_2_i);
            sin_u = sin(-2*(u*pi/pow_2_i));
            cos_u = cos(-2*(u*pi/pow_2_i));
            a_real[n] =
                    real[(n + u*N/pow_2_i) % N * index_multiplier] +
                    real[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*cos_u -
                    imag[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*sin_u;
            a_imag[n] =
                    imag[(n + u*N/pow_2_i) % N * index_multiplier] +
                    imag[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*cos_u +
                    real[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*sin_u;
        }
        for(int n = 0; n < N; n++) {
            real[n * index_multiplier] = a_real[n];
            imag[n * index_multiplier] = a_imag[n];
        }
    }

    free(a_real);
    free(a_imag);
}

void detangle_and_pack(float *row1, float *row2, const uint32_t N, const uint32_t index_multiplier) {
    register float Fr, Fi, Gr, Gi;
    // printf("N = %d\n", N);
    for (int i = 1; i < N/2; i++) {
        // printf("Splitting at index %d and %d\n", i * index_multiplier, (N-i) * index_multiplier);
        Fr = (row1[i * index_multiplier] + row1[(N-i) * index_multiplier]) / 2;
        // printf("Fr = %f + %f = %f\n", row1[i * index_multiplier], row1[(N-i) * index_multiplier], Fr);
        Fi = (row2[i * index_multiplier] - row2[(N-i) * index_multiplier]) / 2;
        Gr = (row2[i * index_multiplier] + row2[(N-i) * index_multiplier]) / 2;
        Gi = -(row1[i * index_multiplier] - row1[(N-i) * index_multiplier]) / 2;

        row1[i * index_multiplier] = Fr;
        row1[(N-i) * index_multiplier] = Fi;
        row2[i * index_multiplier] = Gr;
        row2[(N-i) * index_multiplier] = Gi;
    }
}

void real_pair_fft(float *row1, float *row2, const uint32_t len, const uint32_t index_multiplier) {
    complex_fft(row1, row2, len, index_multiplier);
    detangle_and_pack(row1, row2, len, index_multiplier);
}

int matrix_fft(float *matrix, uint32_t rows, uint32_t cols) {
    if (!is_power_of_2(rows) || !is_power_of_2(cols)) {
        fprintf(stderr, "The image dimensions must be a power of 2.\n");
        return -1;
    }

    for (int row = 0; row < rows; row += 2) {
        real_pair_fft(matrix + row*cols, matrix + (row+1)*cols, cols, 1);
    }

    // printf("Matrix after row fft.\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         printf("%f\t", matrix[i*cols + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    // printf("Column\n");
    real_pair_fft(matrix, matrix + cols/2, rows, cols);
    // printf("Matrix after real column fft.\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         printf("%f\t", matrix[i*cols + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    for (int col = 1; col < cols/2; col++) {
        complex_fft(matrix + col, matrix + cols-col, rows, cols);
    }

    // printf("Matrix after column fft.\n");
    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         printf("%f\t", matrix[i*cols + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

    return 0;
}