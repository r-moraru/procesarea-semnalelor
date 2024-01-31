#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "../shared_defs.h"

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

void untangle_and_pack(float *row1, float *row2, const uint32_t N, const uint32_t index_multiplier) {
    register float Fr, Fi, Gr, Gi;
    for (int i = 1; i < N/2; i++) {
        Fr = (row1[i * index_multiplier] + row1[(N-i) * index_multiplier]) / 2;
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
    untangle_and_pack(row1, row2, len, index_multiplier);
}

int matrix_fft(float *matrix, uint32_t rows, uint32_t cols) {
    float startTime = (float)clock()/CLOCKS_PER_SEC, endTime;
    if (!is_power_of_2(rows) || !is_power_of_2(cols)) {
        fprintf(stderr, "The image dimensions must be a power of 2.\n");
        return -1;
    }

    for (int row = 0; row < rows/2; row++) {
        real_pair_fft(matrix + row*cols, matrix + (row+rows/2)*cols, cols, 1);
    }

    real_pair_fft(matrix, matrix + cols/2, rows, cols);

    for (int col = 1; col < cols/2; col++) {
        complex_fft(matrix + col, matrix + cols-col, rows, cols);
    }

    endTime = (float)clock()/CLOCKS_PER_SEC;
    printf("%d by %d image FFT calculated in %f seconds.\n", rows, cols, endTime-startTime);
    return 0;
}