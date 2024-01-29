#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "shared_defs.h"

long greater_power_of_2(long num) {
    if (is_power_of_2(num)) {
        return num << 1;
    }
    uint32_t last_1_pos = 0;
    for (int i = 0; i < 32; i++) {
        if (num & 1) {
            last_1_pos = i;
        }
        num >>= 1;
    }
    return 1 << (last_1_pos+1);
}

long closest_power_of_2(long num) {
    if (is_power_of_2(num)) {
        return num;
    }
    uint32_t last_1_pos = 0;
    for (int i = 0; i < 32; i++) {
        if (num & 1) {
            last_1_pos = i;
        }
        num >>= 1;
    }
    return 1 << (last_1_pos+1);
}

typedef struct {
    float *real;
    float *imag;
    float *a_real;
    float *a_imag;

    uint32_t pow_2_i;
    uint32_t N;
    uint32_t index_multiplier;
} run_complex_fft_block_t; 

int threaded_complex_fft(float *real, float *imag, const uint32_t N,
        const uint32_t index_multiplier, const uint32_t elements_per_thread) {
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

    return 0;
}

int threaded_untangle_and_pack(float *row1, float *row2, const uint32_t N,
        const uint32_t index_multiplier, const uint32_t elements_per_thread) {
    float Fr, Fi, Gr, Gi;
    // pthread_t *thread_pool =
    //         (pthread_t *)malloc((N / elements_per_thread) * sizeof(pthread_t));
    // if (thread_pool == NULL) {
    //     fprintf(stderr, "Unable to allocate memory for thread pool.\n");
    //     return -1;
    // }
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

    return 0;
}

void threaded_real_pair_fft(float *row1, float *row2, const uint32_t len,
        const uint32_t index_multiplier, const uint32_t elements_per_thread) {
    threaded_complex_fft(row1, row2, len, index_multiplier, elements_per_thread);
    threaded_untangle_and_pack(row1, row2, len, index_multiplier, elements_per_thread);
}

typedef struct {
    float *matrix;
    uint32_t start_index;
    uint32_t allocated_row_pairs;
    uint32_t elements_per_thread;
    uint32_t row_len;
    uint32_t num_rows;
} row_fft_payload_t;

void *run_real_pair_fft(void *vargp) {
    row_fft_payload_t *payload = (row_fft_payload_t *)vargp;
    for (int row_num = payload->start_index;
            row_num < payload->start_index + payload->allocated_row_pairs;
            row_num++) {
        threaded_real_pair_fft(payload->matrix + row_num * payload->row_len,
                      payload->matrix + (row_num + payload->num_rows/2) * payload->row_len,
                      payload->row_len, 1, payload->elements_per_thread);
    }
    free(payload);
    pthread_exit(NULL);
}

typedef struct {
    float *matrix;
    uint32_t start_index;
    uint32_t allocated_col_pairs;
    uint32_t elements_per_thread;
    uint32_t col_len;
    uint32_t num_cols;
} col_fft_payload_t;

void *run_complex_fft(void *vargp) {
    col_fft_payload_t *payload = (col_fft_payload_t *)vargp;
    int col_num = payload->start_index;
    if (payload->start_index == 0) {
        threaded_real_pair_fft(payload->matrix, payload->matrix + payload->num_cols/2,
                      payload->col_len, payload->num_cols, payload->elements_per_thread);
        col_num = 1;
    }
    for (; col_num < payload->start_index + payload->allocated_col_pairs; col_num++) {
        threaded_complex_fft(payload->matrix + col_num,
                    payload->matrix + (payload->num_cols - col_num),
                    payload->col_len, payload->num_cols, payload->elements_per_thread);
    }
    free(payload);
    pthread_exit(NULL);
}

int threaded_matrix_fft(float *matrix, uint32_t rows, uint32_t cols) {
    float startTime = (float)clock()/CLOCKS_PER_SEC, endTime;
    if (!is_power_of_2(rows) || !is_power_of_2(cols) || rows < 16 || cols < 16) {
        fprintf(stderr, "The image dimensions must be a power of 2.\n");
        return -1;
    }

    long number_of_processors = sysconf(_SC_NPROCESSORS_ONLN);
    printf("Active processors: %ld\n", number_of_processors);
    long number_of_threads = greater_power_of_2(number_of_processors);
    
    uint32_t pixels_per_thread = max(4, (rows*cols)/number_of_threads);
    printf("pixels per thread %u\n", pixels_per_thread);
    uint32_t row_pairs_per_thread =
            max(2, closest_power_of_2((uint32_t)round(sqrt(pixels_per_thread))))/2;
    uint32_t col_pairs_per_thread =
            max(1, (pixels_per_thread / row_pairs_per_thread)/4);
    printf("Threads will process %u row pair by %u column pair blocks.\n",
            row_pairs_per_thread, col_pairs_per_thread);

    pthread_t *row_thread_pool = (pthread_t *)malloc(
            sizeof(pthread_t) * (rows/(2*row_pairs_per_thread)));
    if (row_thread_pool == NULL) {
        return -1;
    }
    printf("Threads used for rows: %d\n", rows/(2*row_pairs_per_thread));
    for (int row = 0, i = 0; row < rows/2; row += row_pairs_per_thread, i++) {
        row_fft_payload_t *row_fft_payload = (row_fft_payload_t *)malloc(
            sizeof(row_fft_payload_t));
        if (row_fft_payload == NULL) {
            return -1;
        }
        row_fft_payload->matrix = matrix;
        row_fft_payload->allocated_row_pairs = row_pairs_per_thread;
        row_fft_payload->elements_per_thread = col_pairs_per_thread*2;
        row_fft_payload->start_index = row;
        row_fft_payload->row_len = cols;
        row_fft_payload->num_rows = rows;
        if (pthread_create(row_thread_pool+i, NULL,
                run_real_pair_fft, row_fft_payload) != 0) {
            fprintf(stderr, "Error creating thread.\n");
            return -1;
        }
    }
    for (int i = 0; i < rows/(2*row_pairs_per_thread); i++) {
        if (pthread_join(row_thread_pool[i], NULL) != 0) {
            fprintf(stderr, "Error joining thread.\n");
            return -1;
        }
    }
    free(row_thread_pool);

    pthread_t *col_thread_pool =
            (pthread_t *)malloc(sizeof(pthread_t)* cols/(2*col_pairs_per_thread));
    if (col_thread_pool == NULL) {
        return -1;
    }
    for (int col = 0, i = 0; col < cols/2; col += col_pairs_per_thread, i++) {
        col_fft_payload_t *col_fft_payload = (col_fft_payload_t *)malloc(
            sizeof(col_fft_payload_t));
        if (col_fft_payload == NULL) {
            return -1;
        }
        col_fft_payload->matrix = matrix;
        col_fft_payload->allocated_col_pairs = col_pairs_per_thread;
        col_fft_payload->elements_per_thread = row_pairs_per_thread*2;
        col_fft_payload->start_index = col;
        col_fft_payload->col_len = rows;
        col_fft_payload->num_cols = cols;
        if (pthread_create(col_thread_pool+i, NULL, run_complex_fft, col_fft_payload)) {
            fprintf(stderr, "Error creating thread.\n");
            return -1;
        }
    }
    for (int i = 0; i < cols/(2*col_pairs_per_thread); i++) {
        if (pthread_join(col_thread_pool[i], NULL)) {
            fprintf(stderr, "Error joining thread.\n");
            return -1;
        }
    }
    free(col_thread_pool);

    endTime = (float)clock()/CLOCKS_PER_SEC;
    printf("%d by %d image FFT calculated in %f seconds.\n", rows, cols, endTime-startTime);

    return 0;
}