#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <unistd.h>

#include "../shared_defs.h"

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
    uint32_t start_index;
    uint32_t end_index;
    uint32_t N;
    uint32_t index_multiplier;
    pthread_barrier_t *barrier;
} complex_fft_payload_t;

void *run_complex_fft_block(void *vargp) {
    complex_fft_payload_t *payload = (complex_fft_payload_t *)vargp;
    uint32_t u, iterations, pow_2_i, N;
    float sin_u, cos_u;

    N = payload->N;
    iterations = log_of_pow_of_2(payload->N);
    for (int i = 1; i <= iterations; i++) {
        pow_2_i = pow(2, i);
        for (int n = payload->start_index; n < payload->end_index; n++) {
            u = n / (payload->N/pow_2_i);
            sin_u = sin(-2*(u*pi/pow_2_i));
            cos_u = cos(-2*(u*pi/pow_2_i));
            payload->a_real[n] =
                    payload->real[(n + u*N/pow_2_i) % N * payload->index_multiplier] +
                    payload->real[(n + u*N/pow_2_i + N/pow_2_i) % N * payload->index_multiplier]*cos_u -
                    payload->imag[(n + u*N/pow_2_i + N/pow_2_i) % N * payload->index_multiplier]*sin_u;
            payload->a_imag[n] =
                    payload->imag[(n + u*N/pow_2_i) % N * payload->index_multiplier] +
                    payload->imag[(n + u*N/pow_2_i + N/pow_2_i) % N * payload->index_multiplier]*cos_u +
                    payload->real[(n + u*N/pow_2_i + N/pow_2_i) % N * payload->index_multiplier]*sin_u;
        }

        pthread_barrier_wait(payload->barrier);

        for(int n = payload->start_index; n < payload->end_index; n++) {
            payload->real[n * payload->index_multiplier] = payload->a_real[n];
            payload->imag[n * payload->index_multiplier] = payload->a_imag[n];
        }

        pthread_barrier_wait(payload->barrier);
    }

    free(payload);
    pthread_exit(NULL);
}

int threaded_complex_fft(float *real, float *imag, const uint32_t N,
        const uint32_t index_multiplier, const uint32_t elements_per_thread) {
    int i, t, num_threads = N / elements_per_thread;
    pthread_t *thread_pool =
            (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if (thread_pool == NULL) {
        fprintf(stderr, "Unable to allocate memory for thread pool.\n");
        return -1;
    }

    float *a_real, *a_imag;
    a_real = (float *)malloc(N*sizeof(*a_real));
    if (a_real == NULL) {
        return -1;
    }
    a_imag = (float *)malloc(N*sizeof(*a_imag));
    if (a_imag == NULL) {
        return -1;
    }

    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);

    for (i = 0, t = 0; i < N; i += elements_per_thread, t++) {
        complex_fft_payload_t *payload =
                (complex_fft_payload_t *)malloc(sizeof(complex_fft_payload_t));
        if (payload == NULL) {
            fprintf(stderr, "Failed to initialize thread.\n");
            return -1;
        }
        payload->real = real;
        payload->imag = imag;
        payload->a_real = a_real;
        payload->a_imag = a_imag;
        payload->start_index = i;
        payload->end_index = i+elements_per_thread;
        payload->N = N;
        payload->index_multiplier = index_multiplier;
        payload->barrier = &barrier;
        if (pthread_create(thread_pool+t, NULL, run_complex_fft_block, payload) != 0) {
            fprintf(stderr, "Error creating thread.\n");
            return -1;
        }
    }

    for (t = 0; t < num_threads; t++) {
        if (pthread_join(thread_pool[t], NULL)) {
            fprintf(stderr, "Error joining thread.\n");
            return -1;
        }
    }

    pthread_barrier_destroy(&barrier);
    free(thread_pool);
    free(a_real);
    free(a_imag);

    return 0;
}

typedef struct {
    float *row1;
    float *row2;
    float *Fi;
    float *Fr;
    float *Gi;
    float *Gr;
    uint32_t start_index;
    uint32_t end_index;
    uint32_t N;
    uint32_t index_multiplier;
    pthread_barrier_t *barrier;
} untangle_and_pack_payload_t;

void *untangle_and_pack_block(void *vargp) {
    untangle_and_pack_payload_t *payload = (untangle_and_pack_payload_t *)vargp;
    uint32_t index_multiplier = payload->index_multiplier;
    uint32_t N = payload->N;
    float *row1 = payload->row1, *row2 = payload->row2;

    if (payload->start_index == 0) {
        payload->start_index = 1;
    }

    for (int i = payload->start_index; i < payload->end_index; i++) {
        payload->Fr[i] = (row1[i * index_multiplier] + row1[(N-i) * index_multiplier]) / 2;
        payload->Fi[i] = (row2[i * index_multiplier] - row2[(N-i) * index_multiplier]) / 2;
        payload->Gr[i] = (row2[i * index_multiplier] + row2[(N-i) * index_multiplier]) / 2;
        payload->Gi[i] = -(row1[i * index_multiplier] - row1[(N-i) * index_multiplier]) / 2;
    }

    for (int i = payload->start_index; i < payload->end_index; i++) {
        row1[i * index_multiplier] = payload->Fr[i];
        row1[(N-i) * index_multiplier] = payload->Fi[i];
        row2[i * index_multiplier] = payload->Gr[i];
        row2[(N-i) * index_multiplier] = payload->Gi[i];
    }

    free(payload);
    pthread_exit(NULL);
}

int threaded_untangle_and_pack(float *row1, float *row2, const uint32_t N,
        const uint32_t index_multiplier, const uint32_t elements_per_thread) {
    float *Fr, *Fi, *Gr, *Gi;
    Fr = (float *)malloc(sizeof(*Fr) * (N/2));
    if (Fr == NULL) {
        return -1;
    }
    Fi = (float *)malloc(sizeof(*Fi) * (N/2));
    if (Fr == NULL) {
        return -1;
    }
    Gr = (float *)malloc(sizeof(*Gr) * (N/2));
    if (Fr == NULL) {
        return -1;
    }
    Gi = (float *)malloc(sizeof(*Gi) * (N/2));
    if (Fr == NULL) {
        return -1;
    }
    int i, t, num_threads = N / elements_per_thread;
    pthread_t *thread_pool =
            (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    if (thread_pool == NULL) {
        fprintf(stderr, "Unable to allocate memory for thread pool.\n");
        return -1;
    }
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, num_threads);

    for (i = 0, t = 0; i < N/2; i += elements_per_thread/2, t++) {
        untangle_and_pack_payload_t *payload =
                (untangle_and_pack_payload_t *)malloc(sizeof(untangle_and_pack_payload_t));
        if (payload == NULL) {
            fprintf(stderr, "Failed to initialize thread.\n");
            return -1;
        }
        payload->row1 = row1;
        payload->row2 = row2;
        payload->Fi = Fi;
        payload->Fr = Fr;
        payload->Gi = Gi;
        payload->Gr = Gr;
        payload->start_index = i;
        payload->end_index = i + elements_per_thread/2;
        payload->N = N;
        payload->index_multiplier = index_multiplier;
        payload->barrier = &barrier;
        if (pthread_create(thread_pool+t, NULL, untangle_and_pack_block, payload) != 0) {
            fprintf(stderr, "Error creating thread.\n");
            return -1;
        }
    }

    for (t = 0; t < num_threads; t++) {
        if (pthread_join(thread_pool[t], NULL)) {
            fprintf(stderr, "Error joining thread.\n");
            return -1;
        }
    }

    pthread_barrier_destroy(&barrier);
    free(thread_pool);
    free(Fr);
    free(Fi);
    free(Gr);
    free(Gi);

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
    number_of_threads = 4;
    
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