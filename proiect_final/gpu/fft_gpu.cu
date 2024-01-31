#include <stdio.h>
#include <inttypes.h>
#include <cooperative_groups.h>

#include "../shared_defs.h"

using namespace cooperative_groups;

__device__ uint32_t kernel_log_of_pow_of_2(uint32_t num) {
    uint32_t l;
    for (l = 0; l < 32; l++) {
        if (num&1) {
            return l;
        }
        num >>= 1;
    }
}

__global__ void complex_fft_iteration(float *real, float *imag, const uint32_t N,
                            const uint32_t index_multiplier) {
    int iterations = kernel_log_of_pow_of_2(N);
    float a_real, a_imag;
    uint32_t u, pow_2_i;
    float sin_u, cos_u;
    grid_group g = this_grid();
    int n = g.thread_index().x;

    for (int i = 1; i <= iterations; i++) {
        pow_2_i = 1 << i;
        u = n / (N/pow_2_i);
        sin_u = sin(-2 * (u * pi / pow_2_i));
        cos_u = cos(-2 * (u * pi / pow_2_i));

        a_real =
            real[(n + u*N/pow_2_i) % N * index_multiplier] +
            real[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*cos_u -
            imag[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*sin_u;
        a_imag =
            imag[(n + u*N/pow_2_i) % N * index_multiplier] +
            imag[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*cos_u +
            real[(n + u*N/pow_2_i + N/pow_2_i) % N * index_multiplier]*sin_u;

        g.sync();

        real[n * index_multiplier] = a_real;
        imag[n * index_multiplier] = a_imag;

        g.sync();
    }
}

void complex_fft(float *real, float *imag, const uint32_t N, const uint32_t index_multiplier) {
    dim3 threads_per_block(min(1024, N), 1);
    dim3 num_blocks(max(1, N/1024));
    void *params[4];
    params[0] = (void *)&real;
    params[1] = (void *)&imag;
    params[2] = (void *)&N;
    params[3] = (void *)&index_multiplier;
    cudaLaunchCooperativeKernel(complex_fft_iteration, num_blocks, threads_per_block, params, 0, cudaStreamDefault);
}

__global__ void untangle_and_pack(float *row1, float *row2, const uint32_t N,
                                  const uint32_t index_multiplier) {
    float Fr, Fi, Gr, Gi;
    grid_group g = this_grid();
    int i = g.thread_index().x+1;

    Fr = (row1[i * index_multiplier] + row1[(N-i) * index_multiplier]) / 2;
    Fi = (row2[i * index_multiplier] - row2[(N-i) * index_multiplier]) / 2;
    Gr = (row2[i * index_multiplier] + row2[(N-i) * index_multiplier]) / 2;
    Gi = -(row1[i * index_multiplier] - row1[(N-i) * index_multiplier]) / 2;

    row1[i * index_multiplier] = Fr;
    row1[(N-i) * index_multiplier] = Fi;
    row2[i * index_multiplier] = Gr;
    row2[(N-i) * index_multiplier] = Gi;
}

void real_pair_fft(float *row1, float *row2, const uint32_t len, const uint32_t index_multiplier) {
    complex_fft(row1, row2, len, index_multiplier);
    int N = len/2-1;
    dim3 threads_per_block(min(1024, N), 1);
    dim3 num_blocks(max(1, N/1024));
    untangle_and_pack<<<num_blocks, threads_per_block>>>(row1, row2, len, index_multiplier);
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
    cudaDeviceSynchronize();
    printf("%d by %d image FFT calculated in %f seconds.\n", rows, cols, endTime-startTime);
    return 0;
}

float *copy_matrix_to_gpu(float *host_matrix, int32_t rows, int32_t cols) {
    void *gpu_matrix;

    if (cudaMalloc(&gpu_matrix, sizeof(*host_matrix)*rows*cols) != cudaSuccess) {
        fprintf(stderr, "Unable to allocate memory for matrix on the GPU.");
        return NULL;
    }

    if (cudaMemcpy(gpu_matrix, host_matrix, sizeof(*host_matrix)*rows*cols,
        cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Unable to copy image matrix on the GPU.");
        cudaFree(gpu_matrix);
        return NULL;
    }

    return (float *)gpu_matrix;
}

int main() {
    uint32_t rows, cols;
    uint8_t *image_r, *image_g, *image_b;
    float *image_rf, *image_gf, *image_bf;

    if (read_shape(&rows, &cols) != 0) {
        fprintf(stderr, "Unable to read image shape.");
        return -1;
    }

    printf("Allocating memory.\n");
    image_r = (uint8_t *)malloc(rows*cols*sizeof(*image_r));
    if (image_r == NULL) {
        return -1;
    }
    image_g = (uint8_t *)malloc(rows*cols*sizeof(*image_g));
    if (image_g == NULL) {
        return -1;
    }
    image_b = (uint8_t *)malloc(rows*cols*sizeof(*image_b));
    if (image_b == NULL) {
        return -1;
    }

    if (get_color(image_r, rows, cols, IMG_R_FILENAME) != 0 ||
            get_color(image_g, rows, cols, IMG_G_FILENAME) != 0 ||
            get_color(image_b, rows, cols, IMG_B_FILENAME) != 0) {
        fprintf(stderr, "Unable to read image colors.");
        return -1;
    }

    image_rf = (float *)malloc(rows*cols*sizeof(*image_rf));
    if (image_rf == NULL) {
        return -1;
    }
    image_gf = (float *)malloc(rows*cols*sizeof(*image_gf));
    if (image_gf == NULL) {
        return -1;
    }
    image_bf = (float *)malloc(rows*cols*sizeof(*image_bf));
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

    float *gpu_image_rf, *gpu_image_gf, *gpu_image_bf;

    if ((gpu_image_rf = copy_matrix_to_gpu(image_rf, rows, cols)) == NULL) {
        return -1;
    }
    printf("GPU FFT:\n");
    if (matrix_fft(gpu_image_rf, rows, cols) != 0) {
        fprintf(stderr, "Error calculating fft.\n");
        return -1;
    }
    cudaError_t cuda_error = cudaMemcpy((void *)image_rf, (void *)gpu_image_rf, sizeof(*image_rf)*rows*cols,
                    cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Unable to copy results from GPU to host. cuda error: %d\n", cuda_error);
        return -1;
    }
    printf("\n");
    cudaFree(gpu_image_rf);

    if ((gpu_image_gf = copy_matrix_to_gpu(image_gf, rows, cols)) == NULL) {
        return -1;
    }
    if (matrix_fft(gpu_image_gf, rows, cols) != 0) {
        fprintf(stderr, "Error calculating fft.\n");
        return -1;
    }
    
    cuda_error = cudaMemcpy((void *)image_gf, (void *)gpu_image_gf, sizeof(*image_gf)*rows*cols,
                    cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Unable to copy results from GPU to host. cuda error: %d\n", cuda_error);
        return -1;
    }
    printf("\n");
    cudaFree(gpu_image_gf);

    if ((gpu_image_bf = copy_matrix_to_gpu(image_bf, rows, cols)) == NULL) {
        return -1;
    }
    if (matrix_fft(gpu_image_bf, rows, cols) != 0) {
        fprintf(stderr, "Error calculating fft.\n");
        return -1;
    }
    cuda_error = cudaMemcpy((void *)image_bf, (void *)gpu_image_bf, sizeof(*image_bf)*rows*cols,
                    cudaMemcpyDeviceToHost);
    if (cuda_error != cudaSuccess) {
        fprintf(stderr, "Unable to copy results from GPU to host. cuda error: %d\n", cuda_error);
        return -1;
    }
    printf("\n");
    cudaFree(gpu_image_bf);

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