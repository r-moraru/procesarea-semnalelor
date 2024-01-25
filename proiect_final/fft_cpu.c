#include <inttypes.h>
#include <malloc.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define SHAPE_FILENAME "resources/poza_rgb.shape"
#define IMG_R_FILENAME "resources/poza_rgb.r"
#define IMG_G_FILENAME "resources/poza_rgb.g"
#define IMG_B_FILENAME "resources/poza_rgb.b"

#define FFT_R_FILENAME "output/fft.r"
#define FFT_G_FILENAME "output/fft.g"
#define FFT_B_FILENAME "output/fft.b"

#define pi 3.14159265

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
        fprintf(stderr, "Could not read all bytes from %s. Bytes read: %d, Expected: %d\n", color_file, objects_read, rows*cols);
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

int write_matrix(uint8_t *matrix, const uint32_t rows, const uint32_t cols, const char *matrix_file) {
    FILE *fileptr;
    size_t objects_written;

    fileptr = fopen(matrix_file, "wb");
    if (fileptr == NULL) {
        return -1;
    }

    objects_written = fwrite(matrix, sizeof(*matrix), rows*cols, fileptr);
    if (objects_written != rows*cols) {
        return -1;
    }

    if (fclose(fileptr) != 0) {
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

void swap_float(float_t *a, float_t *b) {
    char *a_bytes, *b_bytes;
    if (*a == *b) {
        return;
    }
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

int complex_fft(float_t *real, float_t *imag, const uint32_t N) {
    float_t *a_real, *a_imag;
    uint32_t u, pow_2_i;
    float_t iterations, sin_u, cos_u;

    a_real = malloc(N*sizeof(*a_real));
    if (a_real == NULL) {
        return -1;
    }
    a_imag = malloc(N*sizeof(*a_imag));
    if (a_imag == NULL) {
        return -1;
    }

    iterations = log_of_pow_of_2(N);
    for (int i = 1; i <= iterations; i++) {
        pow_2_i = pow(2, i);
        for (int n = 0; n < N; n++) {
            u = n / (N/pow_2_i);
            sin_u = sin(-2*u*pi*pow_2_i);
            cos_u = cos(-2*u*pi*pow_2_i);
            a_real[n] = real[n+u*N/pow_2_i] + real[n+u*N/pow_2_i+N/pow_2_i]*cos_u +
                imag[n+u*N/pow_2_i+N/pow_2_i]*sin_u;
            a_imag[n] = imag[n+u*N/pow_2_i] + imag[n+u*N/pow_2_i+N/pow_2_i]*cos_u +
                real[n+u*N/pow_2_i+N/pow_2_i]*sin_u;
        }

        for(int n = 0; n < N; n++) {
            real[n] = a_real[n];
            imag[n] = a_imag[n];
        }
    }
}

void detangle_and_pack(float_t *row1, float_t *row2, const uint32_t N) {
    register uint8_t Fr, Fi, Gr, Gi;
    for (int i = 0; i < N; i++) {
        Fr = (row1[i] + (uint32_t)row1[N-i]) / 2;
        Fi = (row2[i] - (uint32_t)row2[N-i]) / 2;
        Gr = (row2[i] + (uint32_t)row2[N-i]) / 2;
        Gi = -(row1[i] - (uint32_t)row1[N-i]) / 2;

        row1[i] = Fr;
        row1[N-i] = Fi;
        row2[i] = Gr;
        row2[N-i] = Gi;
    }
}

void real_pair_fft(float_t *row1, float_t *row2, const uint32_t len) {
    complex_fft(row1, row2, len);
    detangle_and_pack(row1, row2, len);
}

void transpose(float_t *matrix, uint32_t *rows, uint32_t *cols) {
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            swap_float(matrix + i*(*cols) + j, matrix + j*(*rows) + i);
        }
    }
    swap_int(rows, cols);
}

int matrix_fft(float_t *matrix, uint32_t rows, uint32_t cols) {
    if (!is_power_of_2(rows) || !is_power_of_2(cols)) {
        fprintf(stderr, "The image dimensions must be a power of 2.\n");
        return -1;
    }

    for (int row = 0; row < rows; row += 2) {
        real_pair_fft(matrix + row*cols, matrix + (row+1)*cols, cols);
    }

    transpose(matrix, &rows, &cols);

    real_pair_fft(matrix, matrix + (cols/2)*rows, rows);
    for (int col = 1; col < cols/2; col++) {
        complex_fft(matrix + col*rows, matrix + (cols-col)*rows, rows);
    }

    transpose(matrix, &rows, &cols);
}

int main() {
    uint32_t rows, cols;
    uint8_t *image_r, *image_g, *image_b;
    float_t *image_rf, *image_gf, *image_bf;

    if (read_shape(&rows, &cols) != 0) {
        fprintf(stderr, "Unable to read image shape.");
        return -1;
    }
    
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

    if (matrix_fft(image_rf, rows, cols) != 0 ||
            matrix_fft(image_gf, rows, cols) != 0 ||
            matrix_fft(image_bf, rows, cols) != 0) {
        return -1;
    }

    free(image_r);
    free(image_g);
    free(image_b);

    return 0;
}