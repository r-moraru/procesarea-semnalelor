#include <stdio.h>
#include <stdint.h>
#include <malloc.h>
#include <inttypes.h>

#define SHAPE_FILENAME "resources/poza_rgb.shape"
#define IMG_R_FILENAME "resources/poza_rgb.r"
#define IMG_G_FILENAME "resources/poza_rgb.g"
#define IMG_B_FILENAME "resources/poza_rgb.b"

#define FFT_R_FILENAME "output/fft.r"
#define FFT_G_FILENAME "output/fft.g"
#define FFT_B_FILENAME "output/fft.b"

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

int get_color(int16_t *image_color, const uint32_t rows, const uint32_t cols, const char *color_file) {
    FILE *fileptr;
    size_t objects_read;

    fileptr = fopen(color_file, "rb");
    if (fileptr == NULL) {
        return -1;
    }

    objects_read = fread(image_color, sizeof(*image_color), rows*cols, fileptr);
    if (objects_read != rows*cols) {
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

int write_matrix(int16_t *matrix, const uint32_t rows, const uint32_t cols, const char *matrix_file) {
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

void swap(int16_t *a, int16_t *b) {
    if (*a != *b) {
        *a ^= *b;
        *b ^= *a;
        *a ^= *b;
    }
}

void complex_fft(int16_t *real, int16_t *imag, const uint32_t N) {
    register int16_t a;
    // a <- w 2i ^ u
    // w 2i ^ u = e ^ (-j2pi/2^i)
    // = cos(2pi/2^i) + i*sin(2pi/2^i)
}

void detangle_and_pack(int16_t *row1, int16_t *row2, const uint32_t N) {
    register int16_t Fr, Fi, Gr, Gi;
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

void real_pair_fft(int16_t *row1, int16_t *row2, const uint32_t len) {
    complex_fft(row1, row2, len);
    detangle_and_pack(row1, row2, len);
}

void transpose(int16_t *matrix, uint32_t *rows, uint32_t *cols) {
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            swap(matrix + i*(*cols) + j, matrix + j*(*rows) + i);
        }
    }
    swap(rows, cols);
}

int matrix_fft(int16_t *matrix, uint32_t rows, uint32_t cols) {
    if (!is_power_of_2(rows) || !is_power_of_2(cols)) {
        fprintf(stderr, "The image dimensions must be a power of 2");
        return -1;
    }

    for (int row = 0; row < rows; row += 2) {
        real_pair_fft(matrix + row*cols, matrix + (row+1)*cols, cols);
    }

    transpose(matrix, rows, cols);

    real_pair_fft(matrix, matrix + (cols/2)*rows, rows);
    for (int col = 1; col < cols/2; col++) {
        complex_fft(matrix + col*rows, matrix + (cols-col)*rows, rows);
    }

    transpose(matrix, rows, cols);
}

int main() {
    uint32_t rows, cols;
    int16_t *image_r, *image_g, *image_b;

    if (read_shape(&rows, &cols) != 0) {
        fprintf(stderr, "Unable to read image shape.");
        return -1;
    }
    
    image_r = malloc(rows*cols*sizeof(*image_r));
    image_g = malloc(rows*cols*sizeof(*image_g));
    image_b = malloc(rows*cols*sizeof(*image_b));
    if (image_r == NULL || image_g == NULL || image_b == NULL) {
        fprintf(stderr, "Unable to allocate memory for image.");
        return -1;
    }

    if (get_color(image_r, rows, cols, IMG_R_FILENAME) != 0 ||
            get_color(image_g, rows, cols, IMG_G_FILENAME) != 0 ||
            get_color(image_b, rows, cols, IMG_B_FILENAME)) {
        fprintf(stderr, "Unable to read image colors.");
        return -1;
    }

    matrix_fft(image_r, rows, cols);
    matrix_fft(image_g, rows, cols);
    matrix_fft(image_b, rows, cols);

    free(image_r);
    free(image_g);
    free(image_b);

    return 0;
}