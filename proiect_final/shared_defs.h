#ifndef SHARED_CPU_DEFS_
#define SHARED_CPU_DEFS_

#include <inttypes.h>

#define SHAPE_FILENAME "../resources/img.shape"
#define IMG_R_FILENAME "../resources/img_r.mat"
#define IMG_G_FILENAME "../resources/img_g.mat"
#define IMG_B_FILENAME "../resources/img_b.mat"

#define FFT_R_FILENAME "../output/fft_r.mat"
#define FFT_G_FILENAME "../output/fft_g.mat"
#define FFT_B_FILENAME "../output/fft_b.mat"

#define pi 3.14159265

#define max(a, b) ((a) > (b)) ? (a) : (b)
#define min(a, b) ((a) < (b)) ? (a) : (b)

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
        fprintf(stderr, "Could not read all bytes from %s. Bytes read: %ld, Expected: %d\n",
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
        fprintf(stderr, "Could not read all bytes from %s. Bytes read: %ld, Expected: %d\n",
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

uint32_t log_of_pow_of_2(uint32_t num) {
    uint32_t l;
    for (l = 0; l < 32; l++) {
        if (num&1) {
            return l;
        }
        num >>= 1;
    }
}

# endif // SHARED_CPU_DEFS_