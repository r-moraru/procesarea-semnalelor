/*
 * The tests in this file might be outdated.
 */
#include "fft_cpu.h"

void is_power_of_2_test() {
    printf("Is power of 2 test:\n");
    printf("4: %d\n", is_power_of_2(4));
    printf("\n");
}

void swap_float_test() {
    printf("Swap float test:\n");
    float a = 345.123, b = 732.887;
    printf("Before: a = %f, b = %f.\n", a, b);
    swap_float(&a, &b);
    printf("After: a = %f, b = %f.\n", a, b);
    printf("\n");
}

void swap_int_test() {
    printf("Swap int test:\n");
    int a1 = 345, b1 = 11873;
    printf("Before: a = %d, b = %d.\n", a1, b1);
    swap_int(&a1, &b1);
    printf("After: a = %d, b = %d.\n", a1, b1);
    printf("\n");
}

void log_of_pow_of_2_test() {
    printf("Log of pow of 2 test:\n");
    printf("Log of 512: %d\n", log_of_pow_of_2(512));
    printf("\n");
}

void complex_fft_test() {
    printf("Complex fft test:\n");

    float *test_arr = (float *)malloc(8 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        test_arr[i] = i+1;
        test_arr[i+4] = i+2;
    }
    printf("a: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", test_arr[i], test_arr[i+4]);
    }
    printf("\n");
    complex_fft(test_arr, test_arr + 4, 4, 1);
    printf("A: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", test_arr[i], test_arr[i+4]);
    }
    printf("\n");
    printf("Expected: 10. + i14., -4. + i0., -2. i-2.,  0. i-4.\n");
    printf("\n");
    free(test_arr);
}

void column_complex_fft_test() {
    printf("Column complex fft test:\n");
    uint32_t index_multiplier = 2;
    float *test_arr = (float *)malloc(8 * sizeof(float));
    float *real = test_arr, *imag = test_arr+1;
    for (int i = 0; i < 4; i++) {
        real[i*index_multiplier] = i+1;
        imag[i*index_multiplier] = i+2;
    }
    printf("a: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", real[i*index_multiplier], imag[i*index_multiplier]);
    }
    printf("\n");
    complex_fft(real, imag, 4, index_multiplier);
    printf("A: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", real[i*index_multiplier], imag[i*index_multiplier]);
    }
    printf("\n");
    printf("Expected: 10. + i14., -4. + i0., -2. i-2.,  0. i-4.\n");
    printf("\n");
    free(test_arr);
}

void real_fft_test() {
    printf("Real fft test:\n");

    float *test_arr = (float *)malloc(8 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        test_arr[i] = i+1;
        test_arr[i+4] = i+2;
    }
    printf("a: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", test_arr[i], test_arr[i+4]);
    }
    printf("\n");
    real_pair_fft(test_arr, test_arr + 4, 4, 1);
    printf("A: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", test_arr[i], test_arr[i+4]);
    }
    printf("\n");
    printf("Expected: 10. + i14., -2. + i-2., -2. i-2.,  0. i-4.\n");
    printf("\n");
    free(test_arr);
}

void column_real_fft_test() {
    printf("Column real fft test:\n");

    float *test_arr = (float *)malloc(8 * sizeof(float));
    for (int i = 0; i < 4; i++) {
        test_arr[i*2] = i+1;
        test_arr[i*2+1] = i+2;
    }
    printf("a: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", test_arr[i*2], test_arr[i*2+1]);
    }
    printf("\n");
    real_pair_fft(test_arr, test_arr + 1, 4, 2);
    printf("A: ");
    for (int i = 0; i < 4; i++) {
        printf("%f + i%f, ", test_arr[i*2], test_arr[i*2+1]);
    }
    printf("\n");
    printf("Expected: 10. + i14., -2. + i-2., -2. i-2.,  0. i-4.\n");
    printf("\n");
    free(test_arr);
}

int main() {
    is_power_of_2_test();
    swap_float_test();
    swap_int_test();
    log_of_pow_of_2_test();
    complex_fft_test();
    column_complex_fft_test();
    real_fft_test();
    column_real_fft_test();

    return 0;
}