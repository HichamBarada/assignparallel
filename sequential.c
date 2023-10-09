#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 1000

int mandelbrot(double real, double imag) {
    double z_real = 0;
    double z_imag = 0;

    for (int i = 0; i < MAX_ITER; i++) {
        double z_real_squared = z_real * z_real;
        double z_imag_squared = z_imag * z_imag;

        if (z_real_squared + z_imag_squared > 4) {
            return i;
        }

        double new_z_real = z_real_squared - z_imag_squared + real;
        double new_z_imag = 2 * z_real * z_imag + imag;

        z_real = new_z_real;
        z_imag = new_z_imag;
    }

    return 0;
}

int main() {
    FILE* fp;
    char* filename = "mandelbrot.ppm";
    fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);

    for (int row = 0; row < HEIGHT; row++) {
        for (int col = 0; col < WIDTH; col++) {
            double real = (col - WIDTH / 2.0) * 4.0 / WIDTH;
            double imag = (row - HEIGHT / 2.0) * 4.0 / WIDTH;

            int iter = mandelbrot(real, imag);

            unsigned char r, g, b;
            if (iter == 0) {
                r = g = b = 0;
            } else {
                r = (int) (sin(iter * 0.03) * 127 + 128);
                g = (int) (sin(iter * 0.05) * 127 + 128);
                b = (int) (sin(iter * 0.07) * 127 + 128);
            }

            fputc(r, fp);
            fputc(g, fp);
            fputc(b, fp);
        }
    }

    fclose(fp);

    return 0;
}