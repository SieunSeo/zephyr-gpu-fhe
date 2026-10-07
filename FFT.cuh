
#pragma once

#include <math.h>
#include "Parameter.cuh"

struct FFT {
    double zetar[16][N / 4];
    double zetai[16][N / 4];

    FFT() {
        const double PI = double(3.14159265358979323846264338328);
        double theta = PI / double(N);
        for (int i = 0, poweri = 1; i <= logN - 2; i++, poweri = (poweri * 2) % (2 * N)) {
            for (int j = 0, power = poweri; j < (1 << (logN - 2 - i)); j++, power = (power * 5) % (2 * N)) {
                zetar[i][j] = cos(theta * double(power));
                zetai[i][j] = sin(theta * double(power));
            }
            bitReverse(N >> (i + 2), zetar[i]);
            bitReverse(N >> (i + 2), zetai[i]);
        }
    }

    void bitReverse(int N_, double* z) const{
        int logN_ = 0;
        while (N_ > (1 << logN_)) logN_++;
        for (int i = 0; i < N_; i++) {
            int j = 0, x = i;
            for (int k = 0; k < logN_; k++, x >>= 1)
                j = (j << 1) | (x & 1);
            if (j > i) {
                double temp = z[i];
                z[i] = z[j];
                z[j] = temp;
            }
        }
    }


    void fft(double z[N]) const {
        double* zr = z;
        //double* zi = z + N / 2;
        for (int m = 1, logt = logN - 2; m < N / 2; m *= 2, logt--)
            for (int i = 0; i < m; i++) {
                int t = 1 << logt;
                double* zer = zr + 2 * i * t;
                double* zei = zer + N / 2;
                double* zor = zer + t;
                double* zoi = zor + N / 2;
                for (int k = 0; k < t; k++) {
                    double Ur = zor[k] * zetar[logt][i] - zoi[k] * zetai[logt][i];
                    double Ui = zor[k] * zetai[logt][i] + zoi[k] * zetar[logt][i];
                    zor[k] = zer[k] - Ur;
                    zoi[k] = zei[k] - Ui;
                    zer[k] = zer[k] + Ur;
                    zei[k] = zei[k] + Ui;
                }
            }
    }

    void ifft(double z[N]) const {
        double* zr = z;
        //double* zi = z + N / 2;
        for (int m = N / 4, logt = 0; m > 0; m /= 2, logt++)
            for (int i = 0; i < m; i++) {
                int t = 1 << logt;
                double* zer = zr + 2 * i * t;
                double* zei = zer + N / 2;
                double* zor = zer + t;
                double* zoi = zor + N / 2;
                for (int k = 0; k < t; k++) {
                    double Ur = zer[k] - zor[k];
                    double Ui = zei[k] - zoi[k];
                    zer[k] = zer[k] + zor[k];
                    zei[k] = zei[k] + zoi[k];
                    zor[k] = Ur * zetar[logt][i] + Ui * zetai[logt][i];
                    zoi[k] = -Ur * zetai[logt][i] + Ui * zetar[logt][i];
                }
            }
        double ratio = double(2) / double(N);
        for (int i = 0; i < N; i++) z[i] *= ratio;
    }
};