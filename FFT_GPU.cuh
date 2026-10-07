#pragma once

#include "Parameter_GPU.cuh"

#include <cuda_runtime.h>



void bitReverse(int logn, double z[N]);

__global__ void bitReverse_kernel(int logn, double *z);

void initialize_fft();

void deallocate_fft();
	
//----------------------------------------------------------------
// Cooley-Tuckee 
//----------------------------------------------------------------
__host__ __device__ inline
void CT(double& Ur, double& Ui, double& Vr, double& Vi, double zetar, double zetai) {
	double mulr = Vr * zetar - Vi * zetai;
	double muli = Vr * zetai + Vi * zetar;
	Vr = Ur - mulr;
	Vi = Ui - muli;
	Ur += mulr;
	Ui += muli;
}

//----------------------------------------------------------------
// fft_kernel
//----------------------------------------------------------------
__global__ void fft_kernel(double a[N], int kfr, int kto, double table[N]);


//----------------------------------------------------------------
// fft_host
//----------------------------------------------------------------
void fft_host(double a[N]);
	
//----------------------------------------------------------------
// Cooley-Tuckee inverse
//----------------------------------------------------------------
__host__ __device__ inline
void CT_inverse(double& Ur, double& Ui, double& Vr, double& Vi, double zetar, double zetai) {
	double subr = Ur - Vr;
	double subi = Ui - Vi;
	Ur += Vr;
	Ui += Vi;
	Vr = subr * zetar + subi * zetai;
	Vi = -subr * zetai + subi * zetar;
}

//----------------------------------------------------------------
// ifft_kernel
//----------------------------------------------------------------
__global__ void ifft_kernel(double a[N], int kfr, int kto, double table[N], bool flag);

//----------------------------------------------------------------
// ifft_host
//----------------------------------------------------------------
void ifft_host(double a[N]);

