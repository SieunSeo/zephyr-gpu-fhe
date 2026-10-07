#pragma once

#include "Parameter_GPU.cuh"

#include <cuda_runtime.h>



void bitReverse(int logn, float z[N]);

__global__ void bitReverse_kernel(int logn, float *z);

void initialize_fft_FP32();

void deallocate_fft_FP32();
	
//----------------------------------------------------------------
// Cooley-Tuckee 
//----------------------------------------------------------------
__host__ __device__ inline
void CT_FP32(float& Ur, float& Ui, float& Vr, float& Vi, float zetar, float zetai) {
	float mulr = Vr * zetar - Vi * zetai;
	float muli = Vr * zetai + Vi * zetar;
	Vr = Ur - mulr;
	Vi = Ui - muli;
	Ur += mulr;
	Ui += muli;
}

//----------------------------------------------------------------
// fft_kernel
//----------------------------------------------------------------
__global__ void fft_kernel(float a[N], int kfr, int kto, float table[N]);


//----------------------------------------------------------------
// fft_host
//----------------------------------------------------------------
void fft_host(float a[N]);
	
//----------------------------------------------------------------
// Cooley-Tuckee inverse
//----------------------------------------------------------------
__host__ __device__ inline
void CT_FP32_inverse(float& Ur, float& Ui, float& Vr, float& Vi, float zetar, float zetai) {
	float subr = Ur - Vr;
	float subi = Ui - Vi;
	Ur += Vr;
	Ui += Vi;
	Vr = subr * zetar + subi * zetai;
	Vi = -subr * zetai + subi * zetar;
}

//----------------------------------------------------------------
// ifft_kernel
//----------------------------------------------------------------
__global__ void ifft_kernel(float a[N], int kfr, int kto, float table[N], bool flag);

//----------------------------------------------------------------
// ifft_host
//----------------------------------------------------------------
void ifft_host(float a[N]);

