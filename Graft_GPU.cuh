#pragma once
#include "NTT_GPU.cuh"
#include "CKKS_debug.cuh"


extern __constant__ uint32_t graft_constant[3][11];


void initialize_graft();
void deallocate_graft();
void initialize_graft_Ctext();
void deallocate_graft_Ctext();

__global__
void graft_embed_kernel(uint32_t a[N], uint32_t b[N], int alpha, uint32_t table[3][6][N]);

//----------------------------------------------------------------
// graft_ntt_kernel : blockIdx.z = 3*(temp_index) + graft_index
//                    temp_index = 0,1
//                    graft_index = 0,1,2
//                 blockIdx.z = 0,1,2,3,4,5
//----------------------------------------------------------------
__global__
void graft_ntt_kernel(uint32_t graft_table[3][6][N], int kfr, int kto);

__global__
void graft_intt_kernel(uint32_t graft_table[3][6][N], int kfr, int kto, bool flag);

__global__ void graft_mulmod_kernel(uint32_t graft_table[3][6][N]);

__global__ void graft_modulus_kernel(uint32_t graft_table[3][6][N], int alpha, uint32_t out[N]);

void conv_graft_host(uint32_t a[N], uint32_t b[N], int alpha, uint32_t out[N]);

void conv_graft_host(uint32_t a1[N], uint32_t a2[N], uint32_t b1[N], uint32_t b2[N], int alpha, uint32_t out1[N], uint32_t out2[N]);

void conv_add_graft_host(uint32_t a1[N], uint32_t a2[N], 
                         uint32_t b1[N], uint32_t b2[N],  
					     uint32_t c1[N], uint32_t c2[N], int alpha, uint32_t out1[N], uint32_t out2[N]);
