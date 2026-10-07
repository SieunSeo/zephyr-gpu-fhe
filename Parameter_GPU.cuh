#pragma once
#include <cstdint> 

#include "Parameter.cuh"

//-------------------------------------------------------------
// qp_table[j][0] = qp[j]
//         [j][1] = Barrett_k (qp[j])
//         [j][2] = Barrett_mu(qp[j])
//         [j][3] = Shoup_inverse(1,qp[j])
//         [j][4] =         P%qp[j]
//         [j][5] = inv_mod(P%qp[j], qp[j]);
//         [j][6] = Shoup_inverse([j][5],qp[j])
//         [j][7] = Shoup_inverse([j][4],qp[j])
//
// qp_bits[j] = log2(qp[j])
//
// qp_mutual_table[i][j          ] = inv_mod(qp[i],qp[j])
//                [i][j+(L+G+K)  ] = Shoup_inverse( [i][j],qp[j]);
//                [i][j+(L+G+K)*2] = qp[i] % qp[j]
//                [i][j+(L+G+K)*3] = Shoup_inverse(qp[i]%qp[j],qp[j])
//
//  mod_Dld_qpj[l][d][j] = mod(Dld,qp[j])
//
//-------------------------------------------------------------
extern __constant__ uint32_t d_qp_table[L + G + K][8];
extern __constant__ float    d_qp_bits [L + G + K];
extern __constant__ int      d_g_table [DNUM][L+G+K];
extern __constant__ uint32_t d_qp_mutual_table[L+G+K][(L+G+K)*4];
extern __device__   uint32_t d_mod_Dld_qpj[31][L+1][DNUM][L+G+K];

extern __constant__ int  d_gtable_short[DNUM+1];
extern __constant__ int  d_gtable_graft[DNUM  ];




typedef uint32_t Ptext_Gadget[DNUM][L+G+K][N];
#define Ptext_GadgetSize (sizeof(uint32_t)*DNUM*(L+G+K)*N)
typedef uint32_t Swk[DNUM][2][L+G+K][N];
#define SwkSize (sizeof(uint32_t)*DNUM*2*(L+G+K)*N)

void initialize_parameter();

void deallocate_parameter();

void update_qp_mutual_table(int alpha);

void update_mod_Dld_qpj_table(int l, int alpha);

//__device__ 
static Swk* h_rkey [KEYSIZE];

Swk* get_rkey_host(int shift);
void initialize_rkey(int s[N]);
void deallocate_rkey();

