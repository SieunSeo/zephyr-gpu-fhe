#pragma once

#include <assert.h>
#include "NTT.cuh"
#include "Parameter_GPU.cuh"

#include <cuda_runtime.h>
#include <cstdio>


//----------------------------------------------------------------
// tables
//----------------------------------------------------------------
extern Ptext_QP* ntt_Psi;  
extern Ptext_QP* ntt_PsiShoup;
extern Ptext_QP* ntt_PsiInv; 
extern Ptext_QP* ntt_PsiInvShoup;
extern uint32_t* ntt_NInv     ;
extern uint32_t* ntt_NInvShoup;

//----------------------------------------------------------------
// initialize
//----------------------------------------------------------------
void initialize_ntt();
void deallocate_ntt();

//----------------------------------------------------------------
// Cooley-Tuckee 
//----------------------------------------------------------------
__host__ __device__ inline
void CT(uint32_t& U, uint32_t& V, uint32_t Psi, uint32_t PsiShoup, uint32_t q) {
	uint32_t mul = mul_mod(V, Psi, PsiShoup, q);
	V = U + q - mul; if (V >= q) V -= q;
	U += mul;    if (U >= q) U -= q;
}
__host__ __device__ inline
void CT_inverse(uint32_t& U, uint32_t& V, uint32_t PsiInv, uint32_t PsiInvShoup, uint32_t q) {
	uint32_t sub = U + q - V; if (sub >= q) sub -= q;
	U += V; if (U >= q) U -= q;
	V = mul_mod(sub, PsiInv, PsiInvShoup, q);
}

//----------------------------------------------------------------
// ntt for a[N]
//----------------------------------------------------------------
//void ntt_host(uint32_t a[N], int j) ;
//void intt_host(uint32_t a[N], int j);

//----------------------------------------------------------------
// ntt for Ptext, Ptext_QP
//----------------------------------------------------------------
void  ntt_host(Ptext pt, int lfr, int lto) ;
void intt_host(Ptext pt, int lfr, int lto) ;
void  ntt_QP_host(Ptext_QP pt, int l);
void intt_QP_host(Ptext_QP pt, int l);
 
//----------------------------------------------------------------
// ntt for Ctext, Ctext_QP
//----------------------------------------------------------------
void  ntt_host(Ctext ct, int lfr, int lto) ;
void intt_host(Ctext ct, int lfr, int lto) ;
void  ntt_QP_host(Ctext_QP ct, int l);
void intt_QP_host(Ctext_QP ct, int l);



//----------------------------------------------------------------
// intt device functions
//----------------------------------------------------------------
__device__ __forceinline__
void intt_smem( uint32_t* sm, int bid, int tid, uint32_t q, const uint32_t* psi, const uint32_t* psi_s){
    #pragma unroll
    for (int stage = 0; stage < 10; stage++) {
        int m    = 1 << (logN - 1 - stage);
        int t    = 1 << stage;
		//
		int id = ((tid>>stage)<<(stage+1)) + (tid&(t-1));
		int pair = id^t;
		int index = id + (bid<<10);

		int i = index >> (stage + 1);

		uint32_t a = sm[id  ];
		uint32_t b = sm[pair];

		uint32_t sub = a + q - b; if (sub >= q) sub -= q;
		uint32_t sum = a     + b; if (sum >= q) sum -= q;

		sm[id]  = sum;
		sm[pair] = mul_mod(sub, psi[m + i], psi_s[m + i], q);
        
        __syncthreads();
    }
}



__device__ __forceinline__
void intt_fuse3(uint32_t* ptl, int stage, int index, uint32_t q, 
								const uint32_t* psi, const uint32_t* psi_s){
	
	int m = 1 << (logN - 1 - stage);
	int i = index >> (stage + 1);

	uint32_t a[8];
	#pragma unroll
	for(int i1=0; i1<2; i1++)
	for(int i2=0; i2<2; i2++)
	for(int i3=0; i3<2; i3++)
		a[(i1<<2)+(i2<<1)+i3] = ptl[index+(((i1<<2)+(i2<<1)+i3)<<stage)];
	
	#pragma unroll
	for(int i1=0;i1<2;i1++)
	for(int i2=0;i2<2;i2++){
		int idx=(i1<<2)+(i2<<1)+0;  int pair=idx+1;
		//
		uint32_t w  = psi  [m+i+(i1<<1)+i2];
		uint32_t ws = psi_s[m+i+(i1<<1)+i2];
		//
		uint32_t sub = a[idx] + q - a[pair];	if (sub >= q) sub -= q;
		uint32_t sum = a[idx]     + a[pair];	if (sum >= q) sum -= q;
		a[idx ] = sum;
		a[pair] = mul_mod(sub, w, ws, q);
	}
	
	m>>=1; i>>=1;
	
	#pragma unroll
	for(int i1=0;i1<2;i1++){
		uint32_t w  = psi  [m+i+i1];
		uint32_t ws = psi_s[m+i+i1];
		for(int i3=0;i3<2;i3++){
			int idx=(i1<<2)+0+i3;  int pair=idx+2;
			uint32_t sub = a[idx] + q - a[pair];	if (sub >= q) sub -= q;
			uint32_t sum = a[idx]     + a[pair];	if (sum >= q) sum -= q;
			a[idx ] = sum;
			a[pair] = mul_mod(sub, w, ws, q);
		}
	}
	
	m>>=1; i>>=1;
	uint32_t w  = psi  [m+i];
	uint32_t ws = psi_s[m+i];	
	#pragma unroll
	for(int i2=0;i2<2;i2++)	
	for(int i3=0;i3<2;i3++){
		int idx=0+(i2<<1)+i3;  int pair=idx+4;
		uint32_t sub = a[idx] + q - a[pair];	if (sub >= q) sub -= q;
		uint32_t sum = a[idx]     + a[pair];	if (sum >= q) sum -= q;
		a[idx ] = sum;
		a[pair] = mul_mod(sub, w, ws, q);
	}
	
	#pragma unroll
	for(int i1=0; i1<2; i1++)
	for(int i2=0; i2<2; i2++)
	for(int i3=0; i3<2; i3++)
		ptl[index+(((i1<<2)+(i2<<1)+i3)<<stage)]=a[(i1<<2)+(i2<<1)+i3];
	
}




__device__ __forceinline__
void intt_fuse3_last(uint32_t* ptl, int stage, int index, uint32_t q, 
							const uint32_t* psi, const uint32_t* psi_s, uint32_t ninv, uint32_t ninv_s){
	
	int m = 1 << (logN - 1 - stage);
	int i = index >> (stage + 1);

	uint32_t a[8];
	#pragma unroll
	for(int i1=0; i1<2; i1++)
	for(int i2=0; i2<2; i2++)
	for(int i3=0; i3<2; i3++)
		a[(i1<<2)+(i2<<1)+i3] = ptl[index+(((i1<<2)+(i2<<1)+i3)<<stage)];
	
	#pragma unroll
	for(int i1=0;i1<2;i1++)
	for(int i2=0;i2<2;i2++){
		int idx=(i1<<2)+(i2<<1)+0;  int pair=idx+1;
		//
		uint32_t w  = psi  [m+i+(i1<<1)+i2];
		uint32_t ws = psi_s[m+i+(i1<<1)+i2];
		//
		uint32_t sub = a[idx] + q - a[pair];	if (sub >= q) sub -= q;
		uint32_t sum = a[idx]     + a[pair];	if (sum >= q) sum -= q;
		a[idx ] = sum;
		a[pair] = mul_mod(sub, w, ws, q);
	}
	
	m>>=1; i>>=1;
	
	#pragma unroll
	for(int i1=0;i1<2;i1++){
		uint32_t w  = psi  [m+i+i1];
		uint32_t ws = psi_s[m+i+i1];
		for(int i3=0;i3<2;i3++){
			int idx=(i1<<2)+0+i3;  int pair=idx+2;
			uint32_t sub = a[idx] + q - a[pair];	if (sub >= q) sub -= q;
			uint32_t sum = a[idx]     + a[pair];	if (sum >= q) sum -= q;
			a[idx ] = sum;
			a[pair] = mul_mod(sub, w, ws, q);
		}
	}
	
	m>>=1; i>>=1;
	uint32_t w  = psi  [m+i];
	uint32_t ws = psi_s[m+i];	
	#pragma unroll
	for(int i2=0;i2<2;i2++)	
	for(int i3=0;i3<2;i3++){
		int idx=0+(i2<<1)+i3;  int pair=idx+4;
		uint32_t sub = a[idx] + q - a[pair];	if (sub >= q) sub -= q;
		uint32_t sum = a[idx]     + a[pair];	if (sum >= q) sum -= q;
		a[idx ] = sum;
		a[pair] = mul_mod(sub, w, ws, q);
	}
	
	#pragma unroll
	for(int i1=0; i1<2; i1++)
	for(int i2=0; i2<2; i2++)
	for(int i3=0; i3<2; i3++)
		ptl[index+(((i1<<2)+(i2<<1)+i3)<<stage)]=mul_mod(a[(i1<<2)+(i2<<1)+i3],ninv,ninv_s,q);
}



//----------------------------------------------------------------
// ntt device functions
//----------------------------------------------------------------
__device__ __forceinline__
void ntt_smem( uint32_t* sm, int bid, int tid, uint32_t q, const uint32_t* psi, const uint32_t* psi_s){
    #pragma unroll
    for (int stage = 9; stage >= 0; stage--) {
        int m    = 1 << (logN - 1 - stage);
        int t    = 1 << stage;
		//
		int id = ((tid>>stage)<<(stage+1)) + (tid&(t-1));
		int pair = id^t;
		int index = id + (bid<<10);

		int i = index >> (stage + 1);

		uint32_t a = sm[id  ];
		uint32_t b = sm[pair];

		uint32_t mul = mul_mod(b, psi[m + i], psi_s[m + i], q);
        b = a + q - mul; if(b>=q) b-=q;
		a = a     + mul; if(a>=q) a-=q;
		
		sm[id  ] = a;
		sm[pair] = b;
		
        __syncthreads();
    }
}

__device__ __forceinline__
void ntt_fuse3(uint32_t pt[N], int stage, int index, uint32_t q, 
                        const uint32_t psi[N], const uint32_t psi_s[N]){
   
   int m = 1 << stage;
   int logt = logN-1-stage;
   int i0 = index >> (logt + 1);

   uint32_t a[8];
   #pragma unroll
   for(int i1=0; i1<2; i1++)
   for(int i2=0; i2<2; i2++)
   for(int i3=0; i3<2; i3++)
      a[(i1<<2)+(i2<<1)+i3] = pt[index+(((i1<<2)+(i2<<1)+i3)<<(logt-2))];
   
   //
   uint32_t w  = psi  [m+i0];
   uint32_t ws = psi_s[m+i0];   
   #pragma unroll
   for(int i2=0;i2<2;i2++)   
   for(int i3=0;i3<2;i3++){
      int idx=(i2<<1)+i3;  int pair=idx+4;
      uint32_t mul = mul_mod(a[pair], w, ws, q);
        a[pair] = a[idx] + q - mul; if(a[pair]>=q) a[pair]-=q;
      a[idx ] = a[idx]     + mul; if(a[idx ]>=q) a[idx ]-=q;
   }
   //
   m<<=1; 
   #pragma unroll
   for(int i1=0;i1<2;i1++){
      uint32_t w  = psi  [m+(i0<<1)+i1];
      uint32_t ws = psi_s[m+(i0<<1)+i1];
      for(int i3=0;i3<2;i3++){
         int idx=(i1<<2)+i3;  int pair=idx+2;
         uint32_t mul = mul_mod(a[pair], w, ws, q);
         a[pair] = a[idx] + q - mul; if(a[pair]>=q) a[pair]-=q;
         a[idx ] = a[idx]     + mul; if(a[idx ]>=q) a[idx ]-=q;
      }
   }
   //
   m<<=1;
   #pragma unroll
   for(int i1=0;i1<2;i1++)
   for(int i2=0;i2<2;i2++){
      int idx=(i1<<2)+(i2<<1);  int pair=idx+1;
      //
      uint32_t w  = psi  [m+(i0<<2)+(i1<<1)+i2];
      uint32_t ws = psi_s[m+(i0<<2)+(i1<<1)+i2];
      //
      uint32_t mul = mul_mod(a[pair], w, ws, q);
      a[pair] = a[idx] + q - mul; if(a[pair]>=q) a[pair]-=q;
      a[idx ] = a[idx]     + mul; if(a[idx ]>=q) a[idx ]-=q;
   }
   //
   #pragma unroll
   for(int i1=0; i1<2; i1++)
   for(int i2=0; i2<2; i2++)
   for(int i3=0; i3<2; i3++)
      pt[index+(((i1<<2)+(i2<<1)+i3)<<(logt-2))]=a[(i1<<2)+(i2<<1)+i3];
   
}
