#pragma once

#include <cstdint>
#include <stdlib.h>
#include "uint128_t.cuh"


//-------------------------------------------------------------------------------
//       full multiplication, 
//       full       division, 
//       modular multiplication
//-------------------------------------------------------------------------------
__device__ __host__ inline
void mul(uint32_t a, uint32_t b, uint32_t* hi, uint32_t* lo) {
	uint64_t out = uint64_t(a) * b;
	*lo = uint32_t(out);
	*hi = out >> 32;
}

__device__ __host__ inline
uint32_t mul_mod_worst(uint32_t a, uint32_t b, uint32_t mod) {
	return (uint64_t(a) * b) % mod;
}

//----------------------------------------------------------------
// high multiplication (approximation, error =0,1)
//----------------------------------------------------------------
__host__ __device__ inline
void mul_hi_appr(uint32_t a, uint32_t b, uint32_t &hi) {
	static constexpr uint32_t filter_16_l = (uint32_t(1) << 16) - 1;

    uint32_t al = a & filter_16_l; uint32_t ah = a >> 16;
    uint32_t bl = b & filter_16_l; uint32_t bh = b >> 16;

    uint32_t lh = al * bh;
    uint32_t hl = ah * bl;
    uint32_t hh = ah * bh;

    uint32_t c = 0;
    uint32_t temp;
    temp = (hl << 16) + (lh << 16);
    if (temp < (hl << 16)) c++;

    hi = hh + c + (hl >> 16) + (lh >> 16);
}

//-------------------------------------------------------------------------------
// Shoup's modular multiplication
//-------------------------------------------------------------------------------
//__device__ __host__ inline
//uint32_t mul_mod(uint32_t a, uint32_t b, uint32_t bShoup, uint32_t m) {
//	uint32_t q = (uint64_t(a) * bShoup) >> 32;
//	uint32_t r = a * b - q * m;
//	return (r >= m) ? (r - m) : r;
//}
__host__ __device__ inline
uint32_t mul_mod(uint32_t a, uint32_t b, uint32_t bShoup, uint32_t m) {
    uint32_t qhat; mul_hi_appr(a, bShoup, qhat);
    uint32_t rhat = a*b - qhat*m;
    if (rhat >= m) rhat -= m;
    if (rhat >= m) rhat -= m;
	
	return rhat;
}

__device__ __host__ inline
uint32_t Shoup_inverse(uint32_t b, uint32_t m) {
	uint64_t prod = uint64_t(b) << 32;
	return uint32_t(prod / m);
}


__device__ inline
uint32_t mod(uint32_t a, uint32_t oneShoup, uint32_t m) {
    uint32_t qhat= __umulhi(a, oneShoup);
    uint32_t rhat = a - qhat*m;
    if (rhat >= m) rhat -= m;
	return rhat;
}

//-------------------------------------------------------------------------------
// Barrett's modular multiplication
//-------------------------------------------------------------------------------
__device__ __host__ inline
uint32_t mul_mod(uint32_t a, uint32_t b, uint32_t p, int k, uint32_t mu) {
	uint32_t hi, lo; mul(a, b, &hi, &lo);
	uint32_t c = (hi << (33 - k)) + (lo >> (k - 1));
	uint32_t ab = lo;
	mul(c, mu, &hi, &lo);
	uint32_t q1 = (hi << (31 - k));
	uint32_t q2 = (k + 1 == 32) ? 0 : (lo >> (k + 1));
	uint32_t q = q1 + q2;
	uint32_t r = ab - (q * p);
	if (r >= p) r -= p;
	if (r >= p) r -= p;
	return r;
}
__device__ __host__ inline
int      Barrett_exponent(uint32_t p) {
	int k = 0;
	while (p > (1ULL << k))
		k++;
	return k;
}
__device__ __host__ inline
uint32_t Barrett_inverse(uint32_t p, int k) {
	uint32_t a = 1; a <<= k;
	uint64_t prod = uint64_t(a) * a;
	return uint32_t(prod / p);
}



//-------------------------------------------------------------------------------
// Power_mod
//-------------------------------------------------------------------------------
__device__ __host__ inline
uint32_t power_mod(uint32_t a, uint32_t n, uint32_t q) {
	uint32_t m = a % q, out = 1;
	while (n > 0) {
		if (n % 2 == 1) out = mul_mod_worst(out, m, q);
		m = mul_mod_worst(m, m, q);
		n >>= 1;
	}
	return out;
}

__device__ __host__ inline
uint32_t inv_mod(uint32_t a, uint32_t q)
{
	// case when q = 2^k
	if (q % 2 == 0) {
		// DusseKaliski algorithm
		uint32_t x = 1;
		//int i = 2;
		uint32_t power_i = 4;
		while (power_i <= q) {
			uint64_t prod = uint64_t(a) * x;
			uint32_t mod = prod & (power_i - 1);
			if (power_i < mod * 2)
				x += (power_i) >> 1;
			power_i <<= 1;
		}
		return x;
	}
	// case when q is a prime
	else
		return power_mod(a, q - 2, q);
}


//-------------------------------------------------------------------------------
// rand uint32_t
//-------------------------------------------------------------------------------

__device__ __host__ inline
uint32_t rand_uint32() {
	static uint64_t rand_c = 1;
	static uint64_t rand_seq0 = 2;
	static uint64_t rand_seq1 = 3;
	static uint64_t rand_seq2 = 4;
	static uint64_t rand_seq3 = 5;
	uint64_t s = 2111111111 * rand_seq3
		+ 1492 * rand_seq2
		+ 1776 * rand_seq1
		+ 5115 * rand_seq0
		+ rand_c;
	rand_c = s >> 32;
	rand_seq0 = rand_seq1;
	rand_seq1 = rand_seq2;
	rand_seq2 = rand_seq3;
	rand_seq3 = s;
	return (uint32_t)(s);
}

__device__ __host__ inline
uint64_t rand_uint64() {
	return (uint64_t(rand_uint32()) << 32) + rand_uint32();
}
//-------------------------------------------------------------------------------
// bitReverse
//-------------------------------------------------------------------------------

template< int N_ >
__device__ __host__ inline
void bitReverse(uint32_t a[N_]) {
	uint32_t temp[N_]; for (int i = 0;i < N_;i++) temp[i] = a[i];
	int logN_ = 0;
	while (N_ > (1 << logN_))
		logN_++;
	for (int i = 0;i < N_;i++) {
		int j = 0;
		for (int k = 0;k < logN_;k++)
			j += ((i >> k) & 1) << (logN_ - 1 - k);
		a[j] = temp[i];
	}
}

template< int N_ >
__device__ __host__ inline
void bitReverse(double a[N_]) {
	double temp[N_]; for (int i = 0;i < N_;i++) temp[i] = a[i];
	int logN_ = 0;
	while (N_ > (1 << logN_))
		logN_++;
	for (int i = 0;i < N_;i++) {
		int j = 0;
		for (int k = 0;k < logN_;k++)
			j += ((i >> k) & 1) << (logN_ - 1 - k);
		a[j] = temp[i];
	}
}

//-------------------------------------------------------------------------------
// Common mulmod operations
//-------------------------------------------------------------------------------
template<int N_>
void mul_mod_type1(uint32_t a[N_], uint32_t b[N_], uint32_t p, uint32_t q) {
	uint32_t table1 = inv_mod(p, q);
	uint32_t table2 = Shoup_inverse(table1, q);
	for (int i = 0;i < N_; i++) {
		a[i] = a[i] + p - b[i];
		a[i] = mul_mod(a[i], table1, table2, q);
		a[i] = (a[i] == 0) ? q - 1 : a[i] - 1;
	}
}

template<int N_>
void mul_mod_type2(uint32_t a[N_], uint32_t b[N_], uint32_t p, uint32_t q) {
	uint32_t table1 = p % q;
	uint32_t table2 = Shoup_inverse(table1, q);
	for (int i = 0;i < N_; i++) {
		a[i] = mul_mod(a[i], table1, table2, q);
		a[i] = a[i] + b[i];
		if (a[i] >= q) a[i] -= q;
	}
}


template<int N_>
void mul_mod_type3(uint32_t a[N_], uint32_t q) {
	uint32_t table1 = 1;
	uint32_t table2 = Shoup_inverse(1, q);
	for (int i = 0;i < N_; i++) {
		a[i] = mul_mod(a[i], table1, table2, q);
	}
}

template<int N_>
void mul_mod_type4(uint32_t a[N_], uint32_t r[N_], uint32_t table1, uint32_t q) {
	uint32_t table2 = Shoup_inverse(table1, q);
	for (int i = 0;i < N_; i++) {
		uint32_t ri = mul_mod(r[i], table1, table2, q);
		a[i] = a[i] + q - ri;
		if (a[i] >= q) a[i] -= q;
	}
}

template<int N_>
void mul_mod_type5(const uint32_t a[N_], const uint32_t b[N_], uint32_t out[N_], uint32_t q) {
	int k = Barrett_exponent(q);
	uint32_t mu = Barrett_inverse(q, k);
	for (int i = 0;i < N_; i++) {
		out[i] = mul_mod(a[i], b[i], q, k, mu);
	}
}


template<int N_>
void mul_mod_type6(const uint32_t a[N_], uint32_t table1, uint32_t out[N_], uint32_t q) {
	uint32_t table2 = Shoup_inverse(table1, q);
	for (int i = 0;i < N_; i++) {
		out[i] = mul_mod(a[i], table1, table2, q);
	}
}

template<int N_>
void add_mod(const uint32_t a[N_], const uint32_t b[N_], uint32_t out[N_], uint32_t q) {
	for (int i = 0; i < N_; i++) {
		out[i] = a[i] + b[i];
		if (out[i] >= q) out[i] -= q;
	}
}

template<int N_>
void add_mod(const uint32_t a[N_], const uint32_t b, uint32_t out[N_], uint32_t q) {
	for (int i = 0; i < N_; i++) {
		out[i] = a[i] + b;
		if (out[i] >= q) out[i] -= q;
	}
}

template<int N_>
void sub_mod(const uint32_t a[N_], const uint32_t b[N_], uint32_t out[N_], uint32_t q) {
	for (int i = 0; i < N_; i++) {
		out[i] = a[i] + (q<<1) - b[i];
		if (out[i] >= q) out[i] -= q;
		if (out[i] >= q) out[i] -= q;
	}
}

template<int N_>
void sub_mod(const uint32_t a[N_], const uint32_t b, uint32_t out[N_], uint32_t q) {
	for (int i = 0; i < N_; i++) {
		out[i] = a[i] + q - b;
		if (out[i] >= q) out[i] -= q;
	}
}

// out = (a+q-b)*table1 (mod q)
template<int N_>
void sub_mulmod(const uint32_t a[N_], const uint32_t b[N_], uint32_t table1, uint32_t out[N_], uint32_t q) {
	uint32_t table2 = Shoup_inverse(table1, q);
	for (int i = 0;i < N_; i++) {
		// sub
		out[i] = a[i] + q - b[i];
		if (out[i] >= q) out[i] -= q;
		// mul
		out[i] = mul_mod(out[i], table1, table2, q);
	}
}


// out = (a+qb-b)*inv(qb,qa)-1 (mod qa)
template<int N_>
void sub_mulmod(const uint32_t a[N_], uint32_t qa, const uint32_t b[N_], uint32_t qb, uint32_t out[N_]) {
	uint32_t table1 = inv_mod(qb, qa);
	uint32_t table2 = Shoup_inverse(table1, qa);
	for (int i = 0;i < N_; i++) {
		// sub
		out[i] = a[i] + qb - b[i];
		// mul
		out[i] = mul_mod(out[i], table1, table2, qa);
		// minus 1
		out[i] = (out[i] == 0) ? (qa - 1) : (out[i] - 1);
	}
}


// out = a*table1 + b (mod q)
template<int N_>
void mulmod_add(const uint32_t a[N_], uint32_t table1, const uint32_t b[N_], uint32_t out[N_], uint32_t q) {
	uint32_t table2 = Shoup_inverse(table1, q);
	for (int i = 0;i < N_; i++) {
		// mul
		out[i] = mul_mod(a[i], table1, table2, q);
		// add
		out[i] = out[i] + b[i];
		if (out[i] >= q) out[i] -= q;
		if (out[i] >= q) out[i] -= q; //assumption:q[j]<p[k]<2*q[j]
	}
}



template< int N_>
void base_conversion_nested_hicomponent(const uint32_t p[], uint32_t a[][N_], int K_,
									    const uint32_t q[], uint32_t b[][N_], int L_, uint32_t ahi_nested[N_]){
	for (int j = 0;j < K_;j++)
		for (int k = 0;k < j;k++)
			mul_mod_type1<N_>(a[j], a[k], p[k], p[j]);

	memcpy(ahi_nested, a[K_ - 1], sizeof(uint32_t) * N_);
	for (int j = 0;j < L_;j++) {
		memset(b[j], 0, sizeof(uint32_t) * N_);
		for (int k = K_ - 1;k >= 0;k--)
			mul_mod_type2<N_>(b[j], a[k], p[k], q[j]);
		mul_mod_type3<N_>(b[j], q[j]);
	}
}

template< int N_>
void base_conversion_nested_rounded(const uint32_t p[], uint32_t a[][N_], int K_,
	                                const uint32_t q[], uint32_t b[][N_], int L_ ) {
	for (int j = 0;j < K_;j++)
		for (int k = 0;k < j;k++)
			mul_mod_type1<N_>(a[j], a[k], p[k], p[j]);

	uint32_t r[N_];
	for (int i = 0;i < N_;i++)
		r[i] = (a[K_-1][i] >= (p[K_-1] >> 1)) ? 1 : 0;


	for (int j = 0;j < L_;j++)
	{
		memset(b[j], 0, sizeof(uint32_t) * N_);
		uint32_t P = 1;
		for (int k = K_ - 1;k >= 0;k--)
		{
			mul_mod_type2<N_>(b[j], a[k], p[k], q[j]);
			P = uint32_t((uint64_t(P) * p[k]) % q[j]);
		}
		mul_mod_type3<N_>(b[j], q[j]);
		mul_mod_type4<N_>(b[j], r, P, q[j]);
	}
}


template< int N_>
void base_conversion_nested_hicomponent(const uint32_t p[], uint32_t a[][N_], bool amark[], int K_,
							            const uint32_t q[], uint32_t b[][N_], bool bmark[], int L_, uint32_t ahi[N_], uint32_t& qhi){
	for (int j = 0;j < K_;j++) 
		if(amark[j]){
			for (int k = 0;k < j;k++)
				if(amark[k])
					mul_mod_type1<N_>(a[j], a[k], p[k], p[j]);
			memcpy(ahi, a[j], sizeof(uint32_t) * N_); qhi = p[j];
			//printf("j=%d, a[%d][2]=%u, q[%d]=%u\n",j,j,a[j][2],j,p[j]);
		}
	
	for (int j = 0;j < L_;j++) 
		if (bmark[j]) {
			//if (j == 47)
			//	bool stop = true;
			memset(b[j], 0, sizeof(uint32_t) * N_);
			for (int k = K_ - 1;k >= 0;k--)
				if (amark[k]){
					mul_mod_type2<N_>(b[j], a[k], p[k], q[j]);
					//if(j==0)
					//	printf("k=%d, a[%d][2]=%u, b[0][2]=%u, p[k]=%u, q[0]=%u\n",
					//			k,k,a[k][2],b[0][2],p[k],q[0]);
				}
			mul_mod_type3<N_>(b[j], q[j]);
			//if(j==0)
			//	printf("b[0][2]=%u, q[0]=%u\n",b[0][2],q[0]);
		}
		
}

template< int N_>
void base_conversion_round(const uint32_t p[], uint32_t a[][N_], bool amark[], int K_,
	                       const uint32_t q[], uint32_t b[][N_], bool bmark[], int L_) {
	uint32_t r[N_];
	for (int j = 0;j < K_;j++)
		if (amark[j]) {
			for (int k = 0;k < j;k++)
				if (amark[k])
					mul_mod_type1<N_>(a[j], a[k], p[k], p[j]);
			for (int i = 0;i < N_;i++)
				r[i] = (a[j][i] >= (p[j] >> 1)) ? 1 : 0;
		}

	for (int j = 0;j < L_;j++)
		if (bmark[j]) {
			memset(b[j], 0, sizeof(uint32_t) * N_);
			uint32_t P = 1;
			for (int k = K_ - 1;k >= 0;k--)
				if (amark[k]) {
					mul_mod_type2<N_>(b[j], a[k], p[k], q[j]);
					P = uint32_t((uint64_t(P) * p[k]) % q[j]);
				}
			mul_mod_type3<N_>(b[j], q[j]);
			mul_mod_type4<N_>(b[j], r, P, q[j]);
		}
}



/*
template< int N_>
void base_conversion(const uint32_t p[], uint32_t a[][N_], int K_,
	const uint32_t q[], uint32_t b[][N_], int L_, int r[N_]) {
	//------------------------------------------------------------------------------
	// Table calculation
	//------------------------------------------------------------------------------
	uint32_t** table1 = new uint32_t * [K_];
	uint32_t** table2 = new uint32_t * [K_];
	uint32_t** table3 = new uint32_t * [K_];
	uint32_t** table4 = new uint32_t * [K_];
	uint32_t* table5 = new uint32_t[L_];
	uint32_t* table6 = new uint32_t[L_];
	for (int k = 0;k < K_;k++) {
		table1[k] = new uint32_t[K_];
		table2[k] = new uint32_t[K_];
		for (int j = 0;j < K_;j++) {
			table1[k][j] = inv_mod(p[k], p[j]);
			table2[k][j] = Shoup_inverse(table1[k][j], p[j]);
		}
	}

	for (int k = 0;k < K_;k++) {
		table3[k] = new uint32_t[L_];
		table4[k] = new uint32_t[L_];
		for (int j = 0;j < L_;j++) {
			table3[k][j] = p[k] % q[j];
			table4[k][j] = Shoup_inverse(table3[k][j], q[j]);
		}
	}

	for (int j = 0;j < L_;j++) {
		table5[j] = 1;
		table6[j] = Shoup_inverse(1, q[j]);
	}
	//------------------------------------------------------------------------------
	// Nested form
	//------------------------------------------------------------------------------
	uint32_t(*a_)[N_] = new uint32_t[K_][N_];
	for (int j = 0; j < K_; j++)
		for (int i = 0; i < N_; i++) {
			a_[j][i] = a[j][i];
			for (int k = 0;k < j;k++) {
				a_[j][i] = mul_mod(a_[j][i] + p[k] - a_[k][i], table1[k][j], table2[k][j], p[j]);
				a_[j][i] = (a_[j][i] == 0) ? (p[j] - 1) : (a_[j][i] - 1);
			}
		}
	//------------------------------------------------------------------------------
	// Remainder
	//------------------------------------------------------------------------------
	for (int i = 0;i < N_;i++) r[i] = (a_[K_ - 1][i] >= (p[K_ - 1] >> 1)) ? 1 : 0;
	//------------------------------------------------------------------------------
	// Nested form with mod
	//------------------------------------------------------------------------------
	for (int j = 0; j < L_; j++)
		for (int i = 0; i < N_; i++) {
			b[j][i] = 0;
			for (int k = K_ - 1; k >= 0; k--) {
				b[j][i] = mul_mod(b[j][i], table3[k][j], table4[k][j], q[j]);
				b[j][i] += a_[k][i];
			}
			b[j][i] = mul_mod(b[j][i], table5[j], table6[j], q[j]);
		}
	delete[] a_;
	//------------------------------------------------------------------------------
	// Table deletion
	//------------------------------------------------------------------------------
	for (int j = 0;j < K_;j++) {
		delete[] table1[j];
		delete[] table2[j];
		delete[] table3[j];
		delete[] table4[j];
	}
	delete[] table1;
	delete[] table2;
	delete[] table3;
	delete[] table4;
	delete[] table5;
	delete[] table6;
}
*/

//-------------------------------------------------------------------------------
// CRT
//-------------------------------------------------------------------------------
template<int N_>
void crt1(const uint32_t q[1], const uint32_t a[1][N_], double pt[N_]) {
	for (int i = 0;i < N_;i++) {
		uint64_t sum = a[0][i];
		pt[i] = (sum >= (q[0] >> 1)) ? -(double)(q[0] - sum) : (double)(sum);
	}
}

template<int N_>
void crt2(const uint32_t q[2], const uint32_t a[2][N_], double pt[N_]) {
	uint32_t q0inv = inv_mod(q[0], q[1]), q0invS = Shoup_inverse(q0inv, q[1]);
	uint32_t q1inv = inv_mod(q[1], q[0]), q1invS = Shoup_inverse(q1inv, q[0]);
	uint64_t Q = uint64_t(q[0]) * q[1];
	for (int i = 0;i < N_;i++) {
		uint32_t b0 = mul_mod(a[0][i], q1inv, q1invS, q[0]);
		uint32_t b1 = mul_mod(a[1][i], q0inv, q0invS, q[1]);
		uint64_t sum = uint64_t(b0) * q[1] + uint64_t(b1) * q[0];
		if (sum >= Q) sum -= Q;
		pt[i] = (sum >= (Q >> 1)) ? -(double)(Q - sum) : (double)(sum);
	}
}


template<int N_>
void crt3(const uint32_t q[3], const uint32_t a[3][N_], double pt[N_]) {
	uint32_t q0inv = inv_mod(mul_mod_worst(q[1], q[2], q[0]), q[0]), q0invS = Shoup_inverse(q0inv, q[0]);
	uint32_t q1inv = inv_mod(mul_mod_worst(q[2], q[0], q[1]), q[1]), q1invS = Shoup_inverse(q1inv, q[1]);
	uint32_t q2inv = inv_mod(mul_mod_worst(q[0], q[1], q[2]), q[2]), q2invS = Shoup_inverse(q2inv, q[2]);
	uint128_t Q = uint128_t(uint64_t(q[0]) * q[1]) * uint64_t(q[2]);

	for (int i = 0;i < N_;i++) {
		uint32_t b0 = mul_mod(a[0][i], q0inv, q0invS, q[0]);
		uint32_t b1 = mul_mod(a[1][i], q1inv, q1invS, q[1]);
		uint32_t b2 = mul_mod(a[2][i], q2inv, q2invS, q[2]);
		uint128_t A0 = uint128_t(uint64_t(q[1]) * q[2]) * uint64_t(b0);
		uint128_t A1 = uint128_t(uint64_t(q[2]) * q[0]) * uint64_t(b1);
		uint128_t A2 = uint128_t(uint64_t(q[0]) * q[1]) * uint64_t(b2);
		uint128_t sum = A0 + A1 + A2;
		if (sum >= Q) sum = sum - Q;
		if (sum >= Q) sum = sum - Q;
		if (sum >= Q) sum = sum - Q;
		pt[i] = (sum >= (Q >> 1)) ? -(double)(Q - sum) : (double)(sum);
	}
}

template<int N_>
void crt4(const uint32_t q[4], const uint32_t a[4][N_], double pt[N_]) {
	uint32_t q0inv = inv_mod(mul_mod_worst(q[1], mul_mod_worst(q[2], q[3], q[0]), q[0]), q[0]), q0invS = Shoup_inverse(q0inv, q[0]);
	uint32_t q1inv = inv_mod(mul_mod_worst(q[2], mul_mod_worst(q[3], q[0], q[1]), q[1]), q[1]), q1invS = Shoup_inverse(q1inv, q[1]);
	uint32_t q2inv = inv_mod(mul_mod_worst(q[3], mul_mod_worst(q[0], q[1], q[2]), q[2]), q[2]), q2invS = Shoup_inverse(q2inv, q[2]);
	uint32_t q3inv = inv_mod(mul_mod_worst(q[0], mul_mod_worst(q[1], q[2], q[3]), q[3]), q[3]), q3invS = Shoup_inverse(q3inv, q[3]);
	uint128_t Q = uint128_t(uint64_t(q[0]) * q[1]) * (uint64_t(q[2]) * q[3]);

	for (int i = 0;i < N_;i++) {
		uint32_t b0 = mul_mod(a[0][i], q0inv, q0invS, q[0]);
		uint32_t b1 = mul_mod(a[1][i], q1inv, q1invS, q[1]);
		uint32_t b2 = mul_mod(a[2][i], q2inv, q2invS, q[2]);
		uint32_t b3 = mul_mod(a[3][i], q3inv, q3invS, q[3]);
		uint128_t A0 = uint128_t(uint64_t(q[1]) * q[2]) * (uint64_t(q[3]) * b0);
		uint128_t A1 = uint128_t(uint64_t(q[2]) * q[3]) * (uint64_t(q[0]) * b1);
		uint128_t A2 = uint128_t(uint64_t(q[3]) * q[0]) * (uint64_t(q[1]) * b2);
		uint128_t A3 = uint128_t(uint64_t(q[0]) * q[1]) * (uint64_t(q[2]) * b3);
		uint128_t sum = A0 + A1 + A2 + A3;
		if (sum >= Q) sum = sum - Q;
		if (sum >= Q) sum = sum - Q;
		if (sum >= Q) sum = sum - Q;
		if (sum >= Q) sum = sum - Q;
		pt[i] = (sum >= (Q >> 1)) ? -(double)(Q - sum) : (double)(sum);
	}
}




/*
template<int N_>
void crt6(const uint32_t q[6], const uint32_t a[6][N_], double pt[N_]) {
	uint32_t qiinv[6], qiinvS[6]; uint256_t Qi[6], Q;
	for (int i = 0;i < 6;i++) {
		qiinv[i] = 1; Qi[i].lo = 1;
		for (int j = 0;j < 6;j++)
			if (j != i) {
				qiinv[i] = mul_mod_worst(qiinv[i], q[j], q[i]);
				Qi[i] = Qi[i] * q[j];
			}
		//Qi[i].print();
		qiinv[i] = inv_mod(qiinv[i], q[i]);
		qiinvS[i] = Shoup_inverse(qiinv[i], q[i]);
	}
	Q = Qi[0] * q[0]; //Q.print();
	for (int i = 0;i < N_;i++) {
		uint32_t b[6]; uint256_t sum;
		for (int j = 0;j < 6;j++) {
			b[j] = mul_mod(a[j][i], qiinv[j], qiinvS[j], q[j]);
			sum = sum + Qi[j] * b[j];
			if (sum >= Q) sum = sum - Q;
		}
		//sum.print();
		pt[i] = (sum >= (Q >> 1)) ? -(double(Q - sum)) : double(sum);
	}
}

*/