#pragma once

#include <cstring>
#include "NumberTheory.cuh"
#include "Parameter.cuh"


struct NTT {
	uint32_t p, NInv, Psi[N], PsiInv[N];
	uint32_t    NInvShoup, PsiShoup[N], PsiInvShoup[N];

	void bitReverse(uint32_t a[N]) {
		uint32_t temp[N]; for (int i = 0;i < N;i++) temp[i] = a[i];
		for (int i = 0;i < N;i++) {
			int j = 0;
			for (int k = 0;k < logN;k++)
				j += ((i >> k) & 1) << (logN - 1 - k);
			a[j] = temp[i];
		}
	}
	void initialize(uint32_t p, uint32_t prim) {
		this->p = p;
		Psi[0] = 1; Psi[1] = power_mod(prim, (p - 1) / (2 * N), p);
		PsiInv[0] = 1; PsiInv[1] = inv_mod(Psi[1], p);
		for (int i = 2;i < N;i++) {
			PsiShoup[i - 1] = Shoup_inverse(Psi[i - 1], p);
			PsiInvShoup[i - 1] = Shoup_inverse(PsiInv[i - 1], p);
			Psi[i] = mul_mod(Psi[1], Psi[i - 1], PsiShoup[i - 1], p);
			PsiInv[i] = mul_mod(PsiInv[1], PsiInv[i - 1], PsiInvShoup[i - 1], p);
		}
		PsiShoup[N - 1] = Shoup_inverse(Psi[N - 1], p);
		PsiInvShoup[N - 1] = Shoup_inverse(PsiInv[N - 1], p);
		NInv = inv_mod(N, p);
		NInvShoup = Shoup_inverse(NInv, p);
		bitReverse(Psi);
		bitReverse(PsiInv);
		bitReverse(PsiShoup);
		bitReverse(PsiInvShoup);
	}

	void ntt(uint32_t a[N]) const {
		for (int m = 1, t = N / 2;m < N;m *= 2, t /= 2) {
			for (int i = 0;i < m;i++) {
				uint32_t* ae = a + 2 * i * t;
				uint32_t* ao = ae + t;
				for (int k = 0;k < t;k++) {
					uint32_t U = mul_mod(ao[k], Psi[m + i], PsiShoup[m + i], p);
					ao[k] = ae[k] + p - U; if (ao[k] >= p) ao[k] -= p;
					ae[k] = ae[k] + U; if (ae[k] >= p) ae[k] -= p;
				}
			}
		}
	}
	void intt(uint32_t a[N]) const {
		for (int t = 1, m = N / 2;m > 0;t *= 2, m /= 2) {
			for (int i = 0;i < m;i++) {
				uint32_t* ae = a + 2 * i * t;
				uint32_t* ao = ae + t;
				for (int k = 0;k < t;k++) {
					uint32_t U = (ae[k] + p - ao[k]); if (U >= p) U -= p;
					ae[k] = ae[k] + ao[k]; if (ae[k] >= p) ae[k] -= p;
					ao[k] = mul_mod(U, PsiInv[m + i], PsiInvShoup[m + i], p);
				}
			}
		}
		for (int i = 0;i < N;i++)
			a[i] = mul_mod(a[i], NInv, NInvShoup, p);
	}

};

