#pragma once

#include <math.h>
#include <assert.h>
#include <cstdint>
#include "CKKS_debug.cuh"
#include "Parameter.cuh"
#include "NTT.cuh"
#include "FFT.cuh"
#include "SWK.cuh"
#include "NTT.cuh"



//-------------------------------------------------------------------------------------------------
//
//  CKKS_HalfPrecision
//
//-------------------------------------------------------------------------------------------------
struct CKKS_HalfPrecision {
	//=======================================================================
	//  Parameter setting
	//=======================================================================
	FFT fft;
	NTT ntt[L + G + K];
	uint32_t qp[L + G + K];
	double qp_bits[L + G + K];
	int g[DNUM][L + G + K];
	int h; double Delta; int Reverse[N];
	int s[N];
	//=======================================================================
	//  graft
	//=======================================================================
	uint32_t q_graft[3];
	NTT ntt_graft[3];
	void graft_embed(const uint32_t a[N], int alpha, uint32_t b[3][N]) const;
	void graft_modulus( uint32_t b[3][N], uint32_t a[N], int alpha) const;
	void conv_graft(const uint32_t pt1[N], const uint32_t pt2[N], int alpha, uint32_t out[N]) const;
	void convhatQ_convgraft (const Ptext    pt1, const Ptext    pt2, int l, int alpha, Ptext    out)const;
	void convhatQP_convgraft(const Ptext_QP pt1, const Ptext_QP pt2, int l, int alpha, Ptext_QP out)const;
	//=======================================================================
	//  initialization
	//=======================================================================
	CKKS_HalfPrecision(int s[N]);

	//=======================================================================
	//  Elementary Operations : encode / decode / encrypt / decrypt
	//=======================================================================
	void encode   (const double z[N], double scale, int l, int alpha, uint32_t pt[L + G    ][N]) const;
	void encode_QP(const double z[N], double scale, int l, int alpha, uint32_t pt[L + G + K][N]) const;
	void decode      (const uint32_t pt[L + G][N], int l, double scale, double z[N]) const;
	void decode_graft(const uint32_t pt[L + G][N], int l, int alpha, double scale, double z[N]) const;
	void enc   (const uint32_t pt[L + G][N], int l, int alpha, const int sk[N], uint32_t cthat[2][L + G][N]) const;
	void enc_QP(const Ptext_QP pt, int l, int alpha, const int sk[N], Ctext_QP cthat) const;
	void dec   (const uint32_t ct[2][L + G][N], int l, int alpha, const int sk[N], uint32_t pt[L + G][N]) const;
	void dec_decode      (const uint32_t ct[2][L + G][N], int l,            const int sk[N], double scale, double z[N]) const;
	void dec_decode_graft(const uint32_t ct[2][L + G][N], int l, int alpha, const int sk[N], double scale, double z[N]) const;
	
	//=======================================================================
	//  add / sub
	//=======================================================================
	void add   (const uint32_t a[L+G  ][N], const uint32_t b[L+G  ][N], int l, int alpha, uint32_t sum[L+G  ][N])const;
	void add   (const uint32_t a[2][L+G  ][N], const uint32_t b[2][L+G  ][N], int l, int alpha, uint32_t sum[2][L+G  ][N])const;
	void add_QP(const uint32_t a[L+G+K][N], const uint32_t b[L+G+K][N], int l, int alpha, uint32_t sum[L+G+K][N])const;
	void sub   (const uint32_t a[L+G  ][N], const uint32_t b[L+G  ][N], int l, int alpha, uint32_t sum[L+G  ][N])const;
	void sub_QP(const uint32_t a[L+G+K][N], const uint32_t b[L+G+K][N], int l, int alpha, uint32_t sum[L+G+K][N])const;

	//=======================================================================
	//  Key operations
	//=======================================================================
	void sk_gen(int s[N]);
	void evk_gen(int s[N]);
	void ckey_gen(int s[N]);
	SWK evk, ckey, rkey;

	void swkgen(const int sfr[N], const int sto[N], SWK& swk) const;

	void gadget_ginv(Ptext a, int l, int alpha, Ptext_QP ginva[DNUM]) const;
	void decompose(const Ptext ct_hat1, int l, int alpha, Ptext_QP ginva[DNUM]) const;
	void mult_sum(const Ptext_QP ginva[DNUM], int l, int alpha, const SWK& swk, Ctext_QP mulsum) const;
	void mult_sum(const Ptext_QP ginva[DNUM], int l, int alpha, Ptext_QP mulsum) const;
	void mulP(const Ptext ct_hat, int l, int alpha, Ptext_QP Pct_hat) const;
	void ks(const Ctext ct_hat, int l, int alpha, const SWK& swk, Ctext out_hat)const;

	std::string name_rkeytilde(int shift) const {
		if (shift < N / 4) return "./data/rkeytilde+" + std::to_string(shift) + ".dat";
		else           return "./data/rkeytilde-" + std::to_string(N / 2 - shift) + ".dat";
	}
	void calculate_rkeytilde(int shift, const int s[N]) const;
	void rot(const int s[N], int r, int srot[N]) const;
	SWK& get_rkeytilde(int shift) {
		std::string name = name_rkeytilde(shift);
		if (rkey.check_exist(name.c_str()) == false) {
			calculate_rkeytilde(shift, s);
		}

		rkey.load(name.c_str());
		return rkey;
	}
	//=======================================================================
	//  RS operations
	//=======================================================================
	void rshat    (Ptext pthat, int l, int alpha, double scale    , int alpha_rs, int& lout, int& alpha_out, double& scale_out    ) const;
	void rshat_log(Ptext pthat, int l, int alpha, double scale_log, int alpha_rs, int& lout, int& alpha_out, double& scale_out_log) const;

	void rshat_Ql_Qlmm(Ptext pthat, int l, int alpha, int m) const;
	void rshat_PQl_Ql (Ptext_QP pthat, int l, int alpha) const;

	void rshat_sim(int l, int alpha, double scale, int alpha_rs, int& lout, int& alpha_out, double& scale_out) const;

	//=======================================================================
	//  mul, mul_rs
	//=======================================================================
	void mul   (const Ctext ct1, const Ctext ct2, int l, int alpha, Ctext out) const;
	void mul_rs(const Ctext ct1, int l1, int alpha1, double scale_1,
		        const Ctext ct2, int l2, int alpha2, double scale_2, int rs_bits,
		              Ctext out, int& l_out, int& alpha_out, double& scale_out) const;
	void mul_rs_sim(int l1, int alpha1, double scale_1,
	                int l2, int alpha2, double scale_2, int rs_bits,
	                int& l_out, int& alpha_out, double& scale_out) const;
	//=======================================================================
	//  Rot_hat , Conj_hat
	//=======================================================================
	int get_logN(int N_) const;
	uint32_t powerfive(int r, uint32_t N_) const;

	void rot(const uint32_t pt[L + G][N], int l, int alpha, int r, uint32_t pt_rot[L + G][N]) const;
	void rothat(const uint32_t cthat[L + G][N], int l, int alpha, int r, uint32_t cthat_rot[L + G][N]) const;
	void rothat(const uint32_t cthat[2][L + G][N], int l, int alpha, int r, uint32_t cthat_rot[2][L + G][N]) const;
	void rothat_QP(const uint32_t cthat[L + G + K][N], int l, int alpha, int r, uint32_t cthat_rot[L + G + K][N]) const;
	void rothat_QP(const uint32_t cthat[2][L + G + K][N], int l, int alpha, int r, uint32_t cthat_rot[2][L + G + K][N]) const;


	void rothat(const uint32_t cthat[2][L + G][N], int l, int alpha, int shift, uint32_t cthat_rot[2][L + G][N], int s[N]) const;
	void conj_hat(const uint32_t cthat[2][L + G][N], int l, int alpha, uint32_t cthat_conj[2][L + G][N]) const;

	//=======================================================================
	//  debug
	//=======================================================================
	void save_z_graft(Ctext ct, int l, int alpha, double scale, const char* name) const {
		printf("saving : %s with l=%d, alpha=%d and scale=%f\n", name, l, alpha, log(scale) / log(2));
		double z_out[N]; dec_decode_graft(ct, l, alpha, s, scale, z_out);
		save(z_out, name);
	}

	void save_zBR_graft(Ctext ct, int l, int alpha, double scale, const char* name) const {
		printf("saving : %s with l=%d, alpha=%d and scale=%f\n", name, l, alpha, log(scale) / log(2));
		double z_out[N]; dec_decode_graft(ct, l, alpha, s, scale, z_out);
		fft.bitReverse(N / 2, z_out);
		fft.bitReverse(N / 2, z_out + N / 2);
		save(z_out, name);
	}

	void save_pt_graft(Ctext ct, int l, int alpha, double scale, const char* name) const {
		printf("saving : %s with l=%d, alpha=%d and scale=%f\n", name, l, alpha, log(scale) / log(2));
		double z_out[N]; dec_decode_graft(ct, l, alpha, s, scale, z_out);
		fft.bitReverse(N / 2, z_out);
		fft.bitReverse(N / 2, z_out + N / 2);
		fft.ifft(z_out);
		save(z_out, name);
	}

	void save_z_graft_and_debug(Ctext ct, int l, int alpha, double scale, const char* name, double z_ex[N]) const {
		printf("saving : %s with l=%d, alpha=%d and scale=%f\n", name, l, alpha, log(scale) / log(2));
		double z_out[N]; dec_decode_graft(ct, l, alpha, s, scale, z_out);
		save(z_out, name);

		printf("%s error = %e\n", name, L2_sqr(z_ex, z_out));
	}
};
