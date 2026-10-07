#pragma once

#include <math.h>
#include "CKKS_HalfPrecision.cuh"
//#include "CKKS_debug.cuh"


struct CKKS_polynomial {
	//=======================================================================
	//  Initialize
	//=======================================================================
	const CKKS_HalfPrecision& ckks;
	CKKS_polynomial(const CKKS_HalfPrecision& ckks_input) :ckks(ckks_input) {}

	//=======================================================================
	//  basic
	//=======================================================================
	void encode_constant(double c, int l, int alpha, double scale, uint32_t pt[L + G]) const;
	void mul_constant(Ctext ct, int l, int alpha, double c, double scale_c) const;
	void add_constant(Ctext ct, int l, int alpha, double c, double scale_c) const;
	void increase_alpha(Ctext ct, int l, int alpha, double scale, int alpha_out, int& l_out, double& scale_out) const;

	//=======================================================================
	//  sum
	//=======================================================================
	void sum2(double c[2], Ctext T1, int l, int alpha, double scale, int rs_bits, double scale_out,
		                   Ctext out, int& l_out, int& alpha_out);
	void sum2_sim(int l, int alpha, int rs_bits, int& l_out, int& alpha_out);
	void sum4(double c[4], Ctext T1, int l1, int alpha1, double scale1,
                           Ctext T2, int l2, int alpha2, double scale2, 
                           Ctext T3, int l3, int alpha3, double scale3, int rs_bits, double scale_out,
		                   Ctext out, int& l_out, int& alpha_out) ;
	void sum4_sim(int l1, int alpha1, int l2, int alpha2, int l3, int alpha3, int rs_bits, 
		          int& l_out, int& alpha_out);
	void sum8(double c[8], Ctext T1, int l1, int alpha1, double scale1,
                           Ctext T2, int l2, int alpha2, double scale2, 
                           Ctext T3, int l3, int alpha3, double scale3, 
		                   Ctext T4, int l4, int alpha4, double scale4, 
                           Ctext T5, int l5, int alpha5, double scale5,
		                   Ctext T6, int l6, int alpha6, double scale6, 
                           Ctext T7, int l7, int alpha7, double scale7,	int rs_bits, double scale_out,
		                   Ctext out, int& l_out, int& alpha_out);
	void sum8_sim(int l1, int alpha1,
                  int l2, int alpha2, 
                  int l3, int alpha3, 
		          int l4, int alpha4, 
                  int l5, int alpha5,
		          int l6, int alpha6, 
                  int l7, int alpha7, int rs_bits, int& l_out, int& alpha_out);

	
	//=======================================================================
	//  Chebyshev basis
	//=======================================================================
	void Chebyshev_Doubling(const uint32_t T [2][L+G][N], int  l , int  alpha , double  scale , int rs_bits,
	                              uint32_t Td[2][L+G][N], int& ld, int& alphad, double& scaled               ) const;
	void Chebyshev_General(const uint32_t Ta[2][L+G][N], int  la, int  alphaa, double  scalea,
						   const uint32_t Tb[2][L+G][N], int  lb, int  alphab, double  scaleb,
						   const uint32_t Tc[2][L+G][N], int  lc, int  alphac, double  scalec, int rs_bits,
						   	     uint32_t Td[2][L+G][N], int& ld, int& alphad, double& scaled              ) const;

	void Chebyshev_Doubling_sim(int  l , int  alpha , double  scale , int rs_bits,
	                            int& ld, int& alphad, double& scaled               ) const;
	void Chebyshev_General_sim( int  la, int  alphaa, double  scalea,
						        int  lb, int  alphab, double  scaleb,
						        int  lc, int  alphac, double  scalec, int rs_bits,
						   	    int& ld, int& alphad, double& scaled              ) const;

	//=======================================================================
	//  Binary Merge
	//=======================================================================
	void BinaryMerge(const Ctext Left , int  lL  , int  alphaL  , double  scaleL  ,
	                 const Ctext Right, int  lR  , int  alphaR  , double  scaleR  ,
	                 const Ctext T    , int  lT  , int  alphaT  , double  scaleT  , int rs_bits,
	                       Ctext sum  , int& lsum, int& alphasum, double& scalesum	) const;

	void BinaryMerge_sim(int  lL  , int  alphaL  ,
	                     int  lR  , int  alphaR  ,
	                     int  lT  , int  alphaT  , int rs_bits,
	                     int& lsum, int& alphasum              ) const;

	void BinaryMerge_split(int lL  , int alphaL  , double& scaleL  ,
	                       int lR  , int alphaR  , double& scaleR  ,
	                       int lT  , int alphaT  , double  scaleT  , int rs_bits,
	                                               double  scalesum	              ) const;

	//=======================================================================
	//  Evalpoly
	//=======================================================================
	void evalpoly16      (double c[16], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, Ctext res, int& lres, int& alphares, double scaleres          );
	void evalpoly16_Debug(double c[16], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, Ctext res, int& lres, int& alphares, double scaleres, int s[N]);
	void evalpoly64_Debug(double c[64], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, Ctext res, int& lres, int& alphares, double scaleres, double zres_ex[N], int s[N]);
	void evalpoly64_sim  (                        int l1, int alpha1, double scale1, int rs_bits,            int& lres, int& alphares, double scaleres                             );

	//=======================================================================
	//  Debug
	//=======================================================================
	void Chebyshev_Doubling_Debug(const double T[N], double Td[N]) const;
	void Chebyshev_General_Debug(const double Ta[N], const double Tb[N], const double Tc[N], double Td[N]) const;
	void sum2_Debug(double coeff[2], const double T1[N], double sum[N]) const;
	void sum4_Debug(double coeff[4], const double T1[N],
		                            const double T2[N],
		                            const double T3[N], double sum[N]) const;
	void sum8_Debug(double coeff[8], const double T1[N],
	                                const double T2[N],
	                                const double T3[N],
	                                const double T4[N],
	                                const double T5[N],
	                                const double T6[N],
	                                const double T7[N], double sum[N]) const;
	void BinaryMerge_Debug(const double Left[N],
		                   const double Right[N],
		                   const double T[N], double sum[N]) const;

	//=======================================================================
	//  Double angle
	//=======================================================================
	void Double_angle_formula( Ctext g_i, int  l_g_i, int  alpha_g_i, double  delta_g_i, double c, int rs_bits,
		                       Ctext g_ip1, int& l_g_ip1, int& alpha_g_ip1, double& delta_g_ip1);
	void Double_angle_formula_Debug( double g_i[N], double c, double g_ip1[N]);
}; 




//=======================================================================
//  basic operation
//=======================================================================
void CKKS_polynomial::encode_constant(double c, int l, int alpha, double scale, uint32_t pt[L + G]) const {
	uint64_t c_scaled = uint64_t((c < 0 ? -c : c) * scale + double(0.5));
	//
	for (int j = 0; j < l; j++) {
		pt[j] = uint32_t(c_scaled % ckks.qp[j]);
		if (c < 0) pt[j] = ckks.qp[j] - pt[j];
	}
	//
	if (alpha > 0) {
		uint32_t q = uint32_t(1) << alpha;
		uint32_t ptG = uint32_t(c_scaled & (q - 1));
		if (c < 0 && ptG != 0)	ptG = q - ptG;
		pt[L] = ptG;
	}
}

void CKKS_polynomial::mul_constant(Ctext ct, int l, int alpha, double c, double scale_c) const {
	uint32_t pt_c[L + G];
	encode_constant(c, l, alpha, scale_c, pt_c);
	for (int j = 0;j < l;j++) {
		mul_mod_type6<N>(ct[0][j], pt_c[j], ct[0][j], ckks.qp[j]);
		mul_mod_type6<N>(ct[1][j], pt_c[j], ct[1][j], ckks.qp[j]);
	}
	if (alpha > 0) {
		uint32_t p = uint32_t(1) << alpha;
		for (int i = 0;i < N;i++) {
			ct[0][L][i] = (ct[0][L][i] * pt_c[L]) & (p - 1);
			ct[1][L][i] = (ct[1][L][i] * pt_c[L]) & (p - 1);
		}
	}
}

void CKKS_polynomial::add_constant(Ctext ct, int l, int alpha, double c, double scale_c) const {
	uint32_t pt_c[L + G];
	encode_constant(c, l, alpha, scale_c, pt_c);
	for (int j = 0;j < l;j++) 
		add_mod<N>(ct[0][j], pt_c[j], ct[0][j], ckks.qp[j]);
		
	if (alpha > 0) {
		uint32_t p = uint32_t(1) << alpha;
		ct[0][L][0] = (ct[0][L][0] + pt_c[L]) & (p - 1);
	}
}

void CKKS_polynomial::increase_alpha(Ctext ct, int l, int alpha, double scale, int alpha_out, int& l_out, double& scale_out) const {
	const uint32_t* q = ckks.qp;
	assert(alpha < alpha_out);
	uint32_t p = uint32_t(1) << 30;
	for (int j = 0;j < l;j++) {
		mul_mod_type6<N>(ct[0][j], p % q[j], ct[0][j], q[j]);
		mul_mod_type6<N>(ct[1][j], p % q[j], ct[1][j], q[j]);
	}
	memset(ct[0][L], 0, sizeof(uint32_t) * N);
	memset(ct[1][L], 0, sizeof(uint32_t) * N);
	ckks.rshat_Ql_Qlmm(ct[0], l, alpha_out, 1);
	ckks.rshat_Ql_Qlmm(ct[1], l, alpha_out, 1);
	l_out = l - 1;
	scale_out = scale * double(p) / q[l - 1];
}

//=======================================================================
//  sum
//=======================================================================
void CKKS_polynomial::sum2(double c[2], Ctext T1, int l, int alpha, double scale, int rs_bits, double scale_out,
						                Ctext out, int& l_out, int& alpha_out) {
		
	double scale_sim;
	ckks.rshat_sim(l, alpha, scale * scale, rs_bits, l_out, alpha_out, scale_sim);

	memcpy(out, T1, sizeof(uint32_t)*2*(L+G)*N);
	mul_constant(out, l, alpha, c[1], scale * scale_out / scale_sim);
	double scale_check;
	ckks.rshat(out[0], l, alpha, scale * scale * scale_out / scale_sim, rs_bits, l_out, alpha_out, scale_check);
	ckks.rshat(out[1], l, alpha, scale * scale * scale_out / scale_sim, rs_bits, l_out, alpha_out, scale_check);

	add_constant(out, l_out, alpha_out, c[0], scale_out);

	//printf("in Sum2 : scale_out = %e, scale_check = %e \n", scale_out, scale_check);
}

	
void CKKS_polynomial::sum2_sim(int l, int alpha, int rs_bits, int& l_out, int& alpha_out) {
		
	double scale_sim=1;
	ckks.rshat_sim(l, alpha, scale_sim, rs_bits, l_out, alpha_out, scale_sim);
}

void CKKS_polynomial::sum4(double c[4], Ctext T1, int l1, int alpha1, double scale1,
                                        Ctext T2, int l2, int alpha2, double scale2, 
                                        Ctext T3, int l3, int alpha3, double scale3, int rs_bits, double scale_out,
		                                Ctext out, int& l_out, int& alpha_out) {
	int     l123 = (    l1 <     l2) ?     l1 :     l2;     l123 = (    l123 <     l3) ?     l123 :     l3;
	int alpha123 = (alpha1 < alpha2) ? alpha1 : alpha2; alpha123 = (alpha123 < alpha3) ? alpha123 : alpha3;
		
	double scale_sim1; ckks.rshat_sim(l123, alpha123, scale1 * scale1, rs_bits, l_out, alpha_out, scale_sim1);
	double scale_sim2; ckks.rshat_sim(l123, alpha123, scale2 * scale2, rs_bits, l_out, alpha_out, scale_sim2);
	double scale_sim3; ckks.rshat_sim(l123, alpha123, scale3 * scale3, rs_bits, l_out, alpha_out, scale_sim3);

	memcpy(out, T1, sizeof(uint32_t) * 2 * (L + G) * N);
	mul_constant(out, l123, alpha123, c[1], scale1 * scale_out / scale_sim1);
	//
	Ctext temp; memcpy(temp, T2, sizeof(uint32_t) * 2 * (L + G) * N); 
	mul_constant(temp, l123, alpha123, c[2], scale2 * scale_out / scale_sim2);
	ckks.add(temp[0], out[0], l123, alpha123, out[0]);
	ckks.add(temp[1], out[1], l123, alpha123, out[1]);
	//
	memcpy(temp, T3, sizeof(uint32_t) * 2 * (L + G) * N);
	mul_constant(temp, l123, alpha123, c[3], scale3 * scale_out / scale_sim3);
	ckks.add(temp[0], out[0], l123, alpha123, out[0]);
	ckks.add(temp[1], out[1], l123, alpha123, out[1]);
	//
	double scale_check;
	ckks.rshat(out[0], l123, alpha123, scale1 * scale1 * scale_out / scale_sim1, rs_bits, l_out, alpha_out, scale_check);
	ckks.rshat(out[1], l123, alpha123, scale1 * scale1 * scale_out / scale_sim1, rs_bits, l_out, alpha_out, scale_check);
		
	add_constant(out, l_out, alpha_out, c[0], scale_out);

	//printf("in Sum4 : scale_out = %e, scale_check = %e \n", scale_out, scale_check);
}

	
void CKKS_polynomial::sum4_sim(int l1, int alpha1, int l2, int alpha2, int l3, int alpha3, int rs_bits,
		                       int& l_out, int& alpha_out) {

	int     l123 = (    l1 <     l2) ?     l1 :     l2;     l123 = (    l123 <     l3) ?     l123 :     l3;
	int alpha123 = (alpha1 < alpha2) ? alpha1 : alpha2; alpha123 = (alpha123 < alpha3) ? alpha123 : alpha3;
		
	double scale_sim1=1; ckks.rshat_sim(l123, alpha123, scale_sim1, rs_bits, l_out, alpha_out, scale_sim1);
}

void CKKS_polynomial::sum8(double c[8], Ctext T1, int l1, int alpha1, double scale1,
                                        Ctext T2, int l2, int alpha2, double scale2, 
                                        Ctext T3, int l3, int alpha3, double scale3, 
		                                Ctext T4, int l4, int alpha4, double scale4, 
                                        Ctext T5, int l5, int alpha5, double scale5,
		                                Ctext T6, int l6, int alpha6, double scale6, 
                                        Ctext T7, int l7, int alpha7, double scale7,	int rs_bits, double scale_out,
		                                Ctext out, int& l_out, int& alpha_out) {
	int lmin = (l1 <l2) ? l1 : l2;
	if (lmin > l3) lmin = l3;
	if (lmin > l4) lmin = l4;
	if (lmin > l5) lmin = l5;
	if (lmin > l6) lmin = l6;
	if (lmin > l7) lmin = l7;
		
	int alphamin = (alpha1 < alpha2) ? alpha1 : alpha2; 
	if (alphamin > alpha3) alphamin = alpha3;
	if (alphamin > alpha4) alphamin = alpha4;
	if (alphamin > alpha5) alphamin = alpha5;
	if (alphamin > alpha6) alphamin = alpha6;
	if (alphamin > alpha7) alphamin = alpha7;
		
	double scale_sim1; ckks.rshat_sim(lmin, alphamin, scale1 * scale1, rs_bits, l_out, alpha_out, scale_sim1);
	double scale_sim2; ckks.rshat_sim(lmin, alphamin, scale2 * scale2, rs_bits, l_out, alpha_out, scale_sim2);
	double scale_sim3; ckks.rshat_sim(lmin, alphamin, scale3 * scale3, rs_bits, l_out, alpha_out, scale_sim3);
	double scale_sim4; ckks.rshat_sim(lmin, alphamin, scale4 * scale4, rs_bits, l_out, alpha_out, scale_sim4);
	double scale_sim5; ckks.rshat_sim(lmin, alphamin, scale5 * scale5, rs_bits, l_out, alpha_out, scale_sim5);
	double scale_sim6; ckks.rshat_sim(lmin, alphamin, scale6 * scale6, rs_bits, l_out, alpha_out, scale_sim6);
	double scale_sim7; ckks.rshat_sim(lmin, alphamin, scale7 * scale7, rs_bits, l_out, alpha_out, scale_sim7);

	memcpy(out, T1, sizeof(uint32_t) * 2 * (L + G) * N);
	mul_constant(out, lmin, alphamin, c[1], scale1 * scale_out / scale_sim1);
	//
	Ctext temp; 
	memcpy(temp, T2, sizeof(uint32_t) * 2 * (L + G) * N);  mul_constant(temp, lmin, alphamin, c[2], scale2 * scale_out / scale_sim2); ckks.add(temp[0], out[0], lmin, alphamin, out[0]); ckks.add(temp[1], out[1], lmin, alphamin, out[1]);
	memcpy(temp, T3, sizeof(uint32_t) * 2 * (L + G) * N);  mul_constant(temp, lmin, alphamin, c[3], scale3 * scale_out / scale_sim3); ckks.add(temp[0], out[0], lmin, alphamin, out[0]); ckks.add(temp[1], out[1], lmin, alphamin, out[1]);
	memcpy(temp, T4, sizeof(uint32_t) * 2 * (L + G) * N);  mul_constant(temp, lmin, alphamin, c[4], scale4 * scale_out / scale_sim4); ckks.add(temp[0], out[0], lmin, alphamin, out[0]); ckks.add(temp[1], out[1], lmin, alphamin, out[1]);
	memcpy(temp, T5, sizeof(uint32_t) * 2 * (L + G) * N);  mul_constant(temp, lmin, alphamin, c[5], scale5 * scale_out / scale_sim5); ckks.add(temp[0], out[0], lmin, alphamin, out[0]); ckks.add(temp[1], out[1], lmin, alphamin, out[1]);
	memcpy(temp, T6, sizeof(uint32_t) * 2 * (L + G) * N);  mul_constant(temp, lmin, alphamin, c[6], scale6 * scale_out / scale_sim6); ckks.add(temp[0], out[0], lmin, alphamin, out[0]); ckks.add(temp[1], out[1], lmin, alphamin, out[1]);
	memcpy(temp, T7, sizeof(uint32_t) * 2 * (L + G) * N);  mul_constant(temp, lmin, alphamin, c[7], scale7 * scale_out / scale_sim7); ckks.add(temp[0], out[0], lmin, alphamin, out[0]); ckks.add(temp[1], out[1], lmin, alphamin, out[1]);
	//
	double scale_check;
	ckks.rshat(out[0], lmin, alphamin, scale1 * scale1 * scale_out / scale_sim1, rs_bits, l_out, alpha_out, scale_check);
	ckks.rshat(out[1], lmin, alphamin, scale1 * scale1 * scale_out / scale_sim1, rs_bits, l_out, alpha_out, scale_check);
		
	add_constant(out, l_out, alpha_out, c[0], scale_out);

	printf("in Sum8 : scale_out = %e, scale_check = %e \n", scale_out, scale_check);
}

void CKKS_polynomial::sum8_sim(int l1, int alpha1,
                               int l2, int alpha2, 
                               int l3, int alpha3, 
		                       int l4, int alpha4, 
                               int l5, int alpha5,
		                       int l6, int alpha6, 
                               int l7, int alpha7, int rs_bits, int& l_out, int& alpha_out) {
		
	int lmin = (l1 <l2) ? l1 : l2;
	if (lmin > l3) lmin = l3;
	if (lmin > l4) lmin = l4;
	if (lmin > l5) lmin = l5;
	if (lmin > l6) lmin = l6;
	if (lmin > l7) lmin = l7;
		
	int alphamin = (alpha1 < alpha2) ? alpha1 : alpha2; 
	if (alphamin > alpha3) alphamin = alpha3;
	if (alphamin > alpha4) alphamin = alpha4;
	if (alphamin > alpha5) alphamin = alpha5;
	if (alphamin > alpha6) alphamin = alpha6;
	if (alphamin > alpha7) alphamin = alpha7;
		
	double scale_sim1=1; ckks.rshat_sim(lmin, alphamin, scale_sim1, rs_bits, l_out, alpha_out, scale_sim1);
}





//=======================================================================
//  Chebyshev basis
//=======================================================================
void CKKS_polynomial::Chebyshev_Doubling(const Ctext T , int  l , int  alpha , double  scale , int rs_bits,
	                                           Ctext Td, int& ld, int& alphad, double& scaled               ) const {

	
	uint32_t T_mul2[2][L + G][N];
	ckks.add(T[0], T[0], l, alpha, T_mul2[0]);
	ckks.add(T[1], T[1], l, alpha, T_mul2[1]);
	//
	ckks.mul_rs(T_mul2, l, alpha, scale, T, l, alpha, scale, rs_bits, Td, ld, alphad, scaled);
	//
	uint64_t one = uint64_t(scaled + 0.5);
	for (int j = 0; j <=ld; j++) {
		if (j < ld)
			sub_mod<N>(Td[0][j], uint32_t(one%ckks.qp[j]), Td[0][j], ckks.qp[j]);
		else
		{
			uint32_t p = uint32_t(1) << alphad;
			Td[0][L][0] += p - (uint32_t(one) & (p - 1));
			Td[0][L][0] = Td[0][L][0] & (p-1);
		}
	}

}

//Td = 2*Ta*Tb-Tc
void CKKS_polynomial::Chebyshev_General(const uint32_t Ta[2][L+G][N], int  la, int  alphaa, double  scalea,
										const uint32_t Tb[2][L+G][N], int  lb, int  alphab, double  scaleb,
										const uint32_t Tc[2][L+G][N], int  lc, int  alphac, double  scalec, int rs_bits,
											  uint32_t Td[2][L+G][N], int& ld, int& alphad, double& scaled              ) const {

	uint32_t Ta_mul2[2][L + G][N];
	ckks.add(Ta[0], Ta[0], la, alphaa, Ta_mul2[0]);
	ckks.add(Ta[1], Ta[1], la, alphaa, Ta_mul2[1]);
	//
	int l_abc     = (    la <     lb) ?     la :     lb;
	int alpha_abc = (alphaa < alphab) ? alphaa : alphab;
	    l_abc     = (    l_abc<     lc) ?     l_abc :     lc;
	alpha_abc     = (alpha_abc< alphac) ? alpha_abc : alphac;

	ckks.mul(Ta_mul2, Tb, l_abc, alpha_abc, Td);
	//
	uint32_t temp[2][L + G][N];
	uint64_t one = uint64_t(scalea*scaleb/scalec + 0.5);
	for(int j=0;j<=l_abc;j++){
		if (j < l_abc) {
			mul_mod_type6<N>(Tc[0][j], one % ckks.qp[j], temp[0][j], ckks.qp[j]);
			mul_mod_type6<N>(Tc[1][j], one % ckks.qp[j], temp[1][j], ckks.qp[j]);
		}
		else {
			uint32_t p = uint32_t(1) << alpha_abc;
			uint32_t one_p = (uint32_t(one) & (p - 1));
			for(int k=0;k<2;k++)
			for(int i=0;i<N;i++)
				temp[k][L][i] = (Tc[k][L][i]* one_p)&(p-1);
		}
	}
	//
	for (int k = 0; k <2     ;k++)
	for( int j = 0; j <l_abc ;j++)
		sub_mod<N>(Td[k][j], temp[k][j], Td[k][j], ckks.qp[j]);

	for (int k = 0; k <2     ;k++)
		sub_mod<N>(Td[k][L], temp[k][L], Td[k][L], uint32_t(1) << alpha_abc);

	//
	ckks.rshat(Td[0], l_abc, alpha_abc, scalea*scaleb, rs_bits, ld, alphad, scaled);
	ckks.rshat(Td[1], l_abc, alpha_abc, scalea*scaleb, rs_bits, ld, alphad, scaled);
}

void CKKS_polynomial::Chebyshev_Doubling_sim(int  l , int  alpha , double  scale , int rs_bits,
	                                         int& ld, int& alphad, double& scaled               ) const {

	ckks.mul_rs_sim(l, alpha, scale, l, alpha, scale, rs_bits, ld, alphad, scaled);

}

//Td = 2*Ta*Tb-Tc
void CKKS_polynomial::Chebyshev_General_sim(int  la, int  alphaa, double  scalea,
										    int  lb, int  alphab, double  scaleb,
										    int  lc, int  alphac, double  scalec, int rs_bits,
										    int& ld, int& alphad, double& scaled              ) const {

	int l_abc     = (    la <     lb) ?     la :     lb;
	int alpha_abc = (alphaa < alphab) ? alphaa : alphab;
	    l_abc     = (    l_abc<     lc) ?     l_abc :     lc;
	alpha_abc     = (alpha_abc< alphac) ? alpha_abc : alphac;
	//
	ckks.rshat_sim(l_abc, alpha_abc, scalea*scaleb, rs_bits, ld, alphad, scaled);
	ckks.rshat_sim(l_abc, alpha_abc, scalea*scaleb, rs_bits, ld, alphad, scaled);
}

void CKKS_polynomial::Chebyshev_Doubling_Debug(const double z_T[N], double z_Td[N]) const {

	for (int i = 0; i < N / 2; i++) {
		z_Td[i] = double(2) * (z_T[i] * z_T[i] - z_T[i + N / 2] * z_T[i + N / 2]) - double(1);
		z_Td[i + N / 2] = double(4) * z_T[i] * z_T[i + N / 2];
	}

}

//Td = 2*Ta*Tb-Tc
void CKKS_polynomial::Chebyshev_General_Debug(const double z_Ta[N], const double z_Tb[N], const double z_Tc[N], double z_Td[N]) const {

	for (int i = 0; i < N / 2; i++) {
		z_Td[i] = double(2) * (z_Ta[i] * z_Tb[i] - z_Ta[i + N / 2] * z_Tb[i + N / 2]) - z_Tc[i];
		z_Td[i + N / 2] = double(2) * (z_Ta[i] * z_Tb[i + N / 2] + z_Ta[i + N / 2] * z_Tb[i]) - z_Tc[i + N / 2];
	}

}


//=======================================================================
//  Binary Merge
//=======================================================================
void CKKS_polynomial::BinaryMerge(const Ctext Left , int  lL  , int  alphaL  , double  scaleL  ,
	                              const Ctext Right, int  lR  , int  alphaR  , double  scaleR  ,
	                              const Ctext T    , int  lT  , int  alphaT  , double  scaleT  , int rs_bits,
	                                    Ctext sum  , int& lsum, int& alphasum, double& scalesum	) const{

	int l_RT, alpha_RT; double scale_RT;
	ckks.mul_rs(Right, lR, alphaR, scaleR, T, lT, alphaT, scaleT, rs_bits, sum, l_RT, alpha_RT, scale_RT);

	lsum = (l_RT < lL) ? l_RT : lL;
	alphasum = (alpha_RT < alphaL) ? alpha_RT : alphaL;

	ckks.add(Left[0], sum[0], lsum, alphasum, sum[0]);
	ckks.add(Left[1], sum[1], lsum, alphasum, sum[1]);
}

void CKKS_polynomial::BinaryMerge_sim(int  lL  , int  alphaL  ,
	                                  int  lR  , int  alphaR  ,
	                                  int  lT  , int  alphaT  , int rs_bits,
	                                  int& lsum, int& alphasum              ) const{

	int l_RT = (lR < lT) ? lR : lT;
	int alpha_RT = (alphaR < alphaT) ? alphaR : alphaT;

	double scale_sim=1; ckks.rshat_sim(l_RT, alpha_RT, scale_sim, rs_bits, lsum, alphasum, scale_sim);

	lsum = (lsum < lL) ? lsum : lL;
	alphasum = (alphasum < alphaL) ? alphasum : alphaL;

}

void CKKS_polynomial::BinaryMerge_split(int lL  , int alphaL  , double& scaleL  ,
	                                    int lR  , int alphaR  , double& scaleR  ,
	                                    int lT  , int alphaT  , double  scaleT  , int rs_bits,
	                                                            double  scalesum	              ) const{

	int l_RT = (lR < lT) ? lR : lT;
	int alpha_RT = (alphaR < alphaT) ? alphaR : alphaT;

	double scalesim; int lsim, alphasim;
	ckks.rshat_sim(l_RT, alpha_RT, scaleT*scaleT, rs_bits, lsim, alphasim, scalesim);

	scaleR = scaleT * scalesum / scalesim;
	scaleL = scalesum;

	}



//=======================================================================
//  Debug
//=======================================================================
void CKKS_polynomial::sum2_Debug(double coeff[2], const double T1[N], double sum[N]) const {
	for (int i = 0; i < N / 2; i++) {
		sum[i           ] =  coeff[0] + coeff[1]*T1[i    ];
		sum[i + N / 2] =                coeff[1]*T1[i+N/2];
	}
}

void CKKS_polynomial::sum4_Debug(double coeff[4], const double T1[N],
	                                             const double T2[N],
	                                             const double T3[N], double sum[N]) const {
	for (int i = 0; i < N / 2; i++) {
		sum[i           ] =  coeff[0] + coeff[1]*T1[i    ]+coeff[2]*T2[i     ]+coeff[3]*T3[i     ];
		sum[i + N / 2] =                coeff[1]*T1[i+N/2]+coeff[2]*T2[i+N /2]+coeff[3]*T3[i+N /2];
	}
}

void CKKS_polynomial::sum8_Debug(double coeff[8], const double T1[N],
	                                             const double T2[N],
	                                             const double T3[N],
	                                             const double T4[N],
	                                             const double T5[N],
	                                             const double T6[N],
	                                             const double T7[N], double sum[N]) const {
	for (int i = 0; i < N / 2; i++) {
		sum[i        ] = coeff[0] + coeff[1] * T1[i        ] + coeff[2] * T2[i        ] + coeff[3] * T3[i        ] + coeff[4] * T4[i        ] + coeff[5] * T5[i        ] + coeff[6] * T6[i        ] + coeff[7] * T7[i        ] ;
		sum[i + N / 2] =            coeff[1] * T1[i + N / 2] + coeff[2] * T2[i + N / 2] + coeff[3] * T3[i + N / 2] + coeff[4] * T4[i + N / 2] + coeff[5] * T5[i + N / 2] + coeff[6] * T6[i + N / 2] + coeff[7] * T7[i + N / 2] ;
	}
}


void CKKS_polynomial::BinaryMerge_Debug(const double Left[N],
	                                    const double Right[N],
	                                    const double T[N], double sum[N]) const {

	for (int i = 0; i < N / 2; i++) {
		sum[i           ] = T[i] * Right[i           ] - T[i + N / 2] * Right[i + N / 2] + Left[i        ];
		sum[i + N / 2] = T[i] * Right[i + N / 2] + T[i + N / 2] * Right[i           ] + Left[i+N /2];
	}

}



//=======================================================================
//  Evalpoly
//=======================================================================
void CKKS_polynomial::evalpoly16(double c[16], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, Ctext res, int& lres, int& alphares, double scaleres) {
	
	//====================================================
	// Chebyshev 
	//====================================================
	Ctext T2; int l2, alpha2; double scale2;
	Ctext T3; int l3, alpha3; double scale3;
	Ctext T4; int l4, alpha4; double scale4;
	Ctext T5; int l5, alpha5; double scale5;
	Ctext T6; int l6, alpha6; double scale6;
	Ctext T7; int l7, alpha7; double scale7;
	Ctext T8; int l8, alpha8; double scale8;

	Chebyshev_Doubling(T1, l1, alpha1, scale1, rs_bits, T2, l2, alpha2, scale2); 
	Chebyshev_Doubling(T2, l2, alpha2, scale2, rs_bits, T4, l4, alpha4, scale4); 
	Chebyshev_General(T1,l1,alpha1,scale1, T2, l2, alpha2, scale2, T1, l1, alpha1, scale1, rs_bits, T3, l3, alpha3, scale3);
	Chebyshev_General(T2,l2,alpha2,scale2, T3, l3, alpha3, scale3, T1, l1, alpha1, scale1, rs_bits, T5, l5, alpha5, scale5);
	Chebyshev_Doubling(T3,l3,alpha3,scale3, rs_bits, T6, l6, alpha6, scale6); 
	Chebyshev_General (T3,l3,alpha3,scale3,     T4, l4, alpha4, scale4, T1, l1, alpha1, scale1, rs_bits, T7, l7, alpha7, scale7);
	Chebyshev_Doubling(T4, l4, alpha4, scale4, rs_bits, T8, l8, alpha8, scale8);
	

	//====================================================
	// variable declaration
	//====================================================
	Ctext ct2_0; int l2_0, alpha2_0; double scale2_0;
	Ctext ct2_1; int l2_1, alpha2_1; double scale2_1;
	Ctext ct4_0; int l4_0, alpha4_0; double scale4_0;
	Ctext ct4_1; int l4_1, alpha4_1; double scale4_1;
	Ctext ct8_0; int l8_0, alpha8_0; double scale8_0;
	Ctext ct8_1; int l8_1, alpha8_1; double scale8_1;

	//====================================================
	// simulation -> l,alpha,scale
	//====================================================
	sum2_sim(l1, alpha1, rs_bits, l2_0, alpha2_0);
	sum2_sim(l1, alpha1, rs_bits, l2_1, alpha2_1);
	sum4_sim(l1, alpha1, l2, alpha2, l3, alpha3, rs_bits, l4_0, alpha4_0);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_0, alpha8_0);

	BinaryMerge_sim(l2_0, alpha2_0, l2_1, alpha2_1, l2, alpha2, rs_bits, l4_1, alpha4_1);
	BinaryMerge_sim(l4_0, alpha4_0, l4_1, alpha4_1, l4, alpha4, rs_bits, l8_1, alpha8_1);
	BinaryMerge_sim(l8_0, alpha8_0, l8_1, alpha8_1, l8, alpha8, rs_bits, lres, alphares);

	BinaryMerge_split(l8_0, alpha8_0, scale8_0, l8_1, alpha8_1, scale8_1, l8, alpha8, scale8, rs_bits, scaleres);
	BinaryMerge_split(l4_0, alpha4_0, scale4_0, l4_1, alpha4_1, scale4_1, l4, alpha4, scale4, rs_bits, scale8_1);
	BinaryMerge_split(l2_0, alpha2_0, scale2_0, l2_1, alpha2_1, scale2_1, l2, alpha2, scale2, rs_bits, scale4_1);

	//====================================================
	// evalpoly16 
	//====================================================
	sum2(c + 12, T1, l1, alpha1, scale1, rs_bits, scale2_0, ct2_0, l2_0, alpha2_0);
	sum2(c + 14, T1, l1, alpha1, scale1, rs_bits, scale2_1, ct2_1, l2_1, alpha2_1);
	sum4(c + 8 , T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, rs_bits, scale4_0, ct4_0, l4_0, alpha4_0);
	sum8(c     , T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, 
		                          T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_0, ct8_0, l8_0, alpha8_0);
	
	BinaryMerge(ct2_0, l2_0, alpha2_0, scale2_0, ct2_1, l2_1, alpha2_1, scale2_1, T2, l2, alpha2, scale2, rs_bits, ct4_1, l4_1, alpha4_1, scale4_1);
	BinaryMerge(ct4_0, l4_0, alpha4_0, scale4_0, ct4_1, l4_1, alpha4_1, scale4_1, T4, l4, alpha4, scale4, rs_bits, ct8_1, l8_1, alpha8_1, scale8_1);
	BinaryMerge(ct8_0, l8_0, alpha8_0, scale8_0, ct8_1, l8_1, alpha8_1, scale8_1, T8, l8, alpha8, scale8, rs_bits, res  , lres, alphares, scaleres);

}

void CKKS_polynomial::evalpoly16_Debug(double c[16], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, Ctext res, int& lres, int& alphares, double scaleres, int s[N]) {
	
	//====================================================
	// Chebyshev - exact
	//====================================================
	double z_T1_ex[N];
	ckks.dec_decode_graft(T1, l1, alpha1, s, scale1, z_T1_ex);
	ckks.save_z_graft(T1, l1, alpha1, scale1, "T1.txt");
	double z_T2_ex[N], z_T3_ex[N], z_T4_ex[N], z_T5_ex[N], z_T6_ex[N], z_T7_ex[N], z_T8_ex[N];
	Chebyshev_Doubling_Debug(z_T1_ex, z_T2_ex);
	Chebyshev_General_Debug(z_T1_ex, z_T2_ex, z_T1_ex, z_T3_ex);
	Chebyshev_Doubling_Debug(z_T2_ex, z_T4_ex);
	Chebyshev_General_Debug(z_T2_ex, z_T3_ex, z_T1_ex, z_T5_ex);
	Chebyshev_Doubling_Debug(z_T3_ex, z_T6_ex);
	Chebyshev_General_Debug(z_T3_ex, z_T4_ex, z_T1_ex, z_T7_ex);
	Chebyshev_Doubling_Debug(z_T4_ex, z_T8_ex);

	//====================================================
	// Chebyshev 
	//====================================================
	Ctext T2; int l2, alpha2; double scale2;
	Ctext T3; int l3, alpha3; double scale3;
	Ctext T4; int l4, alpha4; double scale4;
	Ctext T5; int l5, alpha5; double scale5;
	Ctext T6; int l6, alpha6; double scale6;
	Ctext T7; int l7, alpha7; double scale7;
	Ctext T8; int l8, alpha8; double scale8;

	Chebyshev_Doubling(T1, l1, alpha1, scale1, rs_bits, T2, l2, alpha2, scale2); ckks.save_z_graft_and_debug(T2, l2, alpha2, scale2, "T2.txt",z_T2_ex);
	//----------------------------------------------------
	// increase alpha
	//----------------------------------------------------
	//printf("l2=%d, alpha2=%d, scale2=%f\n", l2, alpha2, scale2);
	//increase_alpha(T2, l2, alpha2, scale2, 30, l2, scale2); alpha2 = 30;
	//printf("l2=%d, alpha2=%d, scale2=%f\n", l2, alpha2, scale2);
	
	Chebyshev_Doubling(T2, l2, alpha2, scale2, rs_bits, T4, l4, alpha4, scale4); ckks.save_z_graft_and_debug(T4, l4, alpha4, scale4, "T4.txt",z_T4_ex);
	//
	Chebyshev_General(T1,l1,alpha1,scale1, T2, l2, alpha2, scale2, T1, l1, alpha1, scale1, rs_bits, T3, l3, alpha3, scale3);ckks.save_z_graft_and_debug(T3, l3, alpha3, scale3, "T3.txt",z_T3_ex);
	Chebyshev_General(T2,l2,alpha2,scale2, T3, l3, alpha3, scale3, T1, l1, alpha1, scale1, rs_bits, T5, l5, alpha5, scale5);ckks.save_z_graft_and_debug(T5, l5, alpha5, scale5, "T5.txt",z_T5_ex);
	//
	Chebyshev_Doubling(T3,l3,alpha3,scale3, rs_bits, T6, l6, alpha6, scale6); ckks.save_z_graft_and_debug(T6, l6, alpha6, scale6, "T6.txt",z_T6_ex);
	Chebyshev_General (T3,l3,alpha3,scale3,     T4, l4, alpha4, scale4, T1, l1, alpha1, scale1, rs_bits, T7, l7, alpha7, scale7);ckks.save_z_graft_and_debug(T7, l7, alpha7, scale7, "T7.txt",z_T7_ex);
	//
	Chebyshev_Doubling(T4, l4, alpha4, scale4, rs_bits, T8, l8, alpha8, scale8); ckks.save_z_graft_and_debug(T8, l8, alpha8, scale8, "T8.txt", z_T8_ex);

	

	//====================================================
	// variable declaration
	//====================================================
	Ctext ct2_0; int l2_0, alpha2_0; double scale2_0; double z2_0_ex[N];
	Ctext ct2_1; int l2_1, alpha2_1; double scale2_1; double z2_1_ex[N];
	Ctext ct4_0; int l4_0, alpha4_0; double scale4_0; double z4_0_ex[N];
	Ctext ct4_1; int l4_1, alpha4_1; double scale4_1; double z4_1_ex[N];
	Ctext ct8_0; int l8_0, alpha8_0; double scale8_0; double z8_0_ex[N];
	Ctext ct8_1; int l8_1, alpha8_1; double scale8_1; double z8_1_ex[N];
	//Ctext res  ; int lres, alphares; double scaleres; 
	double zres_ex[N];

	//====================================================
	// simulation -> l,alpha,scale
	//====================================================
	sum2_sim(l1, alpha1, rs_bits, l2_0, alpha2_0);
	sum2_sim(l1, alpha1, rs_bits, l2_1, alpha2_1);
	sum4_sim(l1, alpha1, l2, alpha2, l3, alpha3, rs_bits, l4_0, alpha4_0);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_0, alpha8_0);

	BinaryMerge_sim(l2_0, alpha2_0, l2_1, alpha2_1, l2, alpha2, rs_bits, l4_1, alpha4_1);
	BinaryMerge_sim(l4_0, alpha4_0, l4_1, alpha4_1, l4, alpha4, rs_bits, l8_1, alpha8_1);
	BinaryMerge_sim(l8_0, alpha8_0, l8_1, alpha8_1, l8, alpha8, rs_bits, lres, alphares);

	BinaryMerge_split(l8_0, alpha8_0, scale8_0, l8_1, alpha8_1, scale8_1, l8, alpha8, scale8, rs_bits, scaleres);
	BinaryMerge_split(l4_0, alpha4_0, scale4_0, l4_1, alpha4_1, scale4_1, l4, alpha4, scale4, rs_bits, scale8_1);
	BinaryMerge_split(l2_0, alpha2_0, scale2_0, l2_1, alpha2_1, scale2_1, l2, alpha2, scale2, rs_bits, scale4_1);

	printf("\n================== simulation results ====================\n");
	printf("ct2_0 : l=%d, alpha=%d, log2(scale)=%.2f\n", l2_0, alpha2_0, log(scale2_0) / log(2));
	printf("ct2_1 : l=%d, alpha=%d, log2(scale)=%.2f\n", l2_1, alpha2_1, log(scale2_1) / log(2));
	printf("ct4_0 : l=%d, alpha=%d, log2(scale)=%.2f\n", l4_0, alpha4_0, log(scale4_0) / log(2));
	printf("ct4_1 : l=%d, alpha=%d, log2(scale)=%.2f\n", l4_1, alpha4_1, log(scale4_1) / log(2));
	printf("ct8_0 : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_0, alpha8_0, log(scale8_0) / log(2));
	printf("ct8_1 : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_1, alpha8_1, log(scale8_1) / log(2));
	printf("res   : l=%d, alpha=%d, log2(scale)=%.2f\n", lres, alphares, log(scaleres) / log(2));
	printf("\n==========================================================\n");
	
	//====================================================
	// evalpoly16 - exact
	//====================================================
	sum2_Debug(c + 12, z_T1_ex, z2_0_ex);
	sum2_Debug(c + 14, z_T1_ex, z2_1_ex);
	sum4_Debug(c + 8 , z_T1_ex, z_T2_ex, z_T3_ex, z4_0_ex);
	sum8_Debug(c     , z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_0_ex);
	BinaryMerge_Debug(z2_0_ex, z2_1_ex, z_T2_ex, z4_1_ex);
	BinaryMerge_Debug(z4_0_ex, z4_1_ex, z_T4_ex, z8_1_ex);
	BinaryMerge_Debug(z8_0_ex, z8_1_ex, z_T8_ex, zres_ex); 

	//====================================================
	// evalpoly16 
	//====================================================
	sum2(c + 12, T1, l1, alpha1, scale1, rs_bits, scale2_0, ct2_0, l2_0, alpha2_0);
	sum2(c + 14, T1, l1, alpha1, scale1, rs_bits, scale2_1, ct2_1, l2_1, alpha2_1);
	sum4(c + 8 , T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, rs_bits, scale4_0, ct4_0, l4_0, alpha4_0);
	sum8(c     , T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, 
		                          T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_0, ct8_0, l8_0, alpha8_0);
	ckks.save_z_graft_and_debug(ct2_0, l2_0, alpha2_0, scale2_0, "z2_0.txt", z2_0_ex);
	ckks.save_z_graft_and_debug(ct2_1, l2_1, alpha2_1, scale2_1, "z2_1.txt", z2_1_ex);
	ckks.save_z_graft_and_debug(ct4_0, l4_0, alpha4_0, scale4_0, "z4_0.txt", z4_0_ex);
	ckks.save_z_graft_and_debug(ct8_0, l8_0, alpha8_0, scale8_0, "z8_0.txt", z8_0_ex);

	BinaryMerge(ct2_0, l2_0, alpha2_0, scale2_0, ct2_1, l2_1, alpha2_1, scale2_1, T2, l2, alpha2, scale2, rs_bits, ct4_1, l4_1, alpha4_1, scale4_1);
	BinaryMerge(ct4_0, l4_0, alpha4_0, scale4_0, ct4_1, l4_1, alpha4_1, scale4_1, T4, l4, alpha4, scale4, rs_bits, ct8_1, l8_1, alpha8_1, scale8_1);
	BinaryMerge(ct8_0, l8_0, alpha8_0, scale8_0, ct8_1, l8_1, alpha8_1, scale8_1, T8, l8, alpha8, scale8, rs_bits, res  , lres, alphares, scaleres);

	ckks.save_z_graft_and_debug(ct4_1, l4_1, alpha4_1, scale4_1, "z4_1.txt", z4_1_ex);
	ckks.save_z_graft_and_debug(ct8_1, l8_1, alpha8_1, scale8_1, "z8_1.txt", z8_1_ex);
	ckks.save_z_graft_and_debug(res  , lres, alphares, scaleres, "zres.txt", zres_ex); // poly error


}


 
void CKKS_polynomial::evalpoly64_Debug(double c[64], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, Ctext res, int& lres, int& alphares, double scaleres, double zres_ex[N], int s[N]) {
	
	//====================================================
	// Chebyshev - exact
	//====================================================
	double z_T1_ex[N];
	ckks.dec_decode_graft(T1, l1, alpha1, s, scale1, z_T1_ex);
	ckks.save_z_graft(T1, l1, alpha1, scale1, "T1.txt");
	double z_T2_ex[N], z_T3_ex[N], z_T4_ex[N], z_T5_ex[N], z_T6_ex[N], z_T7_ex[N], z_T8_ex[N], z_T16_ex[N], z_T32_ex[N];
	Chebyshev_Doubling_Debug(z_T1_ex, z_T2_ex);
	Chebyshev_General_Debug(z_T1_ex, z_T2_ex, z_T1_ex, z_T3_ex);
	Chebyshev_Doubling_Debug(z_T2_ex, z_T4_ex);
	Chebyshev_General_Debug(z_T2_ex, z_T3_ex, z_T1_ex, z_T5_ex);
	Chebyshev_Doubling_Debug(z_T3_ex, z_T6_ex);
	Chebyshev_General_Debug(z_T3_ex, z_T4_ex, z_T1_ex, z_T7_ex);
	Chebyshev_Doubling_Debug(z_T4_ex, z_T8_ex);
	Chebyshev_Doubling_Debug(z_T8_ex, z_T16_ex);
	Chebyshev_Doubling_Debug(z_T16_ex, z_T32_ex);

	//====================================================
	// Chebyshev 
	//====================================================
	Ctext T2; int l2, alpha2; double scale2;
	Ctext T3; int l3, alpha3; double scale3;
	Ctext T4; int l4, alpha4; double scale4;
	Ctext T5; int l5, alpha5; double scale5;
	Ctext T6; int l6, alpha6; double scale6;
	Ctext T7; int l7, alpha7; double scale7;
	Ctext T8; int l8, alpha8; double scale8;
	Ctext T16; int l16, alpha16; double scale16;
	Ctext T32; int l32, alpha32; double scale32; 

	Chebyshev_Doubling(T1, l1, alpha1, scale1, rs_bits, T2, l2, alpha2, scale2); 
	ckks.save_z_graft_and_debug(T2, l2, alpha2, scale2, "T2.txt",z_T2_ex);
	//----------------------------------------------------
	// increase alpha
	//----------------------------------------------------
	//printf("l2=%d, alpha2=%d, scale2=%f\n", l2, alpha2, scale2);
	//increase_alpha(T2, l2, alpha2, scale2, 30, l2, scale2); alpha2 = 30;
	//printf("l2=%d, alpha2=%d, scale2=%f\n", l2, alpha2, scale2);
	
	Chebyshev_Doubling(T2, l2, alpha2, scale2, rs_bits, T4, l4, alpha4, scale4); ckks.save_z_graft_and_debug(T4, l4, alpha4, scale4, "T4.txt",z_T4_ex);
	//
	
	Chebyshev_General(T1,l1,alpha1,scale1, T2, l2, alpha2, scale2, T1, l1, alpha1, scale1, rs_bits, T3, l3, alpha3, scale3);ckks.save_z_graft_and_debug(T3, l3, alpha3, scale3, "T3.txt",z_T3_ex);
	Chebyshev_General(T2,l2,alpha2,scale2, T3, l3, alpha3, scale3, T1, l1, alpha1, scale1, rs_bits, T5, l5, alpha5, scale5);ckks.save_z_graft_and_debug(T5, l5, alpha5, scale5, "T5.txt",z_T5_ex);
	//
	Chebyshev_Doubling(T3,l3,alpha3,scale3, rs_bits, T6, l6, alpha6, scale6); ckks.save_z_graft_and_debug(T6, l6, alpha6, scale6, "T6.txt",z_T6_ex);
	Chebyshev_General (T3,l3,alpha3,scale3,     T4, l4, alpha4, scale4, T1, l1, alpha1, scale1, rs_bits, T7, l7, alpha7, scale7);ckks.save_z_graft_and_debug(T7, l7, alpha7, scale7, "T7.txt",z_T7_ex);
	//
	Chebyshev_Doubling(T4, l4, alpha4, scale4, rs_bits, T8, l8, alpha8, scale8); ckks.save_z_graft_and_debug(T8, l8, alpha8, scale8, "T8.txt", z_T8_ex);
	// 
	Chebyshev_Doubling(T8 , l8 , alpha8 , scale8 , rs_bits, T16, l16, alpha16, scale16); ckks.save_z_graft_and_debug(T16, l16, alpha16, scale16, "T16.txt", z_T16_ex);
	Chebyshev_Doubling(T16, l16, alpha16, scale16, rs_bits, T32, l32, alpha32, scale32); ckks.save_z_graft_and_debug(T32, l32, alpha32, scale32, "T32.txt", z_T32_ex);
	
	
	//====================================================
	// variable declaration
	//====================================================
	Ctext ct2_0; int l2_0, alpha2_0; double scale2_0; double z2_0_ex[N];
	Ctext ct2_1; int l2_1, alpha2_1; double scale2_1; double z2_1_ex[N];

	Ctext ct4_0; int l4_0, alpha4_0; double scale4_0; double z4_0_ex[N];
	Ctext ct4_1; int l4_1, alpha4_1; double scale4_1; double z4_1_ex[N];

	Ctext ct8_0; int l8_0, alpha8_0; double scale8_0; double z8_0_ex[N];
	Ctext ct8_1; int l8_1, alpha8_1; double scale8_1; double z8_1_ex[N];
	Ctext ct8_2; int l8_2, alpha8_2; double scale8_2; double z8_2_ex[N];
	Ctext ct8_3; int l8_3, alpha8_3; double scale8_3; double z8_3_ex[N];
	Ctext ct8_4; int l8_4, alpha8_4; double scale8_4; double z8_4_ex[N];
	Ctext ct8_5; int l8_5, alpha8_5; double scale8_5; double z8_5_ex[N];
	Ctext ct8_6; int l8_6, alpha8_6; double scale8_6; double z8_6_ex[N];
	Ctext ct8_7; int l8_7, alpha8_7; double scale8_7; double z8_7_ex[N];

	Ctext ct16_0; int l16_0, alpha16_0; double scale16_0; double z16_0_ex[N];
	Ctext ct16_1; int l16_1, alpha16_1; double scale16_1; double z16_1_ex[N];
	Ctext ct16_2; int l16_2, alpha16_2; double scale16_2; double z16_2_ex[N];
	Ctext ct16_3; int l16_3, alpha16_3; double scale16_3; double z16_3_ex[N];

	Ctext ct32_0; int l32_0, alpha32_0; double scale32_0; double z32_0_ex[N];
	Ctext ct32_1; int l32_1, alpha32_1; double scale32_1; double z32_1_ex[N];


	//====================================================
	// simulation -> l,alpha,scale
	//====================================================
	sum2_sim(l1, alpha1, rs_bits, l2_0, alpha2_0);
	sum2_sim(l1, alpha1, rs_bits, l2_1, alpha2_1);
	sum4_sim(l1, alpha1, l2, alpha2, l3, alpha3, rs_bits, l4_0, alpha4_0);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_0, alpha8_0);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_1, alpha8_1);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_2, alpha8_2);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_3, alpha8_3);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_4, alpha8_4);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_5, alpha8_5);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_6, alpha8_6);

	BinaryMerge_sim(l2_0, alpha2_0, l2_1, alpha2_1, l2, alpha2, rs_bits, l4_1, alpha4_1);
	BinaryMerge_sim(l4_0, alpha4_0, l4_1, alpha4_1, l4, alpha4, rs_bits, l8_7, alpha8_7);
	BinaryMerge_sim(l8_0, alpha8_0, l8_1, alpha8_1, l8, alpha8, rs_bits, l16_0, alpha16_0);
	BinaryMerge_sim(l8_2, alpha8_2, l8_3, alpha8_3, l8, alpha8, rs_bits, l16_1, alpha16_1);
	BinaryMerge_sim(l8_4, alpha8_4, l8_5, alpha8_5, l8, alpha8, rs_bits, l16_2, alpha16_2);
	BinaryMerge_sim(l8_6, alpha8_6, l8_7, alpha8_7, l8, alpha8, rs_bits, l16_3, alpha16_3);
	BinaryMerge_sim(l16_0, alpha16_0, l16_1, alpha16_1, l16, alpha16, rs_bits, l32_0, alpha32_0);
	BinaryMerge_sim(l16_2, alpha16_2, l16_3, alpha16_3, l16, alpha16, rs_bits, l32_1, alpha32_1);
	BinaryMerge_sim(l32_0, alpha32_0, l32_1, alpha32_1, l32, alpha32, rs_bits, lres , alphares );


	BinaryMerge_split(l32_0, alpha32_0, scale32_0, l32_1, alpha32_1, scale32_1, l32, alpha32, scale32, rs_bits, scaleres);
	BinaryMerge_split(l16_0, alpha16_0, scale16_0, l16_1, alpha16_1, scale16_1, l16, alpha16, scale16, rs_bits, scale32_0);
	BinaryMerge_split(l16_2, alpha16_2, scale16_2, l16_3, alpha16_3, scale16_3, l16, alpha16, scale16, rs_bits, scale32_1);
	BinaryMerge_split(l8_0, alpha8_0, scale8_0, l8_1, alpha8_1, scale8_1, l8, alpha8, scale8, rs_bits, scale16_0);
	BinaryMerge_split(l8_2, alpha8_2, scale8_2, l8_3, alpha8_3, scale8_3, l8, alpha8, scale8, rs_bits, scale16_1);
	BinaryMerge_split(l8_4, alpha8_4, scale8_4, l8_5, alpha8_5, scale8_5, l8, alpha8, scale8, rs_bits, scale16_2);
	BinaryMerge_split(l8_6, alpha8_5, scale8_6, l8_7, alpha8_7, scale8_7, l8, alpha8, scale8, rs_bits, scale16_3);
	BinaryMerge_split(l4_0, alpha4_0, scale4_0, l4_1, alpha4_1, scale4_1, l4, alpha4, scale4, rs_bits, scale8_7);
	BinaryMerge_split(l2_0, alpha2_0, scale2_0, l2_1, alpha2_1, scale2_1, l2, alpha2, scale2, rs_bits, scale4_1);

	printf("\n================== simulation results ====================\n");
	printf("ct2_0  : l=%d, alpha=%d, log2(scale)=%.2f\n", l2_0, alpha2_0, log(scale2_0) / log(2));
	printf("ct2_1  : l=%d, alpha=%d, log2(scale)=%.2f\n", l2_1, alpha2_1, log(scale2_1) / log(2));
	printf("ct4_0  : l=%d, alpha=%d, log2(scale)=%.2f\n", l4_0, alpha4_0, log(scale4_0) / log(2));
	printf("ct4_1  : l=%d, alpha=%d, log2(scale)=%.2f\n", l4_1, alpha4_1, log(scale4_1) / log(2));
	printf("ct8_0  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_0, alpha8_0, log(scale8_0) / log(2));
	printf("ct8_1  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_1, alpha8_1, log(scale8_1) / log(2));
	printf("ct8_2  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_2, alpha8_2, log(scale8_2) / log(2));
	printf("ct8_3  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_3, alpha8_3, log(scale8_3) / log(2));
	printf("ct8_4  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_4, alpha8_4, log(scale8_4) / log(2));
	printf("ct8_5  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_5, alpha8_5, log(scale8_5) / log(2));
	printf("ct8_6  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_6, alpha8_6, log(scale8_6) / log(2));
	printf("ct8_7  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_7, alpha8_7, log(scale8_7) / log(2));
	printf("ct16_0 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_0, alpha16_0, log(scale16_0) / log(2));
	printf("ct16_1 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_1, alpha16_1, log(scale16_1) / log(2));
	printf("ct16_2 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_2, alpha16_2, log(scale16_2) / log(2));
	printf("ct16_3 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_3, alpha16_3, log(scale16_3) / log(2));
	printf("ct32_0 : l=%d, alpha=%d, log2(scale)=%.2f\n", l32_0, alpha32_0, log(scale32_0) / log(2));
	printf("ct32_1 : l=%d, alpha=%d, log2(scale)=%.2f\n", l32_1, alpha32_1, log(scale32_1) / log(2));
	printf("res    : l=%d, alpha=%d, log2(scale)=%.2f\n", lres, alphares, log(scaleres) / log(2));
	printf("\n==========================================================\n");
	
	//====================================================
	// evalpoly16 - exact
	//====================================================
	sum2_Debug(c + 60, z_T1_ex, z2_0_ex);
	sum2_Debug(c + 62, z_T1_ex, z2_1_ex);
	sum4_Debug(c + 56, z_T1_ex, z_T2_ex, z_T3_ex, z4_0_ex);
	sum8_Debug(c     , z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_0_ex);
	sum8_Debug(c + 8 , z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_1_ex);
	sum8_Debug(c + 16, z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_2_ex);
	sum8_Debug(c + 24, z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_3_ex);
	sum8_Debug(c + 32, z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_4_ex);
	sum8_Debug(c + 40, z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_5_ex);
	sum8_Debug(c + 48, z_T1_ex, z_T2_ex, z_T3_ex, z_T4_ex, z_T5_ex, z_T6_ex, z_T7_ex, z8_6_ex);
	BinaryMerge_Debug(z2_0_ex, z2_1_ex, z_T2_ex, z4_1_ex);
	BinaryMerge_Debug(z4_0_ex, z4_1_ex, z_T4_ex, z8_7_ex);
	BinaryMerge_Debug(z8_0_ex, z8_1_ex, z_T8_ex, z16_0_ex);
	BinaryMerge_Debug(z8_2_ex, z8_3_ex, z_T8_ex, z16_1_ex);
	BinaryMerge_Debug(z8_4_ex, z8_5_ex, z_T8_ex, z16_2_ex);
	BinaryMerge_Debug(z8_6_ex, z8_7_ex, z_T8_ex, z16_3_ex);
	BinaryMerge_Debug(z16_0_ex, z16_1_ex, z_T16_ex, z32_0_ex);
	BinaryMerge_Debug(z16_2_ex, z16_3_ex, z_T16_ex, z32_1_ex);
	BinaryMerge_Debug(z32_0_ex, z32_1_ex, z_T32_ex, zres_ex);


	//====================================================
	// evalpoly16 
	//====================================================
	sum2(c + 60, T1, l1, alpha1, scale1, rs_bits, scale2_0, ct2_0, l2_0, alpha2_0);
	sum2(c + 62, T1, l1, alpha1, scale1, rs_bits, scale2_1, ct2_1, l2_1, alpha2_1);
	sum4(c + 56, T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, rs_bits, scale4_0, ct4_0, l4_0, alpha4_0);
	sum8(c     , T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_0, ct8_0, l8_0, alpha8_0);
	sum8(c + 8 , T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_1, ct8_1, l8_1, alpha8_1);
	sum8(c + 16, T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_2, ct8_2, l8_2, alpha8_2);
	sum8(c + 24, T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_3, ct8_3, l8_3, alpha8_3);
	sum8(c + 32, T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_4, ct8_4, l8_4, alpha8_4);
	sum8(c + 40, T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_5, ct8_5, l8_5, alpha8_5);
	sum8(c + 48, T1, l1, alpha1, scale1, T2, l2, alpha2, scale2, T3, l3, alpha3, scale3, T4, l4, alpha4, scale4, T5, l5, alpha5, scale5, T6, l6, alpha6, scale6, T7, l7, alpha7, scale7, rs_bits, scale8_6, ct8_6, l8_6, alpha8_6);
	ckks.save_z_graft_and_debug(ct2_0, l2_0, alpha2_0, scale2_0, "z2_0.txt", z2_0_ex);
	ckks.save_z_graft_and_debug(ct2_1, l2_1, alpha2_1, scale2_1, "z2_1.txt", z2_1_ex);
	ckks.save_z_graft_and_debug(ct4_0, l4_0, alpha4_0, scale4_0, "z4_0.txt", z4_0_ex);
	ckks.save_z_graft_and_debug(ct8_0, l8_0, alpha8_0, scale8_0, "z8_0.txt", z8_0_ex);
	ckks.save_z_graft_and_debug(ct8_1, l8_1, alpha8_1, scale8_1, "z8_1.txt", z8_1_ex);
	ckks.save_z_graft_and_debug(ct8_2, l8_2, alpha8_2, scale8_2, "z8_2.txt", z8_2_ex);
	ckks.save_z_graft_and_debug(ct8_3, l8_3, alpha8_3, scale8_3, "z8_3.txt", z8_3_ex);
	ckks.save_z_graft_and_debug(ct8_4, l8_4, alpha8_4, scale8_4, "z8_4.txt", z8_4_ex);
	ckks.save_z_graft_and_debug(ct8_5, l8_5, alpha8_5, scale8_5, "z8_5.txt", z8_5_ex);
	ckks.save_z_graft_and_debug(ct8_6, l8_6, alpha8_6, scale8_6, "z8_6.txt", z8_6_ex);

	BinaryMerge(ct2_0, l2_0, alpha2_0, scale2_0, ct2_1, l2_1, alpha2_1, scale2_1, T2, l2, alpha2, scale2, rs_bits, ct4_1, l4_1, alpha4_1, scale4_1);
	BinaryMerge(ct4_0, l4_0, alpha4_0, scale4_0, ct4_1, l4_1, alpha4_1, scale4_1, T4, l4, alpha4, scale4, rs_bits, ct8_7, l8_7, alpha8_7, scale8_7);
	BinaryMerge(ct8_0, l8_0, alpha8_0, scale8_0, ct8_1, l8_1, alpha8_1, scale8_1, T8, l8, alpha8, scale8, rs_bits, ct16_0, l16_0, alpha16_0, scale16_0);
	BinaryMerge(ct8_2, l8_2, alpha8_2, scale8_2, ct8_3, l8_3, alpha8_3, scale8_3, T8, l8, alpha8, scale8, rs_bits, ct16_1, l16_1, alpha16_1, scale16_1);
	BinaryMerge(ct8_4, l8_4, alpha8_4, scale8_4, ct8_5, l8_5, alpha8_5, scale8_5, T8, l8, alpha8, scale8, rs_bits, ct16_2, l16_2, alpha16_2, scale16_2);
	BinaryMerge(ct8_6, l8_6, alpha8_6, scale8_6, ct8_7, l8_7, alpha8_7, scale8_7, T8, l8, alpha8, scale8, rs_bits, ct16_3, l16_3, alpha16_3, scale16_3);
	BinaryMerge(ct16_0, l16_0, alpha16_0, scale16_0, ct16_1, l16_1, alpha16_1, scale16_1, T16, l16, alpha16, scale16, rs_bits, ct32_0, l32_0, alpha32_0, scale32_0);
	BinaryMerge(ct16_2, l16_2, alpha16_2, scale16_2, ct16_3, l16_3, alpha16_3, scale16_3, T16, l16, alpha16, scale16, rs_bits, ct32_1, l32_1, alpha32_1, scale32_1);
	BinaryMerge(ct32_0, l32_0, alpha32_0, scale32_0, ct32_1, l32_1, alpha32_1, scale32_1, T32, l32, alpha32, scale32, rs_bits, res   , lres , alphares , scaleres );

	ckks.save_z_graft_and_debug(ct4_1, l4_1, alpha4_1, scale4_1, "z4_1.txt", z4_1_ex);
	ckks.save_z_graft_and_debug(ct8_7, l8_7, alpha8_7, scale8_7, "z8_7.txt", z8_7_ex);
	ckks.save_z_graft_and_debug(ct16_0, l16_0, alpha16_0, scale16_0, "z16_0.txt", z16_0_ex);
	ckks.save_z_graft_and_debug(ct16_1, l16_1, alpha16_1, scale16_1, "z16_1.txt", z16_1_ex);
	ckks.save_z_graft_and_debug(ct16_2, l16_2, alpha16_2, scale16_2, "z16_2.txt", z16_2_ex);
	ckks.save_z_graft_and_debug(ct16_3, l16_3, alpha16_3, scale16_3, "z16_3.txt", z16_3_ex);
	ckks.save_z_graft_and_debug(ct32_0, l32_0, alpha32_0, scale32_0, "z32_0.txt", z32_0_ex);
	ckks.save_z_graft_and_debug(ct32_1, l32_1, alpha32_1, scale32_1, "z32_1.txt", z32_1_ex);

	ckks.save_z_graft_and_debug(res  , lres, alphares, scaleres, "zres.txt", zres_ex); // poly error
	

}


void CKKS_polynomial::evalpoly64_sim(int l1, int alpha1, double scale1, int rs_bits, int& lres, int& alphares, double scaleres) {
	
	
	//====================================================
	// Chebyshev 
	//====================================================
	int l2 , alpha2 ; double scale2 ; 
	int l3 , alpha3 ; double scale3 ;
	int l4 , alpha4 ; double scale4 ;
	int l5 , alpha5 ; double scale5 ;
	int l6 , alpha6 ; double scale6 ;
	int l7 , alpha7 ; double scale7 ;
	int l8 , alpha8 ; double scale8 ;
	int l16, alpha16; double scale16;
	int l32, alpha32; double scale32; 

	Chebyshev_Doubling_sim(l1, alpha1, scale1, rs_bits, l2, alpha2, scale2); double scale2_log = log(scale2) / log(2);
	Chebyshev_Doubling_sim(l2, alpha2, scale2, rs_bits, l4, alpha4, scale4); double scale4_log = log(scale4) / log(2);
	Chebyshev_General_sim(l1,alpha1,scale1, l2, alpha2, scale2, l1, alpha1, scale1, rs_bits, l3, alpha3, scale3); double scale3_log = log(scale3) / log(2);
	Chebyshev_General_sim(l2,alpha2,scale2, l3, alpha3, scale3, l1, alpha1, scale1, rs_bits, l5, alpha5, scale5); double scale5_log = log(scale5) / log(2);
	Chebyshev_Doubling_sim(l3,alpha3,scale3, rs_bits, l6, alpha6, scale6); double scale6_log = log(scale6) / log(2);
	Chebyshev_General_sim(l3,alpha3,scale3, l4, alpha4, scale4, l1, alpha1, scale1, rs_bits, l7, alpha7, scale7); double scale7_log = log(scale7) / log(2);
	Chebyshev_Doubling_sim(l4 , alpha4 , scale4 , rs_bits, l8 , alpha8 , scale8 ); double scale8_log = log(scale8) / log(2);
	Chebyshev_Doubling_sim(l8 , alpha8 , scale8 , rs_bits, l16, alpha16, scale16); double scale16_log = log(scale16) / log(2);
	Chebyshev_Doubling_sim(l16, alpha16, scale16, rs_bits, l32, alpha32, scale32); double scale32_log = log(scale32) / log(2);
	
	
	//====================================================
	// variable declaration
	//====================================================
	int l2_0, alpha2_0; double scale2_0; 
	int l2_1, alpha2_1; double scale2_1; 
	int l4_0, alpha4_0; double scale4_0; 
	int l4_1, alpha4_1; double scale4_1; 
	int l8_0, alpha8_0; double scale8_0; 
	int l8_1, alpha8_1; double scale8_1; 
	int l8_2, alpha8_2; double scale8_2; 
	int l8_3, alpha8_3; double scale8_3; 
	int l8_4, alpha8_4; double scale8_4; 
	int l8_5, alpha8_5; double scale8_5; 
	int l8_6, alpha8_6; double scale8_6; 
	int l8_7, alpha8_7; double scale8_7; 
	int l16_0, alpha16_0; double scale16_0;
	int l16_1, alpha16_1; double scale16_1;
	int l16_2, alpha16_2; double scale16_2;
	int l16_3, alpha16_3; double scale16_3;
	int l32_0, alpha32_0; double scale32_0;
	int l32_1, alpha32_1; double scale32_1;


	//====================================================
	// simulation -> l,alpha,scale
	//====================================================
	sum2_sim(l1, alpha1, rs_bits, l2_0, alpha2_0);
	sum2_sim(l1, alpha1, rs_bits, l2_1, alpha2_1);
	sum4_sim(l1, alpha1, l2, alpha2, l3, alpha3, rs_bits, l4_0, alpha4_0);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_0, alpha8_0);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_1, alpha8_1);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_2, alpha8_2);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_3, alpha8_3);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_4, alpha8_4);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_5, alpha8_5);
	sum8_sim(l1, alpha1, l2, alpha2, l3, alpha3, l4, alpha4, l5, alpha5, l6, alpha6, l7, alpha7, rs_bits, l8_6, alpha8_6);

	BinaryMerge_sim(l2_0, alpha2_0, l2_1, alpha2_1, l2, alpha2, rs_bits, l4_1, alpha4_1);
	BinaryMerge_sim(l4_0, alpha4_0, l4_1, alpha4_1, l4, alpha4, rs_bits, l8_7, alpha8_7);
	BinaryMerge_sim(l8_0, alpha8_0, l8_1, alpha8_1, l8, alpha8, rs_bits, l16_0, alpha16_0);
	BinaryMerge_sim(l8_2, alpha8_2, l8_3, alpha8_3, l8, alpha8, rs_bits, l16_1, alpha16_1);
	BinaryMerge_sim(l8_4, alpha8_4, l8_5, alpha8_5, l8, alpha8, rs_bits, l16_2, alpha16_2);
	BinaryMerge_sim(l8_6, alpha8_6, l8_7, alpha8_7, l8, alpha8, rs_bits, l16_3, alpha16_3);
	BinaryMerge_sim(l16_0, alpha16_0, l16_1, alpha16_1, l16, alpha16, rs_bits, l32_0, alpha32_0);
	BinaryMerge_sim(l16_2, alpha16_2, l16_3, alpha16_3, l16, alpha16, rs_bits, l32_1, alpha32_1);
	BinaryMerge_sim(l32_0, alpha32_0, l32_1, alpha32_1, l32, alpha32, rs_bits, lres , alphares );

	///*
	BinaryMerge_split(l32_0, alpha32_0, scale32_0, l32_1, alpha32_1, scale32_1, l32, alpha32, scale32, rs_bits, scaleres);
	BinaryMerge_split(l16_0, alpha16_0, scale16_0, l16_1, alpha16_1, scale16_1, l16, alpha16, scale16, rs_bits, scale32_0);
	BinaryMerge_split(l16_2, alpha16_2, scale16_2, l16_3, alpha16_3, scale16_3, l16, alpha16, scale16, rs_bits, scale32_1);
	BinaryMerge_split(l8_0, alpha8_0, scale8_0, l8_1, alpha8_1, scale8_1, l8, alpha8, scale8, rs_bits, scale16_0);
	BinaryMerge_split(l8_2, alpha8_2, scale8_2, l8_3, alpha8_3, scale8_3, l8, alpha8, scale8, rs_bits, scale16_1);
	BinaryMerge_split(l8_4, alpha8_4, scale8_4, l8_5, alpha8_5, scale8_5, l8, alpha8, scale8, rs_bits, scale16_2);
	BinaryMerge_split(l8_6, alpha8_5, scale8_6, l8_7, alpha8_7, scale8_7, l8, alpha8, scale8, rs_bits, scale16_3);
	BinaryMerge_split(l4_0, alpha4_0, scale4_0, l4_1, alpha4_1, scale4_1, l4, alpha4, scale4, rs_bits, scale8_7);
	BinaryMerge_split(l2_0, alpha2_0, scale2_0, l2_1, alpha2_1, scale2_1, l2, alpha2, scale2, rs_bits, scale4_1);

	printf("\n================== simulation results ====================\n");
	printf("ct2_0  : l=%d, alpha=%d, log2(scale)=%.2f\n", l2_0, alpha2_0, log(scale2_0) / log(2));
	printf("ct2_1  : l=%d, alpha=%d, log2(scale)=%.2f\n", l2_1, alpha2_1, log(scale2_1) / log(2));
	printf("ct4_0  : l=%d, alpha=%d, log2(scale)=%.2f\n", l4_0, alpha4_0, log(scale4_0) / log(2));
	printf("ct4_1  : l=%d, alpha=%d, log2(scale)=%.2f\n", l4_1, alpha4_1, log(scale4_1) / log(2));
	printf("ct8_0  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_0, alpha8_0, log(scale8_0) / log(2));
	printf("ct8_1  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_1, alpha8_1, log(scale8_1) / log(2));
	printf("ct8_2  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_2, alpha8_2, log(scale8_2) / log(2));
	printf("ct8_3  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_3, alpha8_3, log(scale8_3) / log(2));
	printf("ct8_4  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_4, alpha8_4, log(scale8_4) / log(2));
	printf("ct8_5  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_5, alpha8_5, log(scale8_5) / log(2));
	printf("ct8_6  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_6, alpha8_6, log(scale8_6) / log(2));
	printf("ct8_7  : l=%d, alpha=%d, log2(scale)=%.2f\n", l8_7, alpha8_7, log(scale8_7) / log(2));
	printf("ct16_0 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_0, alpha16_0, log(scale16_0) / log(2));
	printf("ct16_1 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_1, alpha16_1, log(scale16_1) / log(2));
	printf("ct16_2 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_2, alpha16_2, log(scale16_2) / log(2));
	printf("ct16_3 : l=%d, alpha=%d, log2(scale)=%.2f\n", l16_3, alpha16_3, log(scale16_3) / log(2));
	printf("ct32_0 : l=%d, alpha=%d, log2(scale)=%.2f\n", l32_0, alpha32_0, log(scale32_0) / log(2));
	printf("ct32_1 : l=%d, alpha=%d, log2(scale)=%.2f\n", l32_1, alpha32_1, log(scale32_1) / log(2));
	printf("res    : l=%d, alpha=%d, log2(scale)=%.2f\n", lres, alphares, log(scaleres) / log(2));
	printf("\n==========================================================\n");
	//*/
	
}



void CKKS_polynomial::Double_angle_formula( Ctext g_i  , int  l_g_i  , int  alpha_g_i  , double  scale_g_i  , double c, int rs_bits,
	                                        Ctext g_ip1, int& l_g_ip1, int& alpha_g_ip1, double& scale_g_ip1){
	Ctext g_i_mul2;
	ckks.add(g_i[0], g_i[0], l_g_i, alpha_g_i, g_i_mul2[0]);
	ckks.add(g_i[1], g_i[1], l_g_i, alpha_g_i, g_i_mul2[1]);
	//
	ckks.mul_rs(g_i, l_g_i, alpha_g_i, scale_g_i, g_i_mul2, l_g_i, alpha_g_i, scale_g_i, rs_bits, g_ip1, l_g_ip1, alpha_g_ip1, scale_g_ip1);

	uint32_t pt_c[L + G];  
	encode_constant(c,  l_g_ip1, alpha_g_ip1, scale_g_ip1, pt_c);
	for (int j = 0; j < l_g_ip1; j++)
		sub_mod<N>(g_ip1[0][j], pt_c[j], g_ip1[0][j], ckks.qp[j]);
	if (alpha_g_ip1 > 0) {
		uint32_t p = uint32_t(1) << alpha_g_ip1;
		g_ip1[0][L][0] += p - pt_c[L];
		g_ip1[0][L][0] = g_ip1[0][L][0] & (p - 1);
	}
}

void CKKS_polynomial::Double_angle_formula_Debug( double g_i[N], double c, double g_ip1[N]) {
	for (int i = 0; i < N / 2;i++) {
		g_ip1[i    ] = 2*(g_i[i] * g_i[i] - g_i[i + N / 2] * g_i[i + N / 2])-c;
		g_ip1[i+N/2] = 4* g_i[i]* g_i[i + N / 2];
	}
}