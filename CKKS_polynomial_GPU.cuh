#pragma once

#include "CKKS_GPU.cuh"

void initialize_poly();
void deallocate_poly();

typedef uint32_t Ptext_Constant [L+G];
#define Ptext_ConstantSize (sizeof(uint32_t)*(L+G))

void encode_constant_host(double c, int l, int alpha, double scale, Ptext_Constant pt);
void mul_constant_host(Ctext ct, int l, int alpha, double c, double scale_c);
void add_constant_host(Ctext ct, int l, int alpha, double c, double scale_c);

//====================================================
// 
//====================================================
						   
void Chebyshev_Doubling_host(Ctext T , int  l , int  alpha , double  scale , int rs_bits,
	                         Ctext Td, int& ld, int& alphad, double& scaled               ) ;
void Chebyshev_General_host(Ctext Ta, int  la, int  alphaa, double  scalea,
					        Ctext Tb, int  lb, int  alphab, double  scaleb,
					        Ctext Tc, int  lc, int  alphac, double  scalec, int rs_bits,
					        Ctext Td, int& ld, int& alphad, double& scaled              );
//
void sum2_host(double c[2], Ctext T1, int l, int alpha, double scale, int rs_bits, double scale_out, 
					       Ctext out, int& l_out, int& alpha_out);
void sum2_sim_host(int l, int alpha, int rs_bits, int& l_out, int& alpha_out);
void sum4_host(double c[4], Ctext T1, int l1, int alpha1, double scale1,
					       Ctext T2, int l2, int alpha2, double scale2, 
					       Ctext T3, int l3, int alpha3, double scale3, int rs_bits, double scale_out,
					       Ctext out, int& l_out, int& alpha_out) ;

void sum4_sim_host(int l1, int alpha1, int l2, int alpha2, int l3, int alpha3, int rs_bits, 
			       int& l_out, int& alpha_out);

void sum8_host(double c[8], Ctext T1, int l1, int alpha1, double scale1,
					       Ctext T2, int l2, int alpha2, double scale2, 
					       Ctext T3, int l3, int alpha3, double scale3, 
					       Ctext T4, int l4, int alpha4, double scale4, 
					       Ctext T5, int l5, int alpha5, double scale5,
					       Ctext T6, int l6, int alpha6, double scale6, 
					       Ctext T7, int l7, int alpha7, double scale7,	int rs_bits, double scale_out,
					       Ctext out, int& l_out, int& alpha_out);
void sum8_sim_host(int l1, int alpha1,
			       int l2, int alpha2, 
			       int l3, int alpha3, 
			       int l4, int alpha4, 
			       int l5, int alpha5,
			       int l6, int alpha6, 
			       int l7, int alpha7, int rs_bits, int& l_out, int& alpha_out);							

void BinaryMerge_host(Ctext Left , int  lL  , int  alphaL  , double  scaleL  ,
	                  Ctext Right, int  lR  , int  alphaR  , double  scaleR  ,
	                  Ctext T    , int  lT  , int  alphaT  , double  scaleT  , int rs_bits,
	                  Ctext sum  , int& lsum, int& alphasum, double& scalesum	);

void BinaryMerge_sim_host(int  lL  , int  alphaL  ,
	                      int  lR  , int  alphaR  ,
	                      int  lT  , int  alphaT  , int rs_bits,
	                      int& lsum, int& alphasum              );

void BinaryMerge_split_host(int lL  , int alphaL  , double& scaleL  ,
	                        int lR  , int alphaR  , double& scaleR  ,
	                        int lT  , int alphaT  , double  scaleT  , int rs_bits,
	                                                double  scalesum	              );
							 
//Debug							 
void Chebyshev_Doubling_Debug(const double z_T[N], double z_Td[N]) ;						   
void Chebyshev_General_Debug(const double z_Ta[N], const double z_Tb[N], const double z_Tc[N], double z_Td[N]);

void sum2_Debug(double coeff[2], const double T1[N], double sum[N]);

void sum4_Debug(double coeff[4], const double T1[N],
	                            const double T2[N],
	                            const double T3[N], double sum[N]);

void sum8_Debug(double coeff[8], const double T1[N],
	                            const double T2[N],
	                            const double T3[N],
	                            const double T4[N],
	                            const double T5[N],
	                            const double T6[N],
	                            const double T7[N], double sum[N]);
								
void BinaryMerge_Debug(const double Left[N],
	                   const double Right[N],
	                   const double T[N], double sum[N]) ;
					   

//====================================================
// Evalpoly
//====================================================
void evalpoly64_Debug_host(double c[64], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, 
                                        Ctext res, int& lres, int& alphares, double scaleres, double zres_ex[N], int s[N]) ;
void evalpoly64_sim_host(int  l1  , int  alpha1  , double scale1  , int rs_bits, 
                         int& lres, int& alphares, double scaleres);
						 
void evalpoly64_host(double c[64], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, 
                                        Ctext res, int& lres, int& alphares, double scaleres, double h_zres_ex[N], int s[N]);
										
void evalpoly5odd_host( double c1, double c3, double c5, Ctext ct1, int l1, int alpha1, double scale1, int rs_bits,
                        Ctext out, int& lout, int& alphaout, double scaleout );
void evalpoly7odd_host( double c1, double c3, double c5, double c7, 
                        Ctext ct1, int l1, int a1, double s1, int rs_bits,
                        Ctext out, int& lout, int& aout, double sout );

//====================================================
// Double angle formula
//====================================================
void Double_angle_formula_host( Ctext g_i  , int  l_g_i  , int  alpha_g_i  , double  scale_g_i  , double c, int rs_bits,
	                            Ctext g_ip1, int& l_g_ip1, int& alpha_g_ip1, double& scale_g_ip1);
void Double_angle_formula_sim_host( int  l_g_i  , int  alpha_g_i  , double  scale_g_i  , double c, int rs_bits,
	                                int& l_g_ip1, int& alpha_g_ip1, double& scale_g_ip1);
void Double_angle_formula_Debug_host( double g_i[N], double c, double g_ip1[N]);


//====================================================
//log 
//====================================================
void Double_angle_formula_host_log( Ctext g_i  , int  l_g_i  , int  alpha_g_i  , double  scale_g_i  , double c, int rs_bits,
	                            Ctext g_ip1, int& l_g_ip1, int& alpha_g_ip1, double& scale_g_ip1);
void evalpoly64_host_log(double c[64], Ctext T1, int l1, int alpha1, double scale1, int rs_bits, 
                                        Ctext res, int& lres, int& alphares, double scaleres, double h_zres_ex[N], int s[N]);
										
void evalpoly64_sim_host_log(int  l1  , int  alpha1  , double scale1  , int rs_bits, 
                         int& lres, int& alphares, double scaleres);
						 

//====================================================
// gelu
//====================================================					 
			   
void gelu_1st_host(threeCtext ct , int  l    , int  alpha    , double  scale   , int rs_bits,
                   threeCtext out, int& l_out, int& alpha_out, double& scale_out);

void gelu_2nd_host(threeCtext ct_x, int  lx   , int  alphax   , double  scalex  ,
                   threeCtext ct  , int  l    , int  alpha    , double  scale   , int rs_bits,
                                   int& l_out, int& alpha_out, double& scale_out);
								   
								   
//====================================================
// tanh
//====================================================					 
			   
void tanh_1st_host(threeCtext ct , int  l    , int  alpha    , double  scale   , int rs_bits,
                   threeCtext out, int& l_out, int& alpha_out, double& scale_out);

void tanh_2nd_host(threeCtext ct , int  l    , int  alpha    , double  scale   , int rs_bits,
                   threeCtext out, int& l_out, int& alpha_out, double& scale_out);
	