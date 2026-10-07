#pragma once

#include "MatrixBSGS_GPU.cuh"
#include "MatrixFP32_GPU.cuh"
#include "Parameter_GPU.cuh"

//----------------------------------------------------------------------------------------------------------------
// Lineartransform
//----------------------------------------------------------------------------------------------------------------
void initialize_LT();
void deallocate_LT();
void applyA_host(MatrixBSGS_GPU& d_A, Ctext ct, int l, int alpha, float delta_pt, Ctext res);
void applyA_host(MatrixFP32_GPU& d_A, Ctext ct, int l, int alpha, float delta_pt, Ctext res);

void applyA_rshat_host(MatrixBSGS_GPU& d_A,
                       Ctext ct , int  l    , int  alpha    , double scale    , int rs_bits,
                       Ctext out, int& l_out, int& alpha_out, double scale_out);
					   
//============================================
// CPMM : out = A*B + b
//============================================
void cpmm_host(threeCtext ct_A , Weight B, Bias b, int  l,     int  alpha,     double scale,     int rsbits,
               threeCtext ct_AB,                   int& l_out, int& alpha_out, double scale_out);

void div2_cpmm_host(threeCtext ct_A , Weight B, Bias b, int  l,     int  alpha,     double scale,     int rsbits,
                    threeCtext ct_AB,                   int& l_out, int& alpha_out, double scale_out);
					
void cpmm_QKV_host(Weight Wq, Bias bq,
                   Weight Wk, Bias bk,
				   Weight Wv, Bias bv,
				   threeCtext ct, int  l,     int  alpha,     double scale,     int rsbits,
                   threeCtext ct_Q, 
                   threeCtext ct_K, 
                   threeCtext ct_V, int& l_out, int& alpha_out, double scale_out);
void cpmm_O_host(threeCtext ct_A , Weight B, Bias b, int  l,     int  alpha,     double scale,     int rsbits,
                 threeCtext ct_AB,                   int& l_out, int& alpha_out, double scale_out);
				 
void cpmm_mean_host(Ctext ct     , int  l,     int  alpha,     double scale,     int rsbits,
                    Ctext ct_mean, int& l_out, int& alpha_out, double scale_out);
					
void cpmm_FC1_host(Weight W1, Bias b1,
                   Weight W2, Bias b2,
				   Weight W3, Bias b3,
				   Weight W4, Bias b4,
				   threeCtext ct, int  l,     int  alpha,     double scale,     int rsbits,
                   threeCtext out1, 
                   threeCtext out2, 
                   threeCtext out3, 
                   threeCtext out4, int& l_out, int& alpha_out, double scale_out);
				   
void cpmm_FC2_host(Weight W1,
                   Weight W2,
				   Weight W3,
				   Weight W4, Bias b,
				   threeCtext ct1, 
                   threeCtext ct2, 
                   threeCtext ct3, 
                   threeCtext ct4, int  l,     int  alpha,     double scale,     int rsbits,
				   threeCtext out, int& l_out, int& alpha_out, double scale_out);
				   
void cpmm_pool_host(threeCtext ct_A , Weight B, Bias b, int  l,     int  alpha,     double scale,     int rsbits,
                    threeCtext ct_AB,                   int& l_out, int& alpha_out, double scale_out);

void cpmm_classification_host(threeCtext ct_A , Weight B, Bias b, int  l,     int  alpha,     double scale,     int rsbits,
                              threeCtext ct_AB,                   int& l_out, int& alpha_out, double scale_out);				   