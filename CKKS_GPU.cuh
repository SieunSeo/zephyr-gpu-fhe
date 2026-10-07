#pragma once

#include "Parameter_GPU.cuh"

void initialize_ckks();

void deallocate_ckks();


//------------------------------------------------------
// Encode / Decode
//------------------------------------------------------
void encode_host(double z[N], double scale, int l, int alpha, Ptext pt);
void encode_host(float z[N], double scale, int l, int alpha, Ptext pt);
void encode_QP_host(double z[N], double scale, int l, int alpha, Ptext_QP pt);
void encode_QP_host(double z[N], double scale, int l, int alpha, Ptext_QP pt);
void encode_QP_host(float z[N], double scale, int l, int alpha, Ptext_QP pt);

void crt4_host(uint32_t q[4], uint32_t a[4][N], double pt[N]);
void decode_host(Ptext pt, int l, double scale, double z[N]);
void decode_QP_host(Ptext_QP pt, int l, double scale, double z[N]);
void decode_graft_host(Ptext pt, int l, int alpha, double scale, double z[N]);



//------------------------------------------------------
// convhat
//------------------------------------------------------
void convhatQ_host(Ptext pt1, Ptext pt2, int l, int alpha, Ptext out);
void convhat_QP_host(Ptext_QP pt1, Ptext_QP pt2, int l, int alpha, Ptext_QP out);

void convhatQ_host(Ctext ct1, Ctext ct2, int l, int alpha, Ctext out);
void convhat_QP_host(Ctext_QP ct1, Ctext_QP ct2, int l, int alpha, Ctext_QP out);

void convhat_QP_host(Ptext_QP pt1, Ctext_QP ct2, int l, int alpha, Ctext_QP out);


//----------------------------------------------------------------------------------------------------------------
// add / sub
//----------------------------------------------------------------------------------------------------------------
void add_host(Ptext pt1, Ptext pt2, int l, int alpha, Ptext out);
void sub_host(Ptext pt1, Ptext pt2, int l, int alpha, Ptext out);
void add_host(Ctext ct1, Ctext ct2, int l, int alpha, Ctext out);
void sub_host(Ctext ct1, Ctext ct2, int l, int alpha, Ctext out);
//-------------------------------------------------------------------------
void add_QP_host(Ptext_QP pt1, Ptext_QP pt2, int l, int alpha, Ptext_QP out);
void sub_QP_host(Ptext_QP pt1, Ptext_QP pt2, int l, int alpha, Ptext_QP out);

//------------------------------------------------------
// convhat+add
//------------------------------------------------------
void convhat_add_QP_host(Ptext_QP pt1, Ctext_QP ct2, Ctext_QP ct3, int l, int alpha, Ctext_QP out);


//----------------------------------------------------------------------------------------------------------------
// enc / dec
//----------------------------------------------------------------------------------------------------------------
void enc_host(Ptext pt, int l, int alpha, int s[N], Ctext ct);
void enc_QP_host(Ptext_QP pt, int l, int alpha, const int s[N], Ctext_QP ct);
void dec_host(Ctext ct, int l, int alpha, int s[N], Ptext pt);
void dec_QP_host(Ctext_QP ct, int l, int alpha, int s[N], Ptext_QP pt);

void dec_decode_host(Ctext ct, int l, int s[N], double scale, double z[N]);
void dec_decode_graft_host(Ctext ct, int l, int alpha, int s[N], double scale, double z[N]) ;
void dec_decode_graft_last_host(Ctext ct, int l, int alpha, int s[N], double scale, double z[N]) ;
void dec_decode_graft_ifft_host(Ctext ct, int l, int alpha, int s[N], double scale, double z[N]);

//----------------------------------------------------------------------------------------------------------------
// key switching
//----------------------------------------------------------------------------------------------------------------
extern int num_ks;
void initialize_ks(int s[N]);
void deallocate_ks();
void swkgen_host(int sfr[N], int sto[N], Swk swk);
void evkgen_host(int s[N], Swk evk);
void ckeygen_host(int s[N], Swk ckey);
__global__ void keygen_s_mod_kernel(int s[N], Ptext_QP smod);
__global__ void keygen_mulP_kernel(Ptext_QP smod, Ptext_Gadget pt);
void ks_host(Ctext cthat, int l, int alpha, Swk swk);
void decompose_host(Ptext a, int l, int alpha, Ptext_Gadget ginva);
__global__ void mulP_kernel(Ptext ct_hat, int l, int alpha, Ptext_QP Pct_hat);
void mult_sum_host(Ptext_Gadget ginva, int l, int alpha, Swk swk, Ctext_QP mulsum);

std::string name_rkeytilde_host(int shift);
void calculate_rkeytilde_host(int shift, int s[N], Swk swk);

void conj_hat_host(Ctext cthat, int l, int alpha, Ctext cthat_conj) ;

//----------------------------------------------------------------------------------------------------------------
// gadget
//----------------------------------------------------------------------------------------------------------------
void gadget_ginv_host( Ptext a, int l, int alpha, Ptext_Gadget ginva);

//----------------------------------------------------------------------------------------------------------------
// CKKS_GPU_rs.cu : rs, mul, mul_rs
//----------------------------------------------------------------------------------------------------------------
void initialize_rs(int s[N]);
void deallocate_rs();

void rshat_Ql_Qlmm_host(Ptext pthat, int l, int alpha, int m);
void rshat_Ql_Qlmm_host(Ctext cthat, int l, int alpha, int m);
void rshat_PQl_Ql_host(Ptext_QP pthat, int l, int alpha) ;
void rshat_PQl_Ql_host(Ctext_QP cthat, int l, int alpha) ;

void rshat_host(Ptext pthat, int  l    , int  alpha    , double  scale, int alpha_rs,
                             int& l_out, int& alpha_out, double& scale_out);
void rshat_host(Ctext cthat, int  l    , int  alpha    , double  scale, int alpha_rs,
                             int& l_out, int& alpha_out, double& scale_out);
//void rshat_log_host(Ctext cthat, int  l    , int  alpha    , double  scale_log, int alpha_rs,
//                                 int& l_out, int& alpha_out, double& scale_out_log);
void rshat_sim_host(int  l    , int  alpha    , double  scale, int alpha_rs,
                    int& l_out, int& alpha_out, double& scale_out);	
							
void mul_host(Ctext ct1, Ctext ct2, int l, int alpha, Ctext out);
void mul_rs_host(Ptext pt , int  l1   , int  alpha1   , double  scale1,
				 Ctext ct , int  l2   , int  alpha2   , double  scale2, int rs_bits,
				 Ctext out, int& l_out, int& alpha_out, double& scale_out);
void mul_rs_host(Ctext ct1, int  l1   , int  alpha1   , double  scale1,
				 Ctext ct2, int  l2   , int  alpha2   , double  scale2, int rs_bits,
				 Ctext out, int& l_out, int& alpha_out, double& scale_out);
void mul_rs_sim_host(int  l1   , int  alpha1   , double  scale1,
				     int  l2   , int  alpha2   , double  scale2, int rs_bits,
				     int& l_out, int& alpha_out, double& scale_out);


void mul_host(Ctext ct1, Ctext ct2, int l, int alpha, Ptext d0, Ptext d1, Ptext d2);
void ks_host(Ptext d0, Ptext d1, Ptext d2, int l, int alpha, Ctext out);

//----------------------------------------------------------------------------------------------------------------
// log scale
//----------------------------------------------------------------------------------------------------------------
void rshat_log_host(Ptext pthat, int  l    , int  alpha    , double  log_scale, int alpha_rs,
                                 int& l_out, int& alpha_out, double& log_scale_out);
void rshat_log_host(Ctext cthat, int  l    , int  alpha    , double  scale_log, int alpha_rs,
                                 int& l_out, int& alpha_out, double& scale_out_log);
void mul_rs_log_host(Ptext pt , int  l1   , int  alpha1   , double  log_scale1,
					 Ctext ct , int  l2   , int  alpha2   , double  log_scale2, int rs_bits,
					 Ctext out, int& l_out, int& alpha_out, double& log_scale_out);
void mul_rs_log_host(Ctext ct1, int  l1   , int  alpha1   , double  log_scale1,
				     Ctext ct2, int  l2   , int  alpha2   , double  log_scale2, int rs_bits,
				     Ctext out, int& l_out, int& alpha_out, double& log_scale_out);
void rshat_log_sim_host(int  l    , int  alpha    , double  log_scale, int alpha_rs,
                        int& l_out, int& alpha_out, double& log_scale_out);
void mul_rs_log_sim_host(int  l1   , int  alpha1   , double  log_scale1,
				         int  l2   , int  alpha2   , double  log_scale2, int rs_bits,
				         int& l_out, int& alpha_out, double& log_scale_out);

//----------------------------------------------------------------------------------------------------------------
// rothat
//----------------------------------------------------------------------------------------------------------------
void initialize_rot();
void deallocate_rot();
void rothat_host(Ptext pthat, int l, int alpha, int r, Ptext outhat);
void rothat_host(Ctext cthat, int l, int alpha, int r, Ctext outhat);
void rothat_QP_host(Ptext_QP pthat, int l, int alpha, int r, Ptext_QP outhat);
void rothat_QP_host(Ctext_QP cthat, int l, int alpha, int r, Ctext_QP outhat);
void rot_host(int s[N], int r, int srot[N]);


//===================================================
// combine / extract
//===================================================
void combine_host(Ctext ctR, Ctext ctI, int l, int alpha, Ctext ct_out);
void combine_QP_host(Ctext_QP ctR, Ctext_QP ctI, int l, int alpha, Ctext_QP ct_out);
void extract_host(Ctext ct, int l, int alpha, Ctext ctR, Ctext ctI);
void extract_real_host(Ctext ct, int l, int alpha, Ctext ctR);


//===================================================
// graft
//===================================================
void increase_alpha_host(Ctext ct , int  l    , int  alpha    , double  scale    ,
                         Ctext out, int& l_out, int& alpha_out, double& scale_out);
void increase_alpha_sim_host(int  l    , int  alpha    , double  scale    ,
                             int& l_out, int& alpha_out, double& scale_out);