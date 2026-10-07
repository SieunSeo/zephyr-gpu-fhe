#pragma once


#include "CKKS_GPU.cuh"


void initialize_bts();
void deallocate_bts();


void set_cts_stc_matrix_host();

__global__
void ModRaise_kernel(Ctext ct);

void ModRaise_host(Ctext ct);

void CTS_host(Ctext ct, int l, int alpha, double scale_pt, Ctext ct_cts, double z_cts_ex[N], int s[N]);

void STC_host(Ctext ct, int l, int alpha, double scale_pt, Ctext ct_stc, int s[N]);

void Evalmod_host(Ctext T1, int l_T1, int alpha_T1, double scale_T1, int rs_bits, Ctext out, int& l_out, int& alpha_out, double scale_out);
	
	

void CTS_host_noDebug(Ctext ct, int l, int alpha, double scale_pt, Ctext ct_cts, double z_cts_ex[N], int s[N]);

void STC_host_noDebug(Ctext ct, int l, int alpha, double scale_pt, Ctext ct_stc, int s[N]);

void Evalmod_host_noDebug(Ctext T1, int l_T1, int alpha_T1, double scale_T1, int rs_bits, Ctext out, int& l_out, int& alpha_out, double scale_out);
	
	
void BTS_host(Ctext ct, Ctext out, int& l_out, int& alpha_out, double& scale_out);         // error print o
void BTS_host_noDebug(Ctext ct, double scale_ct, Ctext out, int& l_out, int& alpha_out, double scale_out); // error print x, log scale version
