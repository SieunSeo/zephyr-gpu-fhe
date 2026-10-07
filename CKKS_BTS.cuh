#pragma once


#include "CKKS_HalfPrecision.cuh"


extern int s[N];
extern CKKS_HalfPrecision ckks;

void set_cts_stc_matrix();
void ModRaise(Ctext ct);
void CTS(const Ctext ct, int l, int alpha, Ctext ct_cts, double z_cts_ex[N]);
void STC(const Ctext ct, int l, int alpha, Ctext ct_stc);
void Evalmod(Ctext T1, int l_T1, int alpha_T1, double scale_T1, int rs_bits, Ctext out, int& l_out, int& alpha_out, double scale_out) ;

