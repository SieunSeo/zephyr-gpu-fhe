#pragma once

#include <math.h>
#include "MatrixBSGS.cuh"
#include "CKKS_HalfPrecision.cuh"


struct CKKS_lineartransform : public MatrixBSGS<N> {
	
	//=======================================================================
	//  setting
	//=======================================================================
	CKKS_HalfPrecision& ckks;
	double Delta_pt;

	//=======================================================================
	//  Initialize
	//=======================================================================
	CKKS_lineartransform(CKKS_HalfPrecision& ckks_input, const Matrix<N>& A, int BSGS_ratio, double Delta_pt_input);
	CKKS_lineartransform(CKKS_HalfPrecision& ckks_input) :ckks(ckks_input) {};

	void operator=(const CKKS_lineartransform& A) {
		MatrixBSGS<N>::operator=(A);
		ckks = A.ckks;
		Delta_pt = A.Delta_pt;
	}
	//=======================================================================
	//  lineartransform (BSGS, double hoisting)
	//=======================================================================
	void applyA(const Ctext ct_hat, int l, int alpha, Ctext res_hat) const;
	void applyA(const double x[N], double Ax[N]) const { MatrixBSGS<N>::applyA(x, Ax); }
};

