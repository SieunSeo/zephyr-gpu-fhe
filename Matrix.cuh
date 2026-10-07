#pragma once

#include <cassert>
#include <complex>
#include <vector>

static double PI = 3.141592653589793238462643383279502884197;

//-------------------------------------------------------------------------------
// SparseComplexMatrix
//-------------------------------------------------------------------------------
template<int N_>
struct Matrix {
	int NumDiags;
	int   * shift;
	double** diag;

	Matrix() {
		NumDiags = 0;
		shift = 0;
		diag = 0;
	}
	~Matrix() {
		if(shift != 0) delete[] shift;
		if (diag != 0) {
			for (int i = 0;i < NumDiags;i++)
				delete[] diag[i];
			delete[] diag;
		}
	}
	void operator=(const Matrix<N_>& A) {
		initialize_Numdiags(A.NumDiags);
		memcpy(shift, A.shift, sizeof(int) * A.NumDiags);
		for (int i = 0;i < NumDiags;i++)
			memcpy(diag[i], A.diag[i], sizeof(double) * N_);
	}
	void initialize_Numdiags(int NumDiags_) {
		if (shift != 0) delete[] shift;
		if (diag != 0) {
			for (int i = 0;i < NumDiags;i++)
				delete[] diag[i];
			delete[] diag;
		}
		NumDiags = NumDiags_;
		shift = new int[NumDiags];
		diag = new double*[NumDiags];
		for (int i = 0;i < NumDiags;i++)
			diag[i] = new double[N_];
		memset(shift, 0, sizeof(int) * NumDiags);
		
		for (int k = 0; k < NumDiags; k++) {
			memset(diag[k], 0, sizeof(double) * N_);
		}
	}
	void applyA(const double x[N_],double Ax[N_]) const {
		for (int i = 0; i < N_/2; i++) {
			Ax[i] = Ax[i+N_/2] = 0;
			for (int k = 0; k < NumDiags; k++) {
				int j = (i + shift[k]) % (N_/2);
				double aijr = diag[k][i    ];
				double aiji = diag[k][i+N_/2];
				Ax[i    ] += aijr * x[j] - aiji * x[j+N_/2];
				Ax[i+N_/2] += aijr * x[j+N_/2] + aiji * x[j];
			}
		}
	}
	void transpose() {
		for (int k = 0; k < NumDiags; k++) {
			int s = shift[k];
			shift[k] = (N_/2 - s) % (N_/2);
			double temp[N_];
			for (int i = 0; i < N_/2; i++) { temp[i] = diag[k][i]; temp[i+N_/2] = diag[k][i+N_/2]; }
			for (int i = 0; i < N_/2; i++) {
				diag[k][i    ] = temp[(i + (N_/2) - s) % (N_/2)     ];
				diag[k][i+N_/2] = temp[(i + (N_/2) - s) % (N_/2)+N_/2 ];
			}
		}
	}

	void conjugate() {
		for (int k = 0; k < NumDiags; k++)
			for (int i = 0; i < N_/2; i++) diag[k][i+N_/2] = -diag[k][i+N_/2];
	}

	void merge_diagonals(int Nsub) {
		bool* shift_valid = new bool[NumDiags]; 
		int count = 0;

		for (int i = 0; i < NumDiags; i++) {
			shift_valid[i] = true;
			count++;

			for (int j = 0; j < i; j++) 
				if ((shift[j] % (Nsub / 2) == shift[i] % (Nsub / 2))&& shift_valid[j]) {
					shift_valid[i] = false;
					for (int k = 0; k < N_ / 2; k++) {
						diag[j][(k + N_ / 2 - shift[j]) % (N_ / 2)        ] += diag[i][(k + N_ / 2 - shift[j]) % (N_ / 2)        ];
						diag[j][(k + N_ / 2 - shift[j]) % (N_ / 2) + N_ / 2] += diag[i][(k + N_ / 2 - shift[j]) % (N_ / 2) + N_ / 2];
					}
					count--;
				}
			
		}

		Matrix<N_> copy; copy = *this;
		initialize_Numdiags(count); count = 0;
		for (int i = 0; i < copy.NumDiags; i++) 
			if (shift_valid[i]) {
				shift[count] = copy.shift[i];
				memcpy(diag[count], copy.diag[i], sizeof(double) * N_);
				count++;
			}
		
	}

};

template< int N_ >
void matmul(const Matrix<N_>& A,
			const Matrix<N_>& B,
	              Matrix<N_>& C) {
	int NumDiagsA = A.NumDiags;
	int NumDiagsB = B.NumDiags;
	int count = 0; int buff[N_];
	for (int iA = 0; iA < NumDiagsA; iA++)
	for (int iB = 0; iB < NumDiagsB; iB++) {
		int shiftA = A.shift[iA];
		int shiftB = B.shift[iB];
		int shiftC = (shiftA + shiftB) % (N_/2);
		//
		int iC = -1;
		for (int i = 0; i < count; i++) {
			if (buff[i] == shiftC)
				iC = i;
		}
		if (iC == -1) { buff[count] = shiftC; count++; }
	}
	
	C.initialize_Numdiags(count);
	count = 0;
	for (int iA = 0; iA < NumDiagsA; iA++)
	for (int iB = 0; iB < NumDiagsB; iB++) {
		int shiftA = A.shift[iA];
		int shiftB = B.shift[iB];
		int shiftC = (shiftA + shiftB) % (N_ / 2);
		//
		int iC = -1;
		for (int i = 0; i < count; i++) {
			if (C.shift[i] == shiftC)
				iC = i;
		}
		if (iC == -1) { 
			iC = count; C.shift[iC] = shiftC; count++; 
		}
		for (int i = 0; i < N_ / 2; i++) {
			int k = (i + shiftA) % (N_ / 2);
			C.diag[iC][i] += A.diag[iA][i] * B.diag[iB][k] - A.diag[iA][i + N_ / 2] * B.diag[iB][k + N_ / 2];
			C.diag[iC][i + N_ / 2] += A.diag[iA][i] * B.diag[iB][k + N_ / 2] + A.diag[iA][i + N_ / 2] * B.diag[iB][k];
		}
	}
}


template <int N_>
void mat_conjtranspose(const Matrix<N_>& A,
							 Matrix<N_>& Aconjtranspose) {
	int NumDiags = A.NumDiags;
	Aconjtranspose.initialize_Numdiags(NumDiags);
	for (int d = 0; d < NumDiags; d++) {
		Aconjtranspose.shift[d] = A.shift[d];
		for (int i = 0; i < N_/2; i++) {
			Aconjtranspose.diag[d][i    ] = 0;
			Aconjtranspose.diag[d][i+N_/2] = 0;
		}
	}
	//
	for (int d = 0; d < NumDiags; d++) {
		int d_ = 0;
		for (; d_ < NumDiags; d_++)
			if (Aconjtranspose.shift[d_] == ((N_/2) - A.shift[d]) % (N_/2))
				break;
		assert(d_ != NumDiags);
		//
		int s = A.shift[d];
		for (int i = 0; i < N_/2; i++) {
			Aconjtranspose.diag[d_][(i + s) % (N_/2)      ] += A.diag[d][i    ];
			Aconjtranspose.diag[d_][(i + s) % (N_/2) + N_/2] -= A.diag[d][i+N_/2];
		}
	}
}


template <int N_>
void mat_conjtranspose(const Matrix<N_>& A,
							 Matrix<N_>& Aconjtranspose, int Nsub) {
	int NumDiags = A.NumDiags;
	Aconjtranspose.initialize_Numdiags(NumDiags);
	for (int d = 0; d < NumDiags; d++) {
		Aconjtranspose.shift[d] = A.shift[d];
		for (int i = 0; i < N_/2; i++) {
			Aconjtranspose.diag[d][i    ] = 0;
			Aconjtranspose.diag[d][i+N_/2] = 0;
		}
	}
	//
	for (int d = 0; d < NumDiags; d++) {
		int d_ = 0;
		for (; d_ < NumDiags; d_++)
			if ((Aconjtranspose.shift[d_]) % (Nsub / 2) == ((N_ / 2) - A.shift[d]) % (Nsub / 2))
				break;
		assert(d_ != NumDiags);
		//
		int s = A.shift[d];
		for (int i = 0; i < N_/2; i++) {
			Aconjtranspose.diag[d_][(i + s) % (N_/2)      ] += A.diag[d][i    ];
			Aconjtranspose.diag[d_][(i + s) % (N_/2) + N_/2] -= A.diag[d][i+N_/2];
		}
	}
}


//-------------------------------------------------------------------------------
// splitU0R into FFT Sparse matrices
//-------------------------------------------------------------------------------
template< int logN_ >
void splitU0R(Matrix<(1 << (logN_))> E[logN_ - 1]) {
	int N_ = 1 << logN_;
	double PI_over_N = PI / double(N_);
	for (int k = 2, k_ = 0; k <= N_ / 2; k *= 2, k_++) {
		E[k_].initialize_Numdiags(3);
		int M = N_ / 2 / k; 
		E[k_].shift[2] = 0;
		E[k_].shift[0] = M;
		E[k_].shift[1] = N_ / 2 - M;
		double Wr[1 << (logN_ - 2)];
		double Wi[1 << (logN_ - 2)];
		for (int i = 0, fiveik = k / 2; i < M; i++, fiveik = (fiveik * 5) % (2 * N_)) {
			Wr[i] = cos(PI_over_N * double(fiveik));
			Wi[i] = sin(PI_over_N * double(fiveik));
		}
		for (int b = 0; b < k / 2; b++)
			for (int i = 0; i < M; i++) {
				E[k_].diag[2][2 * M * b + i] = 1;
				E[k_].diag[1][2 * M * b + M + i] = 1;
				E[k_].diag[0][2 * M * b + i] = Wr[i];
				E[k_].diag[0][2 * M * b + i + N_/2] = Wi[i];
				E[k_].diag[2][2 * M * b + M + i] = -Wr[i];
				E[k_].diag[2][2 * M * b + M + i + N_ / 2] = -Wi[i];
			}
	}
}
template< int logN_, int logNsub >
void splitU0R(Matrix< 1 << logN_> E[]) {
	const int N_ = 1 << logN_;
	const int Nsub = 1<<logNsub;
	double PI_over_N = PI / double(N_);
	for (int k = (N_ / Nsub) * 2, k_ = 0; k <= N_ / 2; k *= 2, k_++) {
		E[k_].initialize_Numdiags(3);
		int M = N_ / 2 / k;
		E[k_].shift[2] = 0;
		E[k_].shift[0] = M;
		E[k_].shift[1] = N_ / 2 - M;
		double Wr[N_ / 4];
		double Wi[N_ / 4];
		for (int i = 0, fiveik = k / 2; i < M; i++, fiveik = (fiveik * 5) % (2 * N_)) {
			Wr[i] = cos(PI_over_N * double(fiveik));
			Wi[i] = sin(PI_over_N * double(fiveik));
		}
		for (int b = 0; b < k / 2 / (N_ / Nsub); b++)
			for (int i = 0; i < M; i++) {
				E[k_].diag[2][2 * M * b + i] = 1;
				E[k_].diag[1][2 * M * b + M + i] = 1;
				E[k_].diag[0][2 * M * b + i] = Wr[i];
				E[k_].diag[0][2 * M * b + i + N_ / 2] = Wi[i];
				E[k_].diag[2][2 * M * b + M + i] = -Wr[i];
				E[k_].diag[2][2 * M * b + M + i + N_ / 2] = -Wi[i];
			}
	}
}