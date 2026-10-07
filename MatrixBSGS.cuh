#pragma once
#include <functional>
#include <set>
#include "Matrix.cuh"

template<int N_>
struct MatrixBSGS {
	int NumB, NumG;
	int* shiftB, *shiftG;
	double*** diag;
	bool** BSGS_map;
	int NumDiags;

	MatrixBSGS() {
		shiftB = 0; shiftG = 0; diag = 0; BSGS_map = 0;
	}

	MatrixBSGS(const MatrixBSGS& A) {
		allocate_parameters(A.NumDiags);
		NumB = A.NumB; NumG = A.NumG;
		memcpy(shiftB, A.shiftB, sizeof(int) * NumDiags);
		memcpy(shiftG, A.shiftG, sizeof(int) * NumDiags);
		for (int i = 0; i < NumDiags; i++)
			memcpy(BSGS_map[i], A.BSGS_map[i], sizeof(bool) * NumDiags);
	}

	void operator=(const MatrixBSGS& A) {
		if (shiftB != 0) delete[] shiftB;
		if (shiftG != 0) delete[] shiftG;
		if (diag != 0) {
			for (int j = 0;j < NumDiags;j++) {
				for (int i = 0;i < NumDiags;i++)
					delete[] diag[j][i];
				delete[]diag[j];
			}
			delete[] diag;
		}
		if (BSGS_map != 0) {
			for (int j = 0;j < NumDiags;j++)
				delete[] BSGS_map[j];
			delete[] BSGS_map;
		}
		//
		allocate_parameters(A.NumDiags, A.NumG, A.NumB);
		memcpy(shiftB, A.shiftB, sizeof(int) * NumDiags);
		memcpy(shiftG, A.shiftG, sizeof(int) * NumDiags);
		for (int i = 0; i < NumDiags; i++)
			memcpy(BSGS_map[i], A.BSGS_map[i], sizeof(bool) * NumDiags);

		for (int i = 0;i < NumG;i++)
		for (int j = 0;j < NumB;j++) 
			if (BSGS_map[i][j] == true) 
				memcpy(diag[i][j],A.diag[i][j],sizeof(double)*N_);
	}

	MatrixBSGS(const Matrix<N_>& A, int BSGS_ratio) {
		std::set<int,std::less<int>> setB_optimal, setG_optimal; 
		int interval_optimal; double min = 1E8;
		for (int interval = 1; interval < N_ / 2; interval *= 2) {
			std::set<int, std::less<int>> setB, setG;
			for (int d = 0;d < A.NumDiags;d++) {
				int shift = A.shift[d];
				//shift = (shift < N_ / 4) ? shift : -(N_ / 2 - shift);
				setB.insert(shift % interval);
				setG.insert((shift / interval) * interval);
			}
			double ratio = double(setB.size()) / setG.size();
			if (fabs(ratio - BSGS_ratio) < min) { 
				min = fabs(ratio - BSGS_ratio); interval_optimal = interval;
				setB_optimal = setB; setG_optimal = setG; 
			}
		}
		//
		allocate_parameters(A.NumDiags, int(setG_optimal.size()), int(setB_optimal.size()));
		for (int d = 0;d < NumDiags;d++) {
			int shift = A.shift[d];
			//shift = (shift < N_ / 4) ? shift : -(N_ / 2 - shift);
			int shiftB = shift % interval_optimal;
			int shiftG = (shift / interval_optimal) * interval_optimal;
			int i = std::distance(setB_optimal.begin(), setB_optimal.find(shiftB)); this->shiftB[i] = (shiftB + N_ / 2) % (N_ / 2);
			int j = std::distance(setG_optimal.begin(),setG_optimal.find(shiftG)); this->shiftG[j] = (shiftG + N_ / 2) % (N_ / 2);
			BSGS_map[j][i] = true;
			memcpy(diag[j][i], A.diag[d], sizeof(double) * N_);
		}
	}
	
	MatrixBSGS(const Matrix<N_>& A, int interval, bool with_interval) {
		std::set<int, std::less<int>> setB, setG;
		for (int d = 0;d < A.NumDiags;d++) {
			int shift = A.shift[d];
			int shiftB = shift % interval;
			int shiftG = (shift / interval) * interval;
			if (shiftB < 0) { shiftB += interval; shiftG-=interval; }
			setB.insert(shiftB);
			setG.insert(shiftG);
		}
		//
		allocate_parameters(A.NumDiags, int(setG.size()), int(setB.size()));
		for (int d = 0;d < NumDiags;d++) {
			int shift = A.shift[d];
			//shift = (shift < N_ / 4) ? shift : -(N_ / 2 - shift);
			int shiftB = shift % interval;
			int shiftG = (shift / interval) * interval;
			if (shiftB < 0) { shiftB += interval; shiftG -= interval; }
			int i = std::distance(setB.begin(), setB.find(shiftB)); this->shiftB[i] = (shiftB + N_ / 2) % (N_ / 2);
			int j = std::distance(setG.begin(), setG.find(shiftG)); this->shiftG[j] = (shiftG + N_ / 2) % (N_ / 2);
			BSGS_map[j][i] = true;
			memcpy(diag[j][i], A.diag[d], sizeof(double) * N_);
		}
	}
	~MatrixBSGS() {
		if (shiftB != 0) delete[] shiftB;
		if (shiftG != 0) delete[] shiftG;
		if (diag != 0) {
			for (int j = 0;j < NumG;j++) {
				for (int i = 0;i < NumB;i++)
					delete[] diag[j][i];
				delete []diag[j];
			}
			delete[] diag;
		}
		if (BSGS_map != 0) {
			for (int j = 0;j < NumDiags;j++)
				delete[] BSGS_map[j];
			delete[] BSGS_map;
		}

	}


	void allocate_parameters(int A_NumDiags, int A_NumG, int A_NumB ) {
		NumDiags = A_NumDiags;
		NumG = A_NumG;
		NumB = A_NumB;
		shiftB = new int[NumDiags];  shiftG = new int[NumDiags];
		diag = new double * *[NumG];
		for (int i = 0;i < NumG;i++) {
			diag[i] = new double * [NumB];
			for (int j = 0;j < NumB;j++) {
				diag[i][j] = new double[N_];
				memset(diag[i][j], 0, sizeof(double) * N_);
			}
		}
		BSGS_map = new bool* [NumDiags];
		for (int i = 0;i < NumDiags;i++)
			BSGS_map[i] = new bool[NumDiags];
		
	}
	

	void initialize_parameters() {
		NumB = 0; NumG = 0;
		memset(shiftB, 0, sizeof(int) * NumDiags);
		memset(shiftG, 0, sizeof(int) * NumDiags);
		for (int i = 0; i < NumDiags; i++)
			memset(BSGS_map[i], 0, sizeof(bool) * NumDiags);
	}
		
	void applyA(const double x[N_], double Ax[N_]) const {
		
		memset(Ax, 0, sizeof(double) * N_);

		for(int i=0;i<NumG;i++)
		for(int j=0;j<NumB;j++){
			if (BSGS_map[i][j] == true) {
				for (int k = 0;k < N_ / 2;k++) {
					Ax[k    ] += diag[i][j][k    ]*x[(k+shiftG[i]+shiftB[j])%(N_/2)    ]
						       - diag[i][j][k+N_/2]*x[(k+shiftG[i]+shiftB[j])%(N_/2)+N_/2];
					Ax[k+N_/2] += diag[i][j][k+N_/2]*x[(k+shiftG[i]+shiftB[j])%(N_/2)    ]
						       + diag[i][j][k    ]*x[(k+shiftG[i]+shiftB[j])%(N_/2)+N_/2];
				}
			}
		}
		
	}
	
	void operator*=(double scale) {
		for (int i = 0;i < NumG;i++)
			for (int j = 0;j < NumB;j++)
				if (BSGS_map[i][j])
					for (int k = 0;k < N_;k++)
						diag[i][j][k] *= scale;
	}


	void print_Bsize_Gsize_effective() const {
		int num_babyRot = 0;
		for (int j = 0; j < NumB; j++) {
			int count=0;
			if (shiftB[j] != 0) {
				for (int i = 0; i < NumG; i++)
					if (BSGS_map[i][j] != 0) 
						count++;
			}
			if (count != 0) num_babyRot++;
		}
		int num_giantRot = 0;
		for (int i = 0; i < NumG; i++) {
			int count = 0;
			if (shiftG[i] != 0) {
				for (int j = 0; j < NumB; j++)
					if (BSGS_map[i][j] != 0) count++;
			}
			if (count != 0) num_giantRot++;
		}
		printf("NumBxNumG=%dx%d, effectively %dx%d\n", NumB, NumG, num_babyRot, num_giantRot);
		printf("shift = ");	for (int j = 0;j < NumB;j++) printf("%d, ", shiftB[j]<N_/4 ? shiftB[j] : -(N_/2-shiftB[j]));
		printf("; ");		for (int i = 0;i < NumG;i++) printf("%d, ", shiftG[i]<N_/4 ? shiftG[i] : -(N_/2-shiftG[i]));
		printf("\n");
	}

	void print_sum_dij_sqr(const char* name ) const {
		double sum = 0;
		for(int i=0;i<NumG;i++)
		for(int j=0;j<NumB;j++)
			if (BSGS_map[i][j] == 1 && shiftB[j] != 0) {
				for (int k = 0;k < N_;k++)
					sum += diag[i][j][k] * diag[i][j][k];
			}
		printf("matrix %s : sum_dij_sqr = %f\n", name, sum);
	}
};
