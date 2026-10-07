#pragma once

#include "MatrixBSGS.cuh"
#include "Parameter.cuh"


struct MatrixBSGS_GPU : public MatrixBSGS<N> {
	//device variable
	double* d_diag;
	
	MatrixBSGS_GPU(){ d_diag=0; }
	

	void initialize(const MatrixBSGS<N>& A){
		MatrixBSGS<N>::operator=(A);
		cudaMalloc(&d_diag, sizeof(double)*NumB*NumG*N);
		for (int i = 0;i < NumG;i++)
		for (int j = 0;j < NumB;j++) 
			if (BSGS_map[i][j]) 
				cudaMemcpy(d_diag + ((i*NumB)+j)*N, diag[i][j], sizeof(double)*N, cudaMemcpyHostToDevice );
	}
	
	void initialize_reusable_matrix(int A_NumG, int A_NumB){
		NumG = A_NumG;
		NumB = A_NumB;
		allocate_parameters(NumB*NumG, NumG, NumB);
		cudaMalloc(&d_diag, sizeof(double)*NumB*NumG*N);
	}
	void initialize_reusable_matrix(const MatrixBSGS<N>& A){
		memcpy(shiftB, A.shiftB, sizeof(int) * NumB);
		memcpy(shiftG, A.shiftG, sizeof(int) * NumG);
		for (int i = 0; i < NumG; i++)
			memcpy(BSGS_map[i], A.BSGS_map[i], sizeof(bool) * NumB);

		for (int i = 0;i < NumG;i++)
		for (int j = 0;j < NumB;j++) 
			if (BSGS_map[i][j]) 
				cudaMemcpy(d_diag + ((i*NumB)+j)*N, A.diag[i][j], sizeof(double)*N, cudaMemcpyHostToDevice );
	}
	
	
	double* get_d_diag(int i, int j){ return d_diag+((i*NumB)+j)*N; } 
	
	~MatrixBSGS_GPU() {
		if (d_diag != 0) cudaFree(d_diag);
	}
};
