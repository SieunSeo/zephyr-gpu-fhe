#pragma once
#include "Parameter.cuh"

struct MatrixFP32_GPU {
	int NumB, NumG;
	int* shiftB, *shiftG;
	float* d_diag;
	bool** BSGS_map;

	MatrixFP32_GPU() {
		NumG=NumB=0;
		shiftB = 0; shiftG = 0; d_diag = 0; BSGS_map = 0;
	}

	~MatrixFP32_GPU() {
		if (shiftB != 0) delete[] shiftB;
		if (shiftG != 0) delete[] shiftG;
		if (d_diag != 0) {
			cudaFree(d_diag);
		}
		if (BSGS_map != 0) {
			for (int j = 0;j < NumG;j++)
				delete[] BSGS_map[j];
			delete[] BSGS_map;
		}

	}


	void initialize( int A_NumG, int A_NumB ) {
		if( NumG != A_NumG || NumB != A_NumB ){
			NumG = A_NumG;
			NumB = A_NumB;
			shiftB = new int[NumB];  shiftG = new int[NumG];
			cudaMalloc(&d_diag, sizeof(float)*N*NumG*NumB);
			BSGS_map = new bool* [NumG];
			for (int i = 0;i < NumG;i++)
				BSGS_map[i] = new bool[NumB];
		
		}
	}
	
	float* get_d_diag(int i, int j){ return d_diag+((i*NumB)+j)*N; } 

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
		printf("shift = ");	for (int j = 0;j < NumB;j++) printf("%d, ", shiftB[j]<N/4 ? shiftB[j] : -(N/2-shiftB[j]));
		printf("; ");		for (int i = 0;i < NumG;i++) printf("%d, ", shiftG[i]<N/4 ? shiftG[i] : -(N/2-shiftG[i]));
		printf("\n");
	}
};
