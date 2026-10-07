#pragma once

#include <fstream>
#include "CKKS_lineartransform_GPU.cuh"
#include "CKKS_GPU.cuh"
#include "MatrixFP32_GPU.cuh"

//=================================
// initialize
//=================================
void initialize_bert();
void deallocate_bert();

//=================================
// load
//=================================
template<int n1>
void load1D(const char* name, double w[n1]) {
	std::ifstream is(name);
	int dim; is >> dim; 
	int temp; is >> temp;
	assert(temp == n1);
	for (int i1 = 0;i1 < n1;i1++)
		is >> w[i1];
}

template<int n1, int n2>
void load2D(const char* name, double w[n1][n2]) {
	std::ifstream is(name);
	int dim; is >> dim;
	char comma; int temp1, temp2;
	is >> temp1 >> comma >> temp2;
	assert(temp1 == n1 && temp2 == n2);
	for (int i1 = 0;i1 < n1;i1++)
		for (int i2 = 0;i2 < n2;i2++)
			is >> w[i1][i2];
}

template<int n1>
void load1D(const char* name, float w[n1]) {
	std::ifstream is(name);
	int dim; is >> dim; 
	int temp; is >> temp;
	assert(temp == n1);
	for (int i1 = 0;i1 < n1;i1++)
		is >> w[i1];
}

template<int n1, int n2>
void load2D(const char* name, float w[n1][n2]) {
	std::ifstream is(name);
	int dim; is >> dim;
	char comma; int temp1, temp2;
	is >> temp1 >> comma >> temp2;
	assert(temp1 == n1 && temp2 == n2);
	for (int i1 = 0;i1 < n1;i1++)
		for (int i2 = 0;i2 < n2;i2++)
			is >> w[i1][i2];
}

template<int n1, int n2>
void save2D(const char* name, double w[n1][n2]) {
	std::ofstream os(name);
	assert(os.is_open());

	os << 2 << '\n';
	os << n1 << ',' << n2 << '\n';

	for (int i1 = 0; i1 < n1; i1++) {
		for (int i2 = 0; i2 < n2; i2++) {
			os << w[i1][i2];
			if (i2 + 1 < n2) os << ' ';
		}
		os << '\n';
	}
}

//=================================
// pack / unpack
//=================================
void pack_matrix_to_threeData(double A[128][768], double z[3][N]);
void unpack_threeData_to_matrix(double z[3][N], double A[128][768]);
void unpack_Ctext_to_matrix(double z[N], double A[128][256]); 
void unpack_Ctext_to_matrix(double z[N], double A[128][768], int k);
void pack_matrix_to_threeCtext_host(double A[128][768], int l, int alpha, double scale, int s[N], Ctext ct_Apack[3]);
void pack_matrix_to_threePtext_host(double A[128][768], int l, int alpha, double scale, threePtext pt_Apack);
void pack_bias_to_threePtext_host(double b[768], int l, int alpha, double scale, threePtext pt_bpack);

//=================================
// convert matrix 
//=================================
void convert_B_into_Bij(double B[768][768], Matrix<N> Bs[3][3]);
void convert_B_into_Bij(double B[768][768], MatrixBSGS<N> Bs[3][3]);

//=================================
// set matrices for ccmm
//=================================
void set_Utranspose (Matrix<N>& Utranspose);
void set_Usigma     (Matrix<N>& Usigma);
void set_Utau       (Matrix<N>& Utau);
void set_Uphi(int l, Matrix<N>& Uphi) ;
void set_Upsi(int l, Matrix<N>& Upsi);
void set_Uflip      (Matrix<N>& Uflip);


//=================================
// data operation
//=================================
void rot(double z[N], int r) ;
void rot(double z[N], int r, double zrot[N]);
void hadamard(double z1[N], double z2[N], double out[N]);
void add(double z1[N], double z2[N], double out[N]);

//=================================
// print
//=================================
void print(double A[128][256]);
void print(double A[768][768]);
void print(double A[128][128], const char* name);
void error(double A[128][256], double A_[128][256], const char* name) ;

//=================================
// Ctext operation
//=================================
void rot(Ctext ct, int l, int alpha, int r, Ctext out);
void conv(Ptext pt, Ctext ct, int l, int alpha, Ctext out);

//============================================
// CPMM : out = A*B / out = A*B+b
//============================================
void set_cpmm_matrix_host(Weight W, int r, int c, MatrixFP32_GPU& A);
void set_cpmm_matrix_host(Weight W, int r, int c, MatrixFP32_GPU& A, float constant);
void set_cpmm_matrix_host(Weight Wr,Weight Wi, int r, int c, MatrixFP32_GPU& A) ;
void set_cpmm_matrix_host(Weight W, int r, int c1,int c2, MatrixFP32_GPU& A);
void set_cpmm_FC2_matrix_host(Weight Wr, Weight Wi, int r, int c, MatrixFP32_GPU& A);
void set_sigma_cpmm_matrix_host(Weight Wr,Weight Wi, int r, int c, MatrixFP32_GPU& A) ;
void set_cpmm_pool_matrix_host(Weight W, int r, int c, MatrixFP32_GPU& A, float constant);
void set_cpmm_pool_matrix_host(Weight W, int r, int c1,int c2, MatrixFP32_GPU& A);
void set_cpmm_classification_matrix_host(Weight W, int r, int c, MatrixFP32_GPU& A, float constant);
void set_cpmm_classification_matrix_host(Weight W, int r, int c1,int c2, MatrixFP32_GPU& A);

void add_bias_host(Ctext ct, int l, int alpha, double scale, float b[256]);
void add_bias_host(Ctext ct, int l, int alpha, double scale, float br[256], float bi[256]);
void add_bias_host(Ctext ct, int l, int alpha, double scale, float br[256], float bi[256], float c);
void add_sigma_bias_host(Ctext ct, int l, int alpha, double scale, float br[256], float bi[256], float c);

//============================================
// CCMM 
//============================================
void ccmm_QKT_host(threeCtext ct_A,  int lA, int aA, double sA,
                   threeCtext ct_B,  int lB, int aB, double sB, int rsbits,          
				   threeCtext ct_AB, int& l_out, int& alpha_out, double& scale_out);	
void ccmm_Y_host(threeCtext ct_A,  int lA, int aA, double sA,
                 threeCtext ct_B,  int lB, int aB, double sB, int rsbits,          
				 threeCtext ct_AB, int& l_out, int& alpha_out, double& scale_out);
//============================================
// softmax
//============================================

void make_BPMax_mask_host(int seq_len, double Rd[128][768], int l, int alpha, double scale, 
                          threePtext pt_mask_real, threePtext pt_mask_imag);
void BPMax_host(int seq_len, double Rd[12][128],
                threeCtext ct,  int  l,     int  alpha,     double scale, int rs_bits, 
				threeCtext out, int& l_out, int& alpha_out, double& scale_out);



			   
//============================================
// attention
//============================================
void attention_host(Weight W_Q, Bias b_Q,
					Weight W_K, Bias b_K,
					Weight W_V, Bias b_V,
					Weight W_O, Bias b_O, double h_attn_Rd[12][128],
                    threeCtext ct , int  l    , int  alpha    , double  scale    , int rs_bits,
                    threeCtext out, int& l_out, int& alpha_out, double scale_out );
					

//============================================
// layer norm
//============================================
void BatchLN_host(float Rd[128], Bias gamma, Bias beta,  
                  threeCtext ct , int  l    , int  alpha    , double scale    , int rs_bits,
				  threeCtext out, int& l_out, int& alpha_out, double& scale_out);					


//====================================================
// FeedForward
//====================================================
void feedforward_host(float W_ff1[3072][768], float b_ff1[3072],
		              float W_ff2[768][3072], float b_ff2[768] ,
		              threeCtext ct , int  l    , int  alpha    , double scale    ,int rs_bits,
		              threeCtext out, int& l_out, int& alpha_out, double scale_out );
void feedforward_host_time(float W_ff1[3072][768], float b_ff1[3072],
		              float W_ff2[768][3072], float b_ff2[768] ,
		              threeCtext ct , int  l    , int  alpha    , double scale    , int rs_bits,
		              threeCtext out, int& l_out, int& alpha_out, double scale_out );
					  
//====================================================
// tanh_host
//====================================================					  
void tanh_host(threeCtext ct , int  l    , int  alpha    , double  scale    , int rs_bits,
		       threeCtext out, int& l_out, int& alpha_out, double& scale_out );