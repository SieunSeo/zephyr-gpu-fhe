#include "BERT_GPU.cuh"

#include "CKKS_lineartransform_GPU.cuh"
#include "CKKS_BTS_GPU.cuh"
#include "CKKS_BTS.cuh"
#include "FFT_GPU.cuh"
#include "FFT_FP32_GPU.cuh"
#include "NTT_GPU.cuh"
#include "Graft_GPU.cuh"
#include "CKKS_polynomial_GPU.cuh"

threeCtext *ct_Q;       
threeCtext *ct_K;       
threeCtext *ct_V;       
threeCtext *ct_QKT;     
threeCtext *ct_softmax; 
threeCtext *ct_Y;       
threeCtext *ct_bts;     
threeCtext *ct_O;       
threeCtext *ct_ln;      
threeCtext *ct_ff;      
threeCtext *ct_ln2;     
threeCtext *ct_real;     
threeCtext *ct_imag;     
threeCtext *ct_tanh; 

void encoder_host(//attn
				  Weight Wq, Bias bq,
				  Weight Wk, Bias bk,
				  Weight Wv, Bias bv,
				  Weight Wo, Bias bo, double h_attn_Rd[12][128], int seq_len,
				  //BatchLN1
				  float LN1_Rd[128], Bias LN1_gamma, Bias LN1_beta, 
				  //FeedForward
				  float W_ff1[3072][768], float b_ff1[3072],
		          float W_ff2[768][3072], float b_ff2[768] ,
				  //BatchLN2
				  float LN2_Rd[128], Bias LN2_gamma, Bias LN2_beta, 
				  //data
				  threeCtext ct_x , int  l    , int  alpha    , double  scale    , int rs_bits,
                  threeCtext out, int& l_out, int& alpha_out, double& scale_out  );

void encoder_host_time(//attn
				  Weight Wq, Bias bq,
				  Weight Wk, Bias bk,
				  Weight Wv, Bias bv,
				  Weight Wo, Bias bo, double h_attn_Rd[12][128], int seq_len,
				  //BatchLN1
				  float LN1_Rd[128], Bias LN1_gamma, Bias LN1_beta, 
				  //FeedForward
				  float W_ff1[3072][768], float b_ff1[3072],
		          float W_ff2[768][3072], float b_ff2[768] ,
				  //BatchLN2
				  float LN2_Rd[128], Bias LN2_gamma, Bias LN2_beta, 
				  //data
				  threeCtext ct_x , int  l    , int  alpha    , double  scale    , int rs_bits,
                  threeCtext out, int& l_out, int& alpha_out, double& scale_out  );

int main() {
	ensure_data_directory_exists();
	
	//====================================================
	// initialize
	//====================================================
	initialize_parameter();
	initialize_ckks();
	initialize_graft();
	initialize_graft_Ctext();
	initialize_fft();
	initialize_fft_FP32();
	initialize_ntt();
	initialize_ks(s);
	initialize_rot();
	initialize_LT();
	initialize_poly();
	initialize_bts();
	initialize_rs(s);
	initialize_rkey(s);
	//
	initialize_bert();
	
	set_cts_stc_matrix_host();
	
	
	
	//time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//====================================================
	// Malloc
	//====================================================
	cudaMalloc(&ct_Q      ,CtextSize*3); 
	cudaMalloc(&ct_K      ,CtextSize*3); 
	cudaMalloc(&ct_V      ,CtextSize*3); 
	cudaMalloc(&ct_QKT    ,CtextSize*3); 
	cudaMalloc(&ct_softmax,CtextSize*3); 
	cudaMalloc(&ct_Y      ,CtextSize*3); 
	cudaMalloc(&ct_bts    ,CtextSize*3);
	cudaMalloc(&ct_O      ,CtextSize*3);  
	cudaMalloc(&ct_ln     ,CtextSize*3);  
	cudaMalloc(&ct_ff     ,CtextSize*3);  
	cudaMalloc(&ct_ln2    ,CtextSize*3); 
	cudaMalloc(&ct_real, CtextSize*3);
	cudaMalloc(&ct_imag, CtextSize*3);
	cudaMalloc(&ct_tanh, CtextSize*3);
	//====================================================
	// data declaration
	//====================================================
	//input
	double h_x[128][768];
	//attention
	Weight h_Wq, h_Wk, h_Wv, h_Wo;
	Bias   h_bq, h_bk, h_bv, h_bo;
	double h_attn_Rd[12][128];
	// device variable
	Weight *Wq, *Wk, *Wv, *Wo;
	Bias   *bq, *bk, *bv, *bo;
	cudaMalloc(&Wq, WeightSize); 
	cudaMalloc(&Wk, WeightSize); 
	cudaMalloc(&Wv, WeightSize); 
	cudaMalloc(&Wo, WeightSize); 
	cudaMalloc(&bq,   BiasSize); 
	cudaMalloc(&bk,   BiasSize); 
	cudaMalloc(&bv,   BiasSize); 
	cudaMalloc(&bo,   BiasSize); 
	// LayerNorm
	Bias h_LN1_gamma, h_LN1_beta; float h_ln1_Rd[128];
	// device variable
	Bias *LN1_gamma, *LN1_beta; float *LN1_Rd;
	cudaMalloc(&LN1_gamma, BiasSize);          
	cudaMalloc(&LN1_beta , BiasSize);          
	cudaMalloc(&LN1_Rd   , sizeof(float)*128); 
	// FeedForward
	float h_W_ff1[3072][768], h_b_ff1[3072];
	float h_W_ff2[768][3072], h_b_ff2[768];
	// device variable
	float (*W_ff1)[3072][768], *b_ff1;
	float (*W_ff2)[768][3072], *b_ff2;
	cudaMalloc(&W_ff1, sizeof(float)*3072*768); 
	cudaMalloc(&W_ff2, sizeof(float)*3072*768); 
	cudaMalloc(&b_ff1, sizeof(float)*3072    ); 
	cudaMalloc(&b_ff2, sizeof(float)*768     ); 
	// LayerNorm
	Bias h_LN2_gamma, h_LN2_beta; float h_ln2_Rd[128];
	// device variable
	Bias *LN2_gamma, *LN2_beta; float *LN2_Rd;
	cudaMalloc(&LN2_gamma, BiasSize);          
	cudaMalloc(&LN2_beta , BiasSize);          
	cudaMalloc(&LN2_Rd   , sizeof(float)*128); 
	
	int seq_len_list[]={ 36 , 49 , 128, 128, 44 , 58 , 101 , 51 , 38 , 51 , 50 , 42 , 43 , 79 , 56 , 48 , 34 , 74 , 54 , 99 , 53 , 128, 128, 128, 128, 42 , 54 , 44 , 49 , 80 , 41 , 128, 128, 81 , 41 , 39 , 62 , 68 , 37 , 53 , 46 , 68 , 128, 31 , 73 , 113 , 46 , 36 , 49 , 76 , 53 , 120 , 81 , 47 , 48 , 54 , 128, 39 , 25 , 74 , 36 , 81 , 38 , 118 , 54 , 50 , 42 , 59 , 34 , 114 , 52 , 119 , 38 , 94 , 74 , 60 , 66 , 56 , 128, 81 , 43 , 23 , 38 , 35 , 40 , 42 , 48 , 54 , 48 , 128, 109 , 38 , 125 , 48 , 58 , 128, 72 , 47 , 56 , 128, 128, 46 , 49 , 97 , 65 , 49 , 128, 36 , 31 , 85 , 128, 47 , 62 , 43 , 128, 49 , 34 , 103 , 37 , 78 , 54 , 59 , 42 , 128, 28 , 128, 46 , 41 , 40 , 71 , 36 , 62 , 60 , 48 , 82 , 24 , 44 , 65 , 38 , 128, 21 , 57 , 110 , 29 , 40 , 31 , 35 , 75 , 35 , 95 , 32 , 50 , 69 , 128, 52 , 128, 61 , 128, 38 , 33 , 56 , 53 , 43 , 31 , 67 , 54 , 63 , 58 , 27 , 37 , 67 , 55 , 74 , 127 , 61 , 69 , 100 , 29 , 44 , 40 , 58 , 42 , 128, 67 , 46 , 36 , 79 , 31 , 108 , 56 , 21 , 53 , 49 , 51 , 128, 128, 42 , 20 , 49 , 73 , 60 , 36 , 29 , 40 , 56 , 61 , 108 , 128, 17 , 49 , 113 , 34 , 60 , 128, 53 , 81 , 33 , 109 , 116 , 27 , 35 , 52 , 67 , 79 , 97 , 76 , 35 , 126 , 53 , 71 , 46 , 68 , 46 , 43 , 41 , 35 , 128, 34 , 44 , 128, 45 , 47 , 39 , 60 , 128, 56 , 41 , 33 , 59 , 24 , 103 , 38 , 31 , 48 , 48 , 28 , 117 , 49 , 52 , 49 , 36 , 63 , 54 , 46 , 55 , 114 , 40 , 128, 50 , 41 , 23 , 29 , 36 , 64 , 73 , 54 , 125 };
	
	double (*x)[128][768]; cudaMalloc(&x, sizeof(double)*128*768); 
	threeCtext  *ct_x; cudaMalloc(&ct_x, CtextSize*3);  
		
	{
		//====================================================
		// input -> device variable
		//====================================================
		char name_input[100]; sprintf(name_input, "./bert_weights/sample_%03d.txt",0);
		//printf("%s\n",name_input);
		load2D<128, 768>(name_input, h_x);
		cudaMemcpy(x,h_x,sizeof(double)*128*768, cudaMemcpyHostToDevice);
	
		
		// modulus : 12*40 = 480 =  16*30 in graft system
		int rs_bits = delta_bits;
		int l=16; int alpha=20; double scale = pow(2,rs_bits);	
		pack_matrix_to_threeCtext_host(*x, l, alpha, scale, s, *ct_x);
		
		//debug
		//debug_threeCt(*ct_x,l,alpha,s,scale,"input of attention");
			
		float sum=0;
		
		for(int i=0;i<1;i++)
		{
			//====================================================
			// encoder i : load weights
			//====================================================
			//attention
			load2D<768, 768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_self_query_weight.txt").c_str(), h_Wq);
			load2D<768, 768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_self_key_weight.txt").c_str(), h_Wk);
			load2D<768, 768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_self_value_weight.txt").c_str(), h_Wv);
			load2D<768, 768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_output_dense_weight.txt").c_str(), h_Wo);
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_self_query_bias.txt").c_str(), h_bq);
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_self_key_bias.txt").c_str(), h_bk);
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_self_value_bias.txt").c_str(), h_bv);
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_output_dense_bias.txt").c_str(), h_bo);
			load2D<12,128>(("./bert_weights/BUFFER_bert_encoder_layer_"+std::to_string(i)+"_attention_self_batch_method_running_denominator.txt").c_str(), h_attn_Rd);
			
			// LayerNorm
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_output_LayerNorm_weight.txt").c_str(), h_LN1_gamma);
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_attention_output_LayerNorm_bias.txt").c_str(), h_LN1_beta);
			load1D<128>(("./bert_weights/BUFFER_bert_encoder_layer_"+std::to_string(i)+"_attention_output_LayerNorm_batch_method_running_denominator.txt").c_str(), h_ln1_Rd);
			// FeedForward
			load2D<3072, 768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_intermediate_dense_weight.txt").c_str(), h_W_ff1);
			load1D<3072     >(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_intermediate_dense_bias.txt").c_str(), h_b_ff1);
			load2D<768, 3072>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_output_dense_weight.txt").c_str(), h_W_ff2);
			load1D<768      >(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_output_dense_bias.txt").c_str(), h_b_ff2);
			// LayerNorm
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_output_LayerNorm_weight.txt").c_str(), h_LN2_gamma);
			load1D<768>(("./bert_weights/bert_encoder_layer_"+std::to_string(i)+"_output_LayerNorm_bias.txt").c_str(), h_LN2_beta);
			load1D<128>(("./bert_weights/BUFFER_bert_encoder_layer_"+std::to_string(i)+"_output_LayerNorm_batch_method_running_denominator.txt").c_str(), h_ln2_Rd);
			
			cudaEventRecord(start);
			// device variable
			cudaMemcpy(Wq, h_Wq, WeightSize, cudaMemcpyHostToDevice);
			cudaMemcpy(Wk, h_Wk, WeightSize, cudaMemcpyHostToDevice);
			cudaMemcpy(Wv, h_Wv, WeightSize, cudaMemcpyHostToDevice);
			cudaMemcpy(Wo, h_Wo, WeightSize, cudaMemcpyHostToDevice);
			cudaMemcpy(bq, h_bq,   BiasSize, cudaMemcpyHostToDevice);
			cudaMemcpy(bk, h_bk,   BiasSize, cudaMemcpyHostToDevice);
			cudaMemcpy(bv, h_bv,   BiasSize, cudaMemcpyHostToDevice);
			cudaMemcpy(bo, h_bo,   BiasSize, cudaMemcpyHostToDevice);
			// device variable
			cudaMemcpy(LN1_gamma, h_LN1_gamma, BiasSize, cudaMemcpyHostToDevice);
			cudaMemcpy(LN1_beta , h_LN1_beta , BiasSize, cudaMemcpyHostToDevice);
			cudaMemcpy(LN1_Rd , h_ln1_Rd , sizeof(float)*128, cudaMemcpyHostToDevice);
			// device variable
			cudaMemcpy(W_ff1, h_W_ff1, sizeof(float)*3072*768, cudaMemcpyHostToDevice);
			cudaMemcpy(W_ff2, h_W_ff2, sizeof(float)*3072*768, cudaMemcpyHostToDevice);
			cudaMemcpy(b_ff1, h_b_ff1, sizeof(float)*3072    , cudaMemcpyHostToDevice);
			cudaMemcpy(b_ff2, h_b_ff2, sizeof(float)*768     , cudaMemcpyHostToDevice);
			// device variable
			cudaMemcpy(LN2_gamma, h_LN2_gamma, BiasSize, cudaMemcpyHostToDevice);
			cudaMemcpy(LN2_beta , h_LN2_beta , BiasSize, cudaMemcpyHostToDevice);
			cudaMemcpy(LN2_Rd   , h_ln2_Rd , sizeof(float)*128, cudaMemcpyHostToDevice);
			
			//====================================================
			// Encoder i
			//====================================================
			int l_ln2, alpha_ln2; double scale_ln2;
			encoder_host_time(*Wq,*bq,*Wk,*bk,*Wv,*bv,*Wo,*bo,h_attn_Rd,seq_len_list[0],
						 LN1_Rd, *LN1_gamma, *LN1_beta,
						 *W_ff1,b_ff1,*W_ff2,b_ff2,
						 LN2_Rd, *LN2_gamma, *LN2_beta,
						 *ct_x,l,alpha,scale, rs_bits,
						 *ct_ln2, l_ln2, alpha_ln2, scale_ln2);
						 
			//			 
			l     = 16;
			alpha = 20;
			scale = scale_ln2;
			cudaMemcpy(ct_x,ct_ln2,CtextSize*3,cudaMemcpyDeviceToDevice);
			
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float ms = 0.0f;
			cudaEventElapsedTime(&ms, start, stop);
			sum += ms;
			printf("\nencoder[%d] kernel time = %.3f ms\n", i,ms);
			
		
			//====================================================
			// Encoder i error 
			//====================================================
			DataUnpack encoder_out_unpack_ex; load2D<128, 768>(("./ct_layer"+std::to_string(i)+"_out_exact.txt").c_str(), encoder_out_unpack_ex);
			debug_threeCt(*ct_ln2, l_ln2, alpha_ln2,s,scale_ln2,("ct_layer"+std::to_string(i)+"_out").c_str(),encoder_out_unpack_ex);
		}
	
	}	
	//====================================================
	// cudaFree
	//====================================================
	cudaFree(ct_Q);     
	cudaFree(ct_K);     
	cudaFree(ct_V);     
	
	//====================================================
	// deallocate
	//====================================================
	deallocate_parameter();
	deallocate_ckks();
	deallocate_graft();
	deallocate_graft_Ctext();
	deallocate_fft();
	deallocate_fft_FP32();
	deallocate_ntt();
	deallocate_rs();
	deallocate_ks();
	deallocate_rkey();
	deallocate_rot();
	deallocate_LT();
	deallocate_poly();
	deallocate_bts();
	deallocate_bert();

	//time
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	return 0;


}

//----------------------------------------------------------------------
//
//
//
//
//----------------------------------------------------------------------
void encoder_host(//attn
				  Weight Wq, Bias bq,
				  Weight Wk, Bias bk,
				  Weight Wv, Bias bv,
				  Weight Wo, Bias bo, double h_attn_Rd[12][128], int seq_len,
				  //BatchLN1
				  float LN1_Rd[128], Bias LN1_gamma, Bias LN1_beta, 
				  //FeedForward
				  float W_ff1[3072][768], float b_ff1[3072],
		          float W_ff2[768][3072], float b_ff2[768] ,
				  //BatchLN2
				  float LN2_Rd[128], Bias LN2_gamma, Bias LN2_beta, 
				  //data
				  threeCtext ct_x , int  l    , int  alpha    , double  scale    , int rs_bits,
                  threeCtext out, int& l_out, int& alpha_out, double& scale_out  ){
	//time
	//cudaEvent_t start, stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	
	//====================================================
	// CPMM : Q, K, V
	//====================================================
	int l_qkv, alpha_qkv; double scale_qkv=scale;
	//
	
	//cudaEventRecord(start);
	//
	cpmm_QKV_host(Wq,bq,Wk,bk,Wv,bv,
	              ct_x, l, alpha, scale, rs_bits,
				  *ct_Q, *ct_K, *ct_V, l_qkv, alpha_qkv, scale_qkv);
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	//float ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nCPMM(Q,K,V) kernel time = %.3f ms\n", ms);
	
	
	//error
	//DataUnpack Q_unpack_ex; load2D<128, 768>("./ct_Q_exact.txt", Q_unpack_ex);
	//DataUnpack K_unpack_ex; load2D<128, 768>("./ct_K_exact.txt", K_unpack_ex);
	//DataUnpack V_unpack_ex; load2D<128, 768>("./ct_V_exact.txt", V_unpack_ex);
	//DataUnpack sigmaQ_ex;
	//DataUnpack sigmaK_ex;
	//for(int i=0;i<128;i++)
	//for(int j=0;j<768;j++){
	//	sigmaQ_ex[i][j] = Q_unpack_ex[i][(j/64)*64+((i+j)%64)];
	//	sigmaK_ex[i][j] = K_unpack_ex[i][(j/64)*64+((i+j)%64)];
	//}
	//debug_threeCt(*ct_Q,l_qkv,alpha_qkv,s,scale_qkv,"ct_sigma(Q)", sigmaQ_ex); // 4.369731e-04 
	//debug_threeCt(*ct_K,l_qkv,alpha_qkv,s,scale_qkv,"ct_sigma(K)", sigmaK_ex); // 3.628873e-04
	//debug_threeCt(*ct_V,l_qkv,alpha_qkv,s,scale_qkv,"ct_V", V_unpack_ex); // 3.634281e-04
	//printf("num of ks = %d\n",num_ks);
	
	//====================================================
	// CCMM : QKT = Q * KT / 16
	//====================================================
	int l_QKT, alpha_QKT; double scale_QKT;
	
	//cudaEventRecord(start);
	//
	ccmm_QKT_host(*ct_Q  , l_qkv, alpha_qkv, scale_qkv,
	              *ct_K  , l_qkv, alpha_qkv, scale_qkv, rs_bits, 
				  *ct_QKT, l_QKT, alpha_QKT, scale_QKT);
	//printf("ccmm done, l_QKT=%d, alpha_QKT=%d, log(scale_QKT) = %.3f\n", l_QKT, alpha_QKT, log(scale_QKT)/log(2));
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nCCMM(QKT) kernel time = %.3f ms\n", ms);
	
	//error
	//for(int k=0; k<3; k++)
	//	extract_host((*ct_QKT)[k], l_QKT, alpha_QKT, (*ct_real)[k], (*ct_imag)[k]);
	//DataUnpack QKT_real_unpack_ex; load2D<128, 768>("./ct_QKT_real_exact.txt", QKT_real_unpack_ex);
	//DataUnpack QKT_imag_unpack_ex; load2D<128, 768>("./ct_QKT_imag_exact.txt", QKT_imag_unpack_ex);
	//for(int i=0; i<128; i++)
	//for(int j=0; j<768; j++){
	//	QKT_real_unpack_ex[i][j]/=16.;
	//	QKT_imag_unpack_ex[i][j]/=16.;
	//}
	//debug_threeCt(*ct_real,l_QKT,alpha_QKT,s,scale_QKT*2,"ct_QKT_real", QKT_real_unpack_ex); // 1.357635e-01 
	//debug_threeCt(*ct_imag,l_QKT,alpha_QKT,s,scale_QKT*2,"ct_QKT_imag", QKT_imag_unpack_ex); // 1.297801e-01 
	//printf("num of ks = %d\n",num_ks);
	
	//====================================================
	// softmax
	//====================================================
	int l_softmax, alpha_softmax; double scale_softmax;
	//
	//cudaEventRecord(start);
	//
	BPMax_host(seq_len, h_attn_Rd,
	           *ct_QKT, l_QKT, alpha_QKT, scale_QKT, rs_bits,
			   *ct_softmax, l_softmax, alpha_softmax, scale_softmax);
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nsoftmax kernel time = %.3f ms\n", ms);
	
	//error
	//for(int k=0; k<3; k++)
	//	extract_host((*ct_softmax)[k], l_softmax,alpha_softmax, (*ct_real)[k], (*ct_imag)[k]);
	//DataUnpack softmax_real_unpack_ex; load2D<128, 768>("./ct_softmax_real_exact.txt", softmax_real_unpack_ex);
	//DataUnpack softmax_imag_unpack_ex; load2D<128, 768>("./ct_softmax_imag_exact.txt", softmax_imag_unpack_ex);
	//debug_threeCt(*ct_real,l_softmax,alpha_softmax,s,scale_softmax*2,"ct_softmax_real", softmax_real_unpack_ex); //
	//debug_threeCt(*ct_imag,l_softmax,alpha_softmax,s,scale_softmax*2,"ct_softmax_imag", softmax_imag_unpack_ex); //
	//printf("num_ks after softmax=%d\n", num_ks);
	
	//====================================================
	// Y = softmax  * V 
	//====================================================
	int l_Y, alpha_Y; double scale_Y;
	//
	//cudaEventRecord(start);
	//
	ccmm_Y_host((*ct_softmax), l_softmax,alpha_softmax, scale_softmax,
		        (*ct_V)  , l_qkv    , alpha_qkv    , scale_qkv, rs_bits,
				(*ct_Y)  , l_Y      , alpha_Y      , scale_Y);  
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nCCMM(Y) kernel time = %.3f ms\n", ms);
	
	for(int k=0; k<3; k++)
		extract_host((*ct_Y)[k], l_Y,alpha_Y, (*ct_real)[k], (*ct_imag)[k]);
	cudaMemcpy(ct_Y,ct_real,CtextSize*3,cudaMemcpyDeviceToDevice);
	
	//error
	//DataUnpack Y_unpack_ex; load2D<128, 768>("./ct_Y_exact.txt", Y_unpack_ex);
	//debug_threeCt(*ct_Y,l_Y, alpha_Y,s,scale_Y*2,"ct_Y",Y_unpack_ex);
	//printf("num_ks after Y=%d\n", num_ks);
	
	//====================================================
	// BTS
	//====================================================
	//printf("BTS start\n");
	int l_bts, alpha_bts; 
	//cudaEventRecord(start);
	
	BTS_host_noDebug((*ct_Y)[0],scale_Y*2,(*ct_bts)[0],l_bts,alpha_bts,scale_Y);
	//
	combine_host((*ct_Y)[1],(*ct_Y)[2],l_Y,alpha_Y, (*ct_Y)[0]);
	BTS_host_noDebug((*ct_Y)[0], scale_Y*2, (*ct_Q)[0], l_bts,alpha_bts,scale_Y/2);
	extract_host((*ct_Q)[0], l_bts,alpha_bts, (*ct_bts)[1], (*ct_bts)[2]);
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nbts kernel time = %.3f ms\n", ms);
	
	//error
	//debug_threeCt(*ct_bts,l_bts,alpha_bts,s,scale_Y,"bts_ct_Y", Y_unpack_ex); //
	//printf("num_ks after bts=%d\n", num_ks);
	
	l_Y     = 16;
	alpha_Y = 20;
	cudaMemcpy(ct_Y,ct_bts,CtextSize*3,cudaMemcpyDeviceToDevice);
	
	
	//====================================================
	// CPMM : ct_O = ct_V * Wo + bo
	//====================================================
	int l_O, alpha_O; double scale_O=scale;
	//
	//cudaEventRecord(start);
	//
	cpmm_O_host(*ct_Y, Wo, bo, l_Y, alpha_Y, scale_Y, rs_bits,
				   *ct_O, l_O, alpha_O, scale_O);
				   
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nCPMM(O) kernel time = %.3f ms\n", ms);
	
	//error
	//DataUnpack attn_out_unpack_ex; load2D<128, 768>("./ct_attn_out_exact.txt", attn_out_unpack_ex);
	//debug_threeCt(*ct_O,l_O, alpha_O,s,scale_O,"ct_attn_out", attn_out_unpack_ex); //
	//printf("num_ks after O=%d\n", num_ks);
	
	//====================================================
	// layer norm
	//====================================================
	//add
	int lmin     = min(l,    l_O);
	int alphamin = min(alpha,alpha_O);
	for(int i=0; i<3; i++)
		add_host((*ct_O)[i], ct_x[i], lmin, alphamin,(*ct_O)[i]);
	//error
	//DataUnpack ln1_in_unpack_ex; load2D<128, 768>("./ct_ln1_in_exact.txt", ln1_in_unpack_ex);
	//debug_threeCt(*ct_O,lmin, alphamin, s,scale,"ct_ln1_in",ln1_in_unpack_ex);
	
	//ln
	int l_ln,alpha_ln; double scale_ln;
	//
	//cudaEventRecord(start);
	//
	BatchLN_host(LN1_Rd, LN1_gamma, LN1_beta,
	               *ct_O  , lmin, alphamin, scale ,rs_bits,
				   *ct_ln , l_ln, alpha_ln, scale_ln);	
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nBatchLN kernel time = %.3f ms\n", ms);
	
	//error
	//DataUnpack ln1_out_unpack_ex; load2D<128, 768>("./ct_ln1_out_exact.txt", ln1_out_unpack_ex);
	//debug_threeCt(*ct_ln  , l_ln, alpha_ln,s,scale_ln,"ct_ln1_out",ln1_out_unpack_ex);
	//	   
	//printf("num_ks after BatchLN =%d\n", num_ks);
	
	
	//====================================================
	// FC1
	//====================================================
	int l_ff, alpha_ff; double scale_ff=scale_ln;
	feedforward_host(W_ff1, b_ff1,
	                 W_ff2, b_ff2,
	                 *ct_ln, l_ln, alpha_ln, scale, rs_bits,
				     *ct_ff, l_ff, alpha_ff, scale_ff );
	
	//====================================================
	// layer norm2
	//====================================================
	//add
	lmin     = min(l_ln,l_ff);
	alphamin = min(alpha_ln,alpha_ff);
	for(int i=0; i<3; i++)
		add_host((*ct_ff)[i], (*ct_ln)[i], lmin, alphamin, (*ct_ff)[i]);
	//ln
	int l_ln2,alpha_ln2; double scale_ln2;
	//
	//cudaEventRecord(start);
	//
	BatchLN_host(LN2_Rd, LN2_gamma, LN2_beta,
	             *ct_ff , lmin , alphamin , scale_ln , rs_bits,
				 *ct_ln2, l_ln2, alpha_ln2, scale_ln2);	
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nBatchLN2 kernel time = %.3f ms\n", ms);
	
	//error
	//DataUnpack ln2_out_unpack_ex; load2D<128, 768>("./ct_ln2_out_exact.txt", ln2_out_unpack_ex);
	//debug_threeCt(*ct_ln2, l_ln2, alpha_ln2,s,scale_ln2,"ct_ln2_out",ln2_out_unpack_ex);
	//printf("num_ks after BatchLN2 =%d\n", num_ks);
	
	//====================================================
	// BTS
	//====================================================
	//cudaEventRecord(start);
	
	BTS_host_noDebug((*ct_ln2)[0],scale_ln2, (*ct_bts)[0],l_bts,alpha_bts,scale_ln2);
	//
	combine_host((*ct_ln2)[1],(*ct_ln2)[2],l_ln2,alpha_ln2, (*ct_Y)[0]);
	BTS_host_noDebug((*ct_Y)[0], scale_ln2, (*ct_Q)[0], l_bts,alpha_bts,scale_ln2/2);
	extract_host((*ct_Q)[0], l_bts,alpha_bts, (*ct_bts)[1], (*ct_bts)[2]);
	//
	//cudaEventRecord(stop);
	//cudaEventSynchronize(stop);
	// ms = 0.0f;
	//cudaEventElapsedTime(&ms, start, stop);
	//printf("\nbts kernel time = %.3f ms\n", ms);
	
	//error
	//debug_threeCt(*ct_bts,l_bts,alpha_bts,s,scale_ln2,"bts_ct_ln2", ln2_out_unpack_ex); //
	//printf("num_ks after bts=%d\n", num_ks);
	
	l_out     = 16;
	alpha_out = 20;
	scale_out = scale_ln2;
	cudaMemcpy(out,ct_bts,CtextSize*3,cudaMemcpyDeviceToDevice);
	
}


void encoder_host_time(//attn
				       Weight Wq, Bias bq,
				       Weight Wk, Bias bk,
				       Weight Wv, Bias bv,
				       Weight Wo, Bias bo, double h_attn_Rd[12][128], int seq_len,
				       //BatchLN1
				       float LN1_Rd[128], Bias LN1_gamma, Bias LN1_beta, 
				       //FeedForward
				       float W_ff1[3072][768], float b_ff1[3072],
		               float W_ff2[768][3072], float b_ff2[768] ,
				       //BatchLN2
				       float LN2_Rd[128], Bias LN2_gamma, Bias LN2_beta, 
				       //data
				       threeCtext ct_x , int  l    , int  alpha    , double  scale    , int rs_bits,
                       threeCtext out, int& l_out, int& alpha_out, double& scale_out  ){
	//time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	//====================================================
	// CPMM : Q, K, V
	//====================================================
	int l_qkv, alpha_qkv; double scale_qkv=scale;
	//
	
	cudaEventRecord(start);
	//
	cpmm_QKV_host(Wq,bq,Wk,bk,Wv,bv,
	              ct_x, l, alpha, scale, rs_bits,
				  *ct_Q, *ct_K, *ct_V, l_qkv, alpha_qkv, scale_qkv);
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nCPMM(Q,K,V) kernel time = %.3f ms\n", ms);
	
	printf("num of ks = %d\n",num_ks);
	
	//====================================================
	// CCMM : QKT = Q * KT / 16
	//====================================================
	int l_QKT, alpha_QKT; double scale_QKT;
	
	cudaEventRecord(start);
	//
	ccmm_QKT_host(*ct_Q  , l_qkv, alpha_qkv, scale_qkv,
	              *ct_K  , l_qkv, alpha_qkv, scale_qkv, rs_bits, 
				  *ct_QKT, l_QKT, alpha_QKT, scale_QKT);
	//printf("ccmm done, l_QKT=%d, alpha_QKT=%d, log(scale_QKT) = %.3f\n", l_QKT, alpha_QKT, log(scale_QKT)/log(2));
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nCCMM(QKT) kernel time = %.3f ms\n", ms);
	
	printf("num of ks = %d\n",num_ks);
	
	//====================================================
	// softmax
	//====================================================
	int l_softmax, alpha_softmax; double scale_softmax;
	//
	cudaEventRecord(start);
	//
	BPMax_host(seq_len, h_attn_Rd,
	           *ct_QKT, l_QKT, alpha_QKT, scale_QKT, rs_bits,
			   *ct_softmax, l_softmax, alpha_softmax, scale_softmax);
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nsoftmax kernel time = %.3f ms\n", ms);
	
	printf("num_ks after softmax=%d\n", num_ks);
	
	//====================================================
	// Y = softmax  * V 
	//====================================================
	int l_Y, alpha_Y; double scale_Y;
	//
	cudaEventRecord(start);
	//
	ccmm_Y_host((*ct_softmax), l_softmax,alpha_softmax, scale_softmax,
		        (*ct_V)  , l_qkv    , alpha_qkv    , scale_qkv, rs_bits,
				(*ct_Y)  , l_Y      , alpha_Y      , scale_Y);  
	
	
	for(int k=0; k<3; k++)
		extract_host((*ct_Y)[k], l_Y,alpha_Y, (*ct_real)[k], (*ct_imag)[k]);
	cudaMemcpy(ct_Y,ct_real,CtextSize*3,cudaMemcpyDeviceToDevice);
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nCCMM(Y) kernel time = %.3f ms\n", ms);
	
	printf("num_ks after Y=%d\n", num_ks);
	
	//====================================================
	// BTS
	//====================================================
	//printf("BTS start\n");
	int l_bts, alpha_bts; 
	cudaEventRecord(start);
	
	BTS_host_noDebug((*ct_Y)[0],scale_Y*2,(*ct_bts)[0],l_bts,alpha_bts,scale_Y);
	//
	combine_host((*ct_Y)[1],(*ct_Y)[2],l_Y,alpha_Y, (*ct_Y)[0]);
	BTS_host_noDebug((*ct_Y)[0], scale_Y*2, (*ct_Q)[0], l_bts,alpha_bts,scale_Y/2);
	extract_host((*ct_Q)[0], l_bts,alpha_bts, (*ct_bts)[1], (*ct_bts)[2]);
	//
	l_Y     = 16;
	alpha_Y = 20;
	cudaMemcpy(ct_Y,ct_bts,CtextSize*3,cudaMemcpyDeviceToDevice);
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nbts kernel time = %.3f ms\n", ms);
	
	printf("num_ks after bts=%d\n", num_ks);
	
	
	
	
	//====================================================
	// CPMM : ct_O = ct_V * Wo + bo
	//====================================================
	int l_O, alpha_O; double scale_O=scale;
	//
	cudaEventRecord(start);
	//
	cpmm_O_host(*ct_Y, Wo, bo, l_Y, alpha_Y, scale_Y, rs_bits,
				   *ct_O, l_O, alpha_O, scale_O);
				   
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nCPMM(O) kernel time = %.3f ms\n", ms);
	
	printf("num_ks after O=%d\n", num_ks);
	
	//====================================================
	// layer norm
	//====================================================
	//add
	int lmin     = min(l,    l_O);
	int alphamin = min(alpha,alpha_O);
	//
	cudaEventRecord(start);
	//
	for(int i=0; i<3; i++)
		add_host((*ct_O)[i], ct_x[i], lmin, alphamin,(*ct_O)[i]);
	//ln
	int l_ln,alpha_ln; double scale_ln;
	
	BatchLN_host(LN1_Rd, LN1_gamma, LN1_beta,
	               *ct_O  , lmin, alphamin, scale ,rs_bits,
				   *ct_ln , l_ln, alpha_ln, scale_ln);	
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nBatchLN kernel time = %.3f ms\n", ms);
	
	printf("num_ks after BatchLN =%d\n", num_ks);
	
	
	//====================================================
	// FC1
	//====================================================
	int l_ff, alpha_ff; double scale_ff=scale_ln;
	feedforward_host_time(W_ff1, b_ff1,
	                 W_ff2, b_ff2,
	                 *ct_ln, l_ln, alpha_ln, scale, rs_bits,
				     *ct_ff, l_ff, alpha_ff, scale_ff );
	
	//====================================================
	// layer norm2
	//====================================================
	cudaEventRecord(start);
	//
	lmin     = min(l_ln,l_ff);
	alphamin = min(alpha_ln,alpha_ff);
	for(int i=0; i<3; i++)
		add_host((*ct_ff)[i], (*ct_ln)[i], lmin, alphamin, (*ct_ff)[i]);
	//ln
	int l_ln2,alpha_ln2; double scale_ln2;
	
	BatchLN_host(LN2_Rd, LN2_gamma, LN2_beta,
	             *ct_ff , lmin , alphamin , scale_ln , rs_bits,
				 *ct_ln2, l_ln2, alpha_ln2, scale_ln2);	
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nBatchLN2 kernel time = %.3f ms\n", ms);
	
	printf("num_ks after BatchLN2 =%d\n", num_ks);
	
	//====================================================
	// BTS
	//====================================================
	cudaEventRecord(start);
	
	BTS_host_noDebug((*ct_ln2)[0],scale_ln2, (*ct_bts)[0],l_bts,alpha_bts,scale_ln2);
	//
	combine_host((*ct_ln2)[1],(*ct_ln2)[2],l_ln2,alpha_ln2, (*ct_Y)[0]);
	BTS_host_noDebug((*ct_Y)[0], scale_ln2, (*ct_Q)[0], l_bts,alpha_bts,scale_ln2/2);
	extract_host((*ct_Q)[0], l_bts,alpha_bts, (*ct_bts)[1], (*ct_bts)[2]);
	//
	l_out     = 16;
	alpha_out = 20;
	scale_out = scale_ln2;
	cudaMemcpy(out,ct_bts,CtextSize*3,cudaMemcpyDeviceToDevice);
	//
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	 ms = 0.0f;
	cudaEventElapsedTime(&ms, start, stop);
	printf("\nbts kernel time = %.3f ms\n", ms);
	
	printf("num_ks after bts=%d\n", num_ks);
	
	
}