#pragma once
#include <inttypes.h>
#include <cassert>
#include <complex>
#include <fstream>
#include "Parameter.cuh"


void print(Ptext pt, const char* name) ;
void print(float a[N], const char* name) ;

void print(double a[N], const char* name);


 void set_random_z(double z[N]);
 void set_random_z(float  z[N]);

 double L2_sqr(const double a[N], const double b[N]);
 float  L2_sqr(const float  a[N], const float  b[N]);

double SumProduct(const double a[N] );
float  SumProduct(const float  a[N] );

 void save(const double  z[N], const char* filename);

 void load(double  z[N], const char* filename) ;

 void save(const uint32_t ct[2][L + G][N], const char* filename);
 void load(uint32_t ct[2][L + G][N], const char* filename);


void save(const uint32_t d[DNUM][2][L + G + K][N], const char* filename);
void load(      uint32_t d[DNUM][2][L + G + K][N], const char* filename);

void debug_data( double* z, const char* name );
void debug_pt( Ptext pt, int l, int alpha, double scale, const char* name );
void debug_ct( Ctext ct, int l, int alpha, int s[N], double scale, const char* name );
void debug_ct( Ctext ct, int l, int alpha, int s[N], double scale, const char* name, double x[128][256] );
double debug_ct_classification( Ctext ct, int l, int alpha, int s[N], double scale, const char* name );
void debug_threeCt( threeCtext ct, int l, int alpha, int s[N], double scale, const char* name );
void debug_threeCt( threeCtext ct, int l, int alpha, int s[N], double scale, const char* name, DataUnpack h_z_unpack_ex);
void debug_12Ct( threeCtext ct1, threeCtext ct2, threeCtext ct3, threeCtext ct4, 
				int l, int alpha, int s[N], double scale, const char* name, float h_z_unpack_ex[128][3072]);
void debug_threePt( threePtext pt, int l, int alpha, double scale, const char* name );
void debug_rkeytilde( uint32_t rkeytilde[DNUM][2][L+G+K][N], int shift, int s[N], const char* name );

/*
void save(const uint32_t pt[N], const char* filename) {
	FILE* fp = fopen(filename, "wb");
	fwrite(pt, 1, sizeof(uint32_t) * N, fp);
	fclose(fp);
}

void load(uint32_t pt[N], const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) perror("fopen failed");
	size_t ret = fread((void*)pt, 1, sizeof(uint32_t) * N, fp);
	if (ret != sizeof(uint32_t) * N) perror("fread error");
	(void)fclose(fp);
}


void save(const uint32_t pt[L+G][N], const char* filename) {
	FILE* fp = fopen(filename, "wb");
	fwrite(pt, 1, sizeof(uint32_t)*(L+G) * N , fp);
	fclose(fp);
}


void load(uint32_t pt[L + G][N], const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) perror("fopen failed");
	size_t ret = fread((void*)pt, 1, sizeof(uint32_t) * (L + G) * N, fp);
	if (ret != sizeof(uint32_t) * (L + G) * N) perror("fread error");
	fclose(fp);
}


void save(const uint32_t ct[2][L + G+K][N], const char* filename) {
	FILE* fp = fopen(filename, "wb");
	fwrite(ct, 1, sizeof(uint32_t) * 2 * (L + G+K) * N, fp);
	fclose(fp);
}


void load(uint32_t ct[2][L + G+K][N], const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) perror("fopen failed");
	size_t ret = fread((void*)ct, 1, sizeof(uint32_t) * 2 * (L + G+K) * N, fp);
	if (ret != sizeof(uint32_t) * 2 * (L + G+K) * N) perror("fread error");
	fclose(fp);
}


void save(const uint32_t d[DNUM][L + G + K][N], const char* filename) {
	FILE* fp = fopen(filename, "wb");
	fwrite(d, 1, sizeof(uint32_t) * DNUM * (L + G + K) * N, fp);
	fclose(fp);
}


void load(uint32_t d[DNUM][L + G + K][N], const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) perror("fopen failed");
	size_t ret = fread((void*)d, 1, sizeof(uint32_t) * DNUM * (L + G + K) * N, fp);
	if (ret != sizeof(uint32_t) * DNUM * (L + G + K) * N) perror("fread error");
	fclose(fp);
}


void save(const double  z[N], const char* filename) {
	FILE* fp = fopen(filename, "wt");
	for (int i = 0; i < N; i++) {
		fprintf(fp, "%e\n", z[i]);
	}
	fclose(fp);
}


void load(double  z[N], const char* filename) {
	std::ifstream is(filename);
	for (int i = 0; i < N; i++)
		is >> z[i];
}


void load(int s[N], const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) perror("fopen failed");
	size_t ret = fread((void*)s, 1, sizeof(int) * N, fp);
	if (ret != sizeof(int) * N) perror("fread error");
	fclose(fp);
}


void save(const int s[N], const char* filename) {
	FILE* fp = fopen(filename, "wb");
	fwrite(s, 1, sizeof(int) * N, fp);
	fclose(fp);
}





 double SumProduct(const double* a, const double* b, int N) {
	double Sum = 0;
	for (int i = 0; i < N; i++) {
		double diff = a[i] - b[i];
		Sum += diff * diff;
	}
	return Sum;
}

template<int N>
double L2_sqr(const double a[N], const double b[N]) {
	double Sum = 0;
	for (int i = 0; i < N; i++) {
		double diff = a[i] - b[i];
		Sum += diff * diff;
	}
	return Sum;
}

template<int N, int L, int K, int DNUM>
void save(const SWK<N,L,1,K,DNUM>& swk, const char* filename) {
	FILE* fp = fopen(filename, "wb");
	int SWKSize = sizeof(uint32_t) * DNUM * 2 * (L + 1 + K) * N;
	fwrite(swk.data, 1, SWKSize, fp);
	fclose(fp);
}

template<int N, int L, int K, int DNUM>
void load(SWK<N,L,1,K,DNUM>& swk, const char* filename) {
	FILE* fp = fopen(filename, "rb");
	if (!fp) perror("fopen failed");
	int SWKSize = sizeof(uint32_t) * DNUM * 2 * (L + 1 + K) * N;
	size_t ret = fread((void*)swk.data, 1, SWKSize, fp);
	if (ret != SWKSize) perror("fread error");
	fclose(fp);
}

*/