#pragma once
#include <inttypes.h>
#include <string>
#include "Parameter.cuh"

#ifdef _WIN32
#include <direct.cuh> // For _mkdir on Windows
#define MKDIR(dir) _mkdir(dir)
#else
#include <sys/stat.h> // For mkdir on Linux/macOS
#define MKDIR(dir) mkdir(dir, 0777) // 0777은 권한 설정
#endif


static void ensure_data_directory_exists() {
	std::string data_dir = "./data"; // 데이터 폴더 경로
	struct stat info;

	if (stat(data_dir.c_str(), &info) != 0) { // data 폴더가 존재하지 않는 경우
		printf("Data directory '%s' not found. Creating...\n", data_dir.c_str());
		if (MKDIR(data_dir.c_str()) != 0) { // 폴더 생성 시도
			perror("Error creating data directory"); // 에러 발생 시 출력
			return;
		}
		else {
			printf("Data directory '%s' created successfully.\n", data_dir.c_str());
		}
	}
}

struct SWK {
	typedef uint32_t data_type[DNUM][2][L + G + K][N];
	data_type* data; bool allocated;
	SWK() { allocated = false; }
	~SWK() { if (allocated == true) free(data); }
	void allocate() {
		if(allocated==false){
			data = (data_type*)malloc(sizeof(*data));
		}
		allocated = true;
	}
	
	bool check_exist(const char* file) {
		FILE* fp = fopen(file, "rb");
		if (fp == NULL) return false;
		else {fclose(fp); return true;}
	}
	
	void save(const char* file) {
		FILE* fp = fopen(file, "wb");
		fwrite(data, sizeof(*data), 1, fp);
		fclose(fp);
	}

	void load(const char* file) {
		if (allocated == false)
			allocate();
		FILE* fp = fopen(file, "rb");
		size_t ret = fread(data, sizeof(*data),1,fp);
		if(ret != 1) perror("fread error");
		fclose(fp);
	}
};
