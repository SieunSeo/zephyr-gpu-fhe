NVCC   := nvcc
TARGET := test

MAIN_SRC := test_powerformer.cu
MAIN_OBJ := $(MAIN_SRC:.cu=.o)

CKKS_LIB := libckks.a
BERT_LIB := libbert.a

NVCC_FLAGS := -O2 -rdc=true -gencode arch=compute_89,code=sm_89 \
              -I./CKKS -I./BERT

all: $(TARGET)

$(TARGET): $(MAIN_OBJ) $(BERT_LIB) $(CKKS_LIB)
	$(NVCC) $(NVCC_FLAGS) -o $@ $(MAIN_OBJ) -L. -lbert -lckks

$(MAIN_OBJ): $(MAIN_SRC)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(MAIN_OBJ) $(TARGET)

.PHONY: all clean