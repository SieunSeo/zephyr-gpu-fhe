# Zephyr: GPU-Efficient Homomorphic Encryption for Privacy-Preserving Transformer Inference

This repository contains the implementation package for **Zephyr**, a GPU-efficient homomorphic encryption framework for privacy-preserving Transformer inference.

## Overview

Zephyr accelerates CKKS-based encrypted Transformer inference on GPUs. The implementation includes the components required to reproduce a single BERT encoder execution using the provided model weights and precompiled static libraries.

The implementation supports the full **12-layer BERT encoder architecture**. Due to the size of the model weights, this repository includes only the weights for **Encoder 0** and provides a reproducible single-encoder execution example.

## Source Code Availability

The core CKKS and BERT CUDA implementations are distributed as precompiled static libraries:

- `libckks.a`
- `libbert.a`

The main execution driver, `test_powerformer.cu`, and the required header (`.cuh`) files are included in this repository.

This distribution allows the provided Zephyr experiment to be built and executed without releasing the core CUDA implementation source files (`.cu`) contained in the static libraries.

## Tested Environment

The implementation was tested in the following environment:

- GPU: NVIDIA GeForce RTX 4090
- OS: Ubuntu
- CUDA: 11.x or later
- GPU architecture: `compute_89` / `sm_89`

For a different NVIDIA GPU architecture, modify the corresponding CUDA architecture option in the `Makefile`.

## Build and Run

Before building, increase the stack size:

```bash
ulimit -s unlimited
```

Compile the program:

```bash
make
```

Run the experiment:

```bash
./test
```

## Model Weights

The full BERT model consists of 12 encoder layers. The implementation is structured to support the complete encoder stack; however, including all model weights would substantially increase the repository size.

Therefore, the provided package contains the weights required for **Encoder 0 only**. The main program is configured to execute one encoder:

```cpp
for (int i = 0; i < 1; i++)
```

The encoder index is used when loading the corresponding BERT weights, so the same implementation structure can be used for the remaining encoder layers when their weights are available.

## Reference Outputs

Reference ciphertext/plaintext outputs are included for numerical error evaluation. These files were generated during the full-model experiments and can be used to compare intermediate or final results.

For the provided Encoder 0 execution, the final reported error is approximately:

```text
ct_layer0_out error = 4.82e-03
```

## Performance

In the tested RTX 4090 environment, the implementation achieved approximately **1.65× speedup over the A100 baseline** used in the corresponding experiments.

The provided implementation is designed to run within approximately **24 GB of GPU memory**.

Detailed operation-level timings and Key Switching counts are printed during execution.

## Repository Structure

```text
.
├── bert_weights/          # Model weights for Encoder 0
├── libckks.a              # Precompiled CKKS static library
├── libbert.a              # Precompiled BERT static library
├── test_powerformer.cu    # Main execution driver
├── *.cuh                  # Required CUDA/C++ headers
├── ct_*_exact.txt         # Reference outputs for error evaluation
├── Makefile
└── README.md
```

## Notes

- The current package reproduces the **single-encoder (Encoder 0)** experiment because only Encoder 0 weights are distributed.
- The implementation itself is structured for the full 12-layer encoder stack.
- Core CKKS/BERT CUDA `.cu` source files compiled into `libckks.a` and `libbert.a` are not included.
- `test_powerformer.cu` is included as the executable driver.
- If compilation fails on a GPU other than RTX 4090, update the CUDA architecture flags in the `Makefile` to match the target GPU.

## Related Paper

**Zephyr: GPU-Efficient Homomorphic Encryption for Privacy-Preserving Transformer Inference**

This repository contains the implementation package associated with the Zephyr research project.
