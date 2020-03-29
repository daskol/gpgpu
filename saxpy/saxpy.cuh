/**
 * \file saxpy.cuh
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <saxpy/memory.cuh>

namespace my {

enum VectorExtention : uint8_t {
    kNone = 0,
    kSse,
    kAvx,
    kAvx512,
};

struct CpuExecutor {
    size_t NoThreads = 1;
    VectorExtention VectorExtension = my::VectorExtention::kNone;
};

struct GpuExecutor {
    size_t NoStreams = 1;
    float KernelTime = 0.0f;
    float DtoH = 0.0f;
    float HtoD = 0.0f;
};

/**
 * Function saxpy implements operation which is very similar to saxpy. In fact,
 * it calculates more computation intensive expression \sum sin(a_i b_i + i).
 */

cudaError_t saxpy(
    cuda_ptr<float[]> lhs, cuda_ptr<float[]> rhs, cuda_ptr<float[]> out,
    size_t length, CpuExecutor &executor
);

cudaError_t saxpy(
    cuda_ptr<float[]> lhs, cuda_ptr<float[]> rhs, cuda_ptr<float[]> out,
    size_t length, GpuExecutor &executor
);

} // namespace my
