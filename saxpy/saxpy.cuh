/**
 * \file saxpy.cuh
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <saxpy/memory.cuh>

namespace my {

/**
 * Class cuda_stream_t wraps around cudaStream_t in order to provide simpler
 * API and object which has RAII-like semantic.
 */
class cuda_stream_t {
private:
    cudaStream_t stream;
    cudaError_t error = cudaErrorInvalidValue;

public:
    cuda_stream_t(void)
        : error{cudaStreamCreate(&stream)}
    {}

    cuda_stream_t(cudaStream_t stream) noexcept
        : stream{stream}
        , error{cudaSuccess}
    {}

    ~cuda_stream_t(void) {
        if (error == cudaSuccess) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }

    explicit operator bool(void) const noexcept {
        return error == cudaSuccess;
    }

    operator cudaStream_t(void) const noexcept {
        return stream;
    }

    cudaStream_t operator*(void) const noexcept {
        return stream;
    }
};

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
};

/**
 * Function saxpy implements operation which is very similar to saxpy. In fact,
 * it calculates more computation intensive expression \sum sin(a_i b_i + i).
 */

cudaError_t saxpy(
    cuda_ptr<float[]> lhs, cuda_ptr<float[]> rhs, cuda_ptr<float[]> out,
    size_t length, CpuExecutor executor
);

cudaError_t saxpy(
    cuda_ptr<float[]> lhs, cuda_ptr<float[]> rhs, cuda_ptr<float[]> out,
    size_t length, GpuExecutor executor
);

} // namespace my
