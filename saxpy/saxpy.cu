/**
 * \file saxpy.cu
 */

#include "saxpy.cuh"

namespace my {

constexpr size_t kWindow = 100;

__global__
void saxpy_kernel(float *a, float *b, float *c, size_t length) {
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= length) {
        return;
    }

    float sum = 0.0f, ab = a[tid] * b[tid];
    for (size_t it = 0; it != kWindow; ++it) {
        sum += sinf(ab + static_cast<float>(it));
    }

    c[tid] = sum;
}

cudaError_t saxpy(
    cuda_ptr<float[]> lhs, cuda_ptr<float[]> rhs, cuda_ptr<float[]> out,
    size_t length, GpuExecutor executor
) {
    constexpr auto kDevice = MemoryType::kDevice;

    std::vector<cuda_stream_t> streams(executor.NoStreams);

    for (size_t it = 0; it != streams.size(); ++it) {
        size_t hunk = length / streams.size();
        size_t offset = it * hunk;
        size_t size = it + 1 == streams.size() ? length - offset : hunk;
        lhs.sync(offset, size, SyncDir::kToDevice, streams[it]);
        rhs.sync(offset, size, SyncDir::kToDevice, streams[it]);
    }

    for (size_t it = 0; it != streams.size(); ++it) {
        size_t hunk = length / streams.size();
        size_t offset = it * hunk;
        size_t size = it + 1 == streams.size() ? length - offset : hunk;

        dim3 threads(512);
        dim3 blocks(size / threads.x);
        saxpy_kernel<<<blocks, threads, 0, streams[it]>>>(
            lhs.get(kDevice) + offset,
            rhs.get(kDevice) + offset,
            out.get(kDevice) + offset,
            size);
    }


    for (size_t it = 0; it != streams.size(); ++it) {
        size_t hunk = length / streams.size();
        size_t offset = it * hunk;
        size_t size = it + 1 == streams.size() ? length - offset : hunk;
        out.sync(offset, size, SyncDir::kToHost, streams[it]);
    }

    return cudaSuccess;
}

cudaError_t saxpy(
    cuda_ptr<float[]> lhs, cuda_ptr<float[]> rhs, cuda_ptr<float[]> out,
    size_t length, CpuExecutor executor
) {
    if (executor.VectorExtension != my::VectorExtention::kNone) {
        return cudaErrorNotYetImplemented;
    }

    for (size_t i = 0; i != length; ++i) {
        float ai = lhs[i];
        float bi = rhs[i];
        float ab = ai * bi;

        out[i] = 0;

        for (size_t j = 0; j != kWindow; ++j) {
            out[i] = sinf(ab + j);
        }
    }

    return cudaSuccess;
}

} // namespace my
