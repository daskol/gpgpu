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
    size_t length, GpuExecutor &executor
) {
    constexpr auto kDevice = MemoryType::kDevice;

    std::vector<cuda_event_t> dtoh_events(2 * executor.NoStreams);
    std::vector<cuda_event_t> htod_events(2 * executor.NoStreams);
    std::vector<cuda_event_t> kernel_events(2 * executor.NoStreams);
    std::vector<cuda_stream_t> streams(executor.NoStreams);

    for (size_t it = 0; it != streams.size(); ++it) {
        size_t hunk = length / streams.size();
        size_t offset = it * hunk;
        size_t size = it + 1 == streams.size() ? length - offset : hunk;
        htod_events[2 * it + 0].record(streams[it]);
        lhs.sync(offset, size, SyncDir::kToDevice, streams[it]);
        rhs.sync(offset, size, SyncDir::kToDevice, streams[it]);
        htod_events[2 * it + 1].record(streams[it]);
    }

    for (size_t it = 0; it != streams.size(); ++it) {
        size_t hunk = length / streams.size();
        size_t offset = it * hunk;
        size_t size = it + 1 == streams.size() ? length - offset : hunk;

        kernel_events[2 * it + 0].record(streams[it]);

        dim3 threads(512);
        dim3 blocks(size / threads.x);
        saxpy_kernel<<<blocks, threads, 0, streams[it]>>>(
            lhs.get(kDevice) + offset,
            rhs.get(kDevice) + offset,
            out.get(kDevice) + offset,
            size);

        kernel_events[2 * it + 1].record(streams[it]);
    }

    for (size_t it = 0; it != streams.size(); ++it) {
        size_t hunk = length / streams.size();
        size_t offset = it * hunk;
        size_t size = it + 1 == streams.size() ? length - offset : hunk;
        dtoh_events[2 * it + 0].record(streams[it]);
        out.sync(offset, size, SyncDir::kToHost, streams[it]);
        dtoh_events[2 * it + 1].record(streams[it]);
    }

    // Take the largest time interval which is spent on kernel execution.
    executor.KernelTime = 0;
    executor.DtoH = 0;
    executor.HtoD = 0;

    float elapsed;
    for (size_t it = 0; it != executor.NoStreams; ++it) {
        dtoh_events[2 * it + 1].sync();
        htod_events[2 * it + 1].sync();
        kernel_events[2 * it + 1].sync();

        for (size_t jt = 0; jt != executor.NoStreams; ++jt) {
            elapsed = dtoh_events[2 * it + 1] - dtoh_events[2 * jt + 0];
            executor.DtoH = std::max(executor.DtoH, elapsed);

            elapsed = htod_events[2 * it + 1] - htod_events[2 * jt + 0];
            executor.HtoD = std::max(executor.HtoD, elapsed);

            elapsed = kernel_events[2 * it + 1] - kernel_events[2 * it + 0];
            executor.KernelTime = std::max(executor.KernelTime, elapsed);
        }
    }

    return cudaSuccess;
}

cudaError_t saxpy(
    cuda_ptr<float[]> lhs, cuda_ptr<float[]> rhs, cuda_ptr<float[]> out,
    size_t length, CpuExecutor &executor
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
