/**
 * \file util.cuh
 */

#pragma once

#include <cstddef>
#include <cstdint>

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

class cuda_event_t {
private:
    cudaEvent_t event;
    cudaError_t error = cudaErrorInvalidValue;

public:
    cuda_event_t(void)
        : error{cudaEventCreate(&event)}
    {}

    cuda_event_t(cudaEvent_t event)
        : event{event}
        , error{cudaSuccess}
    {}

    ~cuda_event_t(void) {
        if (error == cudaSuccess) {
            cudaEventDestroy(event);
        }
    }

    explicit operator bool(void) const noexcept {
        return error == cudaSuccess;
    }

    operator cudaEvent_t(void) const noexcept {
        return event;
    }

    cudaEvent_t operator*(void) const noexcept {
        return event;
    }

    float operator-(cudaEvent_t rhs) {
        float elapsed;
        cudaEventElapsedTime(&elapsed, rhs, event);
        return elapsed;
    }

    cudaError_t sync(void) noexcept {
        return cudaEventSynchronize(event);
    }

    cudaError_t record(cudaStream_t stream = nullptr) noexcept {
        return cudaEventRecord(event, stream);
    }
};

} // namespace my
