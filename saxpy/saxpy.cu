/**
 * \file saxpy.cu
 */

#include "saxpy.cuh"

template <typename T>
class cuda_ptr {
public:
    cuda_ptr(void) = default;

    cuda_ptr(T *ptr)
        : Ptr_{ptr}
    {}

    cuda_ptr(cuda_ptr const &cuda_ptr) = delete;

    cuda_ptr(cuda_ptr &&that)
        : Ptr_{std::move(that.Ptr_)}
    {}

    ~cuda_ptr(void) {
        if (Ptr_) {
            cudaFree(Ptr_);
        }
    }

    T *get(void) {
        return Ptr_;
    }

    T const *get(void) const {
        return Ptr_;
    }

private:
    T *Ptr_ = nullptr;
};

template <typename T>
cuda_ptr<T> make_cuda(size_t size) {
    T *ptr = nullptr;
    cudaMalloc(&ptr, size * sizeof(T));
    return ptr;
}

void saxpy_cpu(float *a, float *b, float *c, size_t length) {
    for (size_t it = 0; it != length; ++it) {
        c[it] = a[it] + b[it];
    }
}

__global__
void saxpy_kernel(float *a, float *b, float *c, size_t length) {
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < length) {
        c[tid] = a[tid] + b[tid];
    }
}

void saxpy_gpu(float *a, float *b, float *c, size_t length) {
    auto dev_a = make_cuda<float>(length);
    auto dev_b = make_cuda<float>(length);
    auto dev_c = make_cuda<float>(length);

    cudaMemcpy(dev_a.get(), a, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b.get(), b, length * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_c.get(), c, length * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threads(512);
    dim3 blocks(length / threads.x);
    saxpy_kernel<<<blocks, threads>>>(dev_a.get(), dev_b.get(), dev_c.get(),
                                      length);
}

void saxpy(float *a, float *b, float *c, size_t length, executor_t executor) {
    switch (executor) {
    case executor_t::kCpu:
        saxpy_cpu(a, b, c, length);
    case executor_t::kGpu:
        saxpy_gpu(a, b, c, length);
    }
}
