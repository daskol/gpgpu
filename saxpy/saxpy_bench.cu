/**
 * \file saxpy_bench.cu
 */

#include <memory>
#include <iostream>

#include <benchmark/benchmark.h>
#include <saxpy/saxpy.cuh>

using benchmark::DoNotOptimize;
using Fixture = benchmark::Fixture;
using State = benchmark::State;

template <my::MemoryKind Kind>
class MemoryManager : public Fixture {
private:
    static constexpr auto kMemoryKind = Kind;

public:
    union {
        size_t concurrency;
        size_t nostreams;
        size_t nothreads;
    };

    size_t length;

    my::cuda_ptr<float[]> lhs;
    my::cuda_ptr<float[]> rhs;
    my::cuda_ptr<float[]> dst;

public:
    virtual void SetUp(State const &state) override {
    }

    virtual void TearDown(State const &state) override {
    }

    virtual void SetUp(State &state) override {
        concurrency = state.range(0);
        length = state.range(1);
        lhs = my::make_cuda<float[]>(length, kMemoryKind);
        rhs = my::make_cuda<float[]>(length, kMemoryKind);
        dst = my::make_cuda<float[]>(length, kMemoryKind);

        for (size_t it = 0; it != length; ++it) {
            lhs[it] = std::sin(static_cast<float>(it));
            rhs[it] = std::cos(static_cast<float>(it) * 2 - 5);
        }
    }

    virtual void TearDown(State &state) override {
        lhs.release();
        rhs.release();
        dst.release();
    }
};

BENCHMARK_TEMPLATE_DEFINE_F(MemoryManager, SaxpyCpuDefault, my::MemoryKind::kDefault)(State& state) {
    my::CpuExecutor executor = { nothreads, my::VectorExtention::kNone };
    for (auto _ : state) {
        DoNotOptimize(saxpy(lhs, rhs, dst, length, executor));
    }
}

BENCHMARK_REGISTER_F(MemoryManager, SaxpyCpuDefault)
    ->Args({1, 512 * 50'000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE_DEFINE_F(MemoryManager, SaxpyCpuPinned, my::MemoryKind::kPinned)(State& state) {
    my::CpuExecutor executor = { nothreads, my::VectorExtention::kNone };
    for (auto _ : state) {
        cudaError_t err = saxpy(lhs, rhs, dst, length, executor);
        if (err != cudaSuccess) {
            state.SkipWithError("failed to execute kernel");
        }
    }
}

BENCHMARK_REGISTER_F(MemoryManager, SaxpyCpuPinned)
    ->Args({1, 512 * 50'000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE_DEFINE_F(MemoryManager, SaxpyGpuDefault, my::MemoryKind::kDefault)(State& state) {
    my::GpuExecutor executor = { nostreams };
    for (auto _ : state) {
        cudaError_t err = saxpy(lhs, rhs, dst, length, executor);
        if (err != cudaSuccess) {
            state.SkipWithError("failed to execute kernel");
        }
    }
}

BENCHMARK_REGISTER_F(MemoryManager, SaxpyGpuDefault)
    ->Args({1, 512 * 50'000})
    ->Args({2, 512 * 50'000})
    ->Args({4, 512 * 50'000})
    ->Args({8, 512 * 50'000})
    ->Args({16, 512 * 50'000})
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE_DEFINE_F(MemoryManager, SaxpyGpuPinned, my::MemoryKind::kPinned)(State& state) {
    my::GpuExecutor executor = { nostreams };
    for (auto _ : state) {
        cudaError_t err = saxpy(lhs, rhs, dst, length, executor);
        if (err != cudaSuccess) {
            state.SkipWithError("failed to execute kernel");
        }
    }
}

BENCHMARK_REGISTER_F(MemoryManager, SaxpyGpuPinned)
    ->Args({1, 512 * 50'000})
    ->Args({4, 512 * 50'000})
    ->Args({1, 512 * 50'000})
    ->Args({2, 512 * 50'000})
    ->Args({4, 512 * 50'000})
    ->Args({8, 512 * 50'000})
    ->Args({16, 512 * 50'000})
    ->Unit(benchmark::kMillisecond);
