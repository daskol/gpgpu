/**
 * \file saxpy_bench.cu
 */

#include <memory>
#include <benchmark/benchmark.h>
#include <saxpy/saxpy.cuh>

template <executor_t Executor>
void BM_saxpy(benchmark::State& state) {
    size_t length = state.range();
    std::unique_ptr<float[]> a{new float[length]};
    std::unique_ptr<float[]> b{new float[length]};
    std::unique_ptr<float[]> c{new float[length]};
    for (auto _ : state) {
        saxpy(a.get(), b.get(), c.get(), length, Executor);
    }
}

BENCHMARK_TEMPLATE(BM_saxpy, executor_t::kCpu)
    ->RangeMultiplier(8)
    ->Range(1 << 9, 1 << 24);

BENCHMARK_TEMPLATE(BM_saxpy, executor_t::kGpu)
    ->RangeMultiplier(8)
    ->Range(1 << 9, 1 << 24);
