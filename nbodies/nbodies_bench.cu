/**
 * \file nbodies_bench.cu
 */

#include <memory>
#include <random>

#include <benchmark/benchmark.h>

#include <nbodies/nbodies.cuh>

using Counter = benchmark::Counter;

class CauchyConditionFixture : public benchmark::Fixture {
protected:
    static constexpr size_t kNoTimeSteps = 16;
    static constexpr float kTimeDelta = 1e-3;

protected:
    size_t nbodies;

    std::unique_ptr<float[]> pos_init;
    std::unique_ptr<float[]> vel_init;

    std::unique_ptr<float[]> poss;
    std::unique_ptr<float[]> vels;

public:
    virtual void SetUp(benchmark::State const &state) override {
        // Overriden for backward compatibility.
    }

    virtual void TearDown(benchmark::State const &state) override {
        // Overriden for backward compatibility.
    }

    virtual void SetUp(benchmark::State &state) override {
        nbodies = state.range(0);

        pos_init = std::make_unique<float[]>(3 * nbodies);
        vel_init = std::make_unique<float[]>(3 * nbodies);

        poss = std::make_unique<float[]>(3 * nbodies * kNoTimeSteps);
        vels = std::make_unique<float[]>(3 * nbodies * kNoTimeSteps);

        std::mt19937_64 rng;
        std::uniform_real_distribution<float> uniform(-1, 1);

        for (size_t it = 0; it != nbodies; ++it) {
            pos_init[3 * it + 0] = uniform(rng);
            pos_init[3 * it + 1] = uniform(rng);
            pos_init[3 * it + 2] = uniform(rng);

            vel_init[3 * it + 0] = uniform(rng);
            vel_init[3 * it + 1] = uniform(rng);
            vel_init[3 * it + 2] = uniform(rng);
        }
    }

    virtual void TearDown(benchmark::State &state) override {
        // Do nothing.
    }
};

BENCHMARK_DEFINE_F(CauchyConditionFixture, CpuSolve)(benchmark::State &state) {
    state.counters["exec-time"] = Counter(0, Counter::kAvgIterations);
    auto executor = my::CpuExecutor{};
    for (auto _ : state) {
        auto err = my::solve(
            nbodies, kNoTimeSteps, kTimeDelta,
            pos_init.get(), vel_init.get(),
            poss.get(), vels.get(),
            executor
        );

        if (err) {
            state.SkipWithError("failed to run solver");
            return;
        }

        state.counters["exec-time"] += executor.ElapsedTime;
    }
}

BENCHMARK_REGISTER_F(CauchyConditionFixture, CpuSolve)
    ->Arg(10240)
    ->Arg(20480)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(CauchyConditionFixture, GpuSolveGlobalMemory)(benchmark::State &state) {
    state.counters["exec-time"] = Counter(0, Counter::kAvgIterations);
    auto executor = my::GpuExecutor{ .UseSharedMemory = false };
    for (auto _ : state) {
        auto err = my::solve(
            nbodies, kNoTimeSteps, kTimeDelta,
            pos_init.get(), vel_init.get(),
            poss.get(), vels.get(),
            executor
        );

        if (err) {
            state.SkipWithError("failed to run solver");
            return;
        }

        state.counters["exec-time"] += executor.ElapsedTime;
    }
}

BENCHMARK_REGISTER_F(CauchyConditionFixture, GpuSolveGlobalMemory)
    ->Arg(10240)
    ->Arg(20480)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_DEFINE_F(CauchyConditionFixture, GpuSolveSharedMemory)(benchmark::State &state) {
    state.counters["exec-time"] = Counter(0, Counter::kAvgIterations);
    auto executor = my::GpuExecutor{ .UseSharedMemory = true };
    for (auto _ : state) {
        auto err = my::solve(
            nbodies, kNoTimeSteps, kTimeDelta,
            pos_init.get(), vel_init.get(),
            poss.get(), vels.get(),
            executor
        );

        if (err) {
            state.SkipWithError("failed to run solver");
            return;
        }

        state.counters["exec-time"] += executor.ElapsedTime;
    }
}

BENCHMARK_REGISTER_F(CauchyConditionFixture, GpuSolveSharedMemory)
    ->Arg(10240)
    ->Arg(20480)
    ->Unit(benchmark::kMillisecond);
