/**
 * \file main.cu
 */

#include <iostream>
#include <memory>
#include <random>

#include <nbodies/nbodies.cuh>

namespace {

void make_init_conditions(float *poss, float *vels, size_t nbodies, size_t seed = 42) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uniform(-1, 1);

    for (size_t it = 0; it != nbodies; ++it) {
        poss[3 * it + 0] = uniform(rng);
        poss[3 * it + 1] = uniform(rng);
        poss[3 * it + 2] = uniform(rng);

        vels[3 * it + 0] = uniform(rng);
        vels[3 * it + 1] = uniform(rng);
        vels[3 * it + 2] = uniform(rng);
    }
}

} // namespace

int main(int argc, char *argv[]) {
    size_t const nbodies = 10240;
    size_t const nosteps = 10;
    float const time_delta = 1e-3;

    std::cout << "no particles:  " << nbodies << '\n'
              << "no time steps: " << nosteps << '\n'
              << "time step:     " << time_delta << '\n';

    auto pos_init = std::make_unique<float[]>(3 * nbodies);
    auto vel_init = std::make_unique<float[]>(3 * nbodies);

    make_init_conditions(pos_init.get(), vel_init.get(), nbodies);

    auto poss = std::make_unique<float[]>(3 * nbodies * nosteps);
    auto vels = std::make_unique<float[]>(3 * nbodies * nosteps);

    auto executor = my::GpuExecutor{
        .UseSharedMemory = false,
    };

    auto err = my::solve(nbodies, nosteps, time_delta,
        pos_init.get(), vel_init.get(),
        poss.get(), vels.get(),
        executor);

    if (err != 0) {
        std::cerr << "[err] failed to execute solver: "
                  << cudaGetErrorString(static_cast<cudaError_t>(err))
                  << '\n';
        return 1;
    }

    std::cout << "elapsed time:  " << executor.ElapsedTime << " ms\n";

    return 0;
}
