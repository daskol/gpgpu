/**
 * \file main.cu
 */

#include <iostream>
#include <memory>

#include <nbodies/nbodies.cuh>

int main(int argc, char *argv[]) {
    size_t const nbodies = 128;
    size_t const nosteps = 128;
    float const time_delta = 1e-3;

    auto pos_init = std::make_unique<float[]>(3 * nbodies);
    auto vel_init = std::make_unique<float[]>(3 * nbodies);
    auto poss = std::make_unique<float[]>(3 * nbodies * nosteps);
    auto vels = std::make_unique<float[]>(3 * nbodies * nosteps);

    my::solve(nbodies, nosteps, time_delta,
        pos_init.get(), vel_init.get(),
        poss.get(), vels.get(),
        {});

    return 0;
}
