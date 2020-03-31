/**
 * \file nbodies.cu
 */

#include "nbodies.cuh"

#include <algorithm>

namespace my {

static constexpr float kMinRadius = 1e-2;
static constexpr float kG = 10;

constexpr float cube(float value) {
    return value * value * value;
}

int solve(
    size_t nbodies, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    CpuExecutor executor
) {
    // Use posistion array as a temporary storage for accelaration.
    float *accs = poss;

    std::fill(accs, accs + 3 * nbodies, 0);
    for (size_t it = 0; it < nbodies; ++it) {
        for (size_t jt = it + 1; jt < nbodies; ++jt) {
            float dx = pos_init[3 * jt + 0] - pos_init[3 * it + 0];
            float dy = pos_init[3 * jt + 1] - pos_init[3 * it + 1];
            float dz = pos_init[3 * jt + 2] - pos_init[3 * it + 2];
            float dr = std::sqrt(dx * dx + dy * dy + dz * dz);

            if (dr < kMinRadius) {
                continue;
            }

            float dax = kG * dx  / cube(dr);
            float day = kG * dy  / cube(dr);
            float daz = kG * dz  / cube(dr);

            accs[3 * it + 0] += dax;
            accs[3 * it + 1] += day;
            accs[3 * it + 2] += daz;

            accs[3 * jt + 0] -= dax;
            accs[3 * jt + 1] -= day;
            accs[3 * jt + 2] -= daz;
        }
    }

    float const time_delta_sqr = time_delta * time_delta;

    for (size_t it = 0; it != nbodies; ++it) {
        vels[3 * it + 0] = vel_init[3 * it + 0] + accs[3 * it + 0] * time_delta;
        vels[3 * it + 1] = vel_init[3 * it + 1] + accs[3 * it + 1] * time_delta;
        vels[3 * it + 2] = vel_init[3 * it + 2] + accs[3 * it + 2] * time_delta;

        poss[3 * it + 0] = pos_init[3 * it + 0] + vels[3 * it + 0] * time_delta + 0.5 * accs[3 * it + 0] * time_delta_sqr;
        poss[3 * it + 1] = pos_init[3 * it + 1] + vels[3 * it + 1] * time_delta + 0.5 * accs[3 * it + 1] * time_delta_sqr;
        poss[3 * it + 2] = pos_init[3 * it + 2] + vels[3 * it + 2] * time_delta + 0.5 * accs[3 * it + 2] * time_delta_sqr;
    }

    return 0;
}

int solve(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    CpuExecutor executor
) {
    float const *pos_prev = pos_init;
    float const *vel_prev = vel_init;

    for (size_t it = 0; it != nosteps; ++it) {
        float *pos_next = &poss[3 * nbodies * it];
        float *vel_next = &vels[3 * nbodies * it];

        solve(
            nbodies,
            time_delta,
            pos_prev,
            vel_prev,
            &poss[3 * nbodies * it],
            &vels[3 * nbodies * it],
            executor);

         pos_prev = pos_next;
         vel_prev = vel_next;
    }

    return 0;
}

} // namespace my
