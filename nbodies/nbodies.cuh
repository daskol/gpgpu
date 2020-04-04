/**
 * \file nbodies.h
 */

#pragma once

namespace my {

struct CpuExecutor {
    float ElapsedTime = 0;
};

int solve(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    CpuExecutor &executor
);

struct GpuExecutor {
    bool UseSharedMemory = true;
    float ElapsedTime = 0;
};

int solve(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    GpuExecutor &executor
);

} // namespace my
