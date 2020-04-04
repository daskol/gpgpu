/**
 * \file nbodies.h
 */

#pragma once

namespace my {

struct Executor {
    float ElapsedTime = 0;
};

struct CpuExecutor : Executor {};

int solve(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    CpuExecutor &executor
);

struct GpuExecutor : Executor {
    bool UseSharedMemory = true;
};

int solve(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    GpuExecutor &executor
);

} // namespace my
