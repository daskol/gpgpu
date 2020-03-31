/**
 * \file nbodies.h
 */

#pragma once

namespace my {

struct CpuExecutor {};

int solve(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    CpuExecutor executor
);

} // namespace my
