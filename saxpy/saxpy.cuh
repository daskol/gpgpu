/**
 * \file saxpy.cuh
 */

#pragma once

enum executor_t {
    kCpu = 0,
    kGpu,
};

void saxpy(float *a, float *b, float *c, size_t length, executor_t executor);
