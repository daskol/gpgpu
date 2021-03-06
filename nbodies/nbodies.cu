/**
 * \file nbodies.cu
 */

#include "nbodies.cuh"

#include <algorithm>
#include <cuda.h>

#include <saxpy/util.cuh>

namespace my {

static constexpr float kMinRadius = 1e-2;
static constexpr float kG = 10;

__host__ __device__
constexpr float cube(float value) {
    return value * value * value;
}

int solve(
    size_t nbodies, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    CpuExecutor &executor
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
    CpuExecutor &executor
) {
    cuda_event_t begin, end;
    if (!begin || !end) {
        return static_cast<int>(cudaGetLastError());
    }

    begin.record();

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

    end.record();
    end.sync();
    executor.ElapsedTime = end - begin;

    return 0;
}

int solve_gpu_shared(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    GpuExecutor &executor
) {
    return 0;
}

__global__
void calc_accels(float const *pos_prev, float *acc_next, size_t nbodies) {
    size_t it = static_cast<size_t>(threadIdx.x + blockDim.x * blockIdx.x);
    float acc[3] = { 0.0, 0.0, 0.0 };

    for (size_t jt = 0; jt != nbodies; ++jt) {
        if (it == jt) {
            continue;
        }

        float dx = pos_prev[3 * jt + 0] - pos_prev[3 * it + 0];
        float dy = pos_prev[3 * jt + 1] - pos_prev[3 * it + 1];
        float dz = pos_prev[3 * jt + 2] - pos_prev[3 * it + 2];
        float dr = sqrtf(dx * dx + dy * dy + dz * dz);

        if (dr < kMinRadius) {
            continue;
        }

        acc[0] += kG * dx / cube(dr);
        acc[1] += kG * dy / cube(dr);
        acc[2] += kG * dz / cube(dr);
    }

    acc_next[3 * it + 0] = acc[0];
    acc_next[3 * it + 1] = acc[1];
    acc_next[3 * it + 2] = acc[2];
}

__global__
void calc_accels_shared(
    float const *pos_prev, float *acc_next,
    size_t nbodies, size_t noblocks
) {
    size_t const it = static_cast<size_t>(threadIdx.x + blockDim.x * blockIdx.x);
    size_t const block_size = 256;

    __shared__
    float pos_cache[3 * block_size];
    float acc[3] = { 0.0, 0.0, 0.0 };

    // Position of the current particle.
    float pos[3] = {
        pos_prev[3 * it + 0], pos_prev[3 * it + 1], pos_prev[3 * it + 1],
    };

    for (size_t block = 0; block != noblocks; ++block) {
        pos_cache[3 * threadIdx.x + 0] = pos_prev[3 * threadIdx.x + 0];
        pos_cache[3 * threadIdx.x + 1] = pos_prev[3 * threadIdx.x + 1];
        pos_cache[3 * threadIdx.x + 2] = pos_prev[3 * threadIdx.x + 2];

        // Wait untill all threads in block initialize corresponding block of
        // particle position.
        __syncthreads();

        for (size_t kt = 0; kt != block_size; ++kt) {
            size_t jt = block * block_size + kt;

            if (it == jt) {
                continue;
            }

            float dx = pos_cache[3 * kt + 0] - pos[0];
            float dy = pos_cache[3 * kt + 1] - pos[1];
            float dz = pos_cache[3 * kt + 2] - pos[2];
            float dr = sqrtf(dx * dx + dy * dy + dz * dz);

            if (dr < kMinRadius) {
                continue;
            }

            acc[0] += kG * dx / cube(dr);
            acc[1] += kG * dy / cube(dr);
            acc[2] += kG * dz / cube(dr);
        }

        // Wait untill all threads estimate forces from a block of particles.
        __syncthreads();
    }

    acc_next[3 * it + 0] = acc[0];
    acc_next[3 * it + 1] = acc[1];
    acc_next[3 * it + 2] = acc[2];
}

__global__
void calc_posits(
    float const *pos_prev, float const *vel_prev, float const *acc_prev,
    float *pos_next, float *vel_next,
    float time_delta, size_t nbodies
) {
    size_t it = static_cast<size_t>(threadIdx.x + blockIdx.x * blockDim.x);
    float time_delta_sqr = time_delta * time_delta;
    float acc[3] = {
        acc_prev[3 * it + 0], acc_prev[3 * it + 1], acc_prev[3 * it + 2],
    };

    pos_next[3 * it + 0] = pos_prev[3 * it + 0] + time_delta * vel_prev[3 * it + 0] + 0.5 * time_delta_sqr * acc[0];
    pos_next[3 * it + 1] = pos_prev[3 * it + 1] + time_delta * vel_prev[3 * it + 1] + 0.5 * time_delta_sqr * acc[1];
    pos_next[3 * it + 2] = pos_prev[3 * it + 2] + time_delta * vel_prev[3 * it + 2] + 0.5 * time_delta_sqr * acc[2];

    vel_next[3 * it + 0] = vel_prev[3 * it + 0] + time_delta * acc[0];
    vel_next[3 * it + 1] = vel_prev[3 * it + 1] + time_delta * acc[1];
    vel_next[3 * it + 2] = vel_prev[3 * it + 2] + time_delta * acc[2];
}

int solve(
    size_t nbodies, size_t nosteps, float time_delta,
    float const *pos_init, float const *vel_init, float *poss, float *vels,
    GpuExecutor &executor
) {
    cuda_event_t begin, end;
    if (!begin || !end) {
        return static_cast<int>(cudaGetLastError());
    }

    size_t size = 3 * nbodies;
    size_t size_bytes = size * sizeof(float);

    float *dev_pos_init, *dev_pos_prev, *dev_pos_next;
    float *dev_vel_init, *dev_vel_prev, *dev_vel_next;

    if (cudaMalloc(&dev_pos_prev, size_bytes) != cudaSuccess) {
        return 1;
    }

    if (cudaMalloc(&dev_pos_next, nosteps * size_bytes) != cudaSuccess) {
        return 1;
    }

    if (cudaMalloc(&dev_vel_prev, size_bytes) != cudaSuccess) {
        return 1;
    }

    if (cudaMalloc(&dev_vel_next, nosteps * size_bytes) != cudaSuccess) {
        return 1;
    }

    dev_pos_init = dev_pos_prev;
    dev_vel_init = dev_vel_prev;

    begin.record(); // Start timer.

    cudaMemcpy(dev_pos_init, pos_init, size_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vel_init, vel_init, size_bytes, cudaMemcpyHostToDevice);

    dim3 nothreads(256);
    dim3 noblocks(nbodies / nothreads.x);

    for (size_t step = 0; step < nosteps; ++step) {
        // Use temporary as a buffer for acceleration.
        float *dev_acc_buffer = dev_pos_next + step * size;

        // Call suitable CUDA kernel if usage of shared memory is allowed.
        if (executor.UseSharedMemory) {
            calc_accels_shared<<<noblocks, nothreads>>>(
                dev_pos_prev, dev_acc_buffer,
                nbodies, noblocks.x);
        } else {
            calc_accels<<<noblocks, nothreads>>>(
                dev_pos_prev, dev_acc_buffer,
                nbodies);
        }

        // Update positions of particles.
        calc_posits<<<noblocks, nothreads>>>(
            dev_pos_prev, dev_vel_prev, dev_acc_buffer,
            dev_pos_next + step * size, dev_vel_next + step * size,
            time_delta, nbodies
        );

        dev_pos_prev = dev_pos_next + step * size;
        dev_vel_prev = dev_vel_next + step * size;
    }

    cudaMemcpy(poss, dev_pos_next, nosteps * size_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vels, dev_vel_next, nosteps * size_bytes, cudaMemcpyDeviceToHost);

    // Stop timer and estimate time spent on calculation.
    end.record();
    end.sync();
    executor.ElapsedTime = end - begin;

    return static_cast<int>(cudaGetLastError());
}

} // namespace my
