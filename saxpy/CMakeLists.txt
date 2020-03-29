add_executable(saxpy-benchmark)

target_sources(saxpy-benchmark
    PRIVATE memory.cuh
    PRIVATE saxpy.cu
    PRIVATE saxpy.cuh
    PRIVATE saxpy_bench.cu
    PRIVATE util.cuh
)

target_link_libraries(saxpy-benchmark
    PRIVATE benchmark::benchmark_main
)
