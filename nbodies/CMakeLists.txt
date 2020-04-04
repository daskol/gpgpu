set(POSITION_INDEPENDENT_CODE ON)

add_library("nbodies-lib" STATIC)

target_sources("nbodies-lib"
    PRIVATE nbodies.cu
    PRIVATE nbodies.cuh
)

add_executable("nbodies-bench")

target_sources("nbodies-bench"
    PRIVATE nbodies_bench.cu
)

target_link_libraries("nbodies-bench"
    PRIVATE nbodies-lib
    PRIVATE benchmark::benchmark_main
)

add_executable("nbodies")

target_sources("nbodies"
    PRIVATE main.cu
)

target_link_libraries("nbodies"
    PRIVATE nbodies-lib
)
