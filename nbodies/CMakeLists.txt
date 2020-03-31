add_executable("nbodies")

target_sources("nbodies"
    PRIVATE main.cu
    PRIVATE nbodies.cu
    PRIVATE nbodies.cuh
)
