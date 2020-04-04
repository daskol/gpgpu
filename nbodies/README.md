## N-Bodis Simulation

### Benchmarks

```
$ time nbodies/nbodies-bench
2020-04-04 18:06:38
Running nbodies/nbodies-bench
Run on (8 X 4400 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x4)
  L1 Instruction 32 KiB (x4)
  L2 Unified 256 KiB (x4)
  L3 Unified 8192 KiB (x1)
Load Average: 0.32, 0.35, 0.15
------------------------------------------------------------------------------------------------------------
Benchmark                                                  Time             CPU   Iterations UserCounters...
------------------------------------------------------------------------------------------------------------
CauchyConditionFixture/CpuSolve/10240                   8292 ms         8195 ms            1 exec-time=7.60943k
CauchyConditionFixture/CpuSolve/20480                  30520 ms        30509 ms            1 exec-time=30.5197k
CauchyConditionFixture/GpuSolveGlobalMemory/10240       89.4 ms         89.4 ms            7 exec-time=89.2634
CauchyConditionFixture/GpuSolveGlobalMemory/20480        231 ms          231 ms            3 exec-time=231.171
CauchyConditionFixture/GpuSolveSharedMemory/10240       68.7 ms         68.7 ms           10 exec-time=68.5682
CauchyConditionFixture/GpuSolveSharedMemory/20480        176 ms          176 ms            4 exec-time=176.054
nbodies/nbodies-bench  41,44s user 0,72s system 99% cpu 42,354 total
```
