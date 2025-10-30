## Handwritten Digit Classification — Performance Optimization Report

### Overview
This project optimizes a C++ neural network application for handwritten digit classification. The focus was on improving matrix math efficiency, memory layout, compiler-level optimizations, and I/O behavior to reduce end-to-end training time.

### Baseline
- **Initial training time**: 43 seconds

### Phase 1: Algorithmic and Data Layout Optimizations
- **Matrix multiplication (C = A × B)**: Transposed the `B` matrix to improve cache locality and reduce cache misses during multiplication.
- **Matrix storage layout**: Switched matrix storage from a 2D vector to a contiguous 1D vector. This reduces memory overhead and improves sequential memory access patterns.
- **Result**: Average training time reduced to **26 seconds**.

### Phase 2: Build and I/O Optimizations
- **Compiler flags and SIMD**: Experimented with compilation flags to better utilize hardware capabilities and enable SIMD instructions.
  - **Impact**: Additional performance improvement of about **2–3 seconds**.
- **Image caching**: Implemented an in-memory image cache to avoid reloading images from disk repeatedly.
  - **Impact**: Significant reduction in I/O overhead.
- **Result**: Final average runtime reduced to **11.37 seconds** (averaged over multiple runs).

### End-to-End Performance Summary
| Stage | Avg Time (s) | Delta vs Previous |
|---|---:|---:|
| Baseline | 43.00 | — |
| After Phase 1 (transpose B, 1D storage) | 26.00 | −17.00 |
| After compiler flags (SIMD) | ~23–24 | −2 to −3 |
| Final (with image cache) | 11.37 | ~−12 |

### Key Takeaways
- **Cache-friendly access** (transposing `B`) and **contiguous storage** (1D vectors) substantially improve compute-bound sections like matrix multiplication.
- **Targeted compiler flags and SIMD** yield incremental wins when the algorithm and data layout are already efficient.
- **Avoiding disk I/O** via an image cache delivers a major reduction in overall runtime for data-heavy workloads.


