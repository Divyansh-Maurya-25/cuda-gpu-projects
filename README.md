# CUDA GPU Projects — Parallel Computing on the GPU

> Three progressive GPU programming projects in CUDA C: spatial distance histograms optimized with shared memory and atomic reduction, and a parallel Bloom filter with GPU-ported SipHash. Built for COP 4520 at USF.

[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![C](https://img.shields.io/badge/C-99-blue)](https://en.wikipedia.org/wiki/C99)

---

## Project 1 — Baseline SDH Kernel

**Task:** Compute a Spatial Distance Histogram (SDH) — count all pairwise distances between N 3D points and bin them into a histogram.

**Brute-force O(n²) complexity on GPU:**
```c
__global__ void sdh_kernel(...)  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    for (int j = i + 1; j < num_points; j++) {
        float dist = euclidean(points[i], points[j]);
        int bin = (int)(dist / bucket_width);
        atomicAdd((unsigned long long*)&d_hist[bin].d_cnt, 1ULL);
    }
}
```

One thread per point i; loops over all j > i. Global memory atomics for histogram update. Includes `CUDA_CHECK` macro and `cudaDeviceSynchronize`.

**Result:** Correct baseline. Bottleneck: global memory atomic contention.

---

## Project 2 — Optimized SDH Kernel (96/100)

All optimizations applied on top of Project 1's correctness baseline.

### Optimization 1 — Struct-of-Arrays (SoA) Memory Layout
```c
// Before (AoS): point[i].x, point[i].y, point[i].z  →  non-coalesced
// After  (SoA): dev_x[i], dev_y[i], dev_z[i]        →  coalesced reads
```
SoA ensures threads in a warp access contiguous memory addresses, maximizing memory bus utilization.

### Optimization 2 — Shared Memory Tiling
Each thread block loads a tile of points into `__shared__` memory before computing distances. Shared memory latency is ~100× lower than global memory.

### Optimization 3 — Per-Block Histogram Copies
```c
// num_copies (1–32) histogram copies in shared memory per block
// Each thread updates one copy → eliminates intra-block atomic contention
// Final reduction merges copies into global histogram
```

### Optimization 4 — Cache & Loop Hints
```c
float x_i = __ldg(&dev_x[i]);   // read-only cache (L1 texture cache)
#pragma unroll 4                  // loop unrolling for instruction pipelining
```

### Adaptive Shared Memory Check
```c
if (required_smem > prop.sharedMemPerBlock)
    // fall back to fewer histogram copies
```

**Timing:** CUDA Events measure kernel execution time end-to-end.

---

## Project 3 — Parallel Bloom Filter

**Task:** Implement a Bloom filter entirely on the GPU — both insert and query operations run as parallel kernels.

### SipHash Ported to GPU Device Code
SipHash is a fast, collision-resistant hash function. Porting it to a CUDA `__device__` function required eliminating all host-side constructs and making it stateless:
```c
__device__ uint64_t siphash_device(const uint8_t *data, size_t len, uint64_t k0, uint64_t k1);
```

### Kernels
```c
__global__ void bloom_insert_kernel(uint8_t *filter, const Item *items, int n);
__global__ void bloom_check_kernel(const uint8_t *filter, const Item *queries, int *misses, int n);
```
`atomicAdd` used for miss counting. Each thread handles one item independently — embarrassingly parallel.

### False Positive Rate
Bloom filters trade memory for speed at the cost of false positives. The README includes the theoretical FPR formula: `(1 - e^(-kn/m))^k` where k = hash functions, n = items, m = filter bits.

---

## Core CS Concepts

| Concept | Project |
|---|---|
| **GPU thread hierarchy** (grid/block/warp) | All |
| **Global memory coalescing** | Project 2 (SoA layout) |
| **Shared memory optimization** | Project 2 (tiling + histogram copies) |
| **Atomic operations** | Projects 1, 2, 3 |
| **CUDA Events timing** | Project 2 |
| **Hash function porting** (SipHash → device code) | Project 3 |
| **Probabilistic data structures** (Bloom filter) | Project 3 |

---

## Setup

```bash
git clone https://github.com/Divyansh-Maurya-25/cuda-gpu-projects.git
cd cuda-gpu-projects
```

Requires CUDA Toolkit 11+. Compile with:
```bash
# Project 1
nvcc -o sdh_basic Project1/proj1-dm17.cu

# Project 2
nvcc -o sdh_optimized Project2/proj2-dm17/proj2-dm17.cu

# Project 3
nvcc -o bloom Project3/proj3-dm17/proj3-dm17.cu
```

---

## File Structure

```
cuda-gpu-projects/
├── Project1/
│   └── proj1-dm17.cu          # Baseline SDH kernel
├── Project2/
│   └── proj2-dm17/
│       └── proj2-dm17.cu      # Optimized SDH (SoA + shared memory)
├── Project3/
│   └── proj3-dm17/
│       └── proj3-dm17.cu      # Parallel Bloom filter
└── README.md
```

---

## Course Context

Projects for **CAP 5768 — GPU Computing** at the University of South Florida. Scores: Project 1 (100/100), Project 2 (96/100), Project 3 (100/100).
