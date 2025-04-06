---
title: "TVM: 1D convolution GPU Optimization"
date: 2025-04-06T17:06:56+08:00
categories: ["tvm"]
summary: "This blog demonstrates optimization techniques for GEMM in GPU using TVM, including thread organization and memory hierarchy exploitation."
---

## Summary

This blog demonstrates optimization techniques for GEMM in GPU using TVM, including thread organization and memory hierarchy exploitation.

## 1. Environment

Env: Google Colab T4 GPU

We test based on this configuration:

```py
M = 1024
N = 512
K = 2048
dtype = 'float32'
a_np = np.random.rand(M, K).astype(dtype)
w_np = np.random.rand(K, N).astype(dtype)
ref = np.matmul(a_np, w_np)
```

### 2.1 Baseline and Manual Optimizations

#### 2.1.1 Naive Baseline

The initial implementation runs at **84.52 ms**.

```py
def make_gemm_gpu_scheduler_naive(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)

    # overall index of a thread: 𝑖=blockIdx.x×blockDim.x+threadIdx.x
    block_x = te.thread_axis("blockIdx.y")
    block_y = te.thread_axis("blockIdx.x")

    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    s[C].bind(y, block_y)
    s[C].bind(x, block_x)
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        print("=" * 100)
    return s, A, B, C
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((1024, 2048), "float32"), B: T.Buffer((2048, 512), "float32"), C: T.Buffer((1024, 512), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_y = T.launch_thread("blockIdx.y", 1024)
    blockIdx_x = T.launch_thread("blockIdx.x", 512)
    C_1 = T.Buffer((524288,), data=C.data)
    C_1[blockIdx_y * 512 + blockIdx_x] = T.float32(0)
    for k in range(2048):
        A_1 = T.Buffer((2097152,), data=A.data)
        B_1 = T.Buffer((1048576,), data=B.data)
        # blockIdx_y * 512 + blockIdx_x: output position
        # blockIdx_y * 2048 + k: A_1 position
        # k * 512 + blockIdx_x: B_1 position
        C_1[blockIdx_y * 512 + blockIdx_x] = C_1[blockIdx_y * 512 + blockIdx_x] + A_1[blockIdx_y * 2048 + k] * B_1[k * 512 + blockIdx_x]
```

Here we declare a 2D block region, each block is responsible for calculating an output. Extremely slow.

#### 2.1.2 v1: Tiling + 1D Threads

Now we splits the x-axis into blocks and tiles, binding the outer part to blocks and inner part to threads.

```py
# opt v1: tiling + threads 1D
def make_gemm_gpu_scheduler_v1(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)

    x, y = s[C].op.axis

    # split the axes
    xo, xi = s[C].split(x, factor=32)

    # bind the outer axes to blocks
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(y, te.thread_axis("blockIdx.y"))

    # bind the inner axes to threads
    s[C].bind(xi, te.thread_axis("threadIdx.x"))

    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        print("=" * 100)

    return s, A, B, C
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((1024, 2048), "float32"), B: T.Buffer((2048, 512), "float32"), C: T.Buffer((1024, 512), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 32)
    threadIdx_x = T.launch_thread("threadIdx.x", 32)
    blockIdx_y = T.launch_thread("blockIdx.y", 512)
    C_1 = T.Buffer((524288,), data=C.data)
    C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y] = T.float32(0)
    for k in range(2048):
        A_1 = T.Buffer((2097152,), data=A.data)
        B_1 = T.Buffer((1048576,), data=B.data)
        # lockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y: output
        # blockIdx_x * 65536 + threadIdx_x * 2048 + k: A1
        # k * 512 + blockIdx_y: B1
        C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y] = C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y] + A_1[blockIdx_x * 65536 + threadIdx_x * 2048 + k] * B_1[k * 512 + blockIdx_y]
```

Here we use 1D thread architecture to support more efficient parallelism. This improves performance to **36.98 ms**.

#### 2.1.3 v2: Tiling + 2D Threads

Building on v1, this version implements 2D thread organization by splitting both x and y axes. This organizes threads in a grid of 32×32 threads per block, with more efficient utilization of GPU resources.

```py
# opt v2: tiling + threads 2D
def make_gemm_gpu_scheduler_v2(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)

    x, y = s[C].op.axis

    # split the axes
    xo, xi = s[C].split(x, factor=32)
    yo, yi = s[C].split(y, factor=32)

    # bind the outer axes to blocks
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("blockIdx.y"))

    # bind the inner axes to threads
    s[C].bind(xi, te.thread_axis("threadIdx.x"))
    s[C].bind(yi, te.thread_axis("threadIdx.y"))

    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        print("=" * 100)

    return s, A, B, C

dev = tvm.cuda()
time, res, func, comp = benchmark_gemm_tvm(
    make_gemm_gpu_scheduler_v2, M, K, N, dev, a_np, w_np, num_runs=20, repeat=20
)
np.testing.assert_allclose(res, ref, rtol=1e-4)
print(f"[TVM v2] time: {time*1e3:.4f} ms")
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((1024, 2048), "float32"), B: T.Buffer((2048, 512), "float32"), C: T.Buffer((1024, 512), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 32)
    threadIdx_x = T.launch_thread("threadIdx.x", 32)
    blockIdx_y = T.launch_thread("blockIdx.y", 16)
    threadIdx_y = T.launch_thread("threadIdx.y", 32)
    C_1 = T.Buffer((524288,), data=C.data)
    C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y] = T.float32(0)
    for k in range(2048):
        A_1 = T.Buffer((2097152,), data=A.data)
        B_1 = T.Buffer((1048576,), data=B.data)
        # blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y: output
        # blockIdx_x * 65536 + threadIdx_x * 2048 + k: A1
        # k * 512 + blockIdx_y * 32 + threadIdx_y: B1
        C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y] = C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y] + A_1[blockIdx_x * 65536 + threadIdx_x * 2048 + k] * B_1[k * 512 + blockIdx_y * 32 + threadIdx_y]
```

Now we use the 2D threads architecture to further increase the efficiency. Performance slightly improves to **35.50 ms**.

#### 2.1.4 v3: Shared Memory Cache + Multi-threaded Loading

This version makes use of the GPU memory hierarchy:

- Caches input matrices `A` and `B` in shared memory
- Splits the reduction axis `K` into tiles
- Uses multiple threads to cooperatively load data into shared memory
- Processes data in blocks of 16×16 elements

```py
# opt3: v2 + cache (with multi threads)
def make_gemm_gpu_scheduler_v3(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)
    block_x, block_y = 16, 16
    xo, xi = s[C].split(C.op.axis[0], factor=block_x)
    yo, yi = s[C].split(C.op.axis[1], factor=block_y)

    # split k
    tile_k = 8
    ko, ki = s[C].split(k, factor=tile_k)

    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("blockIdx.y"))
    s[C].bind(xi, te.thread_axis("threadIdx.x"))
    s[C].bind(yi, te.thread_axis("threadIdx.y"))

    AA = s.cache_read(A, "shared", [C])
    BB = s.cache_read(B, "shared", [C])

    s[AA].compute_at(s[C], ko)
    s[BB].compute_at(s[C], ko)

    # multi threads for loading data
    # this increases performance a lot!
    AAxi, AAyi = s[AA].split(s[AA].op.axis[0], nparts=block_x)
    AAxx, AAxy = s[AA].split(s[AA].op.axis[1], nparts=block_y)
    s[AA].bind(AAxi, te.thread_axis("threadIdx.x"))
    s[AA].bind(AAxx, te.thread_axis("threadIdx.y"))

    BBxi, BByi = s[BB].split(s[BB].op.axis[0], nparts=block_x)
    BBxx, BBxy = s[BB].split(s[BB].op.axis[1], nparts=block_y)
    s[BB].bind(BBxi, te.thread_axis("threadIdx.x"))
    s[BB].bind(BBxx, te.thread_axis("threadIdx.y"))

    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, B, C], simple_mode=True))
        print("=" * 100)

    return s, A, B, C
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((1024, 2048), "float32"), B: T.Buffer((2048, 512), "float32"), C: T.Buffer((1024, 512), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 64)
    A_shared = T.allocate([128], "float32", "shared")
    B_shared = T.allocate([128], "float32", "shared")
    threadIdx_x = T.launch_thread("threadIdx.x", 16)
    blockIdx_y = T.launch_thread("blockIdx.y", 32)
    threadIdx_y = T.launch_thread("threadIdx.y", 16)
    C_1 = T.Buffer((524288,), data=C.data)
    C_1[blockIdx_x * 8192 + threadIdx_x * 512 + blockIdx_y * 16 + threadIdx_y] = T.float32(0)
    for k_outer in range(256):
        A_shared_1 = T.Buffer((128,), data=A_shared, scope="shared")
        # load data into A_shared_1 in parallel
        with T.launch_thread("threadIdx.x", 16) as threadIdx_x_1:
            threadIdx_y_1 = T.launch_thread("threadIdx.y", 16)
            if T.likely(threadIdx_y_1 < 8):
                A_1 = T.Buffer((2097152,), data=A.data)
                A_shared_1[threadIdx_x_1 * 8 + threadIdx_y_1] = A_1[blockIdx_x * 32768 + threadIdx_x_1 * 2048 + k_outer * 8 + threadIdx_y_1]
        # load data into B_shared_1 in parallel
        B_shared_1 = T.Buffer((128,), data=B_shared, scope="shared")
        with T.launch_thread("threadIdx.x", 16) as threadIdx_x_1:
            threadIdx_y_1 = T.launch_thread("threadIdx.y", 16)
            if T.likely(threadIdx_x_1 < 8):
                B_1 = T.Buffer((1048576,), data=B.data)
                B_shared_1[threadIdx_x_1 * 16 + threadIdx_y_1] = B_1[k_outer * 4096 + threadIdx_x_1 * 512 + blockIdx_y * 16 + threadIdx_y_1]
        for k_inner in range(8):
            # blockIdx_x * 8192 + threadIdx_x * 512 + blockIdx_y * 16 + threadIdx_y: output
            # threadIdx_x * 8 + k_inner: A1
            # k_inner * 16 + threadIdx_y: B1
            C_1[blockIdx_x * 8192 + threadIdx_x * 512 + blockIdx_y * 16 + threadIdx_y] = C_1[blockIdx_x * 8192 + threadIdx_x * 512 + blockIdx_y * 16 + threadIdx_y] + A_shared_1[threadIdx_x * 8 + k_inner] * B_shared_1[k_inner * 16 + threadIdx_y]
```

As we can see in the IR, we use `A_shared_1` and `B_shared_1` to save the tiles in on-chip memory and reduce the time usage to visit the global memory.

This substantially improves performance to **8.11 ms**.

### 2.2 AutoTVM Optimization

The AutoTVM implementation explores a search space including:

- Different tile sizes for x, y axes (8, 16, or 32)
- Different tile sizes for the reduction axis (8 or 16)
- Whether to vectorize memory accesses
- Cache writing to local memory
- Cache reading to shared memory

After exploring 36 different configurations, the AutoTVM tuner found a solution running at **42.56 ms**.

>Note: Colab always crashes if we search in a larger space (high-cost exploration). So we just search a small space here.

### 2.3 Performance Results

All timings are in milliseconds for a matrix multiplication with `M=1024`, `K=2048`, `N=512` on a GPU:

| **Implementation**       | **Time (ms)** | **Speedup vs. Naive** | **Speedup vs. Previous** |
|--------------------------|---------------|------------------------|--------------------------|
| **Naive**                | 84.52         | 1.0×                   | -                        |
| **v1** (1D Threads)      | 36.98         | 2.3×                   | 2.3×                     |
| **v2** (2D Threads)      | 35.50         | 2.4×                   | 1.04×                    |
| **v3** (Shared Memory)   | 8.11          | 10.4×                  | 4.4×                     |
| **AutoTVM**              | 42.56         | 2.0×                   | -                        |
| **NumPy** (CPU)          | 74.95         | 1.1×                   | -                        |
| **PyTorch CPU**          | 18.74         | 4.5×                   | -                        |
| **PyTorch CUDA**         | 0.70          | 120.7×                 | -                        |

The manual optimization v3 achieves a **10.4× speedup** over the naive baseline by utilizing GPU-specific optimizations like shared memory, tiling, and cooperative thread loading.

The progression from naive to optimized shows the importance of:

- Effective thread organization
- Proper memory hierarchy utilization
- Cooperative data loading

These principles are fundamental to achieving high performance in GPU matrix multiplication implementations.

## 3. Appendix

- Notebook (all of the code used for this blog): [link](gemm.ipynb)
- Summary of the TVM paper: [link](../../TVM)
