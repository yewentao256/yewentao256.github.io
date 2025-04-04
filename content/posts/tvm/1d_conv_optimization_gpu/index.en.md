---
title: "TVM: 1D convolution GPU Optimization"
date: 2025-04-03T21:50:56+08:00
categories: ["tvm"]
summary: "This blog demonstrates optimization techniques for 1D GPU convolution using TVM, including thread organization, memory hierarchy exploitation, and low-level optimizations."
---

## Summary

This blog demonstrates optimization techniques for 1D GPU convolution using TVM, including thread organization, memory hierarchy exploitation, and low-level optimizations

## 1. Environment

Env: Google Colab T4 GPU

```bash
Sun Mar 23 19:26:52 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   46C    P0             26W /   70W |     102MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
+-----------------------------------------------------------------------------------------+
```

We test based on this configuration:

```py
M = 16384
N = 32
dtype = 'float32'
a_np = np.random.rand(M).astype(dtype)
w_np = np.random.rand(N).astype(dtype)
ref = np.convolve(a_np, w_np)
```

## 2. Optimization

### 2.1 Baseline and Manual Optimizations

#### 2.1.1 Naive Baseline

The initial implementation creates a large reduce axis of size `(M + N - 1)` and uses boundary checks inside an `if_then_else` conditional. This runs extremely slowly at **18.29 ms**.

```py

# naive baseline
def make_conv1d_gpu_scheduler_naive(M, N, dtype="float32", verbose=True):
    A = te.placeholder((M,), name="A", dtype=dtype)
    W = te.placeholder((N,), name="W", dtype=dtype)
    k = te.reduce_axis((0, M + N - 1), "k")   # k in [0, M+N-1)
    B = te.compute(
        (M + N - 1,),   # output shape, n from (0, M + N - 1)
        # if_then_else: if satisfy "any" condition, return 0 else A[k] * W[n - k]
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )
    s = te.create_schedule(B.op)
    i = B.op.axis[0]
    s[B].bind(i, te.thread_axis("blockIdx.x"))
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)

    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((16384,), "float32"), W: T.Buffer((32,), "float32"), B: T.Buffer((16415,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 16415)
    B[blockIdx_x] = T.float32(0)
    for k in range(16415):
        # if 16384 <= k or blockIdx_x - k < 0 or 32 <= blockIdx_x - k: += 0
        # else: += A[k] * W[blockIdx_x - k]
        B[blockIdx_x] = B[blockIdx_x] + T.if_then_else(16384 <= k or blockIdx_x - k < 0 or 32 <= blockIdx_x - k, T.float32(0), A[k] * W[blockIdx_x - k])
```

The most naive version, we just use `16415` blocks to calculate the result.

#### 2.1.2 v1: Compute Refactor

The first major optimization refactors the computation to sum over `[0, N)` instead of `[0, M+N-1)`, checking if (i-r) is valid in A.

```py
# optimize v1, compute refactor
def make_conv1d_gpu_scheduler_v1(M, N, dtype="float32", verbose=True):
    A = te.placeholder((M,), name="A", dtype=dtype)
    W = te.placeholder((N,), name="W", dtype=dtype)
    r = te.reduce_axis((0, N), name="r")
    B = te.compute(
        (M + N - 1,),
        lambda i: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(i - r >= 0, i - r < M),
                A[i - r],
                tvm.tir.const(0, dtype)
            ) * W[r],
            axis=r
        ),
        name="B"
    )

    s = te.create_schedule(B.op)
    i = B.op.axis[0]
    s[B].bind(i, te.thread_axis("blockIdx.x"))
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)

    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((16384,), "float32"), W: T.Buffer((32,), "float32"), B: T.Buffer((16415,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 16415)
    B[blockIdx_x] = T.float32(0)
    for r in range(32):
        # r: kernel position
        # if 0 <= blockIdx_x - r and blockIdx_x - r < 16384: += A[blockIdx_x - r] * W[r]
        # else: += 0
        B[blockIdx_x] = B[blockIdx_x] + T.if_then_else(0 <= blockIdx_x - r and blockIdx_x - r < 16384, A[blockIdx_x - r], T.float32(0)) * W[r]
```

Same to CPU, we calculate by `B[n] = Σ(k=16415) A[k] * W[n-k]`. Now we change it to `B[n] = Σ(k=0→32) A[n-k] * W[k]`. Also, we optimize the if statement.

This results in a significant speedup to **0.107 ms**.

#### 2.1.3 v2: Thread-Level Parallelism

Building on v1, this version adds thread-level parallelism by splitting the output axis and binding to both blocks and threads. This improves performance further to **0.0251 ms**

```py
# optimize v2: v1 + basic threads
def make_conv1d_gpu_scheduler_v2(M, N, dtype="float32", verbose=True):
    s, A, W, B = make_conv1d_gpu_scheduler_v1(M, N, dtype, False)

    # out axis
    i = B.op.axis[0]
    block_i, thread_i = s[B].split(i, factor=8)

    # bind to block and thread
    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(thread_i, te.thread_axis("threadIdx.x"))

    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)

    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((16384,), "float32"), W: T.Buffer((32,), "float32"), B: T.Buffer((16415,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 2052)
    threadIdx_x = T.launch_thread("threadIdx.x", 8)    # 8 threads each block
    if T.likely(blockIdx_x * 8 + threadIdx_x < 16415):
        B[blockIdx_x * 8 + threadIdx_x] = T.float32(0)
    for r in range(32):
        if T.likely(blockIdx_x * 8 + threadIdx_x < 16415):
            # blockIdx_x * 8 + threadIdx_x: output position
            # blockIdx_x * 8 + threadIdx_x - r: input position
            # r: kernel position
            # if 0 <= blockIdx_x * 8 + threadIdx_x - r and blockIdx_x * 8 + threadIdx_x - r < 16384: += A[blockIdx_x * 8 + threadIdx_x - r] * W[r]
            # else: += 0
            B[blockIdx_x * 8 + threadIdx_x] = B[blockIdx_x * 8 + threadIdx_x] + T.if_then_else(0 <= blockIdx_x * 8 + threadIdx_x - r and blockIdx_x * 8 + threadIdx_x - r < 16384, A[blockIdx_x * 8 + threadIdx_x - r], T.float32(0)) * W[r]
```

Here we introduce the usage of **threads**. We use `2052` blocks and `8` threads for each block.

#### 2.1.4 v3: 2D Thread Organization

This optimization improves on v2 by organizing threads in a 2D grid (`4×4 = 16` threads per block). Performance improves to **0.0158 ms**

```py
# optimize v3: v1 + 2D threads
def make_conv1d_gpu_scheduler_v3(M, N, dtype="float32", verbose=True):
    s, A, W, B = make_conv1d_gpu_scheduler_v1(M, N, dtype, False)

    i = B.op.axis[0]
    block_i, thread_i = s[B].split(i, factor=16)
    warp_i, lane_i = s[B].split(thread_i, factor=4)

    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(warp_i, te.thread_axis("threadIdx.y"))
    s[B].bind(lane_i, te.thread_axis("threadIdx.x"))

    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)

    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((16384,), "float32"), W: T.Buffer((32,), "float32"), B: T.Buffer((16415,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 1026)
    threadIdx_y = T.launch_thread("threadIdx.y", 4)
    threadIdx_x = T.launch_thread("threadIdx.x", 4)
    if T.likely(blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x < 16415):
        B[blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x] = T.float32(0)
    for r in range(32):
        if T.likely(blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x < 16415):
            # blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x: output position
            # blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x - r: input position
            # r: kernel position
            B[blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x] = B[blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x] + T.if_then_else(0 <= blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x - r and blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x - r < 16384, A[blockIdx_x * 16 + threadIdx_y * 4 + threadIdx_x - r], T.float32(0)) * W[r]
```

Here we use `4*4` threads instead of the `8` threads in one block, which make more use of the ability of parallelism on a GPU.

#### 2.1.5 v4: Memory Hierarchy + Reduction Splitting

This version leverages the GPU memory hierarchy by adding:

- Cache writing to local memory
- Cache reading of kernel weights to shared memory
- Splitting the reduction axis

Performance improves to **0.0147 ms**.

```py
# optimize v4: v1 + 1D thread + cache + split reduce
def make_conv1d_gpu_scheduler_v4(M, N, dtype="float32", verbose=True):
    s, A, W, B = make_conv1d_gpu_scheduler_v1(M, N, dtype, False)

    # IMPORTANT: create caches BEFORE thread binding
    C_local = s.cache_write(B, "local")
    W_shared = s.cache_read(W, "shared", [C_local])

    i = B.op.axis[0]
    block_i, thread_i = s[B].split(i, factor=32)
    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(thread_i, te.thread_axis("threadIdx.x"))

    # schedule the local cache
    s[C_local].compute_at(s[B], thread_i)

    i_local = C_local.op.axis[0]
    rx = C_local.op.reduce_axis[0]
    # split the reduction axis
    rxo, rxi = s[C_local].split(rx, factor=4)

    # schedule shared memory
    s[W_shared].compute_at(s[C_local], rxo)

    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)

    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((16384,), "float32"), W: T.Buffer((32,), "float32"), B: T.Buffer((16415,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 513)
    B_local = T.allocate([1], "float32", "local")
    W_shared = T.allocate([4], "float32", "shared")
    threadIdx_x = T.launch_thread("threadIdx.x", 32)
    B_local_1 = T.Buffer((1,), data=B_local, scope="local", align=4) # tmp variable for accumulation
    B_local_1[0] = T.float32(0)
    for r_outer in range(8):
        W_shared_1 = T.Buffer((4,), data=W_shared, scope="shared", align=16)
        for ax0 in range(4):
            # firstly load kernel to shared memory (size of 4)
            W_shared_1[ax0] = W[r_outer * 4 + ax0]
        for r_inner in range(4):
            if T.likely(blockIdx_x * 32 + threadIdx_x < 16415):
                # B_local_1: out position
                # blockIdx_x * 32 + threadIdx_x - r_inner - r_outer * 4: input position
                # r_inner: weight position of current shared memory
                B_local_1[0] = B_local_1[0] + T.if_then_else(0 <= blockIdx_x * 32 + threadIdx_x - r_inner - r_outer * 4 and blockIdx_x * 32 + threadIdx_x - r_inner - r_outer * 4 < 16384, A[blockIdx_x * 32 + threadIdx_x - r_inner - r_outer * 4], T.float32(0)) * W_shared_1[r_inner]
    if T.likely(blockIdx_x * 32 + threadIdx_x < 16415):
        B[blockIdx_x * 32 + threadIdx_x] = B_local_1[0]
```

Here we introduce the usage of local memory and shared memory: `B_local = T.allocate([1], "float32", "local")`, `W_shared = T.allocate([4], "float32", "shared")`.

Thanks to the local/shared memory, we don't need to rewrite to global memory each time we do a calculation.

#### 2.1.6 v5: Unrolling + 2D Thread Optimization

The final optimization combines 2D thread organization with loop unrolling on the inner reduction axis.

```py
# optimize v5: v4 + 2D threads + unroll
def make_conv1d_gpu_scheduler_v5(M, N, dtype="float32", verbose=True):
    s, A, W, B = make_conv1d_gpu_scheduler_v1(M, N, dtype, False)

    C_local = s.cache_write(B, "local")
    W_shared = s.cache_read(W, "shared", [C_local])

    i = B.op.axis[0]
    block_i, thread_i = s[B].split(i, factor=32)
    # split 2D threads
    warp_i, lane_i = s[B].split(thread_i, factor=4)
    s[B].bind(block_i, te.thread_axis("blockIdx.x"))
    s[B].bind(warp_i, te.thread_axis("threadIdx.y"))
    s[B].bind(lane_i, te.thread_axis("threadIdx.x"))

    s[C_local].compute_at(s[B], lane_i)

    rx = C_local.op.reduce_axis[0]

    # split the reduce axis
    rxo, rxi = s[C_local].split(rx, factor=8)

    s[W_shared].compute_at(s[C_local], rxo)

    # unroll
    s[C_local].unroll(rxi)

    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)

    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((16384,), "float32"), W: T.Buffer((32,), "float32"), B: T.Buffer((16415,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 513)
    B_local = T.allocate([1], "float32", "local")
    W_shared = T.allocate([8], "float32", "shared")
    threadIdx_y = T.launch_thread("threadIdx.y", 8)
    threadIdx_x = T.launch_thread("threadIdx.x", 4)
    B_local_1 = T.Buffer((1,), data=B_local, scope="local", align=4)
    B_local_1[0] = T.float32(0)
    for r_outer in range(4):
        # W_shared_1 with 8 elements
        W_shared_1 = T.Buffer((8,), data=W_shared, scope="shared", align=32)
        for ax0 in range(8):
            # load to shared memory
            W_shared_1[ax0] = W[r_outer * 8 + ax0]
        if T.likely(blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x < 16415):
            # B_local_1: output position
            # blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x - r_outer * 8: input position
            # 0/1/2 ... /7: weight position
            B_local_1[0] = B_local_1[0] + T.if_then_else(0 <= blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x - r_outer * 8 and blockIdx_x * 8 + threadIdx_y - r_outer * 2 < 4096, A[blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x - r_outer * 8], T.float32(0)) * W_shared_1[0]
        if T.likely(blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x < 16415):
            B_local_1[0] = B_local_1[0] + T.if_then_else(1 <= blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x - r_outer * 8 and blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x - r_outer * 8 < 16385, A[blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x - r_outer * 8 - 1], T.float32(0)) * W_shared_1[1]
        # ... similar code block repeat 6 times
    if T.likely(blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x < 16415):
        B[blockIdx_x * 32 + threadIdx_y * 4 + threadIdx_x] = B_local_1[0]
```

Based on v4, here we introduce the 2d threads hierarchy. Also, we unroll the reduce axis (r from `0` to `7`) to further speed up the calculation. This achieves the best performance at **0.0124 ms**

### 2.2 AutoTVM

We define a search space (splits of threads, whether to split reduction, cache usage, etc.) and let AutoTVM run. It tries different configurations, measures them, and picks the best.

```python
@autotvm.template("conv1d_gpu")
def conv1d_gpu_template_simple(M, N, dtype="float32"):
    ...
```

Best Result: 0.0405 ms

### 2.3 Performance Results

All timings are in milliseconds on a Tesla T4 GPU. Key versions:

| **Implementation**    | **Time (ms)** | **Speedup vs. Naive** | **Speedup vs. Previous** |
|-----------------------|---------------|------------------------|--------------------------|
| **Naive**             | 18.286        | 1.0×                   | -                        |
| **v1** (Refactor)     | 0.107         | 170.9×                 | 170.9×                   |
| **v2** (Threads)      | 0.0251        | 728.5×                 | 4.3×                     |
| **v3** (2D Threads)   | 0.0158        | 1157.3×                | 1.6×                     |
| **v4** (Memory Hier.) | 0.0147        | 1244.0×                | 1.1×                     |
| **v5** (+ Unroll)     | 0.0124        | 1474.7×                | 1.2×                     |
| **AutoTVM**       | 0.0405 ms        | 451.5×                |      -                     |
| **NumPy** (CPU)       | 0.2369        | 77.2×                  | -                        |
| **PyTorch** (GPU)     | 0.1491        | 122.6×                 | -                        |

The final optimized implementation is nearly **1,475× faster** than the naive baseline, showing the enormous performance potential of GPU optimization.

The progression from naive to optimized shows how a combination of algorithmic improvements, thread organization, memory hierarchy exploitation, and low-level optimizations can transform performance on GPU architectures.

## 3. Appendix

- Notebook (all of the code used for this blog): [link](conv1d_gpu.ipynb)
- Summary of the TVM paper: [link](../../TVM)
