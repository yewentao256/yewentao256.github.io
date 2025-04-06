---
title: "TVM: 1D convolution GPU Optimization"
date: 2025-04-03T21:50:56+08:00
categories: ["tvm"]
summary: "这篇博客展示了使用TVM对1D GPU卷积的优化技术，包括线程组织、内存层次结构利用和低级优化。"
---

> 本博客使用`claude-3.7-sonet`翻译，如有冲突请优先参考英文原文

## 概述

这篇博客展示了使用TVM对1D GPU卷积的优化技术，包括线程组织、内存层次结构利用和低级优化。

## 环境

环境：Google Colab T4 GPU

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

我们基于以下配置进行测试：

```py
M = 16384
N = 32
dtype = 'float32'
a_np = np.random.rand(M).astype(dtype)
w_np = np.random.rand(N).astype(dtype)
ref = np.convolve(a_np, w_np)
```

## 2. 实验

### 2.1 基线和手动优化

#### 2.1.1 朴素基线

初始实现创建了一个大小为`(M + N - 1)`的规模较大的归约轴，并在`if_then_else`条件语句内使用边界检查。这个版本运行极其缓慢，耗时**18.29毫秒**。

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

在这个最朴素的版本中，我们仅使用`16415`个块来计算结果。

#### 2.1.2 v1: 计算重构

第一个重要优化是重构计算逻辑，将求和范围从`[0, M+N-1)`改为`[0, N)`，并检查(i-r)在A中是否有效。

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

与CPU相同，我们通过`B[n] = Σ(k=16415) A[k] * W[n-k]`进行计算。现在我们将其改为`B[n] = Σ(k=0→32) A[n-k] * W[k]`。同时，我们也优化了if语句。

这带来了显著的速度提升，执行时间降至**0.107毫秒**。

#### 2.1.3 v2: 线程级并行

在v1的基础上，该版本通过拆分输出轴并同时绑定到块和线程，增加了线程级并行性。性能进一步提升至**0.0251毫秒**。

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

在这里我们引入了**线程**的使用。我们使用`2052`个块，每个块使用`8`个线程。

#### 2.1.4 v3: 二维线程组织

这个优化通过将线程组织成二维网格（每个块`4×4 = 16`个线程）来改进v2。性能提升至**0.0158毫秒**。

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

在这里，我们在每个块中使用`4*4`个线程，而不是v2中的`8`个线程，这更充分地利用了GPU的并行能力。

#### 2.1.5 v4: 内存层次结构 + 归约拆分

这个版本通过以下方式利用GPU内存层次结构：

- 将计算结果缓存到本地内存
- 将核权重缓存到共享内存
- 拆分归约轴

性能提升至**0.0147毫秒**。

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

在这里我们引入了本地内存和共享内存的使用：`B_local = T.allocate([1], "float32", "local")`，`W_shared = T.allocate([4], "float32", "shared")`。

得益于本地/共享内存，我们不需要在每次计算时都重写到全局内存。

#### 2.1.6 v5: 循环展开 + 二维线程优化

最终的优化结合了二维线程组织和内部归约轴的循环展开。

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

在v4的基础上，我们引入了二维线程层次结构。同时，我们展开归约轴（r从`0`到`7`）以进一步加速计算。这实现了最佳性能，达到**0.0124毫秒**。

### 2.2 AutoTVM

我们定义了一个搜索空间（线程拆分、是否拆分归约、缓存使用等），并让AutoTVM运行。它尝试不同的配置，测量它们，并选择最佳配置。

```python
@autotvm.template("conv1d_gpu")
def conv1d_gpu_template_simple(M, N, dtype="float32"):
    ...
```

最佳结果：0.0405毫秒

### 2.3 性能结果

所有时间均为在Tesla T4 GPU上的毫秒计时。主要版本：

| **实现方式**          | **时间 (毫秒)** | **相比朴素基线的加速比** | **相比上一版本的加速比** |
|-----------------------|-----------------|--------------------------|--------------------------|
| **朴素基线**          | 18.286          | 1.0×                     | -                        |
| **v1** (重构)         | 0.107           | 170.9×                   | 170.9×                   |
| **v2** (线程)         | 0.0251          | 728.5×                   | 4.3×                     |
| **v3** (二维线程)     | 0.0158          | 1157.3×                  | 1.6×                     |
| **v4** (内存层次)     | 0.0147          | 1244.0×                  | 1.1×                     |
| **v5** (+ 循环展开)   | 0.0124          | 1474.7×                  | 1.2×                     |
| **AutoTVM**           | 0.0405          | 451.5×                   | -                        |
| **NumPy** (CPU)       | 0.2369          | 77.2×                    | -                        |
| **PyTorch** (GPU)     | 0.1491          | 122.6×                   | -                        |

最终优化的实现比朴素基线快近**1,475倍**，展示了GPU优化的巨大性能潜力。

从朴素到优化的演进过程显示了算法改进、线程组织、内存层次结构利用和低级优化如何在GPU架构上提升性能。

## 3. 附录

- 笔记本（本博客使用的所有代码）：[链接](conv1d_gpu.ipynb)
- TVM论文摘要：[链接](../../zh-cn/TVM)
