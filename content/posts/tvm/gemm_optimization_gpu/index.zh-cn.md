---
title: "TVM: GEMM GPU Optimization"
date: 2025-04-06T17:06:56+08:00
categories: ["tvm"]
summary: "本博客展示了使用 TVM 在 GPU 上优化 GEMM（通用矩阵乘法）的技术，包括线程组织和内存层次结构利用"
---

> 本博客使用`claude-3.7-sonet`翻译，如有冲突请优先参考英文原文

## 摘要

本博客展示了使用 TVM 在 GPU 上优化 GEMM（通用矩阵乘法）的技术，包括线程组织和内存层次结构利用。

## 1. 环境

环境：Google Colab T4 GPU

我们基于以下配置进行测试：

```py
M = 1024
N = 512
K = 2048
dtype = 'float32'
a_np = np.random.rand(M, K).astype(dtype)
w_np = np.random.rand(K, N).astype(dtype)
ref = np.matmul(a_np, w_np)
```

### 2.1 基准测试和手动优化

#### 2.1.1 朴素基准

初始实现的运行时间为 **84.52 毫秒**。

```py
def make_gemm_gpu_scheduler_naive(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)

    # 线程的整体索引：𝑖=blockIdx.x×blockDim.x+threadIdx.x
    block_x = te.thread_axis("blockIdx.y")
    block_y = te.thread_axis("blockIdx.x")

    x, y = s[C].op.axis
    (k,) = s[C].op.reduce_axis
    s[C].bind(y, block_y)
    s[C].bind(x, block_x)
    return s, A, B, C
```

IR：

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
        # blockIdx_y * 512 + blockIdx_x: 输出位置
        # blockIdx_y * 2048 + k: A_1 位置
        # k * 512 + blockIdx_x: B_1 位置
        C_1[blockIdx_y * 512 + blockIdx_x] = C_1[blockIdx_y * 512 + blockIdx_x] + A_1[blockIdx_y * 2048 + k] * B_1[k * 512 + blockIdx_x]
```

在这里，我们声明了一个二维块区域，每个块负责计算一个输出。这种方法非常慢。

#### 2.1.2 v1：分块 + 一维线程

现在我们将 x 轴分割成块和瓦片，将外部部分绑定到块，内部部分绑定到线程。

```py
# 优化 v1：分块 + 一维线程
def make_gemm_gpu_scheduler_v1(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)

    x, y = s[C].op.axis

    # 分割轴
    xo, xi = s[C].split(x, factor=32)

    # 将外部轴绑定到块
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(y, te.thread_axis("blockIdx.y"))

    # 将内部轴绑定到线程
    s[C].bind(xi, te.thread_axis("threadIdx.x"))


    return s, A, B, C
```

IR：

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
        # lockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y: 输出
        # blockIdx_x * 65536 + threadIdx_x * 2048 + k: A1
        # k * 512 + blockIdx_y: B1
        C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y] = C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y] + A_1[blockIdx_x * 65536 + threadIdx_x * 2048 + k] * B_1[k * 512 + blockIdx_y]
```

这里我们使用一维线程架构来支持更高效的并行性。这将性能提高到 **36.98 毫秒**。

#### 2.1.3 v2：分块 + 二维线程

在 v1 的基础上，该版本实现二维线程组织，将 x 和 y 轴都进行分割。这样每个块中的线程以 32×32 的网格组织，能更高效地利用 GPU 资源。

```py
# 优化 v2：分块 + 二维线程
def make_gemm_gpu_scheduler_v2(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)

    x, y = s[C].op.axis

    # 分割轴
    xo, xi = s[C].split(x, factor=32)
    yo, yi = s[C].split(y, factor=32)

    # 将外部轴绑定到块
    s[C].bind(xo, te.thread_axis("blockIdx.x"))
    s[C].bind(yo, te.thread_axis("blockIdx.y"))

    # 将内部轴绑定到线程
    s[C].bind(xi, te.thread_axis("threadIdx.x"))
    s[C].bind(yi, te.thread_axis("threadIdx.y"))


    return s, A, B, C

dev = tvm.cuda()
time, res, func, comp = benchmark_gemm_tvm(
    make_gemm_gpu_scheduler_v2, M, K, N, dev, a_np, w_np, num_runs=20, repeat=20
)
np.testing.assert_allclose(res, ref, rtol=1e-4)
print(f"[TVM v2] time: {time*1e3:.4f} ms")
```

IR：

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
        # blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y: 输出
        # blockIdx_x * 65536 + threadIdx_x * 2048 + k: A1
        # k * 512 + blockIdx_y * 32 + threadIdx_y: B1
        C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y] = C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y] + A_1[blockIdx_x * 65536 + threadIdx_x * 2048 + k] * B_1[k * 512 + blockIdx_y * 32 + threadIdx_y]
```

现在我们使用二维线程架构来进一步提高效率。性能略有提升至 **35.50 毫秒**。

#### 2.1.4 v3：共享内存缓存 + 多线程加载

此版本利用 GPU 内存层次结构：

- 将输入矩阵 `A` 和 `B` 缓存在共享内存中
- 将归约轴 `K` 分割成瓦片
- 使用多个线程协作加载数据到共享内存
- 以 16×16 元素的块处理数据

```py
# 优化 v3：v2 + 缓存（多线程）
def make_gemm_gpu_scheduler_v3(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)
    block_x, block_y = 16, 16
    xo, xi = s[C].split(C.op.axis[0], factor=block_x)
    yo, yi = s[C].split(C.op.axis[1], factor=block_y)

    # 分割 k
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

    # 多线程加载数据
    # 这大大提高了性能！
    AAxi, AAyi = s[AA].split(s[AA].op.axis[0], nparts=block_x)
    AAxx, AAxy = s[AA].split(s[AA].op.axis[1], nparts=block_y)
    s[AA].bind(AAxi, te.thread_axis("threadIdx.x"))
    s[AA].bind(AAxx, te.thread_axis("threadIdx.y"))

    BBxi, BByi = s[BB].split(s[BB].op.axis[0], nparts=block_x)
    BBxx, BBxy = s[BB].split(s[BB].op.axis[1], nparts=block_y)
    s[BB].bind(BBxi, te.thread_axis("threadIdx.x"))
    s[BB].bind(BBxx, te.thread_axis("threadIdx.y"))


    return s, A, B, C
```

IR：

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
        # 并行加载数据到 A_shared_1
        with T.launch_thread("threadIdx.x", 16) as threadIdx_x_1:
            threadIdx_y_1 = T.launch_thread("threadIdx.y", 16)
            if T.likely(threadIdx_y_1 < 8):
                A_1 = T.Buffer((2097152,), data=A.data)
                A_shared_1[threadIdx_x_1 * 8 + threadIdx_y_1] = A_1[blockIdx_x * 32768 + threadIdx_x_1 * 2048 + k_outer * 8 + threadIdx_y_1]
        # 并行加载数据到 B_shared_1
        B_shared_1 = T.Buffer((128,), data=B_shared, scope="shared")
        with T.launch_thread("threadIdx.x", 16) as threadIdx_x_1:
            threadIdx_y_1 = T.launch_thread("threadIdx.y", 16)
            if T.likely(threadIdx_x_1 < 8):
                B_1 = T.Buffer((1048576,), data=B.data)
                B_shared_1[threadIdx_x_1 * 16 + threadIdx_y_1] = B_1[k_outer * 4096 + threadIdx_x_1 * 512 + blockIdx_y * 16 + threadIdx_y_1]
        for k_inner in range(8):
            # blockIdx_x * 8192 + threadIdx_x * 512 + blockIdx_y * 16 + threadIdx_y: 输出
            # threadIdx_x * 8 + k_inner: A1
            # k_inner * 16 + threadIdx_y: B1
            C_1[blockIdx_x * 8192 + threadIdx_x * 512 + blockIdx_y * 16 + threadIdx_y] = C_1[blockIdx_x * 8192 + threadIdx_x * 512 + blockIdx_y * 16 + threadIdx_y] + A_shared_1[threadIdx_x * 8 + k_inner] * B_shared_1[k_inner * 16 + threadIdx_y]
```

正如我们在 IR 中看到的，我们使用 `A_shared_1` 和 `B_shared_1` 将瓦片保存在片上内存中，减少访问全局内存的时间消耗。

这大幅提升了性能至 **8.11 毫秒**。

#### 2.1.5 v4：寄存器缓存

> **把 `C` 先写进寄存器，再写回显存**

```python
CL = s.cache_write(C, "local")          # C → register
s[CL].compute_at(s[C], vy)              # 寄存器活到 k-inner 全跑完
ko, ki = s[CL].split(k, factor=4)       # K-tile = 4
```

- **`cache_write`** 会为 `C` 生成一个局部缓冲块，并把它映射到寄存器 (scope = local)
- **`compute_at`** 把这个缓冲嵌进 `threadIdx.y` 循环里，让寄存器在同一行/列线程内复用到 K 轴结束

```py
# opt4: v3 + register caching
def make_gemm_gpu_scheduler_v4(M, K, N, verbose=True):
    k, s, A, B, C = base_declaration(M, K, N)

    block_x, block_y = 32, 32
    tile_k = 4

    CL = s.cache_write(C, "local")

    bx, vx = s[C].split(C.op.axis[0], factor=block_x)
    by, vy = s[C].split(C.op.axis[1], factor=block_y)

    s[C].bind(bx, te.thread_axis("blockIdx.x"))
    s[C].bind(by, te.thread_axis("blockIdx.y"))
    s[C].bind(vx, te.thread_axis("threadIdx.x"))
    s[C].bind(vy, te.thread_axis("threadIdx.y"))

    # schedule CL (local cache for C)
    s[CL].compute_at(s[C], vy)

    # split reduction axis
    ko, ki = s[CL].split(k, factor=tile_k)

    # cache reads for A and B in shared memory
    AA = s.cache_read(A, "shared", [CL])
    BB = s.cache_read(B, "shared", [CL])

    s[AA].compute_at(s[CL], ko)
    s[BB].compute_at(s[CL], ko)

    # cooperative fetching for shared memory
    for load_buffer in [AA, BB]:
        fused = s[load_buffer].fuse(*s[load_buffer].op.axis)
        tz, fused = s[load_buffer].split(fused, nparts=block_y)
        tx, fused = s[load_buffer].split(fused, nparts=block_x)
        s[load_buffer].bind(tz, te.thread_axis("threadIdx.y"))
        s[load_buffer].bind(tx, te.thread_axis("threadIdx.x"))

    return s, A, B, C
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((1024, 2048), "float32"), B: T.Buffer((2048, 512), "float32"), C: T.Buffer((1024, 512), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    blockIdx_x = T.launch_thread("blockIdx.x", 32)
    C_local = T.allocate([1], "float32", "local")
    A_shared = T.allocate([128], "float32", "shared")
    B_shared = T.allocate([128], "float32", "shared")
    threadIdx_x = T.launch_thread("threadIdx.x", 32)
    blockIdx_y = T.launch_thread("blockIdx.y", 16)
    threadIdx_y = T.launch_thread("threadIdx.y", 32)
    C_local_1 = T.Buffer((1,), data=C_local, scope="local", align=4)
    C_local_1[0] = T.float32(0)
    for k_outer in range(512):
        A_shared_1 = T.Buffer((128,), data=A_shared, scope="shared")
        with T.launch_thread("threadIdx.y", 32) as threadIdx_y_1:
            threadIdx_x_1 = T.launch_thread("threadIdx.x", 32)
            if T.likely(threadIdx_x_1 // 4 + threadIdx_y_1 < 32):
                if T.likely(threadIdx_x_1 < 4):
                    A_1 = T.Buffer((2097152,), data=A.data)
                    A_shared_1[threadIdx_y_1 * 4 + threadIdx_x_1] = A_1[blockIdx_x * 65536 + threadIdx_y_1 * 2048 + k_outer * 4 + threadIdx_x_1]
        B_shared_1 = T.Buffer((128,), data=B_shared, scope="shared")
        with T.launch_thread("threadIdx.y", 32) as threadIdx_y_1:
            threadIdx_x_1 = T.launch_thread("threadIdx.x", 32)
            if T.likely(threadIdx_x_1 // 4 + threadIdx_y_1 < 32):
                if T.likely(threadIdx_x_1 < 4):
                    B_1 = T.Buffer((1048576,), data=B.data)
                    B_shared_1[threadIdx_y_1 * 4 + threadIdx_x_1] = B_1[k_outer * 2048 + threadIdx_y_1 // 8 * 512 + blockIdx_y * 32 + threadIdx_y_1 % 8 * 4 + threadIdx_x_1]
        for k_inner in range(4):
            C_local_1[0] = C_local_1[0] + A_shared_1[threadIdx_x * 4 + k_inner] * B_shared_1[k_inner * 32 + threadIdx_y]
    C_1 = T.Buffer((524288,), data=C.data)
    C_1[blockIdx_x * 16384 + threadIdx_x * 512 + blockIdx_y * 32 + threadIdx_y] = C_local_1[0]
```

IR 循环骨架：

```bash
for k_outer:          # 只 load A/B 一次
  for k_inner = 0..3: # 复用 tile 四次
    C_reg += A_sh * B_sh
global_C = C_reg      # 一次性写回
```

优化之后，我们的性能跑到 **≈ 4.56 ms**，比 v3 再快约 1.8 ×。

### 2.2 AutoTVM 优化

AutoTVM 实现探索了一个搜索空间，包括：

- x、y 轴的不同瓦片大小（8、16 或 32）
- 归约轴的不同瓦片大小（8 或 16）
- 是否向量化内存访问
- 缓存写入到本地内存
- 缓存读取到共享内存

在探索了 36 种不同配置后，AutoTVM 调优器找到了一个运行时间为 **42.56 毫秒**的解决方案。

>注意：如果我们在更大的空间中搜索（高成本探索），Colab 总是会崩溃。所以我们这里只搜索一个小空间。

### 2.3 性能结果

以下是在 GPU 上对 `M=1024`、`K=2048`、`N=512` 的矩阵乘法的所有计时（毫秒）：

| **实现**              | **时间（毫秒）** | **相对朴素基准的加速比** | **相对前一版本的加速比** |
|--------------------------|---------------|------------------------|--------------------------|
| **Naive**                | 84.52         | 1.0×                   | -                        |
| **v1** (1D Threads)      | 36.98         | 2.3×                   | 2.3×                     |
| **v2** (2D Threads)      | 35.50         | 2.4×                   | 1.04×                    |
| **v3** (Shared Memory)   | 8.11          | 10.4×                  | 4.4×                     |
| **v4** (Local Memory)     | 4.56          | 18.5×                  | 1.8×                     |
| **AutoTVM**              | 42.56         | 2.0×                   | -                        |
| **NumPy** (CPU)          | 74.95         | 1.1×                   | -                        |
| **PyTorch CPU**          | 18.74         | 4.5×                   | -                        |
| **PyTorch CUDA**         | 0.70          | 120.7×                 | -                        |

手动优化 v4 通过利用 GPU 特定的优化（如共享/寄存器内存、分块和协作线程加载）实现了比朴素基准 **18.5 倍的加速**。

从朴素到优化的进展显示了以下几点的重要性：

- 有效的线程组织
- 适当的内存层次结构利用
- 协作数据加载

这些原则是在 GPU 矩阵乘法实现中实现高性能的基础。

## 3. 附录

- 笔记本（此博客使用的所有代码）：[链接](https://github.com/yewentao256/TVM_tutorial)
- TVM 论文摘要：[链接](../../TVM)
