---
title: "TVM: 1D convolution CPU Optimization"
date: 2025-03-31T17:31:05+08:00
categories: ["tvm"]
summary: "本文演示如何在 TVM 中加速 1-D 卷积：从缩减计算边界、并行化、向量化到显式展开与自动调优。"
---

> 本博客使用`claude-3.7-sonet`翻译，如有冲突请优先参考英文原文

## 概述

这篇博客介绍了使用 TVM 对 CPU conv1D 逐步进行优化的例子，包含 parallelization, loop tiling, vectorization, unrolling 等技巧。

## 环境

环境配置：

Google colab CPU环境。

```bash
架构:                   x86_64
  CPU运行模式:           32-bit, 64-bit
  地址大小:              46位物理地址, 48位虚拟地址
  字节序:                小端序
CPU数量:                 2
  在线CPU列表:           0,1
厂商ID:                  GenuineIntel
  型号名称:              Intel(R) Xeon(R) CPU @ 2.00GHz
    CPU家族:             6
    型号:                85
    每核心线程数:         2
    每插槽核心数:         1
    插槽数:              1
```

我们基于以下设置进行性能测试：

```py
M = 4096
N = 128
dtype = "float32"
np.random.seed(0)
a_np = np.random.rand(M).astype(dtype)
w_np = np.random.rand(N).astype(dtype)
ref = np.convolve(a_np, w_np)
```

## 1 基准测试和手动优化

### 1.1 朴素基准

这个实现创建了一个大小为(M + N - 1)的规约轴，并在`if_then_else`内使用边界检查。运行时间为`24.3525 ms`。

```py
# 朴素基准
def make_conv1d_cpu_scheduler_naive(M, N):
    A = te.placeholder((M,), name="A")  # 输入张量占位符
    W = te.placeholder((N,), name="W")  # 权重张量占位符

    k = te.reduce_axis((0, M + N - 1), "k")   # k在[0, M+N-1)范围内
    B = te.compute(
        (M + N - 1,),   # 输出形状，n从(0, M + N - 1)
        # if_then_else: 如果满足"任意"条件，返回0，否则返回A[k] * W[n - k]
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )
    s = te.create_schedule(B.op)
    print("=" * 100)
    print(tvm.lower(s, [A, W, B], simple_mode=True))
    print("=" * 100)

    return s, A, W, B
```

IR：

```py
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n in range(4223):
        B[n] = T.float32(0)
        for k in range(4223):
            cse_var_1: T.int32 = n - k  # W的位置
            # 如果 4096 <= k 或 cse_var_1 < 0 或 128 <= cse_var_1: += 0
            # 否则: += A[k] * W[cse_var_1]
            B[n] = B[n] + T.if_then_else(4096 <= k or cse_var_1 < 0 or 128 <= cse_var_1, T.float32(0), A[k] * W[cse_var_1])
```

这是计算一维卷积操作的最简单循环。

### 1.2 v0：减少范围

这里我们将规约轴限制在[0, M)范围内，消除了一些不必要的检查。这略微提高了速度至`23.0471 ms`。

代码：

```py
# 优化v0：缩小k的范围以减少if-else
def make_conv1d_cpu_scheduler_v0(M, N):
    A = te.placeholder((M,), name="A", dtype="float32")
    W = te.placeholder((N,), name="W", dtype="float32")

    k = te.reduce_axis((0, M), "k")   # k在[0, M)范围内
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )

    s = te.create_schedule(B.op)

    return s, A, W, B
```

IR：

```py
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
  T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
  for n in range(4223):
      B[n] = T.float32(0)
      for k in range(4096):
          cse_var_1: T.int32 = n - k
          # 如果 cse_var_1 < 0 或 128 <= cse_var_1: += 0
          # 否则: += A[k] * W[cse_var_1]
          B[n] = B[n] + T.if_then_else(cse_var_1 < 0 or 128 <= cse_var_1, T.float32(0), A[k] * W[cse_var_1])
```

我们可以看到k的内循环缩小为`for k in range(4096)`，并且我们也移除了冗余的`4096 <= k`条件。

### 1.3 v1：并行化

我们对外循环使用`parallel()`。每个输出条目(B[n])可以独立计算，现在的结果是`22.9158 ms`。

代码：

```py
# 优化v1: v0 + 并行
def make_conv1d_cpu_scheduler_v1(M, N):
    s, A, W, B = make_conv1d_cpu_scheduler_v0(M, N, False)
    n_axis = B.op.axis[0]   # 输出轴
    s[B].parallel(n_axis)   # 对输出轴并行

    return s, A, W, B
```

IR：

```py
@T.prim_func
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n in T.parallel(4223):
        B[n] = T.float32(0)
        for k in range(4096):
            cse_var_1: T.int32 = n - k
            B[n] = B[n] + T.if_then_else(cse_var_1 < 0 or 128 <= cse_var_1, T.float32(0), A[k] * W[cse_var_1])
```

值得注意的是，我们使用`T.parallel(4223)`来并行计算。通常这样做可以获得很大的性能提升。

> 注意：在colab上，我们只有一个CPU核心和每核两个线程，所以这种并行效果并不是很好。

### 1.4 v2：拆分 + 向量化

我们进一步拆分输出轴，并对内部部分应用`vectorize`以利用**SIMD**指令。这带来了显著的加速，达到`16.0384 ms`。

代码：

```py
# 优化v2: v0 + 并行 + 拆分 + 向量化
def make_conv1d_cpu_scheduler_v2(M, N, factor=8):
    s, A, W, B = make_conv1d_cpu_scheduler_v0(M, N, False)
    n_axis = B.op.axis[0]
    # AVX2，向量化带宽=256。8 * float32 或 16 * float16
    outer, inner = s[B].split(n_axis, factor=factor)
    s[B].parallel(outer)
    s[B].vectorize(inner)   # 使用CPU SIMD

    return s, A, W, B
```

IR：

```py
@T.prim_func
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n_outer in T.parallel(528):
        for n_inner_s in range(8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                B[n_outer * 8 + n_inner_s] = T.float32(0)
        for k, n_inner_s in T.grid(4096, 8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                cse_var_2: T.int32 = n_outer * 8 + n_inner_s  # 输出位置
                cse_var_1: T.int32 = cse_var_2 - k
                # 如果 cse_var_1 < 0 或 128 <= cse_var_1: += 0
                # 否则: += A[k] * W[cse_var_1]
                B[cse_var_2] = B[cse_var_2] + T.if_then_else(cse_var_1 < 0 or 128 <= cse_var_1, T.float32(0), A[k] * W[cse_var_1])
```

最重要的优化是将循环从`T.parallel(4223)`改为`for n_outer in T.parallel(528)`和`for n_inner_s in range(8)`，现在我们可以更好地：

- 命中缓存
- 减少线程开销
- 使用CPU的SIMD技术（编译器隐式实现）

`T.likely(n_outer * 8 + n_inner_s < 4223)`也向编译器暗示对分支进行更多优化，使其性能更好（编译器分支预测）。

### 1.5 v3：循环展开

我们还拆分了规约轴`k`并展开内部部分以减少循环开销。这帮助我们达到了`14.5967 ms`。

代码：

```py
# 优化v3: v2 + k轴拆分 + 展开
def make_conv1d_cpu_scheduler_v3(M, N, factor=8):
    s, A, W, B = make_conv1d_cpu_scheduler_v2(M, N, factor, False)

    k_axis = B.op.reduce_axis[0]
    k_outer, k_inner = s[B].split(k_axis, factor=factor)
    s[B].unroll(k_inner)  # 展开以减少循环开销

    return s, A, W, B
```

IR：

```py
@T.prim_func
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n_outer in T.parallel(528):
        for n_inner_s in range(8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                B[n_outer * 8 + n_inner_s] = T.float32(0)
        # 将k拆分为512 * 8
        for k_outer in range(512):
            for n_inner_s in range(8):
                if T.likely(n_outer * 8 + n_inner_s < 4223): 
                    cse_var_3: T.int32 = k_outer * 8    # 输入位置
                    cse_var_2: T.int32 = n_outer * 8 + n_inner_s  # 输出位置
                    cse_var_1: T.int32 = cse_var_2 - cse_var_3  # 权重位置
                    # 如果 n_outer - k_outer < 0 或 128 <= cse_var_1: += 0
                    # 否则: += A[cse_var_3] * W[cse_var_1]
                    B[cse_var_2] = B[cse_var_2] + T.if_then_else(n_outer - k_outer < 0 or 128 <= cse_var_1, T.float32(0), A[cse_var_3] * W[cse_var_1])
            for n_inner_s in range(8):
                if T.likely(n_outer * 8 + n_inner_s < 4223):
                    cse_var_6: T.int32 = k_outer * 8
                    cse_var_5: T.int32 = n_outer * 8 + n_inner_s
                    cse_var_4: T.int32 = cse_var_5 - cse_var_6
                    B[cse_var_5] = B[cse_var_5] + T.if_then_else(n_outer - k_outer < 0 or 129 <= cse_var_4, T.float32(0), A[cse_var_6 + 1] * W[cse_var_4 - 1])
            # ... 相同的6个展开块
```

这里我们在k拆分后进行了**循环展开**。`for k_outer in range(512):`然后对`k_inner`进行展开。这使编译器可以进行更好的优化。

### 1.6 v4：计算重构

我们重写卷积以在[0, N)范围内求和，检查(n-k)在A中是否有效。然后我们结合并行化、拆分和向量化。这大大减少了边界开销，实现了我们最快的手动调度（`0.5661 ms`）。

代码：

```py
# 优化v4: 计算重构(最小化if-else) + 并行 + 拆分 + 向量化
def make_conv1d_cpu_scheduler_v4(M, N, factor=8):
    A = te.placeholder((M,), name="A", dtype="float32")
    W = te.placeholder((N,), name="W", dtype="float32")
    k = te.reduce_axis((0, N), name="k")

    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(n - k >= 0, n - k < M),
                A[n - k] * W[k],
                tvm.tir.const(0.0, "float32")
            ),
            axis=k
        ),
        name="B"
    )
    s = te.create_schedule(B.op)
    n_axis = B.op.axis[0]
    outer, inner = s[B].split(n_axis, factor=factor)
    s[B].parallel(outer)
    s[B].vectorize(inner)   # CPU SIMD使用

    return s, A, W, B
```

IR：

```py
@T.prim_func
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n_outer in T.parallel(528):
        for n_inner_s in range(8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                B[n_outer * 8 + n_inner_s] = T.float32(0)
        # k只循环[0, 128]
        for k, n_inner_s in T.grid(128, 8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                cse_var_2: T.int32 = n_outer * 8 + n_inner_s  # 输出位置
                cse_var_1: T.int32 = cse_var_2 - k    # 输入位置
                # 如果 0 <= cse_var_1 且 cse_var_1 < 4096: += A[cse_var_1] * W[k]
                # 否则: += 0
                B[cse_var_2] = B[cse_var_2] + T.if_then_else(0 <= cse_var_1 and cse_var_1 < 4096, A[cse_var_1] * W[k], T.float32(0))
```

最初，我们通过`B[n] = Σ(k=0→4095) A[k] * W[n-k]`计算。现在我们改为`B[n] = Σ(k=0→127) A[n-k] * W[k]`。

if语句也优化为`if 0 <= cse_var_1 and cse_var_1 < 4096`，所有这些结合起来带来了巨大的性能提升。

### 1.7 v5：综合优化

在 v5 中添加两条 stage pragma，强制 TVM 将八通道内层循环展开为直线代码。

```py
def make_conv1d_cpu_scheduler_v5(M, N, factor=8):
    A = te.placeholder((M,), name="A", dtype="float32")
    W = te.placeholder((N,), name="W", dtype="float32")
    k = te.reduce_axis((0, N), name="k")
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(
            tvm.tir.if_then_else(
                tvm.tir.all(n - k >= 0, n - k < M),
                A[n - k] * W[k],
                tvm.tir.const(0.0, "float32")
            ),
            axis=k
        ),
        name="B"
    )

    s = te.create_schedule(B.op)
    n_axis = B.op.axis[0]

    outer, inner = s[B].split(n_axis, factor=factor)
    s[B].parallel(outer)
    s[B].vectorize(inner)
    k_axis = B.op.reduce_axis[0]
    s[B].pragma(outer, "auto_unroll_max_step", 16)
    s[B].pragma(outer, "unroll_explicit", True)

    return s, A, W, B
```

IR:

```py
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n_outer in T.parallel(528):
        cse_var_1: T.int32 = n_outer * 8
        B[cse_var_1] = T.float32(0)
        B[cse_var_1 + 1] = T.float32(0)
        B[cse_var_1 + 2] = T.float32(0)
        B[cse_var_1 + 3] = T.float32(0)
        B[cse_var_1 + 4] = T.float32(0)
        B[cse_var_1 + 5] = T.float32(0)
        B[cse_var_1 + 6] = T.float32(0)
        if T.likely(n_outer < 527):
            B[cse_var_1 + 7] = T.float32(0)
        for k in range(128):
            cse_var_8: T.int32 = cse_var_1 - k
            cse_var_7: T.int32 = cse_var_1 + 6
            cse_var_6: T.int32 = cse_var_1 + 5
            cse_var_5: T.int32 = cse_var_1 + 4
            cse_var_4: T.int32 = cse_var_1 + 3
            cse_var_3: T.int32 = cse_var_1 + 2
            cse_var_2: T.int32 = cse_var_1 + 1
            B[cse_var_1] = B[cse_var_1] + T.if_then_else(0 <= cse_var_8 and cse_var_8 < 4096, A[cse_var_8] * W[k], T.float32(0))
            B[cse_var_2] = B[cse_var_2] + T.if_then_else(-1 <= cse_var_8 and cse_var_8 < 4095, A[cse_var_2 - k] * W[k], T.float32(0))
            B[cse_var_3] = B[cse_var_3] + T.if_then_else(-2 <= cse_var_8 and cse_var_8 < 4094, A[cse_var_3 - k] * W[k], T.float32(0))
            # ... same for 4 5 6 7
            if T.likely(n_outer < 527):
                cse_var_9: T.int32 = cse_var_1 + 7
                B[cse_var_9] = B[cse_var_9] + T.if_then_else(-7 <= cse_var_8 and cse_var_8 < 4089, A[cse_var_9 - k] * W[k], T.float32(0))
```

更大的直线代码块仍可容纳于 L1-I 缓存中，因此省去的循环簿记和分支误预测成本弥补了代码体积的增长——同样的 4096×128 卷积从 `0.566 ms` 下降到 `0.411 ms`。

## 2 AutoTVM

我们定义了一个参数化搜索空间（i和k的拆分、向量化开关、展开因子等），并让AutoTVM运行。它尝试不同的配置，测量它们，并选择最佳配置。

```python
@autotvm.template("conv1d_auto_tune")
def conv1d_auto_tune(M, N):
    ...
    # 使用AutoTVM配置(cfg)定义和应用拆分、向量化、展开
    ...
```

最佳结果：0.70 ms

## 3 性能结果

所有时间都是在x86 CPU（"llvm"目标）上以毫秒为单位。关键版本：

| **实现**                | **时间 (ms)** | **相对朴素的加速比** |
|-------------------------|---------------|------------------------|
| **朴素**               | ~24.35        | 1.0×                  |
| **v0** (较小的k)      | ~23.05        | 1.06×                 |
| **v1** (并行)       | ~22.92        | 1.06×                 |
| **v2** (+向量化)     | ~16.04        | 1.52×                 |
| **v3** (+展开)        | ~14.60        | 1.67×                 |
| **v4** (重构)       | ~0.57         | 43.03×                |
| **v5** (Combined)       | ~0.41         | 59.39×                |
| **AutoTVM** (50次尝试) | ~0.70         | 34.79×                |
| **NumPy**               | ~0.21         | ~116× (vs. 朴素)     |

1. **朴素 vs. v5：** 我们看到通过逐步减少开销并利用并行性和SIMD，时间从24.35 ms大幅下降到0.41 ms。
2. **AutoTVM：** 自动调优收敛到0.70 ms—在这种配置下略慢于我们最好的手动调度，但仍然比朴素实现好得多。
3. **NumPy参考：** NumPy为~0.21 ms，表明高度优化的库（如MKL）仍然可以更快。

## 4 附录

- 笔记本（本博客使用的所有代码）：[链接](https://github.com/yewentao256/TVM_tutorial)
- TVM论文摘要：[链接](../../zh-cn/TVM)
