---
title: "TVM: 2D Depth Conv GPU Optimization"
date: 2025-04-07T18:10:56+08:00
categories: ["tvm"]
summary: "这篇博客展示了使用TVM在GPU上进行2D深度卷积的优化技术，包括块和线程组织、内存层次结构利用和维度融合等。"
---

> 本博客使用`claude-3.7-sonet`翻译，如有冲突请优先参考英文原文

## 摘要

这篇博客展示了使用TVM在GPU上进行2D深度卷积的优化技术，包括块和线程组织、内存层次结构利用和维度融合等。

## 1. 环境

环境：Google Colab T4 GPU

我们基于以下配置进行测试：

```py
B, C, H, W, K = 3, 4, 16, 32, 7
dtype = 'float32'
a_np = np.random.rand(B, C, H, W).astype(dtype)
w_np = np.random.rand(C, 1, K, K).astype(dtype)

ref, pytorch_time = pytorch_depthwise_conv2d(a_np, w_np)
print(f"2DConv PyTorch: {pytorch_time:.4f} ms")
```

### 2.1 基线和手动优化

#### 2.1.1 朴素基线

它使用简单的调度，只有基本的块级并行性。

```py
def base_declaration(B, C, H, W, K):
    assert K % 2 == 1
    inp = te.placeholder((B, C, H, W), name="A")
    ker = te.placeholder((C, 1, K, K), name="W")

    ry = te.reduce_axis((0, K), name='ry')
    rx = te.reduce_axis((0, K), name='rx')
    pad_h = (K - 1) // 2
    pad_w = (K - 1) // 2

    padded = te.compute(
        (B, C, H + 2*pad_h, W + 2*pad_w),
        lambda b, c, h, w: tvm.tir.if_then_else(
            tvm.tir.all(h >= pad_h, h < H + pad_h, w >= pad_w, w < W + pad_w),
            inp[b, c, h - pad_h, w - pad_w],
            tvm.tir.const(0.0, "float32")
        ),
        name="padded"
    )

    out = te.compute(
        (B, C, H, W),
        lambda b, c, h, w: te.sum(
            padded[b, c, h + ry, w + rx] * ker[c, 0, ry, rx],
            axis=[ry, rx]
        ),
        name="depthwise_conv"
    )

    s = te.create_schedule(out.op)
    return s, inp, ker, out, padded

def make_dwsp_conv2d_gpu_scheduler_naive(B, C, H, W, K, verbose=True):
    s, inp, ker, out, padded = base_declaration(B, C, H, W, K)
    block_x = te.thread_axis("blockIdx.x")
    b, c, h, w = s[out].op.axis
    s[out].bind(b, block_x)
    # compute inline: only compute padding when calculating the out
    s[padded].compute_inline()

    return s, inp, ker, out
```

IR:

```py
def main(A: T.Buffer((3, 4, 16, 32), "float32"), W: T.Buffer((4, 1, 7, 7), "float32"), depthwise_conv: T.Buffer((3, 4, 16, 32), "float32")):
    blockIdx_x = T.launch_thread("blockIdx.x", 3)
    # 3d parallel for c in [0,4), h in [0, 16) and w in [0, 32)
    for c, h, w in T.grid(4, 16, 32):
        # depthwise_conv_1: output
        depthwise_conv_1 = T.Buffer((6144,), data=depthwise_conv.data)
        depthwise_conv_1[blockIdx_x * 2048 + c * 512 + h * 32 + w] = T.float32(0)
        for ry, rx in T.grid(7, 7):
            # for each position in kernel
            cse_var_2: T.int32 = h + ry  # vertical position of input
            cse_var_1: T.int32 = w + rx  # horizontal position of input
            A_1 = T.Buffer((6144,), data=A.data)
            W_1 = T.Buffer((196,), data=W.data)
            # blockIdx_x * 2048 + c * 512 + h * 32 + w: output position
            # if_then_else: realize a "padding"
            # 99: 3 * 32 + 3, padding shift, to visit the correct memory position
            # c * 49 + ry * 7 + rx: kernel position
            # blockIdx_x * 2048 + c * 512 + h * 32 + ry * 32 + w + rx - 99: input position
            depthwise_conv_1[blockIdx_x * 2048 + c * 512 + h * 32 + w] = depthwise_conv_1[blockIdx_x * 2048 + c * 512 + h * 32 + w] + T.if_then_else(3 <= cse_var_2 and cse_var_2 < 19 and 3 <= cse_var_1 and cse_var_1 < 35, A_1[blockIdx_x * 2048 + c * 512 + h * 32 + ry * 32 + w + rx - 99], T.float32(0)) * W_1[c * 49 + ry * 7 + rx]
```

这是最朴素的实现。该实现的运行时间为**3.2904毫秒**。

#### 2.1.2 v1: 2D块架构

此版本通过使用`blockIdx.x`和`blockIdx.y`将每个批次-通道组合分配给单独的块来改进。这创建了一个可以并行执行的2D块网格。

```py
# opt v1: 2d block architecture
def make_dwsp_conv2d_gpu_scheduler_v1(B, C, H, W, K, verbose=True):
    s, inp, ker, out, padded = base_declaration(B, C, H, W, K)
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")

    b, c, h, w = s[out].op.axis
    s[out].bind(b, block_x)
    s[out].bind(c, block_y)
    s[padded].compute_inline()

    return s, inp, ker, out
```

IR:

```py
def main(A: T.Buffer((3, 4, 16, 32), "float32"), W: T.Buffer((4, 1, 7, 7), "float32"), depthwise_conv: T.Buffer((3, 4, 16, 32), "float32")):
    blockIdx_x = T.launch_thread("blockIdx.x", 3)
    blockIdx_y = T.launch_thread("blockIdx.y", 4)
    for h, w in T.grid(16, 32):
        depthwise_conv_1 = T.Buffer((6144,), data=depthwise_conv.data)
        depthwise_conv_1[blockIdx_x * 2048 + blockIdx_y * 512 + h * 32 + w] = T.float32(0)
        for ry, rx in T.grid(7, 7):
            cse_var_2: T.int32 = h + ry
            cse_var_1: T.int32 = w + rx
            A_1 = T.Buffer((6144,), data=A.data)
            W_1 = T.Buffer((196,), data=W.data)
            # blockIdx_x * 2048 + blockIdx_y * 512 + h * 32 + w: output
            # blockIdx_x * 2048 + blockIdx_y * 512 + h * 32 + ry * 32 + w + rx - 99: input
            # blockIdx_y * 49 + ry * 7 + rx: kernel
            depthwise_conv_1[blockIdx_x * 2048 + blockIdx_y * 512 + h * 32 + w] = depthwise_conv_1[blockIdx_x * 2048 + blockIdx_y * 512 + h * 32 + w] + T.if_then_else(3 <= cse_var_2 and cse_var_2 < 19 and 3 <= cse_var_1 and cse_var_1 < 35, A_1[blockIdx_x * 2048 + blockIdx_y * 512 + h * 32 + ry * 32 + w + rx - 99], T.float32(0)) * W_1[blockIdx_y * 49 + ry * 7 + rx]
```

2D块结构帮助性能提升到**0.7687毫秒**。

#### 2.1.3 v2: 块融合

这种优化**融合了批次和通道维度**（`b`和`c`）并将它们绑定到`blockIdx.x`，同时将高度维度绑定到`blockIdx.y`。

```py
# opt v2: block fuse
def make_dwsp_conv2d_gpu_scheduler_v2(B, C, H, W, K, verbose=True):
    s, inp, ker, out, padded = base_declaration(B, C, H, W, K)
    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")

    b, c, h, w = s[out].op.axis
    bc = s[out].fuse(b, c)
    s[out].bind(bc, block_x)
    s[out].bind(h, block_y)
    s[padded].compute_inline()
    return s, inp, ker, out
```

IR:

```py
def main(A: T.Buffer((3, 4, 16, 32), "float32"), W: T.Buffer((4, 1, 7, 7), "float32"), depthwise_conv: T.Buffer((3, 4, 16, 32), "float32")):
    blockIdx_x = T.launch_thread("blockIdx.x", 12)
    blockIdx_y = T.launch_thread("blockIdx.y", 16)
    for w in range(32):
        depthwise_conv_1 = T.Buffer((6144,), data=depthwise_conv.data)
        depthwise_conv_1[blockIdx_x * 512 + blockIdx_y * 32 + w] = T.float32(0)
        for ry, rx in T.grid(7, 7):
            cse_var_1: T.int32 = w + rx
            A_1 = T.Buffer((6144,), data=A.data)
            W_1 = T.Buffer((196,), data=W.data)
            # blockIdx_x * 512 + blockIdx_y * 32 + w: output position
            # blockIdx_x * 512 + blockIdx_y * 32 + ry * 32 + w + rx - 99: input
            # blockIdx_x % 4 * 49 + ry * 7 + rx: kernel
            depthwise_conv_1[blockIdx_x * 512 + blockIdx_y * 32 + w] = depthwise_conv_1[blockIdx_x * 512 + blockIdx_y * 32 + w] + T.if_then_else(3 <= blockIdx_y + ry and blockIdx_y + ry < 19 and 3 <= cse_var_1 and cse_var_1 < 35, A_1[blockIdx_x * 512 + blockIdx_y * 32 + ry * 32 + w + rx - 99], T.float32(0)) * W_1[blockIdx_x % 4 * 49 + ry * 7 + rx]
```

请注意，`for h, w in T.grid(16, 32)`现在变成了`for w in range(32)`，但功能相同。

这种融合创建了更大的块，可以并行处理更多数据，并减少了块级开销。性能显著提升到**0.0762毫秒**。

#### 2.1.4 v3: 2D线程架构

这个版本通过以下方式实现线程级并行：

- 将h和w维度分割为内部和外部组件
- 将内部h组件绑定到`threadIdx.y`
- 将内部w组件绑定到`threadIdx.x`

```py
# opt v3: v2 + 2d threads
def make_dwsp_conv2d_gpu_scheduler_v3(B, C, H, W, K, verbose=True):
    s, inp, ker, out, padded = base_declaration(B, C, H, W, K)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    b, c, h, w = s[out].op.axis
    bc = s[out].fuse(b, c)
    h_outer, h_inner = s[out].split(h, factor=16)
    w_outer, w_inner = s[out].split(w, factor=16)

    s[out].bind(bc, block_x)
    s[out].bind(h_outer, block_y)
    s[out].bind(h_inner, thread_y)
    s[out].bind(w_inner, thread_x)

    s[padded].compute_inline()
    return s, inp, ker, out
```

IR:

```py
def main(A: T.Buffer((3, 4, 16, 32), "float32"), W: T.Buffer((4, 1, 7, 7), "float32"), depthwise_conv: T.Buffer((3, 4, 16, 32), "float32")):
    blockIdx_x = T.launch_thread("blockIdx.x", 12)
    blockIdx_y = T.launch_thread("blockIdx.y", 1)
    threadIdx_y = T.launch_thread("threadIdx.y", 16)
    for w_outer in range(2):
        threadIdx_x = T.launch_thread("threadIdx.x", 16)
        depthwise_conv_1 = T.Buffer((6144,), data=depthwise_conv.data)
        depthwise_conv_1[blockIdx_x * 512 + threadIdx_y * 32 + w_outer * 16 + threadIdx_x] = T.float32(0)
        for ry, rx in T.grid(7, 7):
            cse_var_1: T.int32 = w_outer * 16
            A_1 = T.Buffer((6144,), data=A.data)
            W_1 = T.Buffer((196,), data=W.data)
            # blockIdx_x * 512 + threadIdx_y * 32 + cse_var_1 + threadIdx_x: output
            # blockIdx_x * 512 + threadIdx_y * 32 + ry * 32 + cse_var_1 + threadIdx_x + rx - 99: input
            # blockIdx_x % 4 * 49 + ry * 7 + rx: kernel
            depthwise_conv_1[blockIdx_x * 512 + threadIdx_y * 32 + cse_var_1 + threadIdx_x] = depthwise_conv_1[blockIdx_x * 512 + threadIdx_y * 32 + cse_var_1 + threadIdx_x] + T.if_then_else(3 <= threadIdx_y + ry and threadIdx_y + ry < 19 and 3 <= cse_var_1 + threadIdx_x + rx and cse_var_1 + threadIdx_x + rx < 35, A_1[blockIdx_x * 512 + threadIdx_y * 32 + ry * 32 + cse_var_1 + threadIdx_x + rx - 99], T.float32(0)) * W_1[blockIdx_x % 4 * 49 + ry * 7 + rx]
```

由于我们在一个块中引入了线程，因此每个线程的工作量减少为`for w_outer in range(2)`

这创建了每个块16×16线程的网格，实现了块内的细粒度并行。性能大幅提升到**0.0101毫秒**

#### 2.1.5 v4: 外部维度融合

此版本通过以下方式增强v3：

- 重新排序操作以实现外部高度和宽度维度的融合
- 创建更高效的映射

```py
# opt v4: v3 + fuse at hw outer
def make_dwsp_conv2d_gpu_scheduler_v4(B, C, H, W, K, verbose=True):
    s, inp, ker, out, padded = base_declaration(B, C, H, W, K)
    b, c, h, w = s[out].op.axis
    bc = s[out].fuse(b, c)

    h_outer, h_inner = s[out].split(h, factor=16)
    w_outer, w_inner = s[out].split(w, factor=16)
    # we must reorder to do the fuse
    s[out].reorder(bc, h_outer, w_outer, h_inner, w_inner)
    hw_outer = s[out].fuse(h_outer, w_outer)

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")

    s[out].bind(bc, block_x)
    s[out].bind(hw_outer, block_y)
    s[out].bind(h_inner, thread_y)
    s[out].bind(w_inner, thread_x)

    s[padded].compute_inline()
    return s, inp, ker, out
```

IR:

```py
def main(A: T.Buffer((3, 4, 16, 32), "float32"), W: T.Buffer((4, 1, 7, 7), "float32"), depthwise_conv: T.Buffer((3, 4, 16, 32), "float32")):
    blockIdx_x = T.launch_thread("blockIdx.x", 12)
    blockIdx_y = T.launch_thread("blockIdx.y", 2)
    threadIdx_y = T.launch_thread("threadIdx.y", 16)
    threadIdx_x = T.launch_thread("threadIdx.x", 16)
    depthwise_conv_1 = T.Buffer((6144,), data=depthwise_conv.data)
    depthwise_conv_1[blockIdx_x * 512 + threadIdx_y * 32 + blockIdx_y * 16 + threadIdx_x] = T.float32(0)
    for ry, rx in T.grid(7, 7):
        A_1 = T.Buffer((6144,), data=A.data)
        W_1 = T.Buffer((196,), data=W.data)
        # blockIdx_x * 512 + threadIdx_y * 32 + blockIdx_y * 16 + threadIdx_x: output
        # blockIdx_x * 512 + threadIdx_y * 32 + ry * 32 + blockIdx_y * 16 + threadIdx_x + rx - 99: input
        # blockIdx_x % 4 * 49 + ry * 7 + rx: kernel
        depthwise_conv_1[blockIdx_x * 512 + threadIdx_y * 32 + blockIdx_y * 16 + threadIdx_x] = depthwise_conv_1[blockIdx_x * 512 + threadIdx_y * 32 + blockIdx_y * 16 + threadIdx_x] + T.if_then_else(3 <= threadIdx_y + ry and threadIdx_y + ry < 19 and 3 <= blockIdx_y * 16 + threadIdx_x + rx and blockIdx_y * 16 + threadIdx_x + rx < 35, A_1[blockIdx_x * 512 + threadIdx_y * 32 + ry * 32 + blockIdx_y * 16 + threadIdx_x + rx - 99], T.float32(0)) * W_1[blockIdx_x % 4 * 49 + ry * 7 + rx]
```

请注意，在v3中，我们仍然有`for w_outer in range(2):`，现在它消失了。这就是为什么我们进一步优化了性能。

这种优化进一步将性能提升到**0.0080毫秒**。

### 2.2 额外内容：AutoTVM优化

AutoTVM实现探索了一个全面的搜索空间，包括：

- 高度和宽度维度的不同瓦片大小
- 嵌套循环的各种重新排序策略
- 减少轴（卷积核应用）的优化
- 本地缓存和共享内存的选项
- 编译器级展开策略

AutoTVM配置空间包含2,880种可能的配置，经过60轮调优探索后，找到的最佳解决方案运行时间为**0.0035毫秒**。

### 2.3 性能结果

所有时间均为GPU上参数为`B=3, C=4, H=16, W=32, K=7`的2D深度卷积的毫秒时间：

| **实现**                | **时间 (ms)** | **相对朴素基线的加速比** | **相对前一版本的加速比** |
|------------------------|---------------|------------------------|--------------------------|
| **朴素基线**            | 3.2904        | 1.0×                   | -                        |
| **v1** (2D块)          | 0.7687        | 2.3×                   | 2.3×                     |
| **v2** (块融合)        | 0.0762        | 43.2×                  | 10.1×                    |
| **v3** (2D线程)        | 0.0101        | 325.8×                 | 7.5×                     |
| **v4** (外部融合)      | 0.0080        | 411.3×                 | 1.26×                    |
| **AutoTVM**           | 0.0035        | 940.1×                 | 2.29×                    |
| **PyTorch**           | 0.0696        | 47.3×                  | -                        |

手动优化的v4相比朴素基线实现了**411.3倍的加速**。

AutoTVM优化版本相比朴素基线实现了更加令人印象深刻的**940.1倍加速**，展示了自动调优在寻找最佳配置方面的强大能力。

值得注意的是，手动优化的v3和v4实现，以及AutoTVM版本，都优于PyTorch CUDA实现（运行时间为0.0696毫秒）。

## 3. 附录

- 笔记本（本博客使用的所有代码）：[链接](conv2d_dw_gpu.ipynb)
- TVM论文摘要：[链接](../../TVM)