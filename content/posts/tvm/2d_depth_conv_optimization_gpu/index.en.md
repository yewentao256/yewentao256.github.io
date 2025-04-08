---
title: "TVM: 2D Depth Conv GPU Optimization"
date: 2025-04-07T18:10:56+08:00
categories: ["tvm"]
summary: "This blog demonstrates optimization techniques for 2D depth Convolution in GPU using TVM, including block and thread organization, memory hierarchy exploitation and dimension fuse, etc."
---

## Summary

This blog demonstrates optimization techniques for 2D depth Convolution in GPU using TVM, including block and thread organization, memory hierarchy exploitation and dimension fuse, etc.

## 1. Environment

Env: Google Colab T4 GPU

We test based on this configuration:

```py
B, C, H, W, K = 3, 4, 16, 32, 7
dtype = 'float32'
a_np = np.random.rand(B, C, H, W).astype(dtype)
w_np = np.random.rand(C, 1, K, K).astype(dtype)

ref, pytorch_time = pytorch_depthwise_conv2d(a_np, w_np)
print(f"2DConv PyTorch: {pytorch_time:.4f} ms")
```

### 2.1 Baseline and Manual Optimizations

#### 2.1.1 Naive Baseline

It uses a simple scheduling with only basic block-level parallelism.

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

The most naive realization. This implementation runs at **3.2904 ms**.

#### 2.1.2 v1: 2D Block Architecture

This version improves by assigning each batch-channel combination to a separate block using both `blockIdx.x` and `blockIdx.y`. This creates a 2D grid of blocks that can execute in parallel.

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

The 2D block structure help the performance improves to **0.7687 ms**.

#### 2.1.3 v2: Block Fusion

This optimization **fuses the batch and channel dimensions** (`b` and `c`) and binds them to `blockIdx.x` while binding the height dimension to `blockIdx.y`.

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

Note that `for h, w in T.grid(16, 32)` now turns into `for w in range(32)` but having the same functionality.

This fusion creates larger blocks that process more data in parallel and reduces block-level overhead. Performance dramatically improves to **0.0762 ms**.

#### 2.1.4 v3: 2D Thread Architecture

This version implements thread-level parallelism by:

- Splitting h and w dimensions into inner and outer components
- Binding the inner h components to `threadIdx.y`
- Binding the inner w components to `threadIdx.x`

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

Since we introduce the threads in one block, so the amount of work for each thread decreases to `for w_outer in range(2)`

This creates a grid of 16×16 threads per block, enabling fine-grained parallelism within each block. Performance substantially improves to **0.0101 ms**

#### 2.1.5 v4: Outer Dimension Fusion

This version enhances v3 by:

- Reordering operations to enable fusion of outer height and width dimensions
- Creating a more efficient mapping

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

Notice that in v3, we still have `for w_outer in range(2):`, now it is gone. That's why we further optimize the performance.

This optimization further improves performance to **0.0080 ms**.

### 2.2 Bonus: AutoTVM Optimization

The AutoTVM implementation explores a comprehensive search space including:

- Different tile sizes for height and width dimensions
- Various reordering strategies for nested loops
- Optimizations for the reduction axis (convolution kernel application)
- Options for local cache and shared memory
- Compiler-level unrolling strategies

The AutoTVM configuration space contained 2,880 possible configurations, and after exploring 60 rounds of tuning, the best found solution runs at **0.0035 ms**.

### 2.3 Performance Results

All timings are in milliseconds for a 2D depthwise convolution with parameters `B=3, C=4, H=16, W=32, K=7` on a GPU:

| **Implementation**       | **Time (ms)** | **Speedup vs. Naive** | **Speedup vs. Previous** |
|--------------------------|---------------|------------------------|--------------------------|
| **Naive**                | 3.2904        | 1.0×                   | -                        |
| **v1** (2D Blocks)       | 0.7687        | 2.3×                   | 2.3×                     |
| **v2** (Block Fusion)    | 0.0762        | 43.2×                  | 10.1×                    |
| **v3** (2D Threads)      | 0.0101        | 325.8×                 | 7.5×                     |
| **v4** (Outer Fusion)    | 0.0080        | 411.3×                 | 1.26×                    |
| **AutoTVM**              | 0.0035        | 940.1×                 | 2.29×                    |
| **PyTorch**         | 0.0696        | 47.3×                  | -                        |

The manual optimization v4 achieves a **411.3× speedup** over the naive baseline.

The AutoTVM-optimized version achieves an even more impressive **940.1× speedup** over the naive baseline, demonstrating the power of automated tuning for finding optimal configurations.

Notably, both the manually optimized v3 and v4 implementations, as well as the AutoTVM version, outperform the PyTorch CUDA implementation (which runs at 0.0696 ms).

## 3. Appendix

- Notebook (all of the code used for this blog): [link](conv2d_dw_gpu.ipynb)
- Summary of the TVM paper: [link](../../TVM)
