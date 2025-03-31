---
title: "TVM: 1D convolution CPU Optimization"
date: 2025-03-31T17:31:05+08:00
categories: ["tvm"]
summary: "This blog demonstrates optimization techniques for 1D convolution using TVM, including parallelization, loop tiling, vectorization, and unrolling."
---

## Summary

This blog demonstrates optimization techniques for 1D convolution using TVM, including parallelization, loop tiling, vectorization, and unrolling.

## Environment

Environment Set Up:

Google colab CPU env.

```bash
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   2
  On-line CPU(s) list:    0,1
Vendor ID:                GenuineIntel
  Model name:             Intel(R) Xeon(R) CPU @ 2.00GHz
    CPU family:           6
    Model:                85
    Thread(s) per core:   2
    Core(s) per socket:   1
    Socket(s):            1
```

We test performance based on this setting:

```py
M = 4096
N = 128
dtype = "float32"
np.random.seed(0)
a_np = np.random.rand(M).astype(dtype)
w_np = np.random.rand(N).astype(dtype)
ref = np.convolve(a_np, w_np)
```

## 1 Baseline and Manual Optimizations

### 1.1 Naive Basline

This creates a large reduce axis of size \((M + N - 1)\) and uses boundary checks inside an `if_then_else`. This runs slowly with a time `24.3525 ms`

```py
# naive baseline
def make_conv1d_cpu_scheduler_naive(M, N):
    A = te.placeholder((M,), name="A")  # input tensor placeholder
    W = te.placeholder((N,), name="W")  # weight tensor placeholder

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
    print("=" * 100)
    print(tvm.lower(s, [A, W, B], simple_mode=True))
    print("=" * 100)

    return s, A, W, B
```

IR:

```py
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n in range(4223):
        B[n] = T.float32(0)
        for k in range(4223):
            cse_var_1: T.int32 = n - k  # the location of W
            # if 4096 <= k or cse_var_1 < 0 or 128 <= cse_var_1: += 0
            # else: += A[k] * W[cse_var_1]
            B[n] = B[n] + T.if_then_else(4096 <= k or cse_var_1 < 0 or 128 <= cse_var_1, T.float32(0), A[k] * W[cse_var_1])
```

The simplest for loop to calculate the conv1d operation.

### 1.2 v0: Reduced Range

Here we limit the reduce axis to \([0, M)\), eliminating some unnecessary checks. This modestly improves speed to `23.0471 ms`

Code:

```py
# optimize v0: shrink the range of k to reduce if else
def make_conv1d_cpu_scheduler_v0(M, N, verbose=True):
    A = te.placeholder((M,), name="A", dtype="float32")
    W = te.placeholder((N,), name="W", dtype="float32")

    k = te.reduce_axis((0, M), "k")   # k in [0, M)
    B = te.compute(
        (M + N - 1,),
        lambda n: te.sum(tvm.tir.if_then_else(
            tvm.tir.any(k < 0, k >= M, n - k < 0, n - k >= N),
            tvm.tir.const(0.0, "float32"),
            A[k] * W[n - k]), axis=k),
        name="B",
    )

    s = te.create_schedule(B.op)
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)
    return s, A, W, B
```

IR:

```py
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
  T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
  for n in range(4223):
      B[n] = T.float32(0)
      for k in range(4096):
          cse_var_1: T.int32 = n - k
          # if cse_var_1 < 0 or 128 <= cse_var_1: += 0
          # else: += A[k] * W[cse_var_1]
          B[n] = B[n] + T.if_then_else(cse_var_1 < 0 or 128 <= cse_var_1, T.float32(0), A[k] * W[cse_var_1])
```

We can see that the inner loop of k shrink to `for k in range(4096)`, and we remove the redundant `4096 <= k` as well

### 1.3 v1: Parallelization

We use `parallel()` on the outer axis. Each output entry \(B[n]\) can be computed independently, the results now is `22.9158 ms`

Code:

```py
# optimize v1: v0 + parallel
def make_conv1d_cpu_scheduler_v1(M, N, verbose=True):
    s, A, W, B = make_conv1d_cpu_scheduler_v0(M, N, False)
    n_axis = B.op.axis[0]   # output axis
    s[B].parallel(n_axis)   # parallel for output axis
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)
    return s, A, W, B
```

IR:

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

Notably, we use `T.parallel(4223)` to parallel the calculation. Usually we can get a lot of performance improvement by doing so.

> Note: on colab, we just have one CPU core and 2 threads per core, so this parallelism performs not really good.

### 1.4 v2: Split + Vectorize

We further split the output axis and apply `vectorize` on the inner part to leverage **SIMD** instructions. This yields a significant speedup to `16.0384 ms`

Code:

```py
# optimize v2: v0 + parallel + split + vectorize
def make_conv1d_cpu_scheduler_v2(M, N, factor=8, verbose=True):
    s, A, W, B = make_conv1d_cpu_scheduler_v0(M, N, False)
    n_axis = B.op.axis[0]
    # AVX2, bw=256 for vectorization. 8 * float32 or 16 * float16
    outer, inner = s[B].split(n_axis, factor=factor)
    s[B].parallel(outer)
    s[B].vectorize(inner)   # CPU SIMD usage
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)
    return s, A, W, B
```

IR:

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
                cse_var_2: T.int32 = n_outer * 8 + n_inner_s  # location of output
                cse_var_1: T.int32 = cse_var_2 - k
                # if cse_var_1 < 0 or 128 <= cse_var_1: += 0
                # else: += A[k] * W[cse_var_1]
                B[cse_var_2] = B[cse_var_2] + T.if_then_else(cse_var_1 < 0 or 128 <= cse_var_1, T.float32(0), A[k] * W[cse_var_1])
```

The most important optimization is make the loop from `T.parallel(4223)` to `for n_outer in T.parallel(528)` and `for n_inner_s in range(8)`, now we can better:

- Hit the cache
- Reduce the overhead of threads
- Use the SIMD tech of CPU (implicitly for Compiler)

`T.likely(n_outer * 8 + n_inner_s < 4223)` also suggests the compiler to do more optimization to the branch so that it can perform better. (Compiler prediction)

### 1.5 v3: Unroll

We also split the reduce axis `k` and unroll the inner part to reduce loop overhead. This help us reach the `14.5967 ms`

Code:

```py
# optimize v3: v2 + k_axis split + unroll
def make_conv1d_cpu_scheduler_v3(M, N, factor=8, verbose=True):
    s, A, W, B = make_conv1d_cpu_scheduler_v2(M, N, factor, False)

    k_axis = B.op.reduce_axis[0]
    k_outer, k_inner = s[B].split(k_axis, factor=factor)
    s[B].unroll(k_inner)  # unroll to reduce loop overhead
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)
    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n_outer in T.parallel(528):
        for n_inner_s in range(8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                B[n_outer * 8 + n_inner_s] = T.float32(0)
        # split k to 512 * 8
        for k_outer in range(512):
            for n_inner_s in range(8):
                if T.likely(n_outer * 8 + n_inner_s < 4223): 
                    cse_var_3: T.int32 = k_outer * 8    # the input position
                    cse_var_2: T.int32 = n_outer * 8 + n_inner_s  # output position
                    cse_var_1: T.int32 = cse_var_2 - cse_var_3  # weight position
                    # if n_outer - k_outer < 0 or 128 <= cse_var_1: += 0
                    # else: += A[cse_var_3] * W[cse_var_1]
                    B[cse_var_2] = B[cse_var_2] + T.if_then_else(n_outer - k_outer < 0 or 128 <= cse_var_1, T.float32(0), A[cse_var_3] * W[cse_var_1])
            for n_inner_s in range(8):
                if T.likely(n_outer * 8 + n_inner_s < 4223):
                    cse_var_6: T.int32 = k_outer * 8
                    cse_var_5: T.int32 = n_outer * 8 + n_inner_s
                    cse_var_4: T.int32 = cse_var_5 - cse_var_6
                    B[cse_var_5] = B[cse_var_5] + T.if_then_else(n_outer - k_outer < 0 or 129 <= cse_var_4, T.float32(0), A[cse_var_6 + 1] * W[cse_var_4 - 1])
            # ... same with 6 unloop block
```

Here we do the **Loop Unrolling** after the split of k. `for k_outer in range(512):` and then unloop for the `k_inner`. This allows the compiler to do better optimization.

### 1.6 v4: Compute Refactor

We rewrite the convolution to sum over \([0, N)\), checking if \((n-k)\) is valid in \(A\). Then we combine parallelization, splitting, and vectorization. This drastically reduces boundary overhead and achieves our fastest manual schedule. (`0.5661 ms`)

Code:

```py
# optimize v4: compute refactor(minimize if-else) + parallel + split + vectorize
def make_conv1d_cpu_scheduler_v4(M, N, factor=8, verbose=True):
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
    s[B].vectorize(inner)   # CPU SIMD usage
    if verbose:
        print("=" * 100)
        print(tvm.lower(s, [A, W, B], simple_mode=True))
        print("=" * 100)
    return s, A, W, B
```

IR:

```py
@T.prim_func
def main(A: T.Buffer((4096,), "float32"), W: T.Buffer((128,), "float32"), B: T.Buffer((4223,), "float32")):
    T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
    for n_outer in T.parallel(528):
        for n_inner_s in range(8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                B[n_outer * 8 + n_inner_s] = T.float32(0)
        # k only loop for [0, 128]
        for k, n_inner_s in T.grid(128, 8):
            if T.likely(n_outer * 8 + n_inner_s < 4223):
                cse_var_2: T.int32 = n_outer * 8 + n_inner_s  # output position
                cse_var_1: T.int32 = cse_var_2 - k    # input position
                # if 0 <= cse_var_1 and cse_var_1 < 4096: += A[cse_var_1] * W[k]
                # else: += 0
                B[cse_var_2] = B[cse_var_2] + T.if_then_else(0 <= cse_var_1 and cse_var_1 < 4096, A[cse_var_1] * W[k], T.float32(0))
```

Originally, we calculate by `B[n] = Σ(k=0→4095) A[k] * W[n-k]`. Now we change it to `B[n] = Σ(k=0→127) A[n-k] * W[k]`.

The if statement is also optimized to `if 0 <= cse_var_1 and cse_var_1 < 4096`, all of this combined make a huge performance improvement

## 2 AutoTVM

We define a parameterized search space (splits of \(i\) and \(k\), vectorization toggles, unroll factors, etc.) and let AutoTVM run. It tries different configurations, measures them, and picks the best.

```python
@autotvm.template("conv1d_auto_tune")
def conv1d_auto_tune(M, N):
    ...
    # define and apply splits, vectorize, unroll using AutoTVM config (cfg)
    ...
```

Best Result: 0.70 ms

## 3 Performance Results

All timings are in milliseconds on an x86 CPU (“llvm” target). Key versions:

| **Implementation**      | **Time (ms)** | **Speedup vs. Naive** |
|-------------------------|---------------|------------------------|
| **Naive**               | ~24.35        | 1.0×                  |
| **v0** (Smaller k)      | ~23.05        | 1.06×                 |
| **v1** (Parallel)       | ~22.92        | 1.06×                 |
| **v2** (+Vectorize)     | ~16.04        | 1.52×                 |
| **v3** (+Unroll)        | ~14.60        | 1.67×                 |
| **v4** (Refactor)       | ~0.57         | 43.03×                |
| **AutoTVM** (50 trials) | ~0.70         | 34.79×                |
| **NumPy**               | ~0.21         | ~116× (vs. Naive)     |

1. **Naive vs. v4:** We see a dramatic drop from ~24.35 ms to ~0.57 ms by progressively removing overhead and leveraging parallelism and SIMD.
2. **AutoTVM:** Automatic tuning converges to ~0.70 ms—slightly slower than our best manual schedule in this configuration, but still vastly better than naive.
3. **NumPy Reference:** NumPy is ~0.21 ms, indicating that heavily optimized libraries (e.g., MKL) can still be faster.

## 4 Appendix

- Notebook (all of the code used for this blog): [link](conv1d_cpu.ipynb)
- Summary of the TVM paper: [link](../../TVM)
