---
title: "GPU Puzzles"
date: 2024-12-02T10:52:12+08:00
categories: ["Pytorch"]
summary: "This article provides solutions to the [GPU Puzzles](https://github.com/srush/GPU-Puzzles) by [Sasha Rush](https://github.com/srush)"
---

## Summary

This article provides solutions to the [GPU Puzzles](https://github.com/srush/GPU-Puzzles) by [Sasha Rush](https://github.com/srush).

For some challenging puzzles, we include explanations as comments within the code.

## Puzzle 1 - Map

```py
def map_spec(a):
    return a + 10


def map_test(cuda):
    def call(out, a) -> None:
        local_i = cuda.threadIdx.x
        out[local_i] = a[local_i] + 10

    return call
```

## Puzzle 2 - Zip

```py
def zip_spec(a, b):
    return a + b

def zip_test(cuda):
    def call(out, a, b) -> None:
        local_i = cuda.threadIdx.x
        out[local_i] = a[local_i] + b[local_i]

    return call
```

## Puzzle 3 - Guards

```py
def map_guard_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        if local_i < size:
            out[local_i] = a[local_i] + 10

    return call
```

## Puzzle 4 - Map 2D

```py
def map_2D_test(cuda):
    def call(out, a, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        if local_i < size and local_j < size:
            out[local_i, local_j] = a[local_i, local_j] + 10

    return call
```

## Puzzle 5 - Broadcast

```py
def broadcast_test(cuda):
    def call(out, a, b, size) -> None:
        local_i = cuda.threadIdx.x
        local_j = cuda.threadIdx.y
        if local_i < size and local_j < size:
            out[local_i, local_j] = a[local_i, 0] + b[0, local_j]

    return call
```

## Puzzle 6 - Blocks

```py
def map_block_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        if i < size:
            out[i] = a[i] + 10

    return call

```

## Puzzle 7 - Blocks 2D

```py
def map_block2D_test(cuda):
    def call(out, a, size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
        if i < size and j < size:
            out[i, j] = a[i, j] + 10

    return call
```

## Puzzle 8 - Shared

```py
def shared_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        if i < size:
            shared[local_i] = a[i]
            cuda.syncthreads()

        out[i] = shared[local_i] + 10

    return call
```

## Puzzle 9 - Pooling

```py
def pool_spec(a):
    out = np.zeros(*a.shape)
    for i in range(a.shape[0]):
        out[i] = a[max(i - 2, 0) : i + 1].sum()
    return out


TPB = 8
def pool_test(cuda):
    def call(out, a, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        if i < size:
            shared[local_i] = a[i]
            cuda.syncthreads()
            total = shared[local_i]
            if local_i >= 1:
                total += shared[local_i - 1]
            if local_i >= 2:
                total += shared[local_i - 2]
            out[i] = total

    return call
```

## Puzzle 10 - Dot Product

```py
def dot_spec(a, b):
    return a @ b

TPB = 8
def dot_test(cuda):
    def call(out, a, b, size) -> None:
        shared = cuda.shared.array(TPB, numba.float32)

        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        partial = 0.0
        if i < size:
            # 2 Global Reads: a[i] and b[i]
            partial = a[i] * b[i]

        shared[local_i] = partial
        cuda.syncthreads()

        # Perform parallel reduction to sum all partial products
        # eg: first round, shared[0] += shared[1], shared[2] += shared[3] ...
        # second round, shared[0] += shared[2], shared[4] += shared[6]
        # third round, shared[0] += shared[4]
        stride = 1
        while stride < TPB:
            index = 2 * stride * local_i
            if index + stride < TPB:
                shared[index] += shared[index + stride]
            stride *= 2
            cuda.syncthreads()

        # After reduction, the first thread has the total dot product
        if local_i == 0:
            out[0] = shared[0]
    return call
```

## Puzzle 11 - 1D Convolution (Hard)

```py
def conv_spec(a, b):
    out = np.zeros(*a.shape)
    len = b.shape[0]
    for i in range(a.shape[0]):
        out[i] = sum([a[i + j] * b[j] for j in range(len) if i + j < a.shape[0]])
    return out


MAX_CONV = 4
TPB = 8
TPB_MAX_CONV = TPB + MAX_CONV
def conv_test(cuda):
    def call(out, a, b, a_size, b_size) -> None:
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        shared_a = cuda.shared.array(TPB_MAX_CONV, numba.float32)
        shared_b = cuda.shared.array(MAX_CONV, numba.float32)

        if i < a_size:
           shared_a[local_i] = a[i]
        if local_i < b_size:
           shared_b[local_i] = b[local_i]
        else:
           # using the idle threads to read extra elements for shared_a
           # this makes sure two global reads
           # eg: local_i=4, b_size=4, local_i2=0
           local_i2 = local_i - b_size
           i2 = i - b_size
           if i2 + TPB < a_size and local_i2 < b_size:
              shared_a[TPB + local_i2] = a[i2 + TPB]
        cuda.syncthreads()

        acc = 0
        for k in range(b_size):
            acc += shared_a[local_i + k] * shared_b[k]
        if i < a_size:
            out[i] = acc

    return call
```

## Puzzle 12 - Prefix Sum

```py
TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0] + TPB - 1) // TPB)
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[j] = a[i : i + TPB].sum()
    return out


def sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x

        if i < size:
            cache[local_i] = a[i]
        cuda.syncthreads()
        
        # Stride shrinks:
        #  1. cache[0] += cache[4], cache[1] += cache[5], ...
        #  2. cache[0] += cache[2], cache[1] += cache[3]
        #  3. cache[0] += cache[1]
        stride = TPB // 2
        while stride > 0:
            if local_i < stride and (local_i + stride) < TPB:
                cache[local_i] += cache[local_i + stride]
            cuda.syncthreads()
            stride = stride // 2
        
        if local_i == 0:
            out[cuda.blockIdx.x] = cache[0]


    return call
```

## Puzzle 13 - Axis Sum

```py
TPB = 8
def sum_spec(a):
    out = np.zeros((a.shape[0], (a.shape[1] + TPB - 1) // TPB))
    for j, i in enumerate(range(0, a.shape[-1], TPB)):
        out[..., j] = a[..., i : i + TPB].sum(-1)
    return out


def axis_sum_test(cuda):
    def call(out, a, size: int) -> None:
        cache = cuda.shared.array(TPB, numba.float32)
        i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
        local_i = cuda.threadIdx.x
        batch = cuda.blockIdx.y

        if i < size:
            cache[local_i] = a[batch, i]

        cuda.syncthreads()
        
        # Almost same to previous one
        stride = TPB // 2
        while stride > 0:
            if local_i < stride and i + stride < size:
                cache[local_i] += cache[local_i + stride]
            cuda.syncthreads()
            stride = stride // 2
        
        if local_i == 0:
            out[batch, cuda.blockIdx.x] = cache[0]

    return call
```

## Puzzle 14 - Matrix Mul (Hard)

```py
def matmul_spec(a, b):
    return a @ b


TPB = 3
def mm_oneblock_test(cuda):
    def call(out, a, b, size: int) -> None:
        a_shared = cuda.shared.array((TPB, TPB), numba.float32)
        b_shared = cuda.shared.array((TPB, TPB), numba.float32)

        row = cuda.blockIdx.y * TPB + cuda.threadIdx.y
        col = cuda.blockIdx.x * TPB + cuda.threadIdx.x
        local_row = cuda.threadIdx.y
        local_col = cuda.threadIdx.x

        tmp = 0.0

        # Run with tiles:
        # this is because threads number are limited in one block
        # so we should read first, calculate, and move to the next region, until done.
        for m in range((size + TPB - 1) // TPB):
            a_row = row
            a_col = m * TPB + local_col
            b_row = m * TPB + local_row
            b_col = col

            if a_row < size and a_col < size:
                a_shared[local_row, local_col] = a[a_row, a_col]
            else:
                a_shared[local_row, local_col] = 0.0

            if b_row < size and b_col < size:
                b_shared[local_row, local_col] = b[b_row, b_col]
            else:
                b_shared[local_row, local_col] = 0.0

            cuda.syncthreads()
  
            for k in range(TPB):
                tmp += a_shared[local_row, k] * b_shared[k, local_col]

        if row < size and col < size:
            out[row, col] = tmp

    return call
```
