---
title: "Introducton to Pytorch Broadcast"
date: 2023-09-10T14:42:23+08:00
categories: ["pytorch"]
summary: "This article introduces the implementation details of pytorch broadcast mechanism."
---

## Summary

This article introduces the implementation details of pytorch broadcast mechanism.

## Introduction

Let's start with code:

```python
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])    # shape: [2, 3]
B = torch.tensor([1, 2, 3])                 # shape: [3]

C = A + B
print(C)    # tensor([[2, 4, 6], [5, 7, 9]])  shape: [2, 3]
```

How could this happen? Let's discover it step by step.

## BroadCast Rule

The following outlines scenarios in which tensors can be broadcasted:

Case 1: **Dimensional Discrepancy**

If tensors A and B have different dimensions, for instance: `A = [2, 3]` and `B = [3]`, then B will be unsqueezed (with an added dimension of 1) to match the shape `[1, 3]`.

Case2: **Size Discrepancy**

When the dimensions are the same but the sizes differ, and one of them is 1, for example: `A = [2, 3]` and `B = [1, 3]`, B will be broadcasted to the shape `[2, 3]`.

Important Note:

If the sizes in any dimension of the tensors are both greater than 1 and do not match, they cannot be broadcasted together.

As an illustration, consider `A = [2, 3]` and `B = [2, 4]`. Attempting to combine A and B will result in an error.

## How Pytorch Calculates for Broadcasting

Still using the example above, after operator dispatch, we come to **add** structure kernel:

Note: If you are interested in op dispatch, you can refer to my document [deep_dice_to_contiguous](../deep_dive_into_contiguous_1/index.en.md) for more details.

```c++
// build/aten/src/ATen/RegisterCPU.cpp
at::Tensor wrapper_CPU_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    structured_ufunc_add_CPU_functional op;
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
```

Note that **structured_ufunc_add_CPU_functional** is a **TensorIterator**.

We mainly focus on `op.meta` function:

```c++
// aten/src/ATen/native/BinaryOps.cpp
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  // self: [2, 3], other: [3]
  // out (maybe_get_output()) here is undefined
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}
```

Then we comes to the `build_borrowing_binary_op` function

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::build_borrowing_binary_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_OP_CONFIG()
      .add_output(out)
      .add_input(a)
      .add_input(b));
}

void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // ... Tensor Iterator build logic
  // compute the broadcasted shape
  compute_shape(config);
  // ...
}
```

Let's step into the `compute_shape` function:

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::compute_shape(const TensorIteratorConfig& config) {
  // ...
  for (auto& op : operands_) {
    // ...
    if (shape_.empty()) {
      shape_ = shape;
    } else if (!shape.equals(shape_)) {
      all_ops_same_shape_ = false;
      shape_ = infer_size_dimvector(shape_, shape);
    }
  }
}

// aten/src/ATen/ExpandUtils.cpp
DimVector infer_size_dimvector(IntArrayRef a, IntArrayRef b) {
  return infer_size_impl<DimVector, IntArrayRef>(a, b);
}

template <typename Container, typename ArrayType>
Container infer_size_impl(ArrayType a, ArrayType b) {
  size_t dimsA = a.size();
  size_t dimsB = b.size();
  size_t ndim = dimsA > dimsB ? dimsA : dimsB;
  Container expandedSizes(ndim);

  // Uses ptrdiff_t to ensure signed comparison
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;  // same as `dimsA - ndim + i`
    ptrdiff_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    TORCH_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "The size of tensor a (", sizeA,
        ") must match the size of tensor b (", sizeB,
        ") at non-singleton dimension ", i);
    // If sizeA and sizeB are the same, either is taken;
    // if sizeA is 1, sizeB is taken (thus selecting the larger value)
    expandedSizes[i] = sizeA == 1 ? std::move(sizeB) : std::move(sizeA);
  }

  return expandedSizes;
}
```

Given that, we derive `expandedSizes = [2, 3]` when `A = [2, 3]` and `B = [3]`.

Upon computation, the value of `expandedSizes` is stored as `shape_` within the **TensorIterator** class. This class offers robust support for various shapes and strides. Subsequently, methods such as `compute_types`, `compute_strides`, and `coalesce` are invoked to fully construct the TensorIterator.

Thereafter, `op.impl` is called to perform the actual addition operation.

```c++
// build/aten/src/ATen/UfuncCPUKernel_add.cpp
void add_kernel(TensorIteratorBase& iter, const at::Scalar & alpha) {
  AT_DISPATCH_SWITCH(iter.common_dtype(), "add_stub",
// ...

AT_DISPATCH_CASE(at::ScalarType::Float,
  [&]() {
    
auto _s_alpha = alpha.to<scalar_t>();
auto _v_alpha = at::vec::Vectorized<scalar_t>(_s_alpha);
cpu_kernel_vec(iter,
  [=](scalar_t self, scalar_t other) { return ufunc::add(self, other, _s_alpha); },
  [=](at::vec::Vectorized<scalar_t> self, at::vec::Vectorized<scalar_t> other) { return ufunc::add(self, other, _v_alpha); }
);

  }
)
  )
}
// ...
```

The `ufunc::add` is eletment-wise operation and seems quite easy:

```c++
// aten/src/ATen/native/ufunc/add.h
namespace at {
namespace native {
namespace ufunc {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
  return self + alpha * other;
}

#if !defined(__CUDACC__) && !defined(__HIPCC__)
using vec::Vectorized;
template <typename T>
C10_ALWAYS_INLINE Vectorized<T> add(Vectorized<T> self, Vectorized<T> other, Vectorized<T> alpha) __ubsan_ignore_undefined__ {
  return vec::fmadd(other, alpha, self);
}
#endif

}}}  // namespace at::native::ufunc
```

A pivotal component enabling PyTorch's TensorIterator to accommodate diverse shapes and strides is the `cpu_kernel_vec`. This leverages the shape computed during the build phase and utilizes functions like `loop2d` and `DimCounter` for its realization.

In this document, we've bypassed these intricate operations. For those keen on delving deeper into these technical specifics, I encourage you to peruse my previous document: [deep_dive_into_contiguous(3)](../deep_dive_into_contiguous_3).
