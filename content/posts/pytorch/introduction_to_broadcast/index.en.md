---
title: "Introducton to Pytorch Broadcast"
date: 2023-09-10T14:42:23+08:00
categories: ["pytorch"]
summary: "This article introduces the implementation details of pytorch broadcast mechanism, including the forward and backward calculation."
---

## Summary

This article introduces the implementation details of pytorch broadcast mechanism, including the forward and backward calculation.

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

## Understanding Gradient Calculation with Broadcasting

Let's see another code example:

```py
import torch

A = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
B = torch.tensor([1.0], requires_grad=True)

C = A + B
C.sum().backward()
print(A.grad)   # tensor([1., 1., 1.])
print(B.grad)   # tensor([3.])
```

It's easy to understand `A.grad = tensor([1., 1., 1.])`, but why does `B.grad = tensor([3.])`?

To intuitively understand this, consider that the value in `B` is utilized three times during the forward `add` operation. Consequently, during the backward pass, this value is similarly involved three times, leading to its accumulation into a total of 3.

**Question**: How does pytorch realize this?

We've introduced the mechanism of autograd engine in [../deep_dive_to_autograd_1]. If you're new to the foundational concepts of autograd in PyTorch, it's recommended to review that article first.

The key point of gradient computation in PyTorch lies within the `validate_outputs` function. Back to our example, the `add_backward(fn)` operation yields outputs `[1, 1, 1]`.

```c++
// torch/csrc/autograd/engine.cpp
static variable_list call_function(
    std::shared_ptr<GraphTask>& graph_task,
    Node* func,
    InputBuffer& inputBuffer) {
  // ...
  if (has_post_hooks) {
    auto inputs_copy = inputs;
    outputs = fn(std::move(inputs_copy));
  } else {
    outputs = fn(std::move(inputs));
  }

  validate_outputs(fn.next_edges(), outputs, [&](const std::string& msg) { /* ... */ });
  // ...
  return outputs;
}

void validate_outputs(
    const edge_list& edges,
    variable_list& grads,
    const std::function<std::string(const std::string&)>& format_error) {
  // ...
  for (const auto i : c10::irange(grads.size())) {
    const auto& edge = edges[i];
    if (!edge.is_valid())
      continue;

    const auto& metadata = edge.function->input_metadata(edge.input_nr);
    auto& grad = grads[i];
    if (!grad.defined()) {
      continue;
    }

    if (!metadata.is_same_shape(grad)) {
      // Ensuring that the gradient's shape aligns with the original tensor.
      if (metadata.is_expandable_to_shape(grad)) {
        // Calculating the rediced gradients of inputs
        grad = metadata.reduce_grad(grad);
      } else {
        const auto message = metadata.incompatible_shape_error_message(i, grad);
        TORCH_CHECK(false, format_error(message.str()));
      }
    }
    // ...
  }
}
```

In `validate_outputs`, a critical aspect in handling broadcasted tensors during gradient calculation is `reduce_grad`.

```c++
// torch/include/torch/csrc/autograd/input_metadata.h
struct InputMetadata {
  // ...

  at::Tensor reduce_grad(at::Tensor& grad) const {
    TORCH_INTERNAL_ASSERT(!grad.is_nested() && !is_nested_)
    return at::sum_to(std::move(grad), shape_as_dim_vector());
  }
}
```

This leads us to comprehend that the operation is accomplished through a summation.

```c++
// torch/include/ATen/ExpandUtils.h
inline Tensor sum_to(
    Tensor tensor,
    const c10::SymIntArrayRef shape,
    bool always_return_non_view = false) {
  // In our example, shape here is [1] (original one)
  return _sum_to(std::move(tensor), shape, always_return_non_view);
}

template <typename T>
inline Tensor _sum_to(
    Tensor tensor,
    const c10::ArrayRef<T> shape,
    bool always_return_non_view = false) {
  if (shape.size() == 0) {
    return tensor.sum();
  }

  // Get the sizes of our gradient tensor, in our example, it's [3]
  auto sizes = at::symint::sizes<T>(tensor);
  c10::SmallVector<int64_t, 8> reduce_dims;
  const int64_t leading_dims = sizes.size() - shape.size();
  // Add all leading dimensions to the reduction list.
  for (const auto i : c10::irange(leading_dims)) {
    reduce_dims.push_back(i);
  }
  // Check remaining dimensions and see if they need reduction.
  for (int64_t i = leading_dims; i < static_cast<int64_t>(sizes.size()); ++i) {
    if (shape[i - leading_dims] == 1 && sizes[i] != 1) {
      reduce_dims.push_back(i);
    }
  }

  if (!reduce_dims.empty()) {
    tensor = tensor.sum(reduce_dims, /*keepdim=*/true);
  }

  if (always_return_non_view) {
    // ...
  } else {
    return leading_dims > 0 ? at::symint::view<T>(tensor, shape) : tensor;
  }
}
```

In our example, the gradient is computed through `[1, 1, 1].sum([0], true)`, resulting in the final gradient `[3]` for Tensor B.

Congratulations! You now have a clearer understanding of PyTorch's mechanism for broadcasting.

## Referrences

- [pytorch](https://github.com/pytorch/pytorch)
- [autograd](../deep_dive_to_autograd_1)
- [deep_dive_into_contiguous](../deep_dive_into_contiguous_3)
