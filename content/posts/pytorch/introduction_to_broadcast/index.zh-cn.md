---
title: "Introducton to Pytorch Broadcast"
date: 2023-09-10T14:42:23+08:00
categories: ["pytorch"]
summary: "这篇文章介绍了pytorch broadcast机制的基本概念与开发细节，包含前向和反向的详细计算过程。"
---

## AI翻译注意

本篇文档的中文版本由AI（chatgpt o1-preview）进行翻译，如有冲突请参考英文版本

## 摘要

本文介绍了 PyTorch 广播机制的实现细节，包括前向和后向计算。

## 引言

让我们从代码开始：

```python
import torch

A = torch.tensor([[1, 2, 3], [4, 5, 6]])    # 形状：[2, 3]
B = torch.tensor([1, 2, 3])                 # 形状：[3]

C = A + B
print(C)    # tensor([[2, 4, 6], [5, 7, 9]])  形状：[2, 3]
```

这是如何实现的呢？让我们一步步探索。

## 广播规则

以下是张量可以广播的情况：

**情况1：维度不一致**

如果张量 A 和 B 的维度不同，例如：`A = [2, 3]` 和 `B = [3]`，那么 B 将被扩展（添加一个维度 1）以匹配形状 `[1, 3]`。

**情况2：尺寸不一致**

当维度相同但尺寸不同，并且其中一个尺寸为 1 时，例如：`A = [2, 3]` 和 `B = [1, 3]`，B 将被广播到形状 `[2, 3]`。

**重要注意事项：**

如果张量在任何维度上的尺寸都大于 1 且不匹配，则无法一起广播。

例如，考虑 `A = [2, 3]` 和 `B = [2, 4]`。尝试组合 A 和 B 将导致错误。

## PyTorch 如何计算广播

仍然使用上面的例子，在操作符分派后，我们来到 **add** 结构内核：

注意：如果您对操作符分派感兴趣，可以参考我的文档 [深入理解 contiguous](../deep_dive_into_contiguous_1/index.en.md) 了解更多细节。

```c++
// build/aten/src/ATen/RegisterCPU.cpp
at::Tensor wrapper_CPU_add_Tensor(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) {
    structured_ufunc_add_CPU_functional op;
    op.meta(self, other, alpha);
    op.impl(self, other, alpha, *op.outputs_[0]);
    return std::move(op.outputs_[0]).take();
}
```

注意，**structured_ufunc_add_CPU_functional** 是一个 **TensorIterator**。

我们主要关注 `op.meta` 函数：

```c++
// aten/src/ATen/native/BinaryOps.cpp
TORCH_META_FUNC2(add, Tensor) (
  const Tensor& self, const Tensor& other, const Scalar& alpha
) {
  // self: [2, 3], other: [3]
  // out (maybe_get_output()) 这里是未定义的
  build_borrowing_binary_op(maybe_get_output(), self, other);
  native::alpha_check(dtype(), alpha);
}
```

然后我们来到 `build_borrowing_binary_op` 函数：

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
  // ... Tensor Iterator 构建逻辑
  // 计算广播后的形状
  compute_shape(config);
  // ...
}
```

让我们深入 `compute_shape` 函数：

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

  // 使用 ptrdiff_t 来确保有符号比较
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;  // 等同于 `dimsA - ndim + i`
    ptrdiff_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    TORCH_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "张量 a 的尺寸 (", sizeA,
        ") 必须与张量 b 的尺寸 (", sizeB,
        ") 在非单例维度 ", i, " 上匹配");
    // 如果 sizeA 和 sizeB 相同，任取其一；
    // 如果 sizeA 为 1，取 sizeB（因此选择较大的值）
    expandedSizes[i] = sizeA == 1 ? std::move(sizeB) : std::move(sizeA);
  }

  return expandedSizes;
}
```

由此，我们得出当 `A = [2, 3]` 和 `B = [3]` 时，`expandedSizes = [2, 3]`。

在计算完成后，`expandedSizes` 的值被存储为 **TensorIterator** 类中的 `shape_`。该类为各种形状和步幅提供了强大的支持。随后，调用 `compute_types`、`compute_strides` 和 `coalesce` 等方法来完整地构建 TensorIterator。

之后，调用 `op.impl` 来执行实际的加法操作。

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
```

`ufunc::add` 是逐元素操作，看起来相当简单：

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

使 PyTorch 的 TensorIterator 能够适应各种形状和步幅的关键组件是 `cpu_kernel_vec`。它利用构建阶段计算的形状，并使用诸如 `loop2d` 和 `DimCounter` 等函数来实现。

在本文中，我们略过了这些复杂的操作。对于那些渴望深入了解这些技术细节的人，我鼓励您阅读我之前的文档：[深入理解 contiguous (3)](../deep_dive_into_contiguous_3)。

## 理解带有广播的梯度计算

让我们看另一个代码示例：

```python
import torch

A = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
B = torch.tensor([1.0], requires_grad=True)

C = A + B
C.sum().backward()
print(A.grad)   # tensor([1., 1., 1.])
print(B.grad)   # tensor([3.])
```

理解 `A.grad = tensor([1., 1., 1.])` 很容易，但为什么 `B.grad = tensor([3.])`？

为了直观地理解这一点，考虑到在前向 `add` 操作中，`B` 的值被使用了三次。因此，在反向传播过程中，该值也同样涉及三次，导致它的累积总和为 3。

**问题**：PyTorch 是如何实现这一点的？

我们在 [deep_dive_to_autograd_1](../deep_dive_to_autograd_1) 中介绍了自动求导引擎的机制。如果您对 PyTorch 中自动求导的基本概念不熟悉，建议您先阅读那篇文章。

PyTorch 中梯度计算的关键点在于 `validate_outputs` 函数。回到我们的例子，`add_backward(fn)` 操作产生输出 `[1, 1, 1]`。

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
      // 确保梯度的形状与原始张量对齐
      if (metadata.is_expandable_to_shape(grad)) {
        // 计算输入的缩减梯度
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

在 `validate_outputs` 中，处理梯度计算中广播张量的关键方面是 `reduce_grad`。

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

这使我们理解到，该操作是通过求和完成的。

```c++
// torch/include/ATen/ExpandUtils.h
inline Tensor sum_to(
    Tensor tensor,
    const c10::SymIntArrayRef shape,
    bool always_return_non_view = false) {
  // 在我们的例子中，shape 是 [1]（原始形状）
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

  // 获取我们的梯度张量的尺寸，在我们的例子中是 [3]
  auto sizes = at::symint::sizes<T>(tensor);
  c10::SmallVector<int64_t, 8> reduce_dims;
  const int64_t leading_dims = sizes.size() - shape.size();
  // 将所有前导维度添加到缩减列表中
  for (const auto i : c10::irange(leading_dims)) {
    reduce_dims.push_back(i);
  }
  // 检查剩余维度，看看是否需要缩减
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

在我们的例子中，梯度是通过 `[1, 1, 1].sum([0], true)` 计算的，最终得到张量 B 的梯度 `[3]`。

恭喜！您现在对 PyTorch 的广播机制有了更清晰的理解。

## 参考资料

- [PyTorch](https://github.com/pytorch/pytorch)
- [自动求导](../deep_dive_to_autograd_1)
- [深入理解 contiguous](../deep_dive_into_contiguous_3)