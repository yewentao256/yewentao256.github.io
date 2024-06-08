---
title: "Deep Dive to Pytorch Contiguous Operator(4)"
date: 2024-05-25T09:09:23+08:00
categories: ["pytorch"]
summary: "This blog covers PyTorch's TensorIterator, focusing on fast setup and stride calculation for both normal and ambiguous tensors."
---

## Summary

This blog covers PyTorch's TensorIterator, focusing on fast setup and stride calculation for both normal and ambiguous tensors.

## Introduction

In our previous discussion on [Contiguous](../deep_dive_into_contiguous_3/), we primarily covered the top-down call chain of contiguous operations. Some readers expressed confusion regarding the stride calculations for memory format. In response, we have written this supplementary document to further explain this aspect of **TensorIterator**.

## Fast Setup

### Process of Fast Setup

In PyTorch's TensorIterator, if the shapes are consistent and the inputs adhere to the same memory format and stride, the **fast setup path** is utilized to quickly construct the output tensor. This path bypasses the need for stride calculations, directly passing the memory format to `set_output_raw_strided`.

```c++
// aten/src/ATen/TensorIterator.cpp
bool TensorIteratorBase::fast_set_up(const TensorIteratorConfig& config) {
  FastSetupType setup_type = compute_fast_setup_type(config);
  if (setup_type == FastSetupType::NONE) {
    return false;
  }
  switch (setup_type) {
    case FastSetupType::CONTIGUOUS:
      {
        for (const auto i : c10::irange(num_outputs_)) {
          auto& op = operands_[i];
          if (!op.tensor_base().defined()) {
            TORCH_INTERNAL_ASSERT(op.is_type_defined(), "no type for operand", i);
          }
          // directly set the output
          set_output_raw_strided(i, shape_, {}, original_options(op).memory_format(MemoryFormat::Contiguous), names_);
        }
        break;
      }
    case FastSetupType::CHANNELS_LAST: { /* ... */ }
    case FastSetupType::NON_OVERLAPPING_DENSE: { /* ... */ }
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", c10::to_string((int)setup_type));
  }
  // coalescing ...
}

FastSetupType TensorIteratorBase::compute_fast_setup_type(const TensorIteratorConfig& config) {
  if (is_reduction_ || !all_ops_same_shape_) {
    return FastSetupType::NONE;
  }

  // ...

  bool is_contiguous = true;
  bool is_channels_last = true;
  bool is_non_overlapping_and_dense = true;
  for (const auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      is_contiguous &= op.tensor_base().is_contiguous(at::MemoryFormat::Contiguous);
      is_channels_last &= op.tensor_base().is_contiguous(at::MemoryFormat::ChannelsLast);
      is_non_overlapping_and_dense &= op.tensor_base().is_non_overlapping_and_dense();
    }
  }
  if (is_contiguous) {
    return FastSetupType::CONTIGUOUS;
  }
  if (is_channels_last) {
    return FastSetupType::CHANNELS_LAST;
  }
  if (is_non_overlapping_and_dense) {
    int64_t prev = -1;
    // Only allowed when all the defined tensors have the same shape and strides
    for (int64_t i = ntensors() - 1; i >= 0; --i) {
      const auto& op = operands_[i];
      if (op.tensor_base().defined() && !op.will_resize) {
        if (prev < 0) {
          prev = i;
          continue;
        }
        if (!tensor_base(prev).strides().equals(op.tensor_base().strides())) {
          return FastSetupType::NONE;
        }
      }
    }
    return FastSetupType::NON_OVERLAPPING_DENSE;
  }
  return FastSetupType::NONE;
}
```

### Non Overlapping and Dense

Here, `non_overlapping_dense` refers to a densely packed tensor without any gaps in memory. A **contiguous tensor is always a non-overlapping and dense tensor**.

Similar to flags like `is_contiguous`, there is a dedicated function (called by `refresh`) to set this property. The underlying calculation logic is as follows:

```c++
// c10/core/TensorImpl.h
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  bool compute_is_non_overlapping_and_dense_dim5(identity<bool> type_id) {
    return is_contiguous_ || is_channels_last_contiguous_ ||
        is_channels_last_3d_contiguous_ ||
        compute_non_overlapping_and_dense(type_id);
  }

  bool compute_is_non_overlapping_and_dense_anydim(identity<bool> type_id) {
    return is_contiguous_ || compute_non_overlapping_and_dense(type_id);
  }
}

// c10/core/Contiguity.h
template <typename T>
bool _compute_non_overlapping_and_dense(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  auto dim = sizes.size();
  if (dim == 1) {
    return sizes[0] < 2 || strides[0] == 1;
  }
  SmallVector<int64_t, 5> perm;
  perm.resize(dim);
  for (const auto i : c10::irange(dim)) {
    perm[i] = i;
  }
  // Sort by strides, leaving 0 and 1 sized dims at the end of the array
  std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
    if (sizes[a] < 2) {
      return false;
    } else if (sizes[b] < 2) {
      return true;
    }
    return strides[a] < strides[b];
  });

  T require_stride = 1;
  for (const auto i : c10::irange(dim)) {
    const auto& size_perm_i = sizes[perm[i]];
    if (size_perm_i < 2) {
      return true;
    }
    if (strides[perm[i]] != require_stride) {
      return false;
    }
    require_stride *= size_perm_i;
  }
  return true;
}
```

The calculation logic first obtains a permutation that sorts the stride in **ascending order**. Then, based on this permutation, it repeatedly calculates the stride layer by layer, ensuring that the stride for each dimension meets the requirements. We will not delve into the detailed calculation process here; interested readers can refer to the appendix.

A `non_overlapping_and_dense` tensor is not necessarily `contiguous`. For example, a tensor with `shape=[3, 4]` and `stride=[1, 3]` is non-overlapping and dense but not contiguous.

## Stride Calculation

If the conditions for fast setup are not met, TensorIterator will follow the stride calculation logic, utilizing the `perm_` to achieve the desired strides.

The calculation follows these principles (where "ambiguous" refers to tensors with undetermined memory formats, "ct" refers to contiguous, and "cl" refers to channels last): **Left-hand value first** and **ambiguous tensors have the lowest priority**.

| Left Value \ Right Value           | Result          |
|------------------------------------|-----------------|
| ambiguous + ct                     | ct              |
| ambiguous + cl                     | cl              |
| ct + ambiguous                     | ct              |
| cl + ambiguous                     | cl              |
| ct + cl                            | ct              |
| cl + ct                            | cl              |
| ambiguous(ct) + ambiguous(cl)      | ambiguous(ct)   |
| ambiguous(cl) + ambiguous(ct)      | ambiguous(cl)   |

Due to the need to consider coalesce, PyTorch's implementation is complex. Therefore, we will use a simplified version of the code from [DIPU_OPInferrer](https://github.com/DeepLink-org/deeplink.framework/blob/ca91b12f4404f0ab30547e311ff88fa0135e45f3/dipu/torch_dipu/csrc_dipu/aten/ops/DIPUOpInferrer.cpp) to explain the process. This is equivalent to PyTorch's code.

### Normal Case

Let's take an example where we add a `channels_last` tensor to a `contiguous` tensor to illustrate the stride calculation process.

```py
import torch

device = "cuda"

cl = torch.rand(2, 3, 4, 5, device=device).to(memory_format=torch.channels_last)
ct = torch.rand(3, 4, 5, device=device)
result = cl + ct

# cl: torch.Size([2, 3, 4, 5]), (60, 1, 15, 3)
# ct: torch.Size([3, 4, 5]), (20, 5, 1)
# result shape: torch.Size([2, 3, 4, 5]), result stride: (60, 1, 15, 3)
```

First, `perm_` is calculated. `perm_` represents a transpose that makes the first dimension the fastest progressing in memory (sorting strides in **ascending order**).

```c++
// dipu/torch_dipu/csrc_dipu/aten/ops/DIPUOpInferrer.cpp
// Calculate perm_ to sort the dimensions based on strides in ascending order.
// Then we can use the perm_ to calculate the output stride
void OpInferrer::compute_perm() {
  perm_.resize(ndim());
  if (ndim() == 1) {
    perm_[0] = 0;
    return;
  }

  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  auto strides = compute_effective_strides();

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {
    for (const auto i : c10::irange(ntensors())) {
      int64_t stride0 = strides[i][dim0];
      int64_t stride1 = strides[i][dim1];
      if (stride0 == 0 || stride1 == 0) {
        // move on to the next input if one of the dimensions is broadcasted
        continue;
      }
      if (stride0 < stride1) {
        return -1;
      }
      if (stride0 > stride1) {
        return 1;
      }
      // equal strides, use dimensions themselves as the tie-breaker.
      if (shape_[dim0] > shape_[dim1]) {
        return 1;
      }
    }
    return 0;
  };

  // insertion sort with support for ambiguous comparisons
  for (const auto i : c10::irange(1, ndim())) {
    size_t dim1 = i;
    // dim0 >= 0; dim0-- causes overflow
    for (size_t dim0 = i; dim0-- > 0;) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }
}
```

For a contiguous `[3,4,5]` tensor, it will be broadcast to the shape `[1,3,4,5]`, with effective strides of `[0, 20, 5, 1]`. Then, `perm_`, initially `[3, 2, 1, 0]`, is insertion-sorted using `should_swap` as the comparator.

In `should_swap`, we prioritize the stride of the first input, which is why we say **left-hand value first**. If the strides are the same, we consider the shape, and then the second tensor. Here, `shape_` is the common shape after broadcasting (for more on broadcasting, you can read the previous document on [broadcast](../introduction_to_broadcast/)).

After insertion sorting (detailed process in the appendix), we get `perm_` as `[1, 3, 2, 0]`, which represents a transpose making strides ascending. In PyTorch, inputs need to apply this transpose for coalescing and subsequent loops. DIPU simplifies this process and can directly use `perm_` to derive the output's original stride.

Once we have `perm_`, we apply this transpose:

```c++
// dipu/dipu/torch_dipu/csrc_dipu/aten/ops/DIPUOpInferrer.cpp
void OpInferrer::compute_memory_format() {
  if (fast_compute_memory_format()) {
    return;
  }
  compute_perm();

  // Calculate strides based on perm_
  auto strides = c10::DimVector();
  int64_t next_stride = 1;
  for (const auto dim : c10::irange(ndim())) {
    strides.push_back(next_stride);
    next_stride *= shape_[perm_[dim]];
  }

  // calculate the final strides_
  strides_.resize(strides.size());
  for (const auto dim : c10::irange(ndim())) {
    strides_[perm_[dim]] = strides[dim];
  }
}
```

First, calculate the strides in ascending order: `[1, 3, 15, 60]`.

Then, apply the `perm_` transpose to get the final output stride: `[60, 1, 15, 3]`.

Some may wonder why we don't directly use input1's memory format to get the output stride, as input1's stride is also `(60, 1, 15, 3)`. This is because PyTorch may deal with ambiguous tensors, where ambiguous + broadcasting can result in strides different from any tensor input.

### Ambiguous Case

In PyTorch, **ambiguous tensors** refer to those that are both channels last and contiguous in memory format.

There are mainly two types of ambiguous stride tensors. The first type has `c=1`, such as a tensor with shape `(2, 1, 4, 4)`. The second type has `h=1, w=1`, such as a tensor with shape `(2, 4, 1, 1)`.

```py
import torch

tensor1 = torch.randn(2, 1, 4, 4)# .to(memory_format=torch.channels_last)

print(f"tensor 1, stride: [{tensor1.stride()}]")    # [(16, 16, 4, 1)]
print(f"contiguous: {tensor1.is_contiguous()}")     # True
print(f"channels last: {tensor1.is_contiguous(memory_format=torch.channels_last)}")     # True

tensor2 = torch.randn(2, 4, 1, 1)# .to(memory_format=torch.channels_last)
print(f"tensor 2, stride: [{tensor2.stride()}]")    # [(4, 1, 1, 1)]
print(f"contiguous: {tensor2.is_contiguous()}")     # True
print(f"channels last: {tensor2.is_contiguous(memory_format=torch.channels_last)}")     # True
```

Another important point is that only by calling the `.to()` method can the ambiguous stride be transformed (underlying allocation of a new tensor and then `to_copy`). Calling the `.contiguous` method will return early because it first checks `is_contiguous(memory_format)`.

```py
import torch

tensor1 = torch.randn(2, 1, 4, 4)
print(f"tensor 1, stride: [{tensor1.stride()}]")    # [(16, 16, 4, 1)]
tensor1 = tensor1.contiguous(memory_format=torch.channels_last)
print(f"tensor 1, stride: [{tensor1.stride()}]")    # [(16, 16, 4, 1)]
tensor1 = tensor1.to(memory_format=torch.channels_last)
print(f"tensor 1, stride: [{tensor1.stride()}]")    # [(16, 1, 4, 1)]
```

For ambiguous tensors, our calculation logic is the same as for normal tensors. The `perm_` calculation process also supports ambiguous tensors. Let's look at an example.

```py
import torch

device = "cuda"

cl = torch.rand(2, 3, 1, 1, device=device).to(memory_format=torch.channels_last)
ct = torch.rand(3, 1, 1, device=device)
result = cl + ct
# cl: torch.Size([2, 3, 1, 1]), (3, 1, 3, 3)
# ct: torch.Size([3, 1, 1]), (1, 1, 1)
# result shape: torch.Size([2, 3, 1, 1]), result stride: (3, 1, 3, 3)
```

For an ambiguous cl tensor, it is both contiguous and channels last. Therefore, we cannot directly use input1's memory format as the output. Instead, we need to follow the calculation process. The `perm_` calculation process is detailed in the appendix, resulting in `1, 3, 2, 0`.

Then, the logic is the same as in the normal case, yielding `Ascending strides: 1, 3, 3, 3` and the output `Final strides: 3, 1, 3, 3`.

It is worth noting that, according to [PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html), this ambiguous tensor issue is expected to be fixed in the future.

### UnContiguous Case

This mechanism also supports the case of **uncontiguous** tensors.

```py
cl = torch.rand(2, 3, 1, 1, device=device).to(memory_format=torch.channels_last)
ct = torch.rand(3, 1, 3, device=device).transpose(0, 2)

print(f"ct is contiguous: {ct.is_contiguous()}")    # False

result = cl + ct
# cl: torch.Size([2, 3, 1, 1]), (3, 1, 3, 3)
# ct: torch.Size([3, 1, 3]), (1, 3, 3)
# result shape: torch.Size([2, 3, 1, 3]), result stride: (9, 1, 3, 3)
```

The intermediate calculation results for such cases are as follows:

```bash
effective strides: `[3, 1, 3, 0]` and `[0, 1, 3, 3]`
Computed permutation: 1 2 3 0
Ascending strides: 1 3 3 9
Final strides: 9 1 3 3
```

## Extra: `suggest_memory_format`

It is worth noting that due to the existence of ambiguous tensors, the `suggest_memory_format` method of a tensor introduces the `exact_match` parameter.

```c++
// aten/src/ATen/core/TensorBase.h
at::MemoryFormat suggest_memory_format(
      bool channels_last_strides_exact_match = false) const {
    // Setting channels_last_strides_exact_match to true forces function to
    // check 0,1 - sized dimension strides.
    if (layout() == at::kStrided) {
      if (impl_->is_strides_like_channels_last()) {
        if (!channels_last_strides_exact_match ||
            get_channels_last_strides_2d(sizes()) == strides()) {
          return at::MemoryFormat::ChannelsLast;
        }
      }
      else if (impl_->is_strides_like_channels_last_3d()) {
        if (!channels_last_strides_exact_match ||
            get_channels_last_strides_3d(sizes()) == strides()) {
          return at::MemoryFormat::ChannelsLast3d;
        }
      }
    }
    return at::MemoryFormat::Contiguous;
  }
```

Only when `channels_last_strides_exact_match` is set to True will it generate a channels last stride and compare each element. Otherwise, it will directly take the "like" memory format flag set by `refresh`.

## Appendix

### Calculation of `non_overlaping_and_dense`

For example, for a tensor with `sizes = [4, 2, 3]`, `strides = [8, 3, 1]`, and `perm = [2, 1, 0]`:

1. **First iteration (i = 0)**:
   - `perm[0] = 2`, so `size_perm_i = sizes[2] = 3` and `strides[2] = 1`.
   - `strides[2] == required_stride` (1 == 1), condition is met, continue.
   - Update `required_stride`: `required_stride *= size_perm_i`, thus `required_stride = 1 * 3 = 3`.

2. **Second iteration (i = 1)**:
   - `perm[1] = 1`, so `size_perm_i = sizes[1] = 2` and `strides[1] = 3`.
   - `strides[1] == required_stride` (3 == 3), condition is met, continue.
   - Update `required_stride`: `required_stride *= size_perm_i`, thus `required_stride = 3 * 2 = 6`.

3. **Third iteration (i = 2)**:
   - `perm[2] = 0`, so `size_perm_i = sizes[0] = 4` and `strides[0] = 8`.
   - `strides[0] != required_stride` (8 != 6), condition is not met, return `false`.

### Calculation of `perm_`

The initial value of `perm_` is `[3, 2, 1, 0]`.

- Tensor 1 effective strides: `[3, 1, 3, 3]`
- Tensor 2 effective strides: `[0, 1, 1, 1]`

Step 1: **i = 1**

Current `dim1 = 1`, i.e., index 1, `perm_ = [3, 2, 1, 0]`.

Inner loop compares `perm_[0]` and `perm_[1]`:

- `should_swap(3, 2)`: strides are equal, continue comparing dimension size, equal, return 0, do not swap.

Step 2: **i = 2**

Current `dim1 = 2`, i.e., index 2, `perm_ = [3, 2, 1, 0]`.

Inner loop compares `perm_[1]` and `perm_[2]`:

- `should_swap(2, 1)`: stride 3 > 1, return 1, swap:
  - `perm_ = [3, 1, 2, 0]`

Continue comparing `perm_[0]` and `perm_[1]`:

- `should_swap(3, 1)`: stride 3 > 1, return 1, swap:
  - `perm_ = [1, 3, 2, 0]`

Step 3: **i = 3**

Current `dim1 = 3`, i.e., index 3, `perm_ = [1, 3, 2, 0]`.

Inner loop compares `perm_[2]` and `perm_[3]`:

- `should_swap(2, 0)`: strides are equal, compare dimension size, 1 < 10, return -1, do not swap.

Compare `perm_[1]` and `perm_[2]`:

- `should_swap(3, 2)`: strides are equal, compare dimension size, equal, return 0, do not swap.

Compare `perm_[0]` and `perm_[1]`:

- `should_swap(1, 3)`: stride 1 < 3, return -1, do not swap.

The final `perm_` is `[1, 3, 2, 0]`, representing the dimension order sorted by stride.

## Referrence

- [DeepLink.framework](https://github.com/DeepLink-org/deeplink.framework)
- [Channels Last Memory Format in PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
- [Deep Dive to Pytorch Contiguous Operator](../deep_dive_into_contiguous_1/)
