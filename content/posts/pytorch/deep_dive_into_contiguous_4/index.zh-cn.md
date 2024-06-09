---
title: "Deep Dive to Pytorch Contiguous Operator(4)"
date: 2024-05-25T09:09:23+08:00
categories: ["pytorch"]
summary: "这篇博客介绍了PyTorch TensorIterator 针对任意输入tensor计算output stride的过程。"
---

## Summary

这篇博客介绍了PyTorch TensorIterator 针对任意输入tensor计算output stride的过程。

## Introduction

我们在前文介绍[Contiguous](../deep_dive_into_contiguous_3/)的时候主要介绍了由上至下contiguous的调用链，然后有小伙伴对memory format的stride计算产生了疑惑，这里再续写一篇文档对**Tensoriterator**的这部分内容进行补充。

## Fast Setup

### Process of Fast Setup

在Pytorch Tensoriterator中，如果shape一致、input满足相同的memory format和stride，那么就会走**fast setup**的路径快速构建output tensor。在这条路径下不需要进行stride计算，直接`set_output_raw_strided`传参memory format即可。

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

这里`non_overlapping_dense`指的是内存没有空隙的密集tensor，**是contiguous的tensor一定是overlapping_and_dense的tensor**。

和`is_contiguous`等标记位一样，有一个专门的set函数来设置（在`refresh`函数中调用），最底层的计算逻辑为：

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

这里的计算逻辑为先拿到一个让stride**升序排列**的perm，然后依据perm逐层重复stride计算，确保每一维度的stride都符合要求。详细计算过程我们这里不展开，有兴趣的同学可以看附录。

`non_overlapping_and_dense`的tensor不一定是`contiguous`的，如`shape=[3,4], stride=[1, 3]`

## Stride Calculation

但如果不满足fast setup条件，那么 Tensoriterator 就会进入计算stride的逻辑，通过`perm_`这个转置的应用来实现stride计算。

计算遵循以下规律（ambiguous指无法判断memory format的tensor，ct指contiguous、cl指channels last）：**左值优先**、**ambiguous tensor优先级最低**

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

这里pytorch实现由于要考虑coalesce，代码较为复杂，我们使用[DIPU_OPInferrer](https://github.com/DeepLink-org/deeplink.framework/blob/ca91b12f4404f0ab30547e311ff88fa0135e45f3/dipu/torch_dipu/csrc_dipu/aten/ops/DIPUOpInferrer.cpp)的简化版代码来说明，与pytorch代码等价

### Normal Case

我们以一个`channels_last` tensor相加一个`contiguous` tensor为例说明stride计算流程。

```py
import torch

device = "cuda"

cl = torch.rand(2, 3, 4, 5, device=device).to(memory_format=torch.channels_last)
ct = torch.rand(3, 4, 5, device=device)
result = cl + ct

print(f"cl: {cl.shape}, {cl.stride()}, ct: {ct.shape}, {ct.stride()}, result shape: {result.shape}, result stride: {result.stride()}")
# cl: torch.Size([2, 3, 4, 5]), (60, 1, 15, 3)
# ct: torch.Size([3, 4, 5]), (20, 5, 1)
# result shape: torch.Size([2, 3, 4, 5]), result stride: (60, 1, 15, 3)
```

首先会计算`perm_`，`perm_`表示让第一维作为内存递进最快的转置（让stride呈现**递增序列**的转置）

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

对于ct `[3,4,5]` tensor会被广播至shape `[1,3,4,5]`，其effective stride为`[0,20,5,1]`，然后使用`should_swap`作为comparer对初始化为`[3,2,1,0]`的`perm_`进行插入排序。

在`should_swap`中，我们优先考虑第一个input的stride，所以为什么**左值优先**。如果stride相同进而考虑shape、第二个tensor以此类推。此处`shape_`为广播后的公共shape（不了解广播的同学可以阅读之前的文档[broadcast](../introduction_to_broadcast/)）

插入排序后（计算过程我们会在附录中详细展示），我们得到了让stride升序排列（第一维为内存中步进最快的dim）的转置`perm_` `[1 3 2 0]`，pytorch中input需要应用这个转置以进行coalesce和之后的loop，DIPU中简化了这一过程，直接可以利用这个`perm_`推出output的origin stride。

得到`perm_`后，我们应用该转置：

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

先按升序排列计算得到一个`Calculated strides`: `[1 3 15 60]`

然后应用`perm_` 转置便得到了最终output的stride `[60 1 15 3]`

有的小伙伴可能会问了，input1的stride也是`(60, 1, 15, 3)`，那为什么不直接取input1的memory format直接得到output的stride呢？这是因为pytorch中有ambiguous的tensor存在，ambiguous + 广播会导致stride结果与任意的tensor input不同

### Ambiguous Case

pytorch中**ambiguous**指的是既是channels last，又是contiguous的memory format的tensor。

主要有两种ambiguous的stride的tensor，第一种为`c=1`：如shape 为 `(2, 1, 4, 4)`，第二种为`h=1, w=1`，如shape为`(2, 4, 1, 1)`

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

还有一点值得指出的是，调用`.to()`方法才能使ambiguous的stride转换（底层allocate一个新tensor然后`to_copy`），调用`.contiguous`方法因为底层先检查`is_contiguous(memory_format)`所以会提前return。

```py
import torch

tensor1 = torch.randn(2, 1, 4, 4)
print(f"tensor 1, stride: [{tensor1.stride()}]")    # [(16, 16, 4, 1)]
tensor1 = tensor1.contiguous(memory_format=torch.channels_last)
print(f"tensor 1, stride: [{tensor1.stride()}]")    # [(16, 16, 4, 1)]
tensor1 = tensor1.to(memory_format=torch.channels_last)
print(f"tensor 1, stride: [{tensor1.stride()}]")    # [(16, 1, 4, 1)]
```

对于ambiguous的tensor我们的计算逻辑与normal tensor相同，`perm_`计算这一套流程同样支持ambiguous tensor，我们看这个例子

```py
import torch

device = "cuda"

cl = torch.rand(2, 3, 1, 1, device=device).to(memory_format=torch.channels_last)
ct = torch.rand(3, 1, 1, device=device)
result = cl + ct

print(f"cl: {cl.shape}, {cl.stride()}, ct: {ct.shape}, {ct.stride()}, result shape: {result.shape}, result stride: {result.stride()}")
# cl: torch.Size([2, 3, 1, 1]), (3, 1, 3, 3)
# ct: torch.Size([3, 1, 1]), (1, 1, 1)
# result shape: torch.Size([2, 3, 1, 1]), result stride: (3, 1, 3, 3)
```

对于ambiguous cl来说，它既是contiguous又是channels last，所以我们不能直接用input1的memory format作为output的，而是需要走计算流程，`perm_`计算流程我们在附录中展示，结果为`1,3,2,0`。

随后便是和normal case同样的逻辑，得到`Ascending strides: 1 3 3 3`和output
`Final strides: 3 1 3 3`

值得指出的是，根据[pytorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)所说，这部分ambiguous的tensor将在未来被修复。

### UnContiguous Case

Pytorch这套机制同样支持**Uncontiguous**的case

```py
cl = torch.rand(2, 3, 1, 1, device=device).to(memory_format=torch.channels_last)
ct = torch.rand(3, 1, 3, device=device).transpose(0, 2)

print(f"ct is contiguous: {ct.is_contiguous()}")    # False

result = cl + ct

print(f"cl: {cl.shape}, {cl.stride()}, ct: {ct.shape}, {ct.stride()}, result shape: {result.shape}, result stride: {result.stride()}")

# cl: torch.Size([2, 3, 1, 1]), (3, 1, 3, 3)
# ct: torch.Size([3, 1, 3]), (1, 3, 3)
# result shape: torch.Size([2, 3, 1, 3]), result stride: (9, 1, 3, 3)
```

其计算中间结果为：

```bash
effective strides: `[3, 1, 3, 0]` and `[0, 1, 3, 3]`
Computed permutation: 1 2 3 0
Calculated strides: 1 3 3 9
Final strides: 9 1 3 3
```

## Extra: `suggest_memory_format`

值得指出的是，由于ambiguous tensor的存在，tensor的`suggest_memory_format`方法也引入了`exact_match`参数

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

只有在`channels_last_strides_exact_match`设置为True的情况下，才会去generate一个channels last的stride逐一比较，否则就是直接取"like"即`refresh`设置的memory format标记位。

## Appendix

### Calculation of `non_overlaping_and_dense`

例如一个tensor `sizes = [4, 2, 3] strides = [8, 3, 1], perm = [2, 1, 0]`

- 第一次循环 `i = 0`：
  - `perm[0] = 2`，即 `size_perm_i = sizes[2] = 3`，`strides[2] = 1`。
  - `strides[2] == require_stride`（1 == 1），条件满足，继续。
  - 更新 `require_stride`：`require_stride *= size_perm_i`，即 `require_stride = 1 * 3 = 3`。
- 第二次循环 `i = 1`：
  - `perm[1] = 1`，即 `size_perm_i = sizes[1] = 2`，`strides[1] = 3`。
  - `strides[1] == require_stride`（3 == 3），条件满足，继续。
  - 更新 `require_stride`：`require_stride *= size_perm_i`，即 `require_stride = 3 * 2 = 6`。
- 第三次循环 `i = 2`：
  - `perm[2] = 0`，即 `size_perm_i = sizes[0] = 4`，`strides[0] = 8`。
  - `strides[0] != require_stride`（8 != 6），条件不满足，返回 `false`。

### Calculation of `perm_`

`perm_` 的初始值为 `[3, 2, 1, 0]`

- Tensor 1 有效stride：`[3, 1, 3, 3]`
- Tensor 2 有效stride：`[0, 1, 1, 1]`

step1: **i = 1**

当前 `dim1 = 1`，即索引 1，`perm_ = [3, 2, 1, 0]`。

内层循环比较 `perm_[0]` 和 `perm_[1]`：

- `should_swap(3, 2)`，stride相等，继续比较dim大小，相等，返回 0，不交换。

step2: **i = 2**

当前 `dim1 = 2`，即索引 2，`perm_ = [3, 2, 1, 0]`。

内层循环比较 `perm_[1]` 和 `perm_[2]`：

- `should_swap(2, 1)`，stride 3 > 1，返回 1，交换：
  - `perm_ = [3, 1, 2, 0]`

继续比较 `perm_[0]` 和 `perm_[1]`：

- `should_swap(3, 1)`，stride 3 > 1，返回 1，交换：
  - `perm_ = [1, 3, 2, 0]`

step3: **i = 3**

当前 `dim1 = 3`，即索引 3，`perm_ = [1, 3, 2, 0]`。

内层循环比较 `perm_[2]` 和 `perm_[3]`：

- `should_swap(2, 0)`，stride相等，比较dim大小，1 < 10，返回 -1，不交换。

比较 `perm_[1]` 和 `perm_[2]`：

- `should_swap(3, 2)`，stride相等，比较dim大小，相等，返回 0，不交换。

比较 `perm_[0]` 和 `perm_[1]`：

- `should_swap(1, 3)`，stride 1 < 3，返回 -1，不交换。

最终 `perm_` 为 `[1, 3, 2, 0]`，表示按stride排序后的dim顺序。

## Referrence

- [DeepLink.framework](https://github.com/DeepLink-org/deeplink.framework)
- [Channels Last Memory Format in PyTorch](https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html)
- [Deep Dive to Pytorch Contiguous Operator](../deep_dive_into_contiguous_1/)
