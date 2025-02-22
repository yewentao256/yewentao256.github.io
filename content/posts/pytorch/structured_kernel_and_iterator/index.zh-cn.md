---
title: "Exploring Structured Kernel and Tensor Iterator in PyTorch"
date: 2023-12-03T13:56:11+08:00
categories: ["pytorch"]
summary: "在本文中，我们将深入探讨 PyTorch 中的**结构化内核（Structured Kernel）**和**张量迭代器（TensorIterator）**，包括在Structured Kernel中的`meta`、`impl`函数及 TensorIterator 的构建和算子计算调用的过程。"
---

## Summary

在本文中，我们将深入探讨 PyTorch 中的**结构化内核（Structured Kernel）**和**张量迭代器（TensorIterator）**，包括在Structured Kernel中的`meta`、`impl`函数及 TensorIterator 的构建和算子计算调用的过程。

>这篇文章使用`O3-mini-high`翻译，如有困惑请参考英文原文

---

## 1. 引言

在上一篇文章中，我们简要介绍了[结构化内核和 Stub](../deep_dive_to_autograd_1/#structured-kernel-and-stub)中提到的结构化内核概念。同时，在[Copy 和 TensorIterator](../deep_dive_into_contiguous_3/#9-the-copy_-operator-and-tensoriterator)中，我们也深入探讨了结构化内核的基础——**TensorIterator**。  
本文将这两个概念融合在一起，全面探讨结构化内核与 TensorIterator 的实现过程。

我们从下面这段代码开始：

```py
import torch

A = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
B = A.sum(dim=0, keepdim=True)
```

执行这段代码时，会调用 `TensorBody.h` 中的 `sum_dim_IntList`。

```c++
// torch/include/ATen/core/TensorBody.h
inline at::Tensor Tensor::sum(at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) const {
    return at::_ops::sum_dim_IntList::call(const_cast<Tensor&>(*this), dim, keepdim, dtype);
}
```

经过 dispatch 调度后，我们最终进入了 CPU 上 `sum_dim_IntList` 的结构化内核实现（具体位置可能因编译选项不同而有所差异）。

> 注意：如果你对 dispatch 流程感兴趣但还不太熟悉，建议先阅读[Dispatching Contiguous Operators](../deep_dive_into_contiguous_1/#4-dispatch-contiguous-operator-find-schema-and-call-kernel)以打下基础。

---

## 2. 结构化内核

编译 PyTorch 后才能查看生成的代码。下面展示的是 CPU 端的实现：

```c++
// build/aten/src/ATen/RegisterCPU.cpp
at::Tensor wrapper_CPU_sum_dim_IntList(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
  structured_sum_out_functional op;
  op.meta(self, dim, keepdim, dtype);
  op.impl(self, dim, keepdim, dtype, op.outputs_[0]);
  return std::move(op.outputs_[0]);
}
```

结构化内核框架主要包括三个部分：

1. **操作声明（Op Declaration）**：例如声明 `structured_sum_out_functional`。
2. **Op Meta**：为操作做准备，包括推断输出形状、数据类型等。
3. **Op 实现（Op Implementation）**：基于 **TensorIterator** 执行具体计算。

经过这几个步骤，计算结果被生成并返回。

### 2.1 操作声明：MetaBase

先看一下 `structured_sum_out_functional` 的声明：

```c++
// build/aten/src/ATen/RegisterCPU.cpp
struct structured_sum_out_functional final : public at::native::structured_sum_out {
    void set_output_strided(/* params */) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        // ...
    }
    void set_output_raw_strided(/* params */) override {
        outputs_[output_idx] = create_out(sizes, strides, options);
        // ...
    }
    const Tensor& maybe_get_output(int64_t output_idx) override {
      return outputs_[output_idx];
    }
    std::array<Tensor, 1> outputs_;
};

Tensor create_out(IntArrayRef sizes, IntArrayRef strides, const TensorOptions &options) {
  if (strides.empty()) {
      return at::detail::empty_cpu(sizes, options);
  } else {
      return at::detail::empty_strided_cpu(sizes, strides, options);
  }
}
```

这个操作继承自 `at::native::structured_sum_out`：

```c++
// build/aten/src/ATen/ops/sum_native.h
struct TORCH_API structured_sum_out : public at::meta::structured_sum_dim_IntList {
  void impl(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype, const at::Tensor & out);
};

// torch/include/ATen/ops/sum_meta.h
struct TORCH_API structured_sum_dim_IntList : public at::impl::MetaBase {
    void meta(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype);
};
```

通过代码分析可以看出，声明一个操作实际上就是定义了一个 **MetaBase** 类的实例。  
**MetaBase** 是所有结构化内核类（包括 `structured_sum_out_functional` 和 **TensorIteratorBase**）的基础类，其主要接口如下：

```c++
// torch/include/ATen/TensorMeta.h
struct TORCH_API MetaBase {
  // ...
  virtual const Tensor& maybe_get_output(int64_t output_idx) = 0;
  virtual void set_output_strided(/* params */) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  virtual void set_output_raw_strided(/* params */) {
    TORCH_INTERNAL_ASSERT(false, "set_output_strided not implemented.");
  }

  // contiguous 情况下的 set_output_strided 别名
  void set_output_contiguous(/* params */) {
    auto strides = c10::contiguous_strides(sizes);
    set_output_strided(output_idx, sizes, strides, options, names);
  }

  // 当没有预设输出时返回一个未定义的 tensor
  const Tensor& maybe_get_output() {
    return maybe_get_output(0);
  }
  virtual ~MetaBase() = default;
};
```

在 **MetaBase** 中通常会重写的关键函数有：

1. `set_output_raw_strided`：当内核能够处理任意 strides 的输出时调用。
2. `set_output_strided`：用于其他情况（对于连续内存的情况，会调用 `set_output_contiguous`，进而调用该函数）。

对于 `structured_sum_out_functional`，这两个函数均被重写为 `outputs_[output_idx] = create_out(sizes, strides, options);`。后续我们将进一步讨论它们的调用。

---

### 2.2 Op.meta

回到结构化内核流程，第二步调用的是 `op.meta(self, dim, keepdim, dtype);`：

```c++
// aten/src/ATen/native/ReduceOps.cpp

// structured_sum_dim_IntList::meta 的实现
TORCH_META_FUNC2(sum, dim_IntList)
(const Tensor& self, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  // 通过 maybe_get_output() 获得一个未定义的输出
  // infer_dtype_from_optional 根据 self 和（如果已定义）输出推断 dtype
  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
  resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}
```

接着会调用 `resize_reduction`：

```c++
// aten/src/ATen/native/ReduceOpsUtils.h
static void resize_reduction(
    impl::MetaBase& meta,
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype) {
  // 从 opt_dims（如果定义）或 self.dim() 生成 DimVector
  DimVector dims_ = at::native::make_dim_vector(opt_dims, self.dim());
  // 将每个维度就地“包装”，支持负索引
  maybe_wrap_dims(dims_, self.dim());
  // 根据 dims_ 推断 sum 操作的输出形状（基于 std::bitset）
  auto shape = get_reduction_shape(self, dims_, keepdim);
  // 使用推断的形状来声明输出
  // 输出经过此步骤后被分配并定义
  meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype));
  // ...
}
```

在 `op.meta` 函数中，一个关键步骤是调用 `meta.set_output_raw_strided` 来定义结构化内核的输出。  
当输出成功分配后，我们就可以调用 `op.impl` 了。

---

### 2.3 Op.impl

第三步是实际执行计算的部分：

```c++
// aten/src/ATen/native/ReduceOps.cpp

// structured_sum_out::impl 的实现
TORCH_IMPL_FUNC(sum_out) (/* params... */) {
  auto iter = meta::make_reduction_from_out_ty(self, result, opt_dim, keepdim, result.scalar_type());
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    sum_stub(iter.device_type(), iter);
  }
}
```

这里使用 `meta::make_reduction_from_out_ty` 来构建一个 **TensorIterator**：

```c++
// aten/src/ATen/native/ReduceOpsUtils.h
static C10_UNUSED TensorIterator make_reduction_from_out_ty(/* params... */) {
  // ...
  auto in_dtype = gpu_lowp_to_f32 ? self.scalar_type() : out_dtype;
  return make_reduction(self, result, opt_dims, keepdim, in_dtype);
}

static TensorIterator make_reduction(
    const Tensor& self,
    const Tensor& result,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType in_dtype) {
  int64_t ndim = self.dim();
  auto mask = at::native::make_dim_mask(opt_dims, ndim);
  // 如果 keepdim 为 false，则需要对结果进行 view（扩展一个维度）
  auto viewed_result = at::native::review_reduce_result(result, ndim, mask, keepdim);
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}
```

为什么需要对结果进行 view？  
当 `keepdim` 为 false 时，推断出的结果形状（经过 reduction 后）与 TensorIterator 预期的形状不一致，因此需要通过扩展一个维度来“恢复”形状，好像 `keepdim` 为 true 一样。

调用 `TensorIterator::reduce_op` 后，就创建了一个 **TensorIterator** 实例。后续我们将详细介绍这一部分。

接着，调用 `sum_stub`，经过设备 dispatch 后，我们进入了 sum 内核：

```c++
// aten/src/ATen/native/cpu/SumKernel.cpp
void sum_kernel_impl(TensorIterator &iter) {
  // 如果 dtype 为 bool 时……
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "sum_cpu", [&] {
    cascade_sum</*ignore_nan=*/false, scalar_t>(iter);
  });

  // 注：AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(...) 的逻辑等同于下面这种形式：
  [&] {
    const auto& the_type = iter.dtype();
    constexpr const char* at_dispatch_name = "sum_cpu";
    at::ScalarType _st = ::detail::scalar_type(the_type);
    switch (_st) {
      case at::ScalarType::Double:  { /* ... */ }
      case at::ScalarType::Float: {
        do {
          // 一些检查逻辑……
        using scalar_t __attribute__((__unused__)) =
            c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>;
        return [&] { cascade_sum<false, scalar_t>(iter); }();
        }
      }
      case at::ScalarType::ComplexDouble: { /* ... */ }
      // ...
      default: { /* ... */ }
    }
  }()
}
```

最终计算由 `cascade_sum` 完成，其中一个关键点是将匿名函数传递给 **TensorIterator** 的 `parallel_reduce` 方法（后续会详细讨论）。

```c++
// 为了更高的精度，定制了浮点数求和操作
template <bool ignore_nan, typename scalar_t>
void cascade_sum(TensorIterator &iter) {
  iter.output_base().fill_(scalar_t(0));
  iter.parallel_reduce(
    [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
      /* 匿名函数的实现…… */
    });
}
```

当 `cascade_sum` 执行完成后，sum 的计算结果便生成了，并返回给用户。

---

## 3. TensorIterator

回到结构化内核部分：

```c++
// build/aten/src/ATen/RegisterCPU.cpp
at::Tensor wrapper_CPU_sum_dim_IntList(/* params */) {
  structured_sum_out_functional op;
  op.meta(self, dim, keepdim, dtype);
  op.impl(self, dim, keepdim, dtype, op.outputs_[0]);
  return std::move(op.outputs_[0]);
}
```

在 `op.impl` 中，我们使用 **TensorIterator** 来执行 `sum` 操作。那么，TensorIterator 是如何做到这一点的呢？

使用 TensorIterator 包含两个主要步骤：

1. 构建 TensorIterator，为后续计算做好准备。
2. 调用计算，使用 `cpu_kernel` / `gpu_kernel` 或 `parallel_reduce` 执行实际运算。

> 注：TensorIterator 系统非常复杂，此处不会详细展开所有实现细节。如需更深入了解，可以参考[PyTorch 源码](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.cpp)或我的简化版本 [MicroTorch](https://github.com/yewentao256/MicroTorch/blob/main/src/core/tensorIterator.cpp)。

### 3.1 构建 TensorIterator

构建 TensorIterator 有多种方式，这里以 `reduce_op` 为例：

```c++
// aten/src/ATen/TensorIterator.cpp
TensorIterator TensorIterator::reduce_op(TensorBase& out, const TensorBase& a) {
  TORCH_INTERNAL_ASSERT(out.defined());
  return TensorIteratorConfig()
    .set_check_mem_overlap(false)
    .add_owned_output(out)
    .add_owned_input(a)
    .resize_outputs(false)
    .is_reduction(true)
    .promote_inputs_to_common_dtype(true)
    .build();
}
```

我们创建一个 **TensorIteratorConfig** 实例，设置相关属性，然后调用 `build()` 得到一个 TensorIterator。

```c++
// torch/include/ATen/TensorIterator.h
class TORCH_API TensorIteratorConfig final {
 public:
  // ...
  // 重要：必须先添加输出，再添加输入。
  TensorIteratorConfig& add_output(const TensorBase& output) {
    return add_borrowed_output(output);
  }
  TensorIteratorConfig& add_input(const TensorBase& input) {
    return add_borrowed_input(input);
  }

  // ...

  TensorIteratorConfig& is_reduction(const bool _is_reduction) {
    is_reduction_ = _is_reduction;
    return *this;
  }

  // ...
  TensorIterator build() {
    TensorIterator iter;
    iter.build(*this);
    return iter;
  }

 private:
  SmallVector<c10::MaybeOwned<TensorBase>, 4> tensors_;
  int num_outputs_ = 0;
  int num_inputs_ = 0;

  // ...
  bool check_mem_overlap_ = true;
  bool allow_cpu_scalars_ = false;
  bool is_reduction_ = false;
  bool resize_outputs_ = true;
  bool check_all_same_dtype_ = true;
  bool check_all_same_device_ = true;
  bool enforce_safe_casting_to_output_ = false;
  bool enforce_linear_iteration_ = false;
  bool promote_inputs_to_common_dtype_ = false;
  bool promote_integer_inputs_to_float_ = false;
  bool cast_common_dtype_to_outputs_ = false;
  // ...
};
```

配置参数说明：

- **check_mem_overlap (默认：true)**：检查输入和输出 tensor 是否存在内存重叠，若重叠则报错。
- **allow_cpu_scalars (默认：false)**：允许 CPU 标量（通常为包装数值）作为 kernel 参数传递给设备代码（如 CUDA 内核）。
- **is_reduction (默认：false)**：标识 TensorIterator 是否用于归约操作，如求和或求最大值。
- **resize_outputs (默认：true)**：允许根据运算需求调整输出 tensor 的大小。
- **check_all_same_dtype (默认：true)**：确保所有输入与输出 tensor 的数据类型一致，若不一致可能需要进行类型提升或转换。
- **check_all_same_device (默认：true)**：确保所有 tensor 均位于相同设备上。
- **enforce_safe_casting_to_output (默认：false)**：启用后会检查用于计算的 common_dtype 是否能安全地转换为输出 tensor 的数据类型，以防止因不安全的类型转换而导致数据损坏。
- **enforce_linear_iteration (默认：false)**：若为 true，则按照 C 风格连续内存（最后一个维度最快）的迭代顺序进行遍历；这种顺序可能效率较低，且可能阻碍向量化，仅当核函数依赖正确迭代顺序时使用。
- **promote_inputs_to_common_dtype (默认：false)**：若设置，该配置会先计算出 common_dtype，然后将所有输入 tensor 提升为 common_dtype 后再执行运算。
- **promote_integer_inputs_to_float (默认：false)**：若启用，当 common_dtype 为整数类型时会将其提升为默认浮点类型。例如，`int_tensor / 3` 最终得到 float_tensor。
- **cast_common_dtype_to_outputs (默认：false)**：若为 true，计算会先在一个临时的 common_dtype 中进行，随后再转换回输出 tensor 的原始数据类型。

调用 `config.build()` 后，内部会执行 TensorIterator 的 `build()` 方法：

```c++
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // ...
  // 将 config 中的 tensors_ 转移到迭代器的 operands_ 中
  populate_operands(config);
  // 设置输出和读写标志
  mark_outputs();
  // 检查输出内存是否重叠
  compute_mem_overlaps(config);
  // 计算命名信息
  compute_names(config);
  // 根据广播规则计算形状（赋值给 shape_）
  compute_shape(config);
  // 标记需要调整大小的输出
  mark_resize_outputs(config);
  // 根据输入输出计算设备和数据类型
  compute_types(config);
  // 尝试快速设置输出 tensor
  if (!fast_set_up(config)) {
    // 计算每个 tensor 的 stride（以字节为单位）
    compute_strides(config);
    // 重排 shape 与 stride，使得 strides[0] 为最快移动维度（按升序排列）
    reorder_dimensions();
    // 如果输出未定义则分配输出
    allocate_or_resize_outputs();
    // 尽可能合并相邻的维度
    if (!is_meta_) coalesce_dimensions();
  }
  if (is_meta_) return;
  // ...
  for (auto& op : operands_) {
    TORCH_INTERNAL_ASSERT(op.tensor_base().defined());
    op.data = op.tensor_base().data_ptr();
  }
}
```

这里我们不再详细讲解广播逻辑，而重点关注 `fast_set_up`、`stride_bytes` 以及 PyTorch 如何进行输出分配/调整和维度合并。

#### 快速设置（fast_set_up）

```c++
// aten/src/ATen/TensorIterator.cpp

// 尝试快速设置，以避免不必要的维度重排或合并
bool TensorIteratorBase::fast_set_up(const TensorIteratorConfig& config) {
  FastSetupType setup_type = compute_fast_setup_type(config);
  if (setup_type == FastSetupType::NONE) return false;

  // 根据 setup_type 分配输出内存，内存格式取决于 setup_type
  switch (setup_type) {
    case FastSetupType::CONTIGUOUS:
      {
        for (const auto i : c10::irange(num_outputs_)) {
          auto& op = operands_[i];
          // ...
          set_output_raw_strided(i, shape_, {}, original_options(op).memory_format(MemoryFormat::Contiguous), names_);
        }
        break;
      }
    // 其他类型例如 channels last
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", c10::to_string((int)setup_type));
  }
  // 如果能够快速设置，则合并维度到 1
  if (ndim() > 1){
    has_coalesced_dimensions_ = true;
  }
  if (ndim() >= 1) {
    shape_[0] = numel();
    shape_.resize(1);
  }
  for (auto& op : operands_ ) {
    auto element_size_in_bytes = op.tensor_base().element_size();
    op.stride_bytes.resize(ndim());
    if (ndim() > 0) {
      op.stride_bytes[0] = element_size_in_bytes;
    }
  }
  return true;
}
```

`compute_fast_setup_type` 会检测所有 tensor 的内存布局，如果全部连续，则返回 `FastSetupType::CONTIGUOUS`，进而可将所有维度合并为一个线性存储。

若无法快速设置，则需要计算 `stride_bytes`、重排维度，再合并维度为 2D：

```c++
if (!fast_set_up(config)) {
  compute_strides(config);
  reorder_dimensions();
  allocate_or_resize_outputs();
  if (!is_meta_) coalesce_dimensions();
}
```

#### 计算 stride_bytes

```c++
// aten/src/ATen/TensorIterator.cpp

// 计算每个 tensor 的 stride_bytes
// 例如，一个 shape 为 [2, 3] 且 strides 为 [3, 1] 的 float tensor，得到的 stride_bytes 为 [12, 4]
void TensorIteratorBase::compute_strides(const TensorIteratorConfig& config) {
  for (auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      // ...
      for (const auto i : c10::irange(original_shape.size())) {
        if (original_shape[i] == 1 && shape_[offset + i] != 1) {
          op.stride_bytes[offset + i] = 0;
        } else {
          op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
        }
      }
    }
  }
}
```

#### 重排维度

```c++
// aten/src/ATen/TensorIterator.cpp

// 根据 stride_bytes 升序对各维度进行重排，使得 strides[0] 为最快移动的维度。
// 例如：一个输入 tensor，shape=[3, 2] 且 stride_bytes=[8, 4]，重排后为 [2, 3]，stride_bytes=[4, 8]
void TensorIteratorBase::reorder_dimensions() {
  perm_.resize(ndim());
  // 初始化 perm，依次为 n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // should_swap 用于比较两个维度的先后顺序
  auto should_swap = [&](size_t dim0, size_t dim1) { /* ... */ };
  
  // 根据 should_swap 得到最终的排列顺序
  for (const auto i : c10::irange(1, ndim())) {
    int dim1 = i;
    for (int dim0 = i - 1; dim0 >= 0; dim0--) {
      int comparison = should_swap(perm_[dim0], perm_[dim1]);
      if (comparison > 0) {
        std::swap(perm_[dim0], perm_[dim1]);
        dim1 = dim0;
      } else if (comparison < 0) {
        break;
      }
    }
  }

  // 根据 perm 重排 shape 和 stride_bytes
  permute_dimensions(perm_);
}
```

#### 输出的分配或调整

当输出 tensor 未定义或需要调整大小时，使用 `invert_perm` 计算原始形状和 stride_bytes，再调用 `set_output_raw_strided`：

```c++
// aten/src/ATen/TensorIterator.cpp

void TensorIteratorBase::allocate_or_resize_outputs() {
  for (const auto i : c10::irange(num_outputs_)) {
    auto& op = operands_[i];
    if (!op.tensor_base().defined() || op.will_resize) {
      // ...
      int element_size = elementSize(op.target_dtype);
      // 初始化输出的 stride_bytes
      op.stride_bytes = compatible_stride(element_size);
      // 检查当前排列是否为完全反序（例如连续输出）
      bool inverted = true;
      for (const auto j : c10::irange(ndim())) {
        if (perm_[j] != ndim() - j - 1) {
          inverted = false;
          break;
        }
      }
      // 反转 reorder_dimensions 产生的排列
      auto tensor_shape = invert_perm(shape_);
      if (inverted) {
        set_output_raw_strided(i, tensor_shape, {}, original_options(op), names_);
      } else {
        auto tensor_stride = invert_perm(op.stride_bytes);
        for (const auto dim : c10::irange(ndim())) {
          tensor_stride[dim] /= element_size;
        }
        set_output_raw_strided(i, tensor_shape, tensor_stride, original_options(op), names_);
      }
      op.current_dtype = op.target_dtype;
    } else if (op.tensor_base().defined()) {
      // 即使不需要调整大小，也必须调用 set_output_raw_strided 以设置 guard 并传播 names
      set_output_raw_strided(i, op.tensor_base().sizes(), {}, original_options(op), names_);
    }
  }
}
```

#### 合并相邻维度

通过 `coalesce_dimensions` 将相邻可以合并的维度合并，从而降低后续计算的维度复杂度：

```c++
// aten/src/ATen/TensorIterator.cpp

// 尝试合并相邻维度。例如：
// shape_ = [64, 4, 5, 1]，output.stride_bytes = [4, 256, 1024, 5120]，
// input.stride_bytes = [80, 4, 16, 5120]
// 合并后 shape_ = [64, 20]，output.stride_bytes = [4, 256]，input.stride_bytes = [80, 4]
void TensorIteratorBase::coalesce_dimensions() {
  if (ndim() <= 1) return;

  // 若满足以下条件可合并两相邻维度：
  // shape[n] / shape[n+1] == 1 或者
  // 对所有 tensor 都满足：shape[n] * stride[n] == stride[n + 1]
  auto can_coalesce = [&](int dim0, int dim1) { /* ... */ };

  // 合并后将 dim0 的 stride 替换为 dim1 的 stride
  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  int prev_dim = 0;
  for (const auto dim : c10::irange(1, ndim())) {
    if (can_coalesce(prev_dim, dim)) {
      if (shape_[prev_dim] == 1) {
        replace_stride(prev_dim, dim);
      }
      shape_[prev_dim] *= shape_[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        replace_stride(prev_dim, dim);
        shape_[prev_dim] = shape_[dim];
      }
    }
  }
  
  // 缩减 shape_ 和 stride_bytes
  shape_.resize(prev_dim + 1);
  for (const auto i : c10::irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}
```

### 3.2 调用 TensorIterator 进行计算

当 TensorIterator 构建完成后，它会根据广播规则推断输出形状、分配输出，并合并维度。接下来，我们就可以进行具体计算了——以 `sum` 操作为例：

```c++
// aten/src/ATen/native/cpu/SumKernel.cpp

// 为了更高精度定制的浮点数求和
template <bool ignore_nan, typename scalar_t>
void cascade_sum(TensorIterator &iter) {
  iter.output_base().fill_(scalar_t(0));
  iter.parallel_reduce(
    [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
      int64_t in_strides[] = { strides[1], strides[3] };
      int64_t out_strides[] = { strides[0], strides[2] };

      // 利用 stride_bytes 与数据指针计算求和……
    });
}
```

在这一过程中，我们将一个匿名函数作为 `loop2d_t` 参数传递给 `iter.parallel_reduce()`。  
`parallel_reduce` 内部会将数据分割为若干范围（range）以并行计算，若数据量较小时则走串行路径。  
例如，当 `numel < GRAIN_SIZE` 时，函数会调用 `serial_for_each`：

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) return;

  const auto ntensors = this->ntensors();
  const auto ndim = this->ndim();

  c10::SmallBuffer<char*, 4> ptrs(ntensors);
  c10::SmallBuffer<int64_t, 8> strides(ntensors * std::max(ndim, 2));
 
  // 将所有 tensor 的数据指针转换为 char* 类型并存储于 ptrs
  at::get_base_ptrs(ptrs.data(), operands_);
  // 提取每个 operand 的 stride_bytes 并存入 strides
  at::get_strides(strides.data(), operands_, ndim);
  at::internal::serial_for_each(
      shape_, strides, ptrs.data(), ptrs.size(), loop, range);
}
```

`serial_for_each` 会根据当前 range 利用 **DimCounter** 将数据划分为多个 batch，然后将对应的指针传给匿名函数执行计算。

更多关于如何计算每个 batch 内的数据指针和步长的细节，可参考我的[详细文章](../deep_dive_into_contiguous_3/#11-underlying-operation-of-cpu_kernel_vec)。

当所有 batch 的数据都处理完毕后，`cascade_sum` 便完成了求和操作，最终返回结果。

---

## 参考资料

- [PyTorch 源码仓库](https://github.com/pytorch/pytorch)
- [MicroTorch 项目](https://github.com/yewentao256/MicroTorch)
