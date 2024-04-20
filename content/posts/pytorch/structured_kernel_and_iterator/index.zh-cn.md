---
title: "Exploring Structured Kernel and Tensor Iterator in PyTorch"
date: 2023-12-03T13:56:11+08:00
categories: ["pytorch"]
summary: "在本文中，我们将深入探讨 PyTorch 中的**结构化内核（Structured Kernel）**和**张量迭代器（TensorIterator）**，包括在Structured Kernel中的`meta`、`impl`函数及 TensorIterator 的构建和算子计算调用的过程。"
---

## Summary

在本文中，我们将深入探讨 PyTorch 中的**结构化内核（Structured Kernel）**和**张量迭代器（TensorIterator）**，包括在Structured Kernel中的`meta`、`impl`函数及 TensorIterator 的构建和算子计算调用的过程。

## 等待被翻译

非常抱歉，看起来这篇博文还没有被翻译成中文，请等待一段时间

## 1. Introduction

In a previous article, we briefly introduced the concept of the structured kernel in [Structured Kernel and Stub](../deep_dive_to_autograd_1/#structured-kernel-and-stub). We also delved into the foundation of structured kernel, the **TensorIterator**, in [Copy and TensorIterator](../deep_dive_into_contiguous_3/#9-the-copy_-operator-and-tensoriterator).

This article aims to intertwine these two concepts, offering a comprehensive exploration into the implementation of both the structured kernel and the TensorIterator.

Let's start with code:

```py
import torch

A = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
B = A.sum(dim=0, keepdim=True)
```

Upon execution, the `sum_dim_IntList` within `TensorBody.h` is invoked.

```c++
// torch/include/ATen/core/TensorBody.h
inline at::Tensor Tensor::sum(at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) const {
    return at::_ops::sum_dim_IntList::call(const_cast<Tensor&>(*this), dim, keepdim, dtype);
}
```

After dispatch, we reach the CPU's structured kernel implementation of `sum_dim_IntList`. The exact location might vary depending on the compilation options used.

Note: If you're intrigued but not yet familiar with the dispatch process, I recommend starting with [Dispatching Contiguous Operators](../deep_dive_into_contiguous_1/#4-dispatch-contiguous-operator-find-schema-and-call-kernel) for a foundational understanding.

## 2. Structured Kernel

Note: Compiling PyTorch is necessary to view the generated code.

```c++
// build/aten/src/ATen/RegisterCPU.cpp
at::Tensor wrapper_CPU_sum_dim_IntList(const at::Tensor & self, at::OptionalIntArrayRef dim, bool keepdim, c10::optional<at::ScalarType> dtype) {
structured_sum_out_functional op;
op.meta(self, dim, keepdim, dtype);
op.impl(self, dim, keepdim, dtype, op.outputs_[0]);
return std::move(op.outputs_[0]);
}
```

The structured kernel framework in PyTorch has three primary components:

1. **Op Declaration**: Declaring an op, like `structured_sum_out_functional`.
2. **Op Meta**: Prepares the operation.
3. **Op Implementation**: Executes the computation based on **TensorIterator**.

After these steps, the computed result is obtained and returned.

### 2.1. Op Declaration: MetaBase

Let's see `structured_sum_out_functional` first

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

This operation originates from `at::native::structured_sum_out`.

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

Through code analysis, it's apparent that declaring an operation involves defining an instance of the **MetaBase** class.

**MetaBase** serves as the foundational class for all structured kernel classes. This includes both `structured_sum_out_functional` and **TensorIteratorBase**.

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

  // Alias for `set_output_strided`, but with contiguous strides.
  void set_output_contiguous(/* params */) {
    auto strides = c10::contiguous_strides(sizes);
    set_output_strided(output_idx, sizes, strides, options, names);
  }

  // Returns a reference to an undefined tensor if there is no presupplied output
  const Tensor& maybe_get_output() {
    return maybe_get_output(0);
  }
  virtual ~MetaBase() = default;
};
```

Key functions within **MetaBase** that are typically overridden include:

1. `set_output_raw_strided`: Employed when the kernel is capable of handling outputs with arbitrary strides.
2. `set_output_strided`: Utilized in other cases. (For contiguous strides, `set_output_contiguous` is invoked, which in turn calls this function.)

In the case of `structured_sum_out_functional`, these functions are overridden with `outputs_[output_idx] = create_out(sizes, strides, options);`. The invocation of these functions will be further explored subsequently.

### 2.2. `Op.meta`

Let's look back to the structured kernel, the second step is to invoke `op.meta(self, dim, keepdim, dtype);`

```c++
// aten/src/ATen/native/ReduceOps.cpp

// void structured_sum_dim_IntList::meta
TORCH_META_FUNC2(sum, dim_IntList)
(const Tensor& self, OptionalIntArrayRef opt_dim, bool keepdim, optional<ScalarType> opt_dtype) {
  // `maybe_get_output()` gets an undefined output
  // `infer_dtype_from_optional` infers the dtype from self and output(if defined)
  auto out_dtype = infer_dtype_from_optional(self, opt_dtype, maybe_get_output());
  resize_reduction(*this, self, opt_dim, keepdim, out_dtype);
}
```

Then we call `resize_reduction`:

```c++
// aten/src/ATen/native/ReduceOpsUtils.h
static void resize_reduction(
    impl::MetaBase& meta,
    const Tensor& self,
    OptionalIntArrayRef opt_dims,
    bool keepdim,
    ScalarType out_dtype) {
  // Generate DimVector from opt_dims (if defined) or ndim 
  DimVector dims_ = at::native::make_dim_vector(opt_dims, self.dim());
  // "Wraps" each dim in-place to support negative index
  maybe_wrap_dims(dims_, self.dim());
  // Infer the output shape for sum with `dims_` (based on std::bitset)
  auto shape = get_reduction_shape(self, dims_, keepdim);
  // Using inferred shape to declare an output
  // After doing this, the output is allocated and defined
  meta.set_output_raw_strided(0, shape, {}, self.options().dtype(out_dtype));
  // ...
}
```

Within the `op.meta` function, a critical step is the use of `meta.set_output_raw_strided` to define the output of structured kernel.

Once the output is successfully allocated, we are ready to call `op.impl` function.

### 2.3. `op.impl`

The third step is where the actual computation takes place.

```c++
// aten/src/ATen/native/ReduceOps.cpp

// void structured_sum_out::impl
TORCH_IMPL_FUNC(sum_out) (/* params... */) {
  auto iter = meta::make_reduction_from_out_ty(self, result, opt_dim, keepdim, result.scalar_type());
  if (iter.numel() == 0) {
    result.zero_();
  } else {
    sum_stub(iter.device_type(), iter);
  }
}
```

Here, `meta::make_reduction_from_out_ty` is utilized to construct a **TensorIterator**:

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
  // View the result (expanding one dim if the keepdim is false)
  auto viewed_result = at::native::review_reduce_result(result, ndim, mask, keepdim);
  if (self.scalar_type() == in_dtype) {
    return TensorIterator::reduce_op(viewed_result, self);
  }
  return TensorIterator::reduce_op(viewed_result, self.to(in_dtype));
}
```

Why we need to **view** the result?

When `keepdim` is set to false, the resultant shape (previously inferred) is reduced, which does not align with the shape expected by TensorIterator. So the result must be appropriately **viewed** accordingly (expanding one dim), as if `keep_dim` were true.

Post invocation of `TensorIterator::reduce_op`, a **TensorIterator** instance is established. We will introduce this later.

Next, the `sum_stub` is called. After dispatching through the device, we reach the kernel of the sum:

```c++
// aten/src/ATen/native/cpu/SumKernel.cpp
void sum_kernel_impl(TensorIterator &iter) {
  // if bool dtype ...
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      ScalarType::BFloat16, ScalarType::Half, iter.dtype(), "sum_cpu", [&] {
    cascade_sum</*ignore_nan=*/false, scalar_t>(iter);
  });

  // Note: The AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(...) is same as:
  [&] {
    const auto& the_type = iter.dtype();
    constexpr const char* at_dispatch_name = "sum_cpu";
    at::ScalarType _st = ::detail::scalar_type(the_type);
    switch (_st) {
      case at::ScalarType::Double:  { /* ... */ }
      case at::ScalarType::Float: {
        do {
          // some check logic ...
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

The execution of the computation is then carried out by `cascade_sum`. A key aspect here is the passing of an anonymous function to the **TensorIterator**'s `parallel_reduce` member function, which will also be discussed in more detail later.

```c++
// Custom floating point sum for better accuracy
template <bool ignore_nan, typename scalar_t>
void cascade_sum(TensorIterator &iter) {
  iter.output_base().fill_(scalar_t(0));
  iter.parallel_reduce(
    [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
      /* anonymous function implementation... */
    });
}
```

Once the `cascade_sum` function completes its execution, the sum result is obtained. All things done, return back to user.

Having grasped the basic mechanism of the **structured kernel**, we can now delve deeper into the **TensorIterator**.

## 3. TensorIterator

Back to our structured kernel:

```c++
// build/aten/src/ATen/RegisterCPU.cpp
at::Tensor wrapper_CPU_sum_dim_IntList(/* params */) {
structured_sum_out_functional op;
op.meta(self, dim, keepdim, dtype);
op.impl(self, dim, keepdim, dtype, op.outputs_[0]);
return std::move(op.outputs_[0]);
}
```

In `op.impl`, we utilize **TensorIterator** for the `sum` operation. But how does **TensorIterator** achieve this?

Utilizing a TensorIterator involves two main steps:

1. Build a TensorIterator, preparing for calculation.
2. call for calculation, use `cpu_kernel` / `gpu_kernel` or `parallel_reduce`

Note: The **TensorIterator** system is intricate. We won't delve into all implementation specifics here. For a deeper understanding, explore the [PyTorch source code](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/TensorIterator.cpp), or check out my simplified version in [MicroTorch](https://github.com/yewentao256/MicroTorch/blob/main/src/core/tensorIterator.cpp).

### 3.1. Constructing a Tensor Iterator

There are multiple ways to build a tensor; let's consider `reduce_op`:

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

We create a **TensorIteratorConfig** instance, set attributes, then invoke `build()` to obtain a TensorIterator.

```c++
// torch/include/ATen/TensorIterator.h
class TORCH_API TensorIteratorConfig final {
 public:
  // ...
  // Important: the outputs have to be added before the inputs.
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
  bool check_mem_overlap_ = true;

};
```

The config properties:

- **check_mem_overlap (default: true)**: checks for memory overlap between input and output tensors. If detected, an error is thrown.

- **allow_cpu_scalars (default: false)**: When set to true, this allows CPU scalar values (**Wrapped number** usually) to be passed as kernel parameters when executing device code, like within CUDA kernels.

- **is_reduction (default: false)**: Indicates whether the TensorIterator is being used for reduction operations, such as summing or finding maximum values.

- **resize_outputs (default: true)**: This allows output tensors to be resized as needed to match the expected output of the operation.

- **check_all_same_dtype (default: true)**: This ensures that all input and output tensors have the same data type. If they differ, type promotion or conversion might be necessary to proceed with the operation.

- **check_all_same_device (default: true)**: Verifies that all tensors are located on the same device.

- **enforce_safe_casting_to_output (default: false)**: When enabled, this checks that the `common_dtype_` used in computations can be safely cast to the output tensor’s data type, safeguarding against data corruption through unsafe type conversions.

- **enforce_linear_iteration (default: false)**: If true, tensor iteration follows a C-style contiguous memory layout (last dimension iterates fastest). This iteration order can be less efficient and may even prevent vectorization. So only use if the correctness of your kernel depends on it.

- **promote_inputs_to_common_dtype (default: false)**: If set, the `common_dtype_` is computed and all input tensors are promoted to `common_dtype_` before the operation.

- **promote_integer_inputs_to_float (default: false)**: If enabled, and if the `common_dtype_` of the iterator is an integer type, it will be promoted to a default floating-point type. Eg. `int_tensor / 3 = float_tensor`

- **cast_common_dtype_to_outputs (default: false)**: If true, the results of operations are first calculated in a temporary common data type, then converted back to the original data type of the output tensors.

The `config.build()` function internally calls `iterator.build()`:

```c++
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // ...
  // Transfers `tensors_` from config to the iterator's `SmallVector<OperandInfo, 4> operands_`
  populate_operands(config);
  // Set is_output and is_read_write flags on appropriate tensors
  mark_outputs();
  // Checks that the output memory does not overlap
  compute_mem_overlaps(config);
  // Compute outnames.
  compute_names(config);
  // Computes the shape of broadcasting. (setting `shape_` variable)
  compute_shape(config);
  // If output needs resizing (different from `shape_`), it's marked
  mark_resize_outputs(config);
  // Computes device (taking the first non-CPU device as common device) and dtype
  compute_types(config);
  // Attempts to quickly build output tensor
  if (!fast_set_up(config)) {
    // Computes the stride(stride_bytes) for each tensors
    compute_strides(config);
    // Re-order tensor's shape and stride, with `stride[0]` as the fastest progression dimension (stride ascending)
    reorder_dimensions();
    // allocate the output tensor if it's not provided
    allocate_or_resize_outputs();
    // Coalesce adjacent dimensions when possible
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

We previously discussed [broadcast logic](../introduction_to_broadcast/). Here, our focus is on `fast_set_up`, `stride_bytes`, and PyTorch's methods for output allocation/resizing and dimension `coalescing`.

First, for `fast_set_up`:

```c++
// aten/src/ATen/TensorIterator.cpp

// This function tries to do a fast setup to avoid needless reordering of dimensions or coalecsing
bool TensorIteratorBase::fast_set_up(const TensorIteratorConfig& config) {
  FastSetupType setup_type = compute_fast_setup_type(config);
  if (setup_type == FastSetupType::NONE) return false;

  // allocate memory for output, memory format depends on setup_type
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
    // ... channels last
    default:
      TORCH_INTERNAL_ASSERT(false, "Unsupported fast setup type", c10::to_string((int)setup_type));
  }
  // If we can do a fast setup, coalescing dimensions to 1
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
    if (ndim()>0) {
      op.stride_bytes[0] = element_size_in_bytes;
    }
  }
  return true;
}
```

We use `compute_fast_setup_type` to assess the memory layout of tensors. If all of the tensors are contiguous, we get a `FastSetupType::CONTIGUOUS`.

Then we can coalesce the dimensions to `1` directly, implying the tensor can be treated as linear storage.

If fast setup isn't feasible, we calculate `stride_bytes` for tensors, reorder the dimensions, then coalesce dimensions to **2D** for simpler computation.

```c++
  if (!fast_set_up(config)) {
    compute_strides(config);
    reorder_dimensions();
    allocate_or_resize_outputs();
    if (!is_meta_) coalesce_dimensions();
  }
```

Firstly, calculate the operation's `stride_bytes`:

```c++
// aten/src/ATen/TensorIterator.cpp

// Set the operation's `stride_bytes`
// Eg: a float tensor with shape[2, 3], strides[3, 1] and we get [12, 4]
void TensorIteratorBase::compute_strides(const TensorIteratorConfig& config) {
  for (auto& op : operands_) {
    if (op.tensor_base().defined() && !op.will_resize) {
      // ...
      for (const auto i : c10::irange(original_shape.size())) {
        if (original_shape[i] == 1 && shape_[offset + i] !=1) {
          op.stride_bytes[offset + i] = 0;
        } else {
          op.stride_bytes[offset + i] = original_stride[i] * element_size_in_bytes;
        }
      }
    }
  }
}
```

Next, reorder dimensions using `reorder_dimensions` and resize or allocate outputs with `allocate_or_resize_outputs()`.

Note: In `allocate_or_resize_outputs()`, if output requires resizing or is undefined, we utilize `invert_perm` to determine the original shape and strides, then configure the output using `set_output_raw_strided`.

```c++
// aten/src/ATen/TensorIterator.cpp

// Sort dimensions based on `stride_bytes` in ascending order.
// The fastest moving dimension is strides[0] instead of strides[ndim - 1].
// Eg: An input tensor with shape=[3, 2] -> [2, 3], stride_bytes=[8, 4] -> [4, 8]
void TensorIteratorBase::reorder_dimensions() {
  perm_.resize(ndim());
  // ...
  // initialize perm with n-1, n-2, ..., 1, 0
  std::iota(perm_.rbegin(), perm_.rend(), 0);

  // returns 1 if the dim0 should come after dim1, -1 if dim0 should come
  // before dim1, and 0 if the comparison is ambiguous.
  auto should_swap = [&](size_t dim0, size_t dim1) {/* ... */};
  
  // calculation for get a permute order
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

  // perform re-ordering of shape and strides
  permute_dimensions(perm_);
}

// If the output is not defined or marked `should_resize`, we will use `perm_`
// to compute the original shape and stride_bytes for it,
// then `set_output_raw_strided`
void TensorIteratorBase::allocate_or_resize_outputs() {
  for (const auto i : c10::irange(num_outputs_)) {
    auto& op = operands_[i];
    if (!op.tensor_base().defined() || op.will_resize) {
      // ...
      int element_size = elementSize(op.target_dtype);
      // initialize output's stride_bytes
      op.stride_bytes = compatible_stride(element_size);
      // check if permutation is an fully inverted order
      // for example: contiguous output
      bool inverted = true;
      for (const auto j : c10::irange(ndim())) {
        if (perm_[j] != ndim() - j - 1) {
          inverted = false;
          break;
        }
      }
      // Invert the permutation caused by reorder_dimensions.
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
      // Even if we don't need to resize, we still need to call
      // set_output_raw_strided so that we properly set guard and propagate names
      set_output_raw_strided(i, op.tensor_base().sizes(), {}, original_options(op), names_);
    }
  }
}
```

Lastly, `coalesce_dimensions` is applied to minimize dimensions, enhancing later computation efficiency.

```c++
// aten/src/ATen/TensorIterator.cpp

// Try coalescing the adjacent dims.
// For example:
// `shape_` = [64, 4, 5, 1], `output.stride_bytes` = [4, 256, 1024, 5120],
// `input.stride_bytes` = [80, 4, 16, 5120]
// Changes to `shape_` = [64, 20],
// `output.stride_bytes` = [4, 256], `input.stride_bytes` = [80, 4]
void TensorIteratorBase::coalesce_dimensions() {
  if (ndim() <= 1) return;

  // We can coalesce two adjacent dimensions if:
  // shape[n] / shape[n+1] == 1 or
  // shape[n] * stride[n] == stride[n + 1] for all of the tensors
  auto can_coalesce = [&](int dim0, int dim1) { /* ... */ };

  // replace all of the operand's stride at dim0 with its stride at dim1
  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  // Starting from the `prev_dim` pointer, traversing each dim afterwards,
  // trying to coalesce as many dimensions as possible.
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
  
  // shrink shape_ and stride_bytes
  shape_.resize(prev_dim + 1);
  for (const auto i : c10::irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}
```

### 3.2. Calling TensorIterator for Calculation

Once a **TensorIterator** instance is constructed, it infers the broadcast shape, allocates the output, and coalesces dimensions. With these preparations, we can proceed to the actual computation.

We continue with the `sum` operation as our example.

```c++
// aten/src/ATen/native/cpu/SumKernel.cpp

// Custom floating point sum for better accuracy
template <bool ignore_nan, typename scalar_t>
void cascade_sum(TensorIterator &iter) {
  iter.output_base().fill_(scalar_t(0));
  iter.parallel_reduce(
    [&](char** data, const int64_t* strides, int64_t size0, int64_t size1) {
      int64_t in_strides[] = { strides[1], strides[3] };
      int64_t out_strides[] = { strides[0], strides[2] };

      // Use stride_bytes and pointers to calculate sum ...
    });
}
```

In this process, we provide an anonymous function as a `loop2d_t` parameter to `iter.parallel_reduce()`:

```c++
// aten/src/ATen/native/TensorIteratorReduce.cpp
void TensorIteratorBase::parallel_reduce(loop2d_t loop) {
  // ...
  int64_t numel = this->numel();
  if (numel < at::internal::GRAIN_SIZE || at::get_num_threads() == 1 ||
      at::in_parallel_region()) {
    serial_for_each(loop, {0, numel});
  } else if (use_two_pass_reduction(*this)) {
    // ...
  } else {
    // ...
  }
}
```

PyTorch typically employs a parallel mechanism for computation. Data are segmented into several `ranges` within `parallel_reduce`. We won't delve into the specifics of this segmentation here.

Assuming `numel < GRAIN_SIZE`, we examine the `serial_for_each` function:

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) return;

  const auto ntensors = this->ntensors();
  const auto ndim = this->ndim();

  c10::SmallBuffer<char*, 4> ptrs(ntensors);
  c10::SmallBuffer<int64_t, 8> strides(ntensors * std::max(ndim, 2));
 
  // convert data ptrs to char* type, and store in `tensor_ptrs`.
  at::get_base_ptrs(ptrs.data(), operands_);
  // extract op.stride_bytes and store in `strides`
  at::get_strides(strides.data(), operands_, ndim);
  at::internal::serial_for_each(
      shape_, strides, ptrs.data(), ptrs.size(), loop, range);
}

// torch/include/ATen/TensorIteratorInternal.h
inline void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t ntensors,
    typename TensorIteratorBase::loop2d_t loop,
    Range range) {
  const auto ndim = shape.size();
  if (ndim <= 1) {
    // ...
  } else {
    // `ptrs` stores the addresses that need to be processed in current batch.
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    // DimCounter divides range into several parts for calculation
    auto counter = DimCounter(shape, range);
    // `is_done` judges whether the offset is greater than range.end
    while (!counter.is_done()) {
      // Calculating the starting address of each tensor under the current batch
      get_data_ptrs(
          ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);
      // Get the steps that should be processed in current batch.
      // Try to fetch the **maximum** range of steps
      auto step = counter.max_2d_step();
      // call for the anonymous function to calculate the result
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      // updates offset and dim_offsets according to the steps we fetched
      counter.increment(step);
    }
  }
}
```

`serial_for_each` primarily functions to divide the data in the current `range` into multiple **batches**, guided by **DimCounter**. It then passes the data pointers to the anonymous function for computation.

This discussion omits the intricate calculations of pointers and addresses for specific batches. Those interested in a deeper dive can refer to my detailed article [here](../deep_dive_into_contiguous_3/#11-underlying-operation-of-cpu_kernel_vec), which presents a comprehensive example for clarity.

Once all batches have been processed, the `cascade_sum` call concludes, thereby completing the sum kernel calculation.

## References

- [PyTorch](https://github.com/pytorch/pytorch)
- [MicroTorch](https://github.com/yewentao256/MicroTorch)
