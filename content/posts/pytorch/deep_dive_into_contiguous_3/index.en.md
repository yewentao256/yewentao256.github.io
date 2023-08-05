---
title: "PyTorch Under the Hood: A Deep Dive into the Contiguous Operator(3)"
date: 2023-05-13T09:53:09+08:00
categories: ["pytorch"]
summary: "Uncover the inner workings of PyTorch through a deep dive into the `contiguous` operator, from its Python interface to its dispatching and registration process, and finally how it is executed."
---

## Summary

Uncover the inner workings of PyTorch through a deep dive into the `contiguous` operator, from its Python interface to its dispatching and registration process, and finally how it is executed.

## 9. The `copy_` operator and TensorIterator

In the clone operator, we call the `copy_` operator after creating a tensor with a specified memory format.

```c++
// aten/src/ATen/native/Copy.cpp
Tensor& copy_(Tensor& self, const Tensor& src, bool non_blocking) {
  // ...
  if (src._is_zerotensor()) {
    return self.zero_();
  }
  copy_impl(self, src, non_blocking);
  return self;
}

static Tensor & copy_impl(Tensor & self, const Tensor & src, bool non_blocking) {
  // ...
  if (self.is_same(src)) {
    return self;
  }
  
  // ... 
  // If self and src are views of the same storage
  const bool is_same_data = (
      self.is_alias_of(src) &&                          // same storage
      self.storage_offset() == src.storage_offset() &&  // same storage offset
      self.strides().equals(src.strides()) &&
      self.sizes().equals(src.sizes()) &&
      self.scalar_type() == src.scalar_type()
    );
  if (is_same_data) {
    return self;
  }

  // We construct an `at::TensorIterator` which delineates inputs and outputs to handle device, dtype
  auto iter = TensorIteratorConfig()
    // The tensor is stored in the `SmallVector<c10::MaybeOwned<TensorBase>, 4> tensors_` of TensorIteratorConfig
    // Note the order here, output is added first, to ensure it is at the top of the list
    .add_output(self)
    .add_input(src)
    .resize_outputs(false)    // Setting the variable, same below
    .check_all_same_dtype(false)
    .check_all_same_device(false)
    .build();   // Creates a new TensorIterator, using the config built above

  if (iter.numel() == 0) {
    return self;
  }

  DeviceType device_type = iter.device_type(0);
  if (iter.device_type(1) == kCUDA) {
    device_type = kCUDA;
  } else if (iter.device_type(1) == kHIP) {
    device_type = kHIP;
  } else if (iter.device_type(1) == kMPS) {
    device_type = kMPS;
  }

  // ...
  copy_stub(device_type, iter, non_blocking);
  return self;
}
```

Let's dive into how **TensorIterator** is built

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // Transfers `tensors_` from config to the iterator's `SmallVector<OperandInfo, 4> operands_`
  populate_operands(config);
  // Sets flags such as `is_output`, to determine whether the input and output are the same tensor (inplace operation)
  mark_outputs();
  // Checks that the output memory does not overlap, and does not share storage with the input
  compute_mem_overlaps(config);
  // Computes the output name
  compute_names(config);
  // Computes the shape of broadcasting.
  // The logic is to first take the shape of the output and store it as `shape_`.
  // If the shape of the input tensor is different from `shape_`, a new shape is inferred, as described below.
  compute_shape(config);
  // If output needs resizing (different from `shape_`), it's marked
  mark_resize_outputs(config);
  // Computes device (taking the first non-CPU device as common device) and dtype
  // (taking the dtype of the first input tensor as `conmon_dtype_`, and the dtype of the first output tensor as `output_dtype_`)
  compute_types(config);
  // Attempts to quickly build output tensor
  // e.g., if all tensors are contiguous, channels last, then it can quickly infer output/resize (if needed), set name, etc.
  if (!fast_set_up(config)) {
    // Computes the stride after each tensor's broadcasting (in fact, it calculates `op.stride_bytes` (stride * element_size).
    // For example, here, `shape_` is [1, 64, 5, 4], `output.stride_bytes` = [5120, 4, 1024, 256] (NHWC)
    compute_strides(config);
    // Here, the tensor's shape and stride are reordered, with `stride[0]` as the fastest progression dimension (stride ascending)
    // As in this example, `shape_` changes to [64, 4, 5, 1], `output.stride_bytes` = [4, 256, 1024, 5120]
    reorder_dimensions();
    // If the output is not defined, it's allocated here
    allocate_or_resize_outputs();
    // If each tensor's corresponding dim size is 1 or `shape[n] * stride[n] == stride[n + 1]`, merge adjacent dimensions.
    // Why merge adjacent dimensions? This can increase addressing efficiency and facilitate subsequent stride traversal. We'll further expand on this in the code below.
    if (!is_meta_) coalesce_dimensions();
  }

  // ...
}
```

Now, let's further expand on how broadcasting is computed:

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
  // eg: a = {2, 1, 3}, b = {4, 3}
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

  // eg: a = {2, 1, 3}, b = {4, 3}, and we get {2,4,3} finally
  return expandedSizes;
}
```

Next, let's expand on how `coalesce_dimensions` merges dimensions:

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }

  // If dim == 1 or shape[n] * stride[n] == stride[n + 1], then it can be merged
  // Note that all input and output tensors must meet the conditions simultaneously
  auto can_coalesce = [&](int dim0, int dim1) {
    auto shape0 = shape_[dim0];
    auto shape1 = shape_[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      if (shape0 * stride[dim0] != stride[dim1]) {
        return false;
      }
    }
    return true;
  };

  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  // The logic here is dual-pointer, starting from the `prev_dim` pointer,
  // traversing each dimension afterwards, and trying to merge as many dimensions as possible
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

  // Finally resize. In our example tensor:
  // `shape_` = [64, 4, 5, 1], `output.stride_bytes` = [4, 256, 1024, 5120], `input.stride_bytes` = [80, 4, 16, 5120]
  // Changes to `shape_` = [64, 20], `output.stride_bytes` = [4, 256], `input.stride_bytes` = [80, 4]
  shape_.resize(prev_dim + 1);
  for (const auto i : c10::irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}
```

After merging adjacent dimensions, the **TensorIterator** is constructed. Note that there are no `input` and `output` parameters at this point, and all subsequent operations are based on this iterator (**pointer and strides_bytes**).

## 10. Copy Operator: Kernel Execution

Going back to the previous call, after one round of dispatch, `copy_stub` calls `copy_kernel`.

```c++
// aten/src/ATen/native/cpu/CopyKernel.cpp
void copy_kernel(TensorIterator& iter, bool /*non_blocking*/) {
  ScalarType dtype = iter.dtype(0);
  // ...
  auto strides_out = iter.strides(0);
  auto strides_in = iter.strides(1);
  if (dtype == iter.dtype(1)) {
    copy_same_dtype(iter, requires_conj, requires_neg);
  } else if (/* bfloat16 */) {
    float_bfloat16_copy_kernel(iter, requires_neg);
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::ComplexHalf, ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, dtype, "copy_", [&] {
      using dest_t = scalar_t;
      AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(ScalarType::ComplexHalf, ScalarType::Half, ScalarType::Bool, ScalarType::BFloat16, iter.dtype(1), "copy_", [&] {
        if (iter.has_contiguous_first_dim()) {
          TORCH_INTERNAL_ASSERT(iter.ninputs() == 1);
          TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

          iter.for_each([](char **data, const int64_t *strides, int64_t size) {
            auto src = reinterpret_cast<const scalar_t*>(data[1]);
            auto dst = reinterpret_cast<dest_t*>(data[0]);
            at::vec::convert(src, dst, size);
          });
        } else {
          cpu_kernel(iter, [](scalar_t x) -> dest_t {
            return c10::convert<dest_t>(x);
          });
        }
      });
    });

    // ...
  }
}
```

If the types are inconsistent, it goes to `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4`, which is a switch macro that handles data types.

If **has_contiguous_first_dim** is true (all input tensors meet `stride[0] == elementsize`), then `iter.for_each` is directly called and a lambda function with `reinterpret_cast` type conversion is passed. Otherwise, `cpu_kernel` is called, which essentially also calls `iter.for_each` and passes a lambda function with `c10::convert`.

The difference between these two lambda functions is that the former just changes the way the data is interpreted, without changing the original data bit pattern, while the latter actually performs vectorized type conversion.

Let's focus on the case where the types are consistent (for `contiguous`, the types are definitely consistent here). In `copy_same_dtype`, we call the `direct_copy_kernel` function.

```c++
// aten/src/ATen/native/cpu/CopyKernel.cpp
void direct_copy_kernel(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    // ...
  } else if (dtype == ScalarType::ComplexHalf) {
    // ...
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
        kBool, kHalf, kBFloat16, dtype, "copy_kernel", [&] {
      cpu_kernel_vec(
          iter,
          [=](scalar_t a) -> scalar_t { return a; },
          [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; });
    });
  }
}
```

The code expanded after the `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3` macro is as follows:

```c++
// aten/src/ATen/native/cpu/CopyKernel.cpp
[&] {
      const auto& the_type = dtype;
      constexpr const char* at_dispatch_name = "copy_kernel";
      at::ScalarType _st = ::detail::scalar_type(the_type);
      switch (_st) {
        case at::ScalarType::Byte: { /* ... */ }
        case at::ScalarType::Char: { /* ... */ }
        case at::ScalarType::Int: { /* ... */ }
        case at::ScalarType::Long: { /* ... */ }
        case at::ScalarType::Short: { /* ... */ }
        case at::ScalarType::Double: { /* ... */ }
        case at::ScalarType::Float: {
            if constexpr (!at::should_include_kernel_dtype(
                              at_dispatch_name, at::ScalarType::Float)) {
                // error check
            }
          using scalar_t __attribute__((__unused__)) =
              c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>;
          return [&] {
            cpu_kernel_vec(
                iter,
                [=](scalar_t a) -> scalar_t { return a; },
                [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
                  return a;
                });
          }();
        }
        case at::ScalarType::ComplexDouble: { /* ... */ }
        case at::ScalarType::ComplexFloat: { /* ... */ }
        case kBool: { /* ... */ }
        case kHalf: { /* ... */ }
        case kBFloat16: { /* ... */ }
      } 
  }
```

At this point, the content of the copy kernel itself has been executed, and the two lambda functions are passed to `cpu_kernel_vec`.

```c++
cpu_kernel_vec(
    iter,
    [=](scalar_t a) -> scalar_t { return a; },
    [=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> {
      return a;
    });
```

## 11. Underlying Operation of `cpu_kernel_vec`

`cpu_kernel_vec` supports scalar and vectorized functions as parameters, let's delve into its details.

```c++
// aten/src/ATen/native/cpu/Loops.h
template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  // ... some check

  // make_vectorized_loop2d integrates scalar op and vectorized op
  // into an object `VectorizedLoop2d` for `for_each`
  iter.for_each(make_vectorized_loop2d(op, vop), grain_size);
  // The cast_outputs method is used to forcibly convert the output tensor according to the current data type
  iter.cast_outputs();
}

// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();
  if (numel == 0) {
    return;
  } else if (numel < grain_size || at::get_num_threads() == 1) {
    // If the number of elements is less than grain_size or the number of threads is 1, serial operations are used.
    return serial_for_each(loop, {0, numel});
  } else {
    // Otherwise, it is divided into multiple tasks of grain_size, and serial iteration is performed within each task.
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}

```

We continue to delve into the serial operation `serial_for_each`. Pay attention to the `operands_` variable, which is the collection of operands `SmallVector<OperandInfo, 4> operands_;` of **TensorIterator**, including input and output tensors, where output is definitely at the first place (this part was introduced in the previous section on the construction process of iterator).

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {
    return;
  }

  const auto ntensors = this->ntensors();
  // Since the dimensions were coalesced when the TensorIterator was constructed above, the ndim here is 2.
  const auto ndim = this->ndim();

  c10::SmallBuffer<char*, 4> ptrs(ntensors);
  // The total strides length here is 4.
  c10::SmallBuffer<int64_t, 8> strides(ntensors * std::max(ndim, 2));
  
  // Here, `operands_` is the tensor list of TensorIteratorBase.
  // `get_base_ptrs` gets the storage pointers of all tensors,
  // converts them to char* type, and stores them in the pointer list.
  at::get_base_ptrs(ptrs.data(), operands_);
  // `get_strides` stores the strides of all tensors in order in `strides` (from low dimension to high dimension).
  // The arrangement from low to high dimension is for easy value calculation below.
  // For example, in our case, the final `strides` is [4, 80, 256, 4] (note that the data type here is int64_t).
  // Where the stride of the output tensor is [4, 256], and the stride of the input tensor is [80, 4].
  at::get_strides(strides.data(), operands_, ndim);
  at::internal::serial_for_each(
      shape_, strides, ptrs.data(), ptrs.size(), loop, range);
}

// aten/src/ATen/TensorIteratorInternal.h
inline void serial_for_each(
    IntArrayRef shape,
    IntArrayRef strides,
    char** base_ptrs,
    size_t ntensors,
    typename TensorIteratorBase::loop2d_t loop,
    Range range) {
  const auto ndim = shape.size();
  // ...

  if (ndim <= 1) {
    // ...
  } else {
    // // Here, `ptrs` is created again, similar to the above declaration,
    // but the `ptrs` here stores the addresses that need to be processed in the current batch.
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    // DimCounter ensures that every element is processed.
    // `is_done` judges whether the offset is greater than range.end.
    while (!counter.is_done()) {
      get_data_ptrs(
          ptrs.data(), {base_ptrs, ntensors}, strides, counter.values);
      auto step = counter.max_2d_step();
      loop(ptrs.data(), strides.data(), step[0], step[1]);
      counter.increment(step);
    }
  }
}
```

What is **DimCounter** here?

```c++
// aten/src/ATen/TensorIteratorInternal.h
struct DimCounter {
  DimCounter(IntArrayRef shape, Range range);

  void increment(const std::array<int64_t, 2>& step);
  bool is_done() const;   // return offset >= range.end;
  std::array<int64_t, 2> max_2d_step() const;

  IntArrayRef shape;
  // `range` is the range of elements to be processed, like {0, numel()}.
  Range range;
  // The offset on each dimension, note that it's the offset of the elements, not stride.
  c10::SmallBuffer<int64_t, 4> values;  
  int64_t offset;  // The offset of the current element being processed.
};

DimCounter::DimCounter(IntArrayRef shape, Range range)
  : shape(shape)
  , range(range)
  , values(shape.size())
  , offset(range.begin) {
  std::fill(values.begin(), values.end(), 0);
  if (range.begin == 0) {
    return;
  }

  int64_t linear_offset = range.begin;
  int64_t ndim = values.size();
  for (const auto dim : c10::irange(ndim)) {
    int64_t size = shape[dim];
    if (size > 0) {
      // Let's take a new example, where `begin` = 1066670, `size` = [64, 2000, 10],
      // here `values` stores the remainder offset [46, 666, 8].
      // The offset here can be multiplied by the stride to calculate the total offset,
      // in order to directly find the begin location of the current range.
      values[dim] = linear_offset % size;
      linear_offset /= size;
    }
  }
  TORCH_INTERNAL_ASSERT(linear_offset == 0);
}
```

Note the `get_data_ptrs` method, here it gets the starting pointer of the current range and stores it in `ptrs` (`ptrs[0]` is the output pointer, the rest are input pointers).

```c++
// aten/src/ATen/TensorIteratorInternal.h
inline void get_data_ptrs(
    char** ptrs,
    ArrayRef<char*> base,
    IntArrayRef strides,
    IntArrayRef counter) {
  const int64_t ntensors = base.size();
  const int64_t ndim = counter.size();
  std::copy(base.begin(), base.end(), ptrs);
  // Here, the starting pointer of each tensor under the current range is calculated.
  // The algorithm is the sum of the product of the offset and stride byte on all dimensions.
  // For example, `offset` = [46, 666, 8], `output stride` (corresponding to `strides` dimension = [0,2,4]) = [4, 256, 512000].
  // The starting address is `base + 46 * 4 + 666 * 256 + 8 * 512000 = base + 4266680`.
  // Note that 4266680 / 4 is exactly our previous 1066670th element (`range.begin`).
  for (const auto dim : c10::irange(ndim)) {
    int64_t value = counter[dim];
    for (const auto arg : c10::irange(ntensors)) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}
```

Then it goes to the `max_2d_step` method. Here, regardless of the number of dimensions, only two dimensions are taken, i.e., the step `(m, n)` of the data that should be processed in the current batch is obtained. We hope to fetch the data as quickly as possible. In an ideal case (without offset), for example, if the shape is `[64, 2000, 10]`, we take step = `{64, 2000}`, so that we can fetch the data 10 times to finish.

However, due to the offset restriction, we may not be able to fetch as much as `{64, 2000}`, but after adjusting for alignment once by fetching a small amount of data `shape[0] - values[0]`, the first dimension is aligned, and then the first dimension can be fetched full 64. Similarly, the second dimension may need to be adjusted once after `shape[1] - values[1]` to align.

For example, when we get here for the first time, step = `{18, 1}`, then enter `loop` (**we will expand the loop logic in the following text**), call `increment` to update the offset to `1066670 + 18 = 1066688`, and update `values=[0, 667, 8]`

The second time we get here, the step is `{64, 1333}`, and we can fetch more data. Then call `increment` to update the offset to `1066688 + 64 * 1333 = 1152000`, and update `values=[0,0,9]`

The third time we get here, the offset is fully corrected, and the maximum step is `{64, 2000}`, then update the offset to `1280000`, and update values = `[0,0,0]`. Because our range is exactly `{1066670, 1280000}`, the call ends.

Note that there might be a shortage of data when fetching for the last time, then fetch the small batch of data `range.end - offset` to finish.

```c++
// aten/src/ATen/TensorIterator.cpp
std::array<int64_t, 2> DimCounter::max_2d_step() const {
  // Try to fetch the maximum range of data.
  // Note that if the offset is already close to end, fetch the remaining data through `range.end - offset`.
  int64_t step0 = std::min(shape[0] - values[0], range.end - offset);
  int64_t step1 = 1;
  if (step0 == shape[0] && !shape.empty()) {
    step1 = std::min(shape[1] - values[1], (range.end - offset) / shape[0]);
  }
  return {step0, step1};
}

void DimCounter::increment(const std::array<int64_t, 2>& step) {
  offset += step[0] * step[1];
  int64_t ndim = values.size();
  int64_t overflow = step[0];
  int i = 0;
  if (step[1] != 1) {
    // step[1] != 1 indicates that the first dimension is already adjusted, jump to the second
    TORCH_INTERNAL_ASSERT(step[0] == shape[0] && values[0] == 0);
    i = 1;
    overflow = step[1];
  }
  for (; i < ndim && overflow > 0; i++) {
    auto size = shape[i];
    auto prev = values[i];
    auto value = prev + overflow;
    // overflow marks, if overflow happens, next dimension should increment by 1
    if (value >= size) {
      overflow = 1;
      value -= size;
      TORCH_INTERNAL_ASSERT(value < size);
    } else {
      overflow = 0;
    }
    values[i] = value;
  }
  TORCH_INTERNAL_ASSERT(overflow == 0 || overflow == 1);
}
```

Finally, it calls the `loop` method. Note the `cpu_kernel_vec` above, the op passed in here is actually just the lambda expression `[=](scalar_t a) -> scalar_t { return a; },` and `[=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; }`

`loop2d` essentially performs a double loop according to step0 and step1:

- First loop through `step0`, if it can be vectorized (e.g., stride is exactly equal to type size), it will try to call `vectorized_loop` to unroll (similar to unrolling for loop, processing multiple data in one loop). Otherwise, call `basic_loop` to loop through `step0`, advancing the stride of the first dimension each time to achieve element-by-element processing.
- Then `advance` the stride of the second dimension to achieve the loop of `step1`.

```c++
// aten/src/ATen/native/cpu/Loops.h
template <typename op_t, typename vop_t>
struct VectorizedLoop2d {

  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    // using data_t = std::array<char*, ntensors>;
    // Here, the addresses of the pointers that need to be processed in the current range are copied to the data array.
    data_t data;
    std::copy_n(base, ntensors, data.data());
    // Get the strides of the second dimension of output, for `advance`.
    const int64_t *outer_strides = &strides[ntensors];

    // using traits = function_traits<op_t>;
    // Note that `op_t` here is `[=](scalar_t a) -> scalar_t { return a; }`,
    if (is_contiguous<traits>(strides)) {
      for (const auto i C10_UNUSED : c10::irange(size1)) {
        vectorized_loop(data.data(), size0, 0, op, vop);
        advance(data, outer_strides);
      }
    } else {
      // `Indices` is a template class, where the template parameter is traits::arity
      // (the number of parameters of `func_t`, the `arity` of our copy kernel is 1).
      // Construct a numerical sequence of `indices`, here it is {0}
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        if (idx) {
          // When `idx` is non-zero, perform a vectorized loop.
          for (const auto i C10_UNUSED : c10::irange(size1)) {
            vectorized_loop(data.data(), size0, idx, op, vop);
            advance(data, outer_strides);
          }
        } else {
          // When `idx` is 0, perform a basic loop.
          for (const auto i C10_UNUSED : c10::irange(size1)) {
            basic_loop(data.data(), strides, 0, size0, op);
            advance(data, outer_strides);
          }
        }
      });
    }
  }
};
```

Before entering any computation, `is_contiguous<traits>(strides)` is called first, what is it doing here?

```c++
// aten/src/ATen/native/cpu/IsContiguous.h
// in our example，traits is function_traits<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::$_5::operator()() const::'lambda5'()::operator()() const::'lambda'(float)>
template <typename traits,
    typename std::enable_if<!std::is_void<typename traits::result_type>::value>::type* = nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  // traits::arity is the number of parameters, which is 1 in this example
  // And it is exactly `ntensors - 1` (pointing to input stride)
  return IsContiguous<traits::arity, traits::arity, traits>::eval(strides);
}

// n: number of function parameters (traits::arity)
// stride_index: 1 in this example
// traits: function_traits (see FunctionTraits.h for details)
// s: scalar parameter index or -1, here it is the default -1 as we did not pass it
template <int n, int stride_index, typename traits, int s=-1>
struct IsContiguous {
  static bool eval(const int64_t* strides) {
    using type = typename traits::template arg<n - 1>::type;
    // Here strides[stride_index] == sizeof(type) is checked,
    // which requires the interval between adjacent elements to be exactly the current element size.
    // Since our strides[1] = 80 (corresponding to the lowest dimension stride of the input tensor) does not equal any type,
    // it directly returns false, and there is no need to recursively judge the lowest dimension stride of the output tensor
    return strides[stride_index] == (s == n ? 0 : sizeof(type)) &&
           IsContiguous<n - 1, stride_index - 1, traits, s>::eval(strides);
  }
};
```

Then it goes to the next branch to judge `unroll_contiguous_scalar_checks`

```c++
// aten/src/ATen/native/cpu/Loops.h
// traits: function_traits<direct_copy_kernel(at::TensorIteratorBase&)>
// cb_t: the anonymous function from above
// INDEX0: 0
template <typename traits, typename cb_t, size_t INDEX0, size_t ...INDEX>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<INDEX0, INDEX...>,
    cb_t&& cb) {
  // Here it is similar to the above, judging whether strides[1]
  // (the lowest dimension stride of the input) is contiguous, it returns false
  if (is_contiguous_scalar<traits, INDEX0 + 1>(strides)) {
    cb(INDEX0 + 1);
  } else {
    // Recursive call to unroll, because INDEX is empty, so it calls cb(0) below
    unroll_contiguous_scalar_checks<traits>(strides, std::index_sequence<INDEX...>{}, std::forward<cb_t>(cb));
  }
}

template <typename traits, typename cb_t>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* /*strides*/,
    std::index_sequence<>,
    cb_t&& cb) {
  cb(0);
}
```

After calling `cb(0)`, the basic loop branch of the anonymous function mentioned above is executed

```c++
// aten/src/ATen/native/cpu/Loops.h
// size1 is the step[1] we mentioned above, here it means to loop over the second dimension data, loop over the first dimension data
// `data` is the pointer address we handled above
for (const auto i C10_UNUSED : c10::irange(size1)) {
  basic_loop(data.data(), strides, 0, size0, op);
  advance(data, outer_strides);
}
```

Then calls `basic_loop`

```c++
// aten/src/ATen/native/cpu/Loops.h
template <typename func_t>
static inline void
basic_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  // Here op is the anonymous function we specified at the earliest, 
  // op: [=](scalar_t a) -> scalar_t { return a; }
  using traits = function_traits<func_t>;
  // Reading from the above to here, ntensors is just arity+1, which is easy to understand.
  // For our copy kernel, op has one parameter, corresponding to ntensors being 2 (one output and one input)
  constexpr int ntensors = traits::arity + 1;

  // Using local variable strides helps to optimize on older compilers
  int64_t strides[ntensors];
  // Note that only the lowest dimension stride of output and input needs to be fetched here,
  // because we are just looping the second dimension
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  // Here i is 0, n is the number of loops required for the first dimension
  // Why do we need to pass down i, instead of directly int i=0 in the for loop below?
  // Because in some cases, it is not necessary to start traversing from 0 (such as when vectorized)
  execute_op(data, strides, i, n, std::forward<func_t>(op));
}

template <typename func_t,
    typename std::enable_if<!std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
static inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  using result_type = typename traits::result_type;
  for (; i < n; i++) {
    // Here the output pointer address that needs to be operated on is calculated
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    // dereference calculates the address of the input data[1] + i * strides[1],
    // which is packed into a tuple as the parameter of apply
    *out_ptr = c10::guts::apply(std::forward<func_t>(op), dereference<traits>(
        &data[1],
        &strides[1],
        i));
  }
}

template <class F, class Tuple>
CUDA_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  // Finally, the std::apply function is called, and the op function 
  // takes out the value of the address in the tuple, return and write out_ptr
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}
```

After `basic_loop` is executed, the data of the current sub-batch has been correctly copied to output, but it is not over yet, we still need to perform the `advance` operation, adding the head address of the operation with the stride of the second dimension, in order to perform the next sub-batch loop.

```c++
// aten/src/ATen/native/cpu/Loops.h
static void advance(data_t &data, const int64_t *outer_strides) {
  // outer_strides is the stride of the second dimension, such as [256, 4] in this example
  for (const auto arg : c10::irange(data.size())) {
    data[arg] += outer_strides[arg];
  }
}
```

At this point, the `copy` operation is completed, and the data of the old tensor is copied to the new tensor according to the specified memory format, and the `contiguous` process is over.

Additionally, if the stride meets the conditions **input output type are the same**, **contiguous** (here contiguous means the size is exactly `sizeof(type)`), or the input is just a scalar (stride0), then `vectorized_loop` will be called, processing as much data as possible in each loop.

You can see that through vectorized loop unrolling, the loop condition becomes `for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size())`, here vec size defaults to 8, reducing the loop count by 16 times.

```c++
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  // ...
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    // ... vectorized loop unrolling logic
  }
  // basic_loop handles the remaining data that has not been unrolled
  if (i < n) {
    // ...
    basic_loop(data, strides, i, n, std::forward<func_t>(op));
  }
}
```

## 12. Review of `contiguous` execution process

Although we have expanded a lot of technical details, such as **how tensor_iterator preprocesses tensors**, **how dim counter takes the steps to be processed**, **how loop2d implements iteration**, etc., from a high-level understanding, the most key call paths are the following two pieces of code:

```c++
// aten/src/ATen/native/TensorProperties.cpp
Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  return self.clone(memory_format);
}
```

```c++
// aten/src/ATen/native/TensorFactories.cpp
Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
  // ...
  Tensor self;
  self = at::empty_like(src, src.options(), memory_format);

  if (src._is_zerotensor()) {
    self.zero_();
  } else {
    self.copy_(src);
  }
  return self;
}
```

That is to say, first judge whether it is `contiguous`, if it is not `contiguous` then `clone` the tensor according to the specified memory format.

In the clone operation, first `empty` a new tensor with the specified memory format, then `copy_` the data of the old tensor to the new tensor according to the stride and other information.

## Reference

- [pybind11-gil](https://pybind11.readthedocs.io/en/stable/advanced/misc.html)
- [pytorch-github](https://github.com/pytorch/pytorch)
- [Pytorch Tensor 加法实现细节](https://zhuanlan.zhihu.com/p/129778637)
