---
title: "How Pytorch 2.0 Call Ops(3)"
date: 2023-05-13T09:53:09+08:00
categories: ["pytorch"]
summary: "This article introduces the process of pytorch 2.0 calling ops, using `contiguous` as an example."
---

## Summary

This article introduces the process of pytorch 2.0 calling ops, using `contiguous` as an example.

## To be translated

Oh Sorry!

This blog has't been translated to English, please wait for a little while...

## 9. copy_算子与TensorIterator

在clone算子中，创建好指定memory format的tensor之后，我们便调用copy算子

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

  // 如果self和src是相同storage的view
  const bool is_same_data = (
      self.is_alias_of(src) &&                          // storage相同
      self.storage_offset() == src.storage_offset() &&  // storage offset相同
      self.strides().equals(src.strides()) &&
      self.sizes().equals(src.sizes()) &&
      self.scalar_type() == src.scalar_type()
    );
  if (is_same_data) {
    return self;
  }

  // 构建了at::TensorIterator，划定input output 便于处理device、dtype等相关内容
  auto iter = TensorIteratorConfig()
    // 将tensor存储到TensorIteratorConfig中的SmallVector<c10::MaybeOwned<TensorBase>, 4> tensors_;
    // 注意此处顺序，先add了output，保证output在list的第一位
    .add_output(self)
    .add_input(src)
    .resize_outputs(false)    // 设置变量，下同
    .check_all_same_dtype(false)
    .check_all_same_device(false)
    .build();   // 新建一个TensorIterator，使用上面构建的config build

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

这里展开补充一下tensor iterator build部分的代码

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  is_reduction_ = config.is_reduction_;
  enforce_linear_iteration_ = config.enforce_linear_iteration_;

  // 将config中的tensors_转存到iterator的SmallVector<OperandInfo, 4> operands_;
  populate_operands(config);
  // 设置 `is_output` 等 flag，判断是否input output是相同tensor(inplace操作)
  mark_outputs();
  // 检查output 内存没有overlap，同时与input不共享存储
  compute_mem_overlaps(config);
  // 计算out name
  compute_names(config);
  // 计算广播的shape，逻辑为首先取output的shape作为shape_存储，如果input的tensor shape与shape_不同，则infer出新的shape，详见下文
  compute_shape(config);
  // 如果output需要resize（与shape_不同），则打上标记
  mark_resize_outputs(config);
  // 计算device（取第一个不为cpu的device作为common device）和dtype（取第一个input tensor的dtype作为conmon_dtype_，取第一个output tensor的dtype作为output_dtype_）
  compute_types(config);
  // 尝试快速构建output tensor（比如如果所有tensor都是contiguous、channals last，那就可以快速infer output/resize（如果需要），set name等）
  if (!fast_set_up(config)) {
    // 计算每个tensor广播后的stride（实际上是计算出op.stride_bytes（stride * element_size）如本例中，shape_为[1, 64, 5, 4], output.stride_bytes = [5120, 4, 1024, 256]
    compute_strides(config);
    // 此处对tensor的shape、stride进行重排序，将stride[0]作为最快的步进维度（stride升序排列），如本例中，shape_变为[64, 4, 5, 1], output.stride_bytes = [4, 256, 1024, 5120]
    reorder_dimensions();
    // 如果output没有defined，这里进行allocate
    allocate_or_resize_outputs();
    // 如果每个tensor对应dim size为1或shape[n] * stride[n] == stride[n + 1]，合并相邻的dimension。为什么要合并相邻dimension呢？这样可以有效提升取址运算效率，也便于之后取stride遍历。我们下文展开说明这段代码
    if (!is_meta_) coalesce_dimensions();
  }

  // ...
}
```

我们再进一步展开如何计算广播的代码

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
  // 例如：a = {2, 1, 3}， b = {4, 3}
  size_t dimsA = a.size();
  size_t dimsB = b.size();
  size_t ndim = dimsA > dimsB ? dimsA : dimsB;  // 取更大的dim，例如 ndim = 3
  Container expandedSizes(ndim);

  // 使用 ptrdiff_t 来确保有符号的比较
  for (ptrdiff_t i = (ptrdiff_t)ndim - 1; i >= 0; --i) {
    ptrdiff_t offset = ndim - 1 - i;
    ptrdiff_t dimA = dimsA - 1 - offset;  // 相当于 dimsA - ndim + i
    ptrdiff_t dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? a[dimA] : 1;   // 如果dimA不是负数那就取值
    auto sizeB = (dimB >= 0) ? b[dimB] : 1;

    TORCH_CHECK(
        sizeA == sizeB || sizeA == 1 || sizeB == 1,
        "The size of tensor a (", sizeA,
        ") must match the size of tensor b (", sizeB,
        ") at non-singleton dimension ", i);

    // 如果sizeA与sizeB相同，那么任意取；如果sizeA为1，那么取sizeB（实现了取更大的值）
    expandedSizes[i] = sizeA == 1 ? std::move(sizeB) : std::move(sizeA);
  }

  // 如上例，最终返回 {2,4,3}
  return expandedSizes;
}
```

我们再展开说明`coalesce_dimensions`如何合并维度

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::coalesce_dimensions() {
  if (ndim() <= 1) {
    return;
  }

  // 如果dim == 1 或 shape[n] * stride[n] == stride[n + 1]则可合并
  // 注意需要所有input和output tensor同时满足条件
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

  // 将stride[dim0]的值赋为stride[dim1]
  auto replace_stride = [&](int dim0, int dim1) {
    for (const auto i : c10::irange(ntensors())) {
      auto& stride = operands_[i].stride_bytes;
      stride[dim0] = stride[dim1];
    }
  };

  // 这里的逻辑是双指针，从prev_dim指针开始，遍历后面每一个维度，尽可能合并更多维度
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

  // 最后resize 如我们上例的tensor：
  // shape_ = [64, 4, 5, 1], output.stride_bytes = [4, 256, 1024, 5120], input.stride_bytes = [80, 4, 16, 5120]
  // 变为 shape_ = [64, 20], input.stride_bytes = [80, 4]
  shape_.resize(prev_dim + 1);
  for (const auto i : c10::irange(ntensors())) {
    operands_[i].stride_bytes.resize(ndim());
  }
  has_coalesced_dimensions_ = true;
}
```

合并相邻维度后，`TensorIterator`就构建完成了，注意此时已经没有了`input`和`output`的参数，之后所有操作全部基于该iterator展开。

## 10. copy算子：kernel执行

回到上文，`copy_stub`进行一轮dispatch后调用到`copy_kernel`(device_type用于dispatch到不同的kernel，到具体kernel时已经没有了这个参数)。此外，

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
    // 如果类型不一致，走到该分支
    // AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4 宏 switch 处理数据类型
    // 如果`has_contiguous_first_dim`为true（两tensor的 stride[0] == elementsize）
    // 则直接调用iter.for_each直接传入一个带类型转化的匿名函数
    // 否则调用cpu_kernel(aten/src/ATen/native/cpu/Loops.h)，本质是调用`iter.for_each`传入`basic_loop`的匿名函数（basic_loop可以支持任意stride的1d slice）
    // basic loop为何能支持任意stride呢？请继续阅读本文，之后会进行介绍
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

我们重点看类型一致的情况（`contiguous`到这里一定是类型一致的），因为我们不需要取负也不需要共轭，所以调用到`direct_copy_kernel`函数

```c++
// aten/src/ATen/native/cpu/CopyKernel.cpp
void direct_copy_kernel(TensorIteratorBase &iter) {
  ScalarType dtype = iter.dtype(0);
  if (isQIntType(dtype)) {
    // 量化整数类型...
  } else if (dtype == ScalarType::ComplexHalf) {
    // 半精度复数
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

`AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3` 宏展开后代码如下

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

到这里copy kernel本身的内容已经调用完成，将两个匿名函数传给了`cpu_kernel_vec`

## 11. `cpu_kernel_vec`底层运行原理

`cpu_kernel_vec`支持标量化函数和向量化函数作参数，我们展开其细节

```c++
// aten/src/ATen/native/cpu/Loops.h
template <bool check_dynamic_cast=true, typename func_t, typename vec_func_t>
void cpu_kernel_vec(TensorIteratorBase& iter, func_t&& op, vec_func_t&& vop, int64_t grain_size = at::internal::GRAIN_SIZE) {
  // ... some check

  // make_vectorized_loop2d将标量op和向量化op整合成一个对象`VectorizedLoop2d`给for_each
  iter.for_each(make_vectorized_loop2d(op, vop), grain_size);
  // cast_outputs方法用于将输出张量按照当前数据类型进行强制类型转换
  iter.cast_outputs();
}

// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::for_each(loop2d_t loop, int64_t grain_size) {
  int64_t numel = this->numel();
  if (numel == 0) {
    return;
  } else if (numel < grain_size || at::get_num_threads() == 1) {
    // 如果元素数量少于grain_size或者线程数为1，使用串行操作
    return serial_for_each(loop, {0, numel});
  } else {
    // 否则，划分为grain_size大小的多个任务，每个任务里再串行迭代
    at::parallel_for(0, numel, grain_size, [&](int64_t begin, int64_t end) {
      serial_for_each(loop, {begin, end});
    });
  }
}

```

我们继续深入展开串行操作`serial_for_each`，注意`operands_`这个变量，它是TensorIterator的operands集合`SmallVector<OperandInfo, 4> operands_;`，包括inputs和outputs的tensor，其中output一定位于第一个（这部分介绍在上文iterator构建过程中）

```c++
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::serial_for_each(loop2d_t loop, Range range) const {
  if (range.size() == 0) {
    return;
  }

  const auto ntensors = this->ntensors();
  // 由于上面构建TensorIterator的时候进行了维度合并，所以这里ndim为2
  const auto ndim = this->ndim();

  c10::SmallBuffer<char*, 4> ptrs(ntensors);
  // 此处总strides长4
  c10::SmallBuffer<int64_t, 8> strides(ntensors * std::max(ndim, 2));
  
  // 这里operands_是 TensorIteratorBase的tensor op列表
  // `get_base_ptrs`拿到了所有tensor的storage指针，转化为char*类型，存储在指针列表中
  at::get_base_ptrs(ptrs.data(), operands_);
  // `get_strides`则将所有tensor stride按序存储到strides中（低维到高维排列）
  // 低维到高维排列是为了方便下面取值计算
  // 如我们的例子中，strides最后为[4, 80, 256, 4] 注意这里是int64_t数据类型
  // 其中out tensor的stride为[4, 256], input tensor的stride为[80, 4]
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
    // 这里又创建了和上面一样声明的ptrs，但此处的ptrs存放的是当前batch中需要处理的地址
    c10::SmallBuffer<char*, 4> ptrs(ntensors);
    auto counter = DimCounter(shape, range);
    // 通过DimCounter确保每一个element都被处理过，is_done判断offset是否大于range.end
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

这里的`DimCounter`是何方神圣呢？

```c++
// aten/src/ATen/TensorIteratorInternal.h
struct DimCounter {
  DimCounter(IntArrayRef shape, Range range);

  void increment(const std::array<int64_t, 2>& step);
  bool is_done() const;   // return offset >= range.end;
  std::array<int64_t, 2> max_2d_step() const;

  IntArrayRef shape;
  Range range;    // range是处理的element范围，如{0, numel()}
  c10::SmallBuffer<int64_t, 4> values;  // 每个维度上的offset，注意是元素的offset，不是stride
  int64_t offset; // 当前处理的元素offset
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
      // 我们举一个新例子，如begin = 1066670, size = [64, 2000, 10], 此处values存下了余数offset [46, 666, 8]，
      // 这里的offset之后乘以stride就可以算出总体offset，来实现直接找到当前range的begin位置
      values[dim] = linear_offset % size;
      linear_offset /= size;
    }
  }
  TORCH_INTERNAL_ASSERT(linear_offset == 0);
}
```

注意`get_data_ptrs`方法，此处取到了当前range的起始指针，存放到`ptrs`中（`ptrs[0]`为output的指针，其余为input指针）

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
  // 这里求出每个tensor在当前range下的起始指针
  // 算法为所有维度上offset与stride byte乘积之和
  // 例如offset = [46, 666, 8], output stride（对应strides维度为[0,2,4]） = [4, 256, 512000]
  // 起始地址为 base + 46 * 4 + 666 * 256 + 8 * 512000 = base + 4266680
  // 注意到4266680 / 4正好是我们之前的第1066670个元素（range.begin）
  for (const auto dim : c10::irange(ndim)) {
    int64_t value = counter[dim];
    for (const auto arg : c10::irange(ntensors)) {
      ptrs[arg] += value * strides[dim * ntensors + arg];
    }
  }
}
```

然后走到`max_2d_step`方法，这里无论有多少维度，只取二维，即获取到当前batch应处理数据的step `(m, n)`，我们希望尽可能快地取完数据，理想情况下（没有offset时），比如shape为`[64, 2000, 10]`，我们取step = `{64, 2000}`，那么取10次就能取完数据（但本例中第三维开始offset已经是8了，所以取三次正好取完）。

但由于offset限制，所以我们不一定能取满`{64, 2000}`那么大，但经过一次取少量数据`shape[0] - values[0]`一次调整对齐之后，第一维就对齐了，之后第一维都能取满64。同理，第二维可能需要经过`shape[1] - values[1]`调整一次后才能对齐。

例如，我们走到这里时第一次step = `{18, 1}`，然后进`loop`（**loop逻辑我们下文展开**），调用`increment`更新offset为`1066670 + 18 = 1066688`, 更新`values=[0, 667, 8]`

第二次走到这里时step就为`{64, 1333}`，可以取的数据变多了，然后调用`increment`更新offset为`1066688 + 64 * 1333 = 1152000`， 更新`values=[0,0,9]`

第三次到这里offset就完全修正完成了，取最大的step为`{64, 2000}`，随后更新offset为`1280000`，更新values = `[0,0,0]`，因为我们的range 正好是`{1066670, 1280000}`，调用结束。

注意最后一次取数据的时候有可能出现数据不足的情况，那取接近end的少批量数据`range.end - offset`取完

```c++
// aten/src/ATen/TensorIterator.cpp
std::array<int64_t, 2> DimCounter::max_2d_step() const {
  // 尝试取最大范围的数据，注意如果offset已经接近end，则通过range.end - offset取完数据
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
    // step[1] 不为1说明第一维已经调整过offset了，直接跳到第二维
    TORCH_INTERNAL_ASSERT(step[0] == shape[0] && values[0] == 0);
    i = 1;
    overflow = step[1];
  }
  for (; i < ndim && overflow > 0; i++) {
    auto size = shape[i];
    auto prev = values[i];
    auto value = prev + overflow;
    // overflow标记是否偏差溢出，如果溢出下一维度要对应处理+1
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

最后调用到`loop`方法。注意看上文的`cpu_kernel_vec`，这里传进来的op其实只是lambda表达式 `[=](scalar_t a) -> scalar_t { return a; },`和`[=](Vectorized<scalar_t> a) -> Vectorized<scalar_t> { return a; }`

`loop2d`本质上是根据step0和step1进行二重循环：

- 首先循环step0，如果可以向量化展开（如stride恰好等于type size）就尽可能调用`vectorized_loop`展开（类似for循环展开，在一次循环中处理多个数据），否则调用`basic_loop`for循环step0，每次步进第一维的stride实现逐元素处理。
- 然后 `advance` 第二维的stride，实现step1的loop。

```c++
// aten/src/ATen/native/cpu/Loops.h
template <typename op_t, typename vop_t>
struct VectorizedLoop2d {

  void operator()(char** base, const int64_t *strides, int64_t size0, int64_t size1) {
    // using data_t = std::array<char*, ntensors>;
    // 这里将上面的当前range需处理的指针地址复制到data array中
    data_t data;
    std::copy_n(base, ntensors, data.data());
    // 拿到第二维 output的strides，便于`advance`
    const int64_t *outer_strides = &strides[ntensors];

    // using traits = function_traits<op_t>;
    // 注意这里op_t为[=](scalar_t a) -> scalar_t { return a; }，
    // 这里is contiguous是在做什么呢？我们下文展开
    if (is_contiguous<traits>(strides)) {
      for (const auto i C10_UNUSED : c10::irange(size1)) {
        vectorized_loop(data.data(), size0, 0, op, vop);
        advance(data, outer_strides);
      }
    } else {
      // Indices是一个模板类，模板参数为traits::arity（func_t参数数量，我们copy kernel arity为1）
      // 构造了indices数值序列（左闭右开），即{0}
      using Indices = std::make_index_sequence<traits::arity>;
      unroll_contiguous_scalar_checks<traits>(strides, Indices{}, [&](size_t idx) {
        // 这一部分{}内都是匿名函数体的内容
        if (idx) {
          // idx非0，进行向量化loop
          for (const auto i C10_UNUSED : c10::irange(size1)) {
            vectorized_loop(data.data(), size0, idx, op, vop);
            advance(data, outer_strides);
          }
        } else {
          // idx为0，表示scalar维度，进行basic loop
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

在进入任何计算前，首先调用了`is_contiguous<traits>(strides)`，这里是在做什么呢？

```c++
// aten/src/ATen/native/cpu/IsContiguous.h
// 在本例中，traits为function_traits<at::native::AVX2::direct_copy_kernel(at::TensorIteratorBase&)::$_5::operator()() const::'lambda5'()::operator()() const::'lambda'(float)>
template <typename traits,
    typename std::enable_if<!std::is_void<typename traits::result_type>::value>::type* = nullptr>
static inline bool is_contiguous(const int64_t* strides) {
  // traits::arity为参数个数，本例中为1，恰好为ntensors - 1（指向input stride）
  return IsContiguous<traits::arity, traits::arity, traits>::eval(strides);
}

// n: 函数参数个数 (traits::arity)
// stride_index: 本例中为1
// traits: function_traits (详见 FunctionTraits.h)
// s: scalar参数索引或-1，我们这里由于没传，所以为默认的-1
template <int n, int stride_index, typename traits, int s=-1>
struct IsContiguous {
  static bool eval(const int64_t* strides) {
    using type = typename traits::template arg<n - 1>::type;
    // 这里对strides[stride_index] == sizeof(type)判断，即要求相邻元素间隔恰好等于当前元素大小。
    // 由于我们strides[1] = 80（对应input tensor的最低维stride）不等于任何一种type，直接返回false，无需递归判断output tensor的最低维stride
    return strides[stride_index] == (s == n ? 0 : sizeof(type)) &&
           IsContiguous<n - 1, stride_index - 1, traits, s>::eval(strides);
  }
};
```

随后走到下一个分支判断`unroll_contiguous_scalar_checks`

```c++
// aten/src/ATen/native/cpu/Loops.h
// traits: function_traits<direct_copy_kernel(at::TensorIteratorBase&)>
// cb_t: 上面的一段匿名函数
// INDEX0: 0
template <typename traits, typename cb_t, size_t INDEX0, size_t ...INDEX>
static inline void unroll_contiguous_scalar_checks(
    const int64_t* strides,
    std::index_sequence<INDEX0, INDEX...>,
    cb_t&& cb) {
  // 这里类似上面，判断strides[1]（input的最低维stride）是否contiguous，return false
  if (is_contiguous_scalar<traits, INDEX0 + 1>(strides)) {
    cb(INDEX0 + 1);
  } else {
    // 递归调用unroll，由于INDEX空了，所以调用到下面的cb(0)
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

调用`cb(0)`后，执行上文中匿名函数的basic loop分支

```c++
// aten/src/ATen/native/cpu/Loops.h
// size1即我们的上文的step[1]，这里的意思是for 第二维数据，loop第一维数据
// data即我们上文处理好的指针首地址
for (const auto i C10_UNUSED : c10::irange(size1)) {
  basic_loop(data.data(), strides, 0, size0, op);
  advance(data, outer_strides);
}
```

然后调用到`basic_loop`

```c++
// aten/src/ATen/native/cpu/Loops.h
template <typename func_t>
static inline void
basic_loop(char* C10_RESTRICT data[], const int64_t* strides_, int64_t i, int64_t n, func_t&& op) {
  // 这里op即我们最早指定的匿名函数  [=](scalar_t a) -> scalar_t { return a; }
  using traits = function_traits<func_t>;
  // 从上文读到这里，ntensors恰好为arity+1就很好理解了，对于我们的copy kernel，op参数为一个，对应ntensors为2（一个output 一个input)
  constexpr int ntensors = traits::arity + 1;

  // 使用局部变量 strides 有利于在老版本的编译器上优化
  int64_t strides[ntensors];
  // 注意这里只用拿output和input最低维的stride即可，因为我们正在for loop 第二维
  for (const auto arg : c10::irange(ntensors)) {
    strides[arg] = strides_[arg];
  }

  // 这里i是0，n是第一维需要loop的数量
  // 为什么要费尽心机把i传下来，而不是直接在下面for循环里int i=0呢？因为有的情况下并不需要从0开始遍历（如vectorized的时候）
  execute_op(data, strides, i, n, std::forward<func_t>(op));
}

template <typename func_t,
    typename std::enable_if<!std::is_void<typename function_traits<func_t>::result_type>::value>::type* = nullptr>
static inline void
execute_op(char* C10_RESTRICT data[], const int64_t* strides, int64_t i, int64_t n, func_t&& op) {
  using traits = function_traits<func_t>;
  using result_type = typename traits::result_type;
  for (; i < n; i++) {
    // 这里算出需要操作的output指针地址
    result_type* out_ptr = (result_type*)(data[0] + i * strides[0]);
    // dereference算出input的地址data[1] + i * strides[1]，封成了tuple作为apply的参数
    *out_ptr = c10::guts::apply(std::forward<func_t>(op), dereference<traits>(
        &data[1],
        &strides[1],
        i));
  }
}

template <class F, class Tuple>
CUDA_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  // 最后调用std::apply函数，调用op函数取出了tuple里地址的值，return并写out_ptr
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}
```

`basic_loop`执行完后，对于当前子batch数据已经正确地复制到了output中，但还没有结束，我们还需要进行`advance`操作，将操作的首地址advance加上第二维的stride，以便进行下一个子batch的循环。

```c++
// aten/src/ATen/native/cpu/Loops.h
static void advance(data_t &data, const int64_t *outer_strides) {
  // outer_strides为第二维的stride，如本例中为[256, 4]
  for (const auto arg : c10::irange(data.size())) {
    data[arg] += outer_strides[arg];
  }
}
```

到此，`copy`算子执行完成，旧tensor的数据按照指定memory format的格式复制到新tensor上，`contiguous`流程结束。

额外指出，如果stride满足条件（**input output type相同**，**contiguous**（这里的contiguous指大小恰好等于`sizeof(type)`），或者input只是一个scalar（stride0），那么会调用`vectorized_loop`，每次循环中一次处理尽可能多的数据。

可以看到，通过向量化循环展开，循环条件变成了`for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size())`，这里vec size默认是8，减少16倍循环次数。

```c++
template <typename func_t, typename vec_func_t>
static inline void
vectorized_loop(char** C10_RESTRICT data_, int64_t n, int64_t S, func_t&& op, vec_func_t&& vop) {
  // ...
  for (; i <= n - 2 * Vec::size(); i += 2 * Vec::size()) {
    // ... 向量化循环展开逻辑
  }
  // basic_loop处理剩余没有被展开的数据
  if (i < n) {
    // ...
    basic_loop(data, strides, i, n, std::forward<func_t>(op));
  }
}
```

## 12. contiguous执行流程回顾

虽然我们展开了很多底层技术细节，如**tensor_iterator如何预处理tensor**，**dim counter如何取需处理的step**，**loop2d如何实现迭代**等，但从上层理解，最关键的调用链路为以下两段代码：

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

即先判断是否contiguous，如果非contiguous则按照指定memory format `clone` tensor，在clone算子中，先`empty`出一个指定memory format的新tensor，然后`copy_`将旧tensor的数据按照stride等信息复制到新tensor上。

## Reference

- [pybind11-gil](https://pybind11.readthedocs.io/en/stable/advanced/misc.html)
- [pytorch-github](https://github.com/pytorch/pytorch)
- [Pytorch Tensor 加法实现细节](https://zhuanlan.zhihu.com/p/129778637)

---
*Confused about some of the content? Feel free to report an issue [here](https://github.com/yewentao256/yewentao256.github.io/issues/new).*
