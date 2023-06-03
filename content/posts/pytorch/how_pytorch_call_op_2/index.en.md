---
title: "How Pytorch 2.0 Call Ops(2)"
date: 2023-04-12T09:53:09+08:00
categories: ["pytorch"]
summary: "This article introduces the process of pytorch 2.0 calling ops, using `contiguous` as an example."
---

## Summary

This article introduces the process of pytorch 2.0 calling ops, using `contiguous` as an example.

## To be translated

Oh Sorry!

This blog has't been translated to English, please wait for a little while...

## 6. register和dispatch的回顾

我们纵观register和dispatch的过程，总结其大体流程为：

1. 注册op schema
2. 注册op下的具体kernel实现（基于dispatch key）
3. 查找op schema
4. 查找op下具体kernel实现并调用（基于dispatch key）

中间几个重要的数据类型：`Dispatcher`, `OperatorHandle`, `OperatorEntry`

- **Dispatcher**
  - `operatorLookupTable_`维护了OperatorName->OperatorHandle的映射

```c++
// aten/src/ATen/core/dispatch/Dispatcher.h
class TORCH_API Dispatcher final {
private:
  friend class impl::OperatorEntry;
  struct OperatorDef final {
    explicit OperatorDef(OperatorName&& op_name)
    : op(std::move(op_name)) {}
    impl::OperatorEntry op;
    size_t def_count = 0;
    size_t def_and_impl_count = 0;
  };
  friend class OperatorHandle;
  template<class> friend class TypedOperatorHandle;

public:
  // ...
  static Dispatcher& realSingleton();

  c10::optional<OperatorHandle> findSchema(const OperatorName& operator_name);

  template<class Return, class... Args>
  Return call(const TypedOperatorHandle<Return (Args...)>& op, Args... args) const;

  RegistrationHandleRAII registerImpl(/* ... */);

private:
  // ...
  std::list<OperatorDef> operators_;
  LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;
  ska::flat_hash_map<std::string, std::string> libraries_;
};
```

- **OperatorHandle**
  - 其内部的`operatorDef_`本质是上面Dispatcher中的`OperatorDef`，是对`OperatorEntry`的封装
  - 更多时候用的是`TypedOperatorHandle`，`OperatorHandle`的子类，可以理解为针对op参数模板化的`OperatorHandle`

```c++
// aten/src/ATen/core/dispatch/Dispatcher.h
class TORCH_API OperatorHandle {
  template <typename T> friend struct std::hash;

public:
  const OperatorName& operator_name() const {
    return operatorDef_->op.operator_name();
  }

  const FunctionSchema& schema() const {
    return operatorDef_->op.schema();
  }
  
  // ...

  template<class FuncType>
  TypedOperatorHandle<FuncType> typed() const {
    // ...
    return TypedOperatorHandle<FuncType>(operatorIterator_);
  }

private:
  // ...
  friend class Dispatcher;
  template<class> friend class TypedOperatorHandle;

  Dispatcher::OperatorDef* operatorDef_;

  std::list<Dispatcher::OperatorDef>::iterator operatorIterator_;
};
```

- **OperatorEntry**：
  - 实际存储op信息的数据结构

```c++

// aten/src/ATen/core/dispatch/OperatorEntry.h
class TORCH_API OperatorEntry final {
public:
  explicit OperatorEntry(OperatorName&& operator_name);

  const FunctionSchema& schema() const {
    return schema_->schema;
  }

  void registerSchema(FunctionSchema&&, std::string&& debug, std::vector<at::Tag> tags = {});
  void deregisterSchema();

  const OperatorName& operator_name() const {
    return name_;
  }

  using AnnotatedKernelContainer = std::list<AnnotatedKernel>;  // linked list

  using AnnotatedKernelContainerIterator = AnnotatedKernelContainer::iterator;

  AnnotatedKernelContainerIterator registerKernel(/* ... */);

  void deregisterKernel_(/* ... */);

  const DispatchKeyExtractor& dispatchKeyExtractor() const { return dispatchKeyExtractor_; }

  const KernelFunction& lookup(DispatchKeySet ks) const {
    const auto idx = ks.getDispatchTableIndexForDispatchKeySet();
    // ...
    const auto& kernel = dispatchTable_[idx];
    // ...
    return kernel;
  }

private:
  OperatorName name_;
  c10::optional<AnnotatedSchema> schema_;
  std::array<KernelFunction, c10::num_runtime_entries> dispatchTable_;
  DispatchKeyExtractor dispatchKeyExtractor_;
  ska::flat_hash_map<DispatchKey, std::list<AnnotatedKernel>> kernels_;

  c10::optional<CppSignatureWithDebug> cpp_signature_;
  // ...
};
```

通过 **Library** -> **Dispatcher** -> **OperatorHandle** -> **OperatorEntry** 这样的调用链路，pytorch完成了op和对应kernel的注册。之后，pytorch就可以基于这条链路查找到所需算子的kernel并轻松实现调用。

## 7. `is_contiguous`判断是否连续

大致了解了pytorch算子的注册和调用流程之后，我们终于进入了contiguous算子实际执行的流程了，这部分相对而言简单很多。

我们将调用路径拉回到上文dispatch末端

```c++
// build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
at::Tensor wrapper_CompositeImplicitAutograd__contiguous(const at::Tensor & self, at::MemoryFormat memory_format) {
  return at::native::contiguous(self, memory_format);
}
```

这里调用aten native的contiguous算子

```c++
// aten/src/ATen/native/TensorProperties.cpp
Tensor contiguous(const Tensor& self, MemoryFormat memory_format) {
  if (self.is_contiguous(memory_format)) {
    return self;
  }
  TORCH_CHECK(
      memory_format != MemoryFormat::Preserve,
      "preserve memory format is unsupported by the contiguous operator");

  return self.clone(memory_format);
}
```

首先判断`is_contiguous(memory_format)`，即在指定memory format下是否已经连续，经过`TensorBase.h`中转来到`TensorImpl.h`中

```c++
// c10/core/TensorImpl.h
struct C10_API TensorImpl : public c10::intrusive_ptr_target {
  bool is_contiguous_default(at::MemoryFormat memory_format) const {
    // ...
    if (memory_format == at::MemoryFormat::ChannelsLast) {
      return is_channels_last_contiguous_;
    } else if (memory_format == at::MemoryFormat::ChannelsLast3d) {
      return is_channels_last_3d_contiguous_;
    }
    return is_contiguous_;
  }
  
  // ...
 protected:
  std::unique_ptr<c10::ExtraMeta> extra_meta_ = nullptr;
  c10::impl::SizesAndStrides sizes_and_strides_;
  int64_t storage_offset_ = 0;
  int64_t numel_ = 1;
  caffe2::TypeMeta data_type_;
  c10::optional<c10::Device> device_opt_;
  // ...
  bool is_contiguous_ : 1;
  bool is_channels_last_ : 1;
  bool is_channels_last_contiguous_ : 1;
  bool is_channels_last_3d_ : 1;
  bool is_channels_last_3d_contiguous_ : 1;
}
```

可以看到，pytorch的判断`is_contiguous`并没有计算，而是将数据直接存储在`TensorImpl`里，每次直接取用即可，这样省去了计算量，但也要求在更改tensor stride或者初始化的时候算出相关bool并存储。

那么，它是如何被设置的呢？我们溯源该变量的set流程，发现它被`refresh_contiguous()`设置，每次修改tensor的shape或stride的时候都要调用该方法。

```c++
// c10/core/TensorImpl.h
void _refresh_contiguous() {
    auto type_id = identity<T>();
    switch (dim()) {
      case 4: {
        _set_is_contiguous(type_id, compute_contiguous(type_id));
        _set_is_channels_last_contiguous(
            type_id, compute_channels_last_contiguous_2d(type_id));
        _set_is_channels_last_3d_contiguous(type_id, false);
        // ...
        break;
      }
      case 5: {
        _set_is_contiguous(type_id, compute_contiguous(type_id));
        _set_is_channels_last_contiguous(
            type_id, compute_channels_last_contiguous_2d(type_id));
        _set_is_channels_last_3d_contiguous(
            type_id, compute_channels_last_contiguous_3d_dim5(type_id));
        // ...
        break;
      }
      default:
        _set_is_contiguous(type_id, compute_contiguous(type_id));
        _set_is_channels_last_contiguous(type_id, false);
        _set_is_channels_last_3d_contiguous(type_id, false);
        // ...
    }
  }
```

我们挑一个`_compute_channels_last_contiguous_2d`展开看看，其本质就是在contiguous（NCHW）标准下，是否符合NHWC（1320置换）：

```c++
template <typename T>
bool _compute_channels_last_contiguous_2d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  switch (sizes.size()) {
    case 4: {
      T expected = 1;
      // const array可以被编译器自动展开加速
      for (auto& d : {1, 3, 2, 0}) {
        const auto& size_d = sizes[d];
        if (size_d != 1) {
          if (strides[d] != expected) {
            return false;
          }
          expected *= size_d;
        }
      }
      return true;
    }
    // ...
    default:
      return false;
  }
}
```

例如一个`N, C, H, W = 2, 2048, 1, 1`的tensor，它的stride为`[2048, 1, 1, 1]` 就是一个channels last的tensor（同时也是contiguous的tensor，因为内存排布刚好h、w都是1）

## 8. clone算子：自动微分与empty tensor

继续我们的调用流程，如果tensor在指定memory format下已经连续，那就直接返回，如果不连续，那就按照指定memory format进行`clone`

```c++
// build/aten/src/ATen/core/TensorBody.h
inline at::Tensor Tensor::clone(c10::optional<at::MemoryFormat> memory_format) const {
    return at::_ops::clone::call(const_cast<Tensor&>(*this), memory_format);
}
// build/aten/src/ATen/Operators_1.cpp
at::Tensor clone::call(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
    static auto op = create_clone_typed_handle();
    return op.call(self, memory_format);
}
```

是不是很熟悉？是的，这就是我们上面调用contiguous算子的入口，再经过类似的dispatch流程（找op schema，然后找kernel）后我们来到了实际clone处。由于上面contiguous的dispatch key是`CompositeImplicitAutograd`，这里clone算子也调用到该disptach key并需要处理自动微分相关逻辑。

这与我们对clone算子的印象也是一致的：完全独立的副本，保留`requires_grad`属性并支持自动求导（`CloneBackward0`放入`grad_fn`中）。

```c++
// torch/csrc/autograd/generated/VariableType_1.cpp
at::Tensor clone(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  // 此处调用`checked_cast_variable`检查tensor是否defined
  // self_和self地址相同
  auto& self_ = unpack(self, "self", 0);

  // 自动求导相关，如果需要自动求导，则设置grad_fn到graph里
  auto _any_requires_grad = compute_requires_grad( self );
  (void)_any_requires_grad;
  auto _any_has_forward_grad_result = (isFwGradDefined(self));
  (void)_any_has_forward_grad_result;
  std::shared_ptr<CloneBackward0> grad_fn;
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<CloneBackward0>(new CloneBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self ));
  }
  #ifndef NDEBUG
  // 拿到self的storage和impl
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  // 将当前dispatchkey和c10::after_autograd_keyset运算后，redisptach clone算子
  // redispatch的结果是拿到了clone的正确结果，redisptach的过程我们下文展开
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::clone(ks & c10::after_autograd_keyset, self_, memory_format);
  })();
  auto result = std::move(_tmp);
  // ...
  return result;
}
```

值得指出的是，在上面代码中redisptach的过程中，在重新计算dispatchkey之后，redisptach到aten的clone算子。`redispatch`和`call`有什么区别呢？

一方面，是函数签名上的差异，`redispatch`带了一个`currentDispatchKeySet`，就不用像call那样从op里取dispatchkey，直接用参数传进来的就好。

```c++
Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const

inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const
```

另一方面，是`redispatch`调用中，一般会将当前dispatchkey再下调一个优先级（如与`c10::after_autograd_keyset`进行与操作），然后调度到实际执行clone的算子上（此处已经处理了自动微分，之后就不再需要考虑自动微分），做了一层分级

redispatch后到`CompositeExplicitAutograd.cpp`

```c++
// build/aten/src/ATen/RegisterCompositeExplicitAutograd.cpp
at::Tensor wrapper_CompositeExplicitAutograd__clone(const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  return at::native::clone(self, memory_format);
}

// aten/src/ATen/native/TensorFactories.cpp
Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
  // ...
  Tensor self;
  if (memory_format == MemoryFormat::Preserve) {
    // ...
  } else {
    // 创建空tensor
    self = at::empty_like(src, src.options(), memory_format);
  }

  if (src._is_zerotensor()) {
    self.zero_();
  } else {
    self.copy_(src);
  }
  return self;
}
```

创建tensor时调用了`empty_like`算子，创建一个和src相同（但memory format为新memory format）的空tensor，一样经过dispatch后来到`TensorFactories.cpp`，然后`empty_like`又调用了empty算子（先call到`build/aten/src/ATen/RegisterBackendSelect.cpp`处，然后redispatch到CPU上——redispatch到哪根据编译选项不同会有差异）

```c++
// aten/src/ATen/native/TensorFactories.cpp
Tensor empty_like(
    const Tensor& self,
    c10::optional<ScalarType> dtype,
    c10::optional<Layout> layout,
    c10::optional<Device> device,
    c10::optional<bool> pin_memory,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  // ...
  Tensor result;
  if (memory_format == MemoryFormat::Preserve) {
    // ...
  } else {
    result = at::empty(self.sizes(), options.memory_format(memory_format), c10::nullopt);
  }
  // ...
  return result;
}
```

`empty`最终dispatch到`empty_cpu上`，首先拿一个cpu的allocator，然后调用`empty_generic`方法

```c++
// aten/src/ATen/EmptyTensor.cpp
TensorBase empty_cpu(IntArrayRef size, ScalarType dtype, bool pin_memory,
                     c10::optional<c10::MemoryFormat> memory_format_opt) {
  auto allocator = GetCPUAllocatorMaybePinned(pin_memory);
  constexpr c10::DispatchKeySet cpu_ks(c10::DispatchKey::CPU);
  return empty_generic(size, allocator, cpu_ks, dtype, memory_format_opt);
}

// c10/core/Allocator.cpp
C10_API at::Allocator* allocator_array[at::COMPILE_TIME_MAX_DEVICE_TYPES];

at::Allocator* GetAllocator(const at::DeviceType& t) {
  // 这里根据devicetype拿到对应类型的allocator，cpu就拿cpu，cuda就拿cuda
  auto* alloc = allocator_array[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}
```

`empty_generic`调用到`_empty_generic`方法

```c++
// pytorch/aten/src/ATen/EmptyTensor.cpp
template <typename T>
TensorBase _empty_generic(
    ArrayRef<T> size,
    c10::Allocator* allocator,
    c10::DispatchKeySet ks,
    ScalarType scalar_type,
    c10::optional<c10::MemoryFormat> memory_format_opt) {
  // ...
  // 计算需要分配的空间，然后实行分配，拿到storage指针
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  // 使用storage指针创建tensor(实际上是TensorBase类型)
  // 此时shape，stride等已经计算好并填入了（NCHW的形式）
  // 我们按照文章一开始的用例，此处stride为 [1280,20,4,1]
  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  // ...

  if (memory_format_opt.has_value()) {
    // 此处仅仅改了stride，并不需要改变tensor内存排布（因为只是空tensor）
    // 如一开始用例的话，此处stride改变为[1280, 1, 256, 64]
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}
```

`_empty_generic`调用完毕后，新tensor便创建好了，对于stride计算有疑问的小伙伴们可以看笔者的另外一篇文章[memory_format](../memory_format/index.zh-cn.md)

创建好后，一路返回到redispatch的clone算子处

```c++
// aten/src/ATen/native/TensorFactories.cpp
Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
  // ...
  Tensor self;
  if (memory_format == MemoryFormat::Preserve) {
    // ...
  } else {
    // 创建空tensor
    self = at::empty_like(src, src.options(), memory_format);
  }

  if (src._is_zerotensor()) {
    self.zero_();
  } else {
    self.copy_(src);
  }
  return self;
}
```

如果源tensor为空，那就直接set zero，如果不是，那么就调用`copy_`算子
