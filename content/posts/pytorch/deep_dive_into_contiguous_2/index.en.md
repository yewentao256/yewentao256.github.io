---
title: "PyTorch Under the Hood: A Deep Dive into the Contiguous Operator(2)"
date: 2023-04-12T09:53:09+08:00
categories: ["pytorch"]
summary: "Uncover the inner workings of PyTorch through a deep dive into the `contiguous` operator, from its Python interface to its dispatching and registration process, and finally how it is executed."
---

## Summary

Uncover the inner workings of PyTorch through a deep dive into the `contiguous` operator, from its Python interface to its dispatching and registration process, and finally how it is executed.

## 6. Review of register and dispatch

Looking at the processes of register and dispatch, we can summarize their overall procedures as:

1. Registering op schema
2. Registering specific kernel under the op schema (based on dispatch key)
3. Looking up op schema
4. Looking up and invoking specific kernel of the op (based on dispatch key)

Several important data types in the middle: `Dispatcher`, `OperatorHandle`, `OperatorEntry`

- **Dispatcher**
  - `operatorLookupTable_` maintains a mapping from `OperatorName` to `OperatorHandle`

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
  - Its internal `operatorDef_` is essentially the `OperatorDef` in the above Dispatcher, which is a wrapper for `OperatorEntry`
  - More often, `TypedOperatorHandle` is used, which is a subclass of `OperatorHandle` and can be understood as an `OperatorHandle` templated for op parameters

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

- **OperatorEntry**
  - The data structure that actually stores op information

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

Through the calling chain of **Library** -> **Dispatcher** -> **OperatorHandle** -> **OperatorEntry**, PyTorch completes the registration of the op and its corresponding kernel. After that, PyTorch can easily call kernel functions based on this chain.

## 7. `is_contiguous` determines whether it is contiguous

Having roughly understood the registration and invocation process of PyTorch operators, we finally enter the actual execution process of the contiguous operator, which is relatively simple.

We bring the call path back to the end of the dispatch section mentioned earlier.

```c++
// build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
at::Tensor wrapper_CompositeImplicitAutograd__contiguous(const at::Tensor & self, at::MemoryFormat memory_format) {
  return at::native::contiguous(self, memory_format);
}
```

Here, we call the `contiguous` operator of aten native.

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

First, it judges `is_contiguous(memory_format)`, that is, whether it is already contiguous under the specified memory format. After going through `TensorBase.h`, we come to `TensorImpl.h`.

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

As we can see, PyTorch's `is_contiguous` judgment does not involve computation. Instead, it directly stores the data in `TensorImpl`, and each time it is directly taken and used. This saves computation, but it also requires calculating the related boolean value and storing it when initializing or changing the tensor stride.

So, how is it set? Tracing the set process of this variable, we find that it is set by `refresh_contiguous()`. This method must be called every time the tensor's shape or stride is modified.

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

Let's look at `_compute_channels_last_contiguous_2d`. Essentially, it determines whether it conforms to NHWC (a permutation of `1 3 2 0`) under the contiguous (NCHW) standard:

```c++
template <typename T>
bool _compute_channels_last_contiguous_2d(
    ArrayRef<T> sizes,
    ArrayRef<T> strides) {
  switch (sizes.size()) {
    case 4: {
      T expected = 1;
      // const array can be `unrolling` and accelerated by compiler
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

For example, a tensor with `N, C, H, W = 2, 2048, 1, 1` has a stride of `[2048, 1, 1, 1]`. It is a channels last tensor (and also a contiguous tensor, because the memory layout happens to have h, w all being 1).

## 8. Clone Operator: Empty Tensor and Copy

Let's continue our call process, if the tensor is already `contiguous` under the specified memory format, it is directly returned. If it is not `contiguous`, it will be `clone` according to the specified memory format.

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

Look familiar? Yes, this is the entry point we used to call the `contiguous` operator above. After a similar dispatch process (finding the op schema and then the kernel), we arrive at the actual clone function. Since the dispatch key of the above contiguous operation is **CompositeImplicitAutograd**, the `clone` operator also calls this dispatch key and needs to handle the related logic of automatic differentiation.

This is consistent with our impression of the clone operator: a completely independent copy that retains the `requires_grad` attribute and supports automatic derivation (`CloneBackward0` is put into `grad_fn`).

```c++
// torch/csrc/autograd/generated/VariableType_1.cpp
at::Tensor clone(c10::DispatchKeySet ks, const at::Tensor & self, c10::optional<at::MemoryFormat> memory_format) {
  // Here `checked_cast_variable` is called to check whether the tensor is defined
  // self_ and self have the same address
  auto& self_ = unpack(self, "self", 0);

  // If automatic differentiation is required, set grad_fn into the graph
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
  // get storage and impl pointer of self
  c10::optional<Storage> self__storage_saved =
    self_.has_storage() ? c10::optional<Storage>(self_.storage()) : c10::nullopt;
  c10::intrusive_ptr<TensorImpl> self__impl_saved;
  if (self_.defined()) self__impl_saved = self_.getIntrusivePtr();
  #endif
  // After computing the current dispatchkey and c10::after_autograd_keyset, the clone operator is redispatched
  // redispatch is to get the actual result of the clone. We will expand on the process of redispatch below
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::clone(ks & c10::after_autograd_keyset, self_, memory_format);
  })();
  auto result = std::move(_tmp);
  // ...
  return result;
}
```

It's worth noting that in the process of **redispatch**, after recalculating the dispatchkey, the clone operator of aten is redispatched. What's the difference between `redispatch` and `call`?

On the one hand, it's a difference in function signatures. `redispatch` carries a `currentDispatchKeySet` parameter, so it doesn't need to take the dispatchkey from the op like `call` does.

```c++
Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const

inline Return Dispatcher::redispatch(const TypedOperatorHandle<Return (Args...)>& op, DispatchKeySet currentDispatchKeySet, Args... args) const
```

On the other hand, in the `redispatch` call, the current dispatchkey is generally downgraded one priority level (such as performing an AND operation with `c10::after_autograd_keyset`), and then scheduled to the operator that actually performs the clone (after handling automatic differentiation, automatic differentiation is no longer needed).

After redispatch, we go to `CompositeExplicitAutograd.cpp`.

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
    // create empty tensor
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

The `empty_like` operator is called to create an empty tensor the same as the src (but the memory format is new). After going through dispatch, we come to `TensorFactories.cpp`, and then `empty_like` calls the `empty` operator (first call to `build/aten/src/ATen/RegisterBackendSelect.cpp`, and then redispatch to **CPU** —— where it redispatches to will vary depending on compilation options and device).

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

`empty` finally dispatches to `empty_cpu`, first getting a CPU allocator, and then calling the `empty_generic` method.

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
   // Here, based on the device type, the corresponding type of allocator is obtained. If it's CPU, get CPU
  auto* alloc = allocator_array[static_cast<int>(t)];
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(alloc, "Allocator for ", t, " is not set.");
  return alloc;
}
```

And then calls `_empty_generic`

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
  // Calculate the space needed to be allocated, then allocate and get the storage pointer.
  caffe2::TypeMeta dtype = scalarTypeToTypeMeta(scalar_type);
  auto size_bytes = computeStorageNbytesContiguous(size, dtype.itemsize());
  auto storage_impl = c10::make_intrusive<StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size_bytes,
      allocator,
      /*resizeable=*/true);

  // Use the storage pointer to create a tensor (actually of type TensorBase)
  // At this point, shape, stride, etc. have been calculated and filled in (in NCHW format)
  // According to our use case, the stride here is [1280,20,4,1].
  auto tensor = detail::make_tensor_base<TensorImpl>(
      std::move(storage_impl), ks, dtype);
  // ...

  if (memory_format_opt.has_value()) {
    // Here, only the stride is changed, and the memory layout of the tensor
    // does not need to be changed (because it is just an empty tensor)
    // the stride here changes to [1280, 1, 256, 64].
    if (*memory_format_opt != MemoryFormat::Contiguous) {
      tensor.unsafeGetTensorImpl()->empty_tensor_restride(*memory_format_opt);
    }
  }

  return tensor;
}
```

After `_empty_generic` is called, the new tensor is created. If you have any questions about stride calculation, you can refer to another article I wrote [tensor_data_layout](../tensor_data_layout).

After creation, we go back to the redispatched clone operator location.

```c++
// aten/src/ATen/native/TensorFactories.cpp
Tensor clone(const Tensor& src, c10::optional<c10::MemoryFormat> optional_memory_format) {
  // ...
  Tensor self;
  if (memory_format == MemoryFormat::Preserve) {
    // ...
  } else {
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

If the source tensor is empty, it is directly set to zero. If not, the `copy_` operator is called.
