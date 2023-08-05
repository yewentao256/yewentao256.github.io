---
title: "PyTorch Under the Hood: A Deep Dive into the Contiguous Operator(1)"
date: 2023-03-11T09:53:09+08:00
categories: ["pytorch"]
summary: "Uncover the inner workings of PyTorch through a deep dive into the `contiguous` operator, from its Python interface to its dispatching and registration process, and finally how it is executed."
---

## Summary

Uncover the inner workings of PyTorch through a deep dive into the `contiguous` operator, from its Python interface to its dispatching and registration process, and finally how it is executed.

## 0. Introduction

Let's begin with code:

```py
import torch

N, C, H, W = 1, 64, 5, 4
x = torch.rand(N, C, H, W)
x = x.contiguous(memory_format=torch.channels_last)
print(x.shape)              # torch.Size([1, 64, 5, 4])
print(x.stride())           # (1280, 1, 256, 64)
print(x.is_contiguous())    # False
```

It converts the NCHW memory layout to the NHWC (channel last) memory layout, thereby achieving better performance improvement in certain specific scenarios (such as `conv2d`).

How is `contiguous` exported to the Python layer? What is the underlying logic of its actual operation? We will go down layer by layer and finally link up the call chain to unveil the process of PyTorch calling operators.

## 1. C++ to Python: How is `contiguous` exported

The Python layer does not have any extra encapsulation for `contiguous`, and directly uses the pyi declaration exported by C++.

```py
# torch/_C/__init__.pyi

# Defined in torch/csrc/autograd/python_variable.cpp
class _TensorMeta(type): ...

# Defined in torch/csrc/autograd/python_variable.cpp
class _TensorBase(metaclass=_TensorMeta):
    def contiguous(self, memory_format=torch.contiguous_format) -> Tensor: ...
```

As we can see, `contiguous` is a class method of `_TensorBase`. `_TensorBase` uses `_TensorMeta` as a metaclass (a Python mechanism that can dynamically modify the properties or methods inside a class).

How is `_TensorBase` exported to the Python layer? PyTorch uses the built-in **PyModuleDef** mechanism of Python to create the `torchmodule`, then calls `THPVariable_initModule` and exports through `PyModule_AddObject`.

```c++
// torch/csrc/Module.cpp
PyObject* initModule() {
  // ...
  static struct PyModuleDef torchmodule = {
      PyModuleDef_HEAD_INIT, "torch._C", nullptr, -1, methods.data()};
  ASSERT_TRUE(module = PyModule_Create(&torchmodule));
  ASSERT_TRUE(THPVariable_initModule(module));
  // ...
}

// torch/csrc/autograd/python_variable.cpp
bool THPVariable_initModule(PyObject* module) {
  // ....
  PyModule_AddObject(module, "_TensorMeta", (PyObject*)&THPVariableMetaType);
  // ....
  static std::vector<PyMethodDef> methods;
  THPUtils_addPyMethodDefs(methods, torch::autograd::variable_methods);
  THPUtils_addPyMethodDefs(methods, extra_methods);
  // add `methods` to `THPVariableType.tp_methods`
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  // ....
  return true;
}
```

Our `contiguous` method is located in `variable_methods`, and is then exported to the Python layer as a member method of `_TensorBase`.

## 2. Brief introduction to code generation: `native_functions.yaml` and `variable_methods`

`variable_methods` is defined in `tools/autograd/templates/python_variable_methods.cpp`.

```c++
// tools/autograd/templates/python_variable_methods.cpp
PyMethodDef variable_methods[] = {
  // ... other functions
  {"contiguous", castPyCFunctionWithKeywords(THPVariable_contiguous), METH_VARARGS | METH_KEYWORDS, NULL},
  ${py_method_defs}
}
```

However, note that this is just a template and not the actual code that is compiled and run. There is a lot of similar function code in operator development. To reduce the amount of duplicate work, PyTorch introduces a **code generation mechanism**. Simply put, it generates code based on `native_functions.yaml` and templates. The specific logic can be seen in `torchgen/gen.py`, but we will not delve into it here.

After compiling PyTorch, we can see more content in the generated folder, such as the newly generated `unsqueeze`.

```c++
// torch/csrc/autograd/generated/python_variable_methods.cpp
PyMethodDef variable_methods[] = {
  // other functions
  {"contiguous", castPyCFunctionWithKeywords(THPVariable_contiguous), METH_VARARGS | METH_KEYWORDS, NULL},

  // generated new functions
  {"unsqueeze", castPyCFunctionWithKeywords(THPVariable_unsqueeze), METH_VARARGS | METH_KEYWORDS, NULL},
  {"unsqueeze_", castPyCFunctionWithKeywords(THPVariable_unsqueeze_), METH_VARARGS | METH_KEYWORDS, NULL},
}
```

`unsqueeze_` comes from the definition in `native_functions.yaml`, replacing `${py_method_defs}` in the template.

```yaml
- func: unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
  variants: method
  device_check: NoCheck
  device_guard: False
  tags: inplace_view
  dispatch:
    CompositeExplicitAutograd: unsqueeze_
```

- `func`: Describes the function name, parameters, output types, etc.
- `variants`: `method` or `function`, indicating whether to generate tensor methods or standalone functions.
- `device_check`: Ensures that all tensors passed to the kernel are on the same device.
- `device_guard`: Ensures that the kernel is executed on the specified device (matching the device of the first tensor argument).
- `dispatch`: Specifies the backend and corresponding function. `CompositeExplicitAutograd` refers to the explicit automatic differentiation dispatch key, and the differentiation rule needs to be stated in `derivative.yaml`. If it is `CompositeImplicitAutograd`, it is not necessary, as this is based on the underlying operators of the operator supporting automatic differentiation, such as `conv2d`.
- `tags`: Operator tags, see [link](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/tags.yaml) for details.

It's worth noting that, due to the complexity of the `contiguous` code, the full content is already in `tools/autograd/templates/python_variable_methods.cpp`, and is not generated via `{py_method_defs}`.

## 3. Calling `contiguous`: Before dispatch

Note: The operator we are following is the ATen operator, not the `torchprim` version. The author compiled PyTorch based on **CPU**, so we are not following the CUDA (cuDNN/Triton) path.

If you want to debug the CPP part using gdb, please set the environment variable `export DEBUG=1` before compiling. If you want to see the call chain during runtime, you can set `export TORCH_SHOW_DISPATCH_TRACE=1`.

As mentioned earlier, the `contiguous` function we put into `tensorbase` is `THPVariable_contiguous`. This is the function that directly interacts with the Python layer, responsible for parsing parameters, executing calls, etc.

```c++
// torch/csrc/autograd/generated/python_variable_methods.cpp
static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static PythonArgParser parser({
    "contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);
  // parse self to `at::Tensor`
  auto& self_ = THPVariable_Unpack(self);
  auto memory_format = r.memoryformat(0);
  if (self_.is_contiguous(memory_format)) {
    // jit::tracer does something ...
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_, memory_format));
}
```

It parses Python parameters, then checks if the current tensor `is_contiguous` for the required `memory_format`. If it is, it returns directly, otherwise it calls `dispatch_contiguous`. We will expand on the specifics of `is_contiguous()` later.

```c++
// torch/csrc/autograd/generated/python_variable_methods.cpp
static Tensor dispatch_contiguous(const Tensor & self, at::MemoryFormat memory_format) {
  // release `Global Interpreter Lock (GIL)`
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.contiguous(memory_format);
}
```

`pybind11::gil_scoped_release` releases the `Global Interpreter Lock (GIL)` to improve performance (pybind11 does not release it implicitly, everything is controlled by the user. If you need to access Python objects after release, you must require it, see [pybind11-gil](https://pybind11.readthedocs.io/en/stable/advanced/misc.html)). Here, since we have parsed all parameters into C++ parameters, we can release the GIL.

`OptionalDeviceGuard device_guard` is a type of **RAII** (Resource Acquisition Is Initialization) guard, which is set to a certain device in the constructor and unset in the destructor. Compared to `DeviceGuard`, `OptionalDeviceGuard` allows passing a nullopt, equivalent to `optional<DeviceGuard>`. We won't expand on this here. Interested readers can refer to `c10/core/DeviceGuard.h`.

Then it calls `self.contiguous()`.

```c++
// build/aten/src/ATen/core/TensorBody.h
class TORCH_API Tensor: public TensorBase {
  // ....
  Tensor contiguous(MemoryFormat memory_format=MemoryFormat::Contiguous) const {
    return TensorBase::contiguous(memory_format);
  }
}

// aten/src/ATen/core/TensorBase.h
class TORCH_API TensorBase {
  // ...
  TensorBase contiguous(MemoryFormat memory_format=MemoryFormat::Contiguous) const {
    if (is_contiguous(memory_format)) {
      return *this;
    } else {
      return __dispatch_contiguous(memory_format);
    }
  }
}
```

Attentive readers may notice that it calls the `is_contiguous` method again in `tensorbase`. Is this a repetition of what's in `THPVariable_contiguous`? For our example where it's called from Python, it is indeed redundant, but `contiguous` doesn't just have one entry point at the Python layer, other tensors in the C++ layer may also call it, so it's necessary to include it here.

Can't we just skip the check at the Python layer and check it here instead? In theory, we can, but that would add some overhead to the call flow and reduce efficiency. We will go over the logic of `is_contiguous` later. Since it's stored as a variable, `is_contiguous` runs very efficiently, so it's ok to call `is_contiguous` multiple times.

Then it calls the `__dispatch_contiguous()` method of `TensorBase`.

```c++
// aten/src/ATen/core/Tensor.cpp
TensorBase TensorBase::__dispatch_contiguous(c10::MemoryFormat memory_format) const {
  OptionalTensorRef self(*this);
  return at::_ops::contiguous::call(*self, memory_format);
}
```

Note that here, `tensorbase` is converted to `OptionalTensorRef self`, which changes the call from a member method to a function method, i.e., `self` becomes a **parameter** for the subsequent call to the `contiguous` operator.

This also corresponds to the parameter declaration in `native_functions.yaml`: `aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)`.

## 4. Dispatch `contiguous` operator: Find schema and call kernel

We call `at::_ops::contiguous::call()` to get to the file `Operators_4.cpp`, which is generated based on `native_functions.yaml`.

**Dispatching** is a two-step process: first, find the function schema, and second, call the kernel that meets the conditions in the schema (e.g., dispatch the CPU tensor to the CPU kernel, dispatch the CUDA tensor to the CUDA kernel, etc. We will expand on this process later).

```c++
// build/aten/src/ATen/Operators_4.cpp
at::Tensor contiguous::call(const at::Tensor & self, at::MemoryFormat memory_format) {
    static auto op = create_contiguous_typed_handle();
    return op.call(self, memory_format);
}

static C10_NOINLINE c10::TypedOperatorHandle<contiguous::schema> create_contiguous_typed_handle() {
  return c10::Dispatcher::singleton()
      .findSchemaOrThrow(contiguous::name, contiguous::overload_name)
      .typed<contiguous::schema>();
}
```

The `contiguous::name/overload_name` here comes from `continuous_ops.h` (generated code).

```c++
// build/aten/src/ATen/ops/contiguous_ops.h
struct TORCH_API contiguous {
  using schema = at::Tensor (const at::Tensor &, at::MemoryFormat);
  using ptr_schema = schema*;
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::contiguous")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "")
  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, "contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)")
  static at::Tensor call(const at::Tensor & self, at::MemoryFormat memory_format);
  static at::Tensor redispatch(c10::DispatchKeySet dispatchKeySet, const at::Tensor & self, at::MemoryFormat memory_format);
};
```

We'll expand on the process of obtaining an op. First, we get a singleton of `Dispatcher`.

```c++
// aten/src/ATen/core/dispatch/Dispatcher.h
class TORCH_API Dispatcher final {
  C10_ALWAYS_INLINE static Dispatcher& singleton() {
    static Dispatcher& s = realSingleton();
    return s;
  }
}
// aten/src/ATen/core/dispatch/Dispatcher.cpp
C10_EXPORT Dispatcher& Dispatcher::realSingleton() {
  static Dispatcher _singleton;
  return _singleton;
}
```

Then we take the singleton of the dispatcher to `findSchemaOrThrow()`.

```c++
// aten/src/ATen/core/dispatch/Dispatcher.cpp
OperatorHandle Dispatcher::findSchemaOrThrow(const char* name, const char* overload_name) {
  // here name = "aten::contiguous", overload_name = ""
  auto it = findSchema({name, overload_name});
  if (!it.has_value()) {
    auto it2 = findOp({name, overload_name});
    // ...
  }
  return it.value();
}
c10::optional<OperatorHandle> Dispatcher::findSchema(const OperatorName& overload_name) {
  // (const c10::OperatorName) (name = "aten::contiguous", overload_name = "")
  auto it = findOp(overload_name);
  if (it.has_value()) {
    if (it->hasSchema()) {
      return it;
    } else {
      return c10::nullopt;
    }
  } else {
    return it;
  }
}
c10::optional<OperatorHandle> Dispatcher::findOp(const OperatorName& overload_name) {
  return operatorLookupTable_.read(
    [&] (const ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) -> c10::optional<OperatorHandle> {
    auto found = operatorLookupTable.find(overload_name);
    if (found == operatorLookupTable.end()) {
      return c10::nullopt;
    }
    return found->second;
  }
  );
}
```

The `operatorLookupTable_` here is a private variable declared in `Dispatcher.h`: `LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;`. In simple terms, it's a hash table. We pass an anonymous function into it to look up the name in the hash table. If it's found, the function returns the found `OperatorHandle`; if it's not found, it returns `nullopt`.

```c++
template <class T>
class LeftRight final {
  template <typename F>
  auto read(F&& readFunc) const -> typename c10::invoke_result_t<F, const T&> {
    // ...

    // through _data[_foregroundDataIndex.load()] we get operatorLookupTable
    return readFunc(_data[_foregroundDataIndex.load()]);
  }
}
```

Here we find the corresponding `c10::OptionalBase<c10::OperatorHandle>` op and return it. After going through `typed()`, it eventually generates `c10::TypedOperatorHandle<at::Tensor (const at::Tensor &, c10::MemoryFormat)>` for the outer static variable `op`.

At this point, the first step of finding the schema is complete. We then start to find and call the kernel.

```c++
// build/aten/src/ATen/Operators_4.cpp
at::Tensor contiguous::call(const at::Tensor & self, at::MemoryFormat memory_format) {
    static auto op = create_contiguous_typed_handle();
    return op.call(self, memory_format);
}
```

Then we call the `call` method.

```c++
// aten/src/ATen/core/dispatch/Dispatcher.h
template<class Return, class... Args>
class TypedOperatorHandle<Return (Args...)> final : public OperatorHandle {

  // ...
  C10_ALWAYS_INLINE Return call(Args... args) const {
    return c10::Dispatcher::singleton().call<Return, Args...>(*this, std::forward<Args>(args)...);
  }
}

template<class Return, class... Args>
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(const TypedOperatorHandle<Return(Args...)>& op, Args... args) const {
  // ...
  // Calculate the optimal dispatch key set based on tensor and other parameters
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
  // Find the kernel in operatorHandle based on the dispatch key set
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
  // ...
  // Finally, call the kernel
  return kernel.template call<Return, Args...>(op, dispatchKeySet, std::forward<Args>(args)...);
}

// aten/src/ATen/core/dispatch/OperatorEntry.h
const KernelFunction& lookup(DispatchKeySet ks) const {
    const auto idx = ks.getDispatchTableIndexForDispatchKeySet();
    const auto& kernel = dispatchTable_[idx];
    // ... some check
    return kernel;
  }
```

In the `call` method, we first calculate a `dispatchKeySet`, then enter `op.lookup` to calculate `idx` based on the `dispatchKeySet`, and finally find the final dispatched kernel function in the `dispatchTable_`, and call its template function `call`.

`dispatchKeySet` is a `uint64_t` bitset, where each dispatch key represents a bit, and a larger bit index indicates a higher priority. For example, if a tensor's device is specified as `cuda`, the dispatch key set might be `{AutogradCUDA | CUDA | ADInplaceOrView}`, and it will first dispatch to `AutogradCUDA` for some automatic differentiation processing, and then `redispatch` to `CUDA`.

Here it is worth pointing out that `ADInplaceOrView` is a special dispatch key, registered specifically for inplace and view operations, to provide additional settings for subsequent autograd calculations.

- For inplace operations, it adds a `version counter`. When the autograd engine performs a backward operation, it checks the version. If the tensor that needs to perform gradient calculation has been operated on inplace, it will report an error to avoid incorrect gradient calculation. This part of the code is in `torch/csrc/autograd/generated/ADInplaceOrViewTypeEverything.cpp`.
- The same principle applies to `view`, which prevents any modifications to the view tensor to avoid incorrect gradient calculations (because the `view` tensor and the original tensor share storage).

```c++
// aten/src/ATen/core/boxing/KernelFunction_impl.h
template<class Return, class... Args>
C10_ALWAYS_INLINE Return KernelFunction::call(const OperatorHandle& opHandle, DispatchKeySet dispatchKeySet, Args... args) const {
    if (guts::disjunction<has_symint<Args>...>::value) {
      // ... get inlined by compiler
    } else {
      if (C10_LIKELY(unboxed_kernel_func_ != nullptr)) {
        auto *functor = boxed_kernel_func_.getFunctor();
        return callUnboxedKernelFunction<Return, Args...>(
            unboxed_kernel_func_, functor, dispatchKeySet, std::forward<Args>(args)...);
      }
    }

    return impl::BoxedKernelWrapper<Return(Args...)>::call(
        boxed_kernel_func_, opHandle, dispatchKeySet, std::forward<Args>(args)...
    );
}
```

Here, if `unboxed_kernel_func_` is not null, it retrieves the `functor` from `boxed_kernel_func_`, and then calls `callUnboxedKernelFunction<Return, Args...>`.

`unboxed` refers to unpackaged functions, which include complete signatures and parameters, etc. Packaged `boxed` functions are intuitively understood as compressing all parameters into a whole, such as the `stack` in `void conjugateFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack)`. This way, you don't have to write a function signature for each parameter variant, you can reuse code to the maximum extent, the compiled binary occupies less space, and it's convenient for deployment on mobile devices, but the process of packing and unpacking can affect efficiency to some extent.

```c++
// aten/src/ATen/core/boxing/KernelFunction_impl.h
template<class Return, class... Args>
inline Return callUnboxedKernelFunction(void* unboxed_kernel_func, OperatorKernel* functor, DispatchKeySet dispatchKeySet, Args&&... args) {
    using ActualSignature = Return (OperatorKernel*, DispatchKeySet, Args...);
    ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func);
    // here functor: &(at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__contiguous(at::Tensor const&, c10::MemoryFormat))>
    return (*func)(functor, dispatchKeySet, std::forward<Args>(args)...);
}
```

Then it enters `wrap_kernel_functor_unboxed_` and calls the `call` function.

```c++
// aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h
template<class KernelFunctor, class ReturnType, class... ParameterTypes>
struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
  static ReturnType call(OperatorKernel* functor, DispatchKeySet, ParameterTypes... args) {
    // Note that there are no dispatch keys here
    KernelFunctor* functor_ = static_cast<KernelFunctor*>(functor);
    return (*functor_)(std::forward<ParameterTypes>(args)...);
  }
};

// aten/src/ATen/core/boxing/impl/WrapFunctionIntoFunctor.h
template<class FuncPtr, class ReturnType, class... Parameters>
class WrapFunctionIntoFunctor_<FuncPtr, ReturnType, guts::typelist::typelist<Parameters...>> final : public c10::OperatorKernel {
public:
  C10_ALWAYS_INLINE decltype(auto) operator()(Parameters... args) {
    return (*FuncPtr::func_ptr())(std::forward<Parameters>(args)...);
  }
};
```

After that, we peel off the layers of encapsulation and dispatch to the actual functor (depending on the compilation options, tensor types, etc., the kernel dispatched here will differ).

Here it dispatches to **CompositeImplicitAutograd**. The meaning of this dispatch key is a combination of non-explicit automatic differentiation. It does not need to write a separate differentiation function like `ExplicitAutograd`. It depends on the underlying other operators being able to implement automatic differentiation.

```c++
// build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
at::Tensor wrapper_CompositeImplicitAutograd__contiguous(const at::Tensor & self, at::MemoryFormat memory_format) {
  return at::native::contiguous(self, memory_format);
}
```

Finally, it calls into the native contiguous (`aten/src/ATen/native/TensorProperties.cpp`), and the operator dispatch process is complete.

## 5. Why can we find the `contiguous` operator in the table: Operator registration

In the previous sections, we traced the dispatch process of `contiguous`. But dispatching necessarily implies registration. How is the schema of the `contiguous` operator registered in the `OperatorHandle`? How is its kernel registered in the `dispatchTable_`?

Before we explain the registration process of the `contiguous` operator, let's first get a brief understanding of the general PyTorch operator registration process, which is done through the `TORCH_LIBRARY(ns, m)` and `TORCH_LIBRARY_IMPL(ns, k, m)` macros in two steps.

```c++
// torch/library.h
#define TORCH_LIBRARY(ns, m)    \
  static void TORCH_LIBRARY_init_##ns(torch::Library&);     \
  static const torch::detail::TorchLibraryInit TORCH_LIBRARY_static_init_##ns( \
      torch::Library::DEF, &TORCH_LIBRARY_init_##ns, \
      #ns,   \
      c10::nullopt,  \
      __FILE__,  \
      __LINE__);     \
  void TORCH_LIBRARY_init_##ns(torch::Library& m)

#define TORCH_LIBRARY_IMPL(ns, k, m) _TORCH_LIBRARY_IMPL(ns, k, m, C10_UID)
```

First, the `TORCH_LIBRARY(ns, m)` macro is called to register the schema under the `ns` namespace (essentially writing into the `OperatorEntry.schema_` field through the **Dispatcher**). At this point, only an empty dispatch table exists, and the specific kernel has not been registered.

```c++
// build/aten/src/ATen/RegisterSchema.cpp
TORCH_LIBRARY(aten, m) {
  m.def("batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor", {});
  m.def("contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)", {});
}
```

Then, `TORCH_LIBRARY_IMPL(ns, k, m)` is called to register the specific implementation of the operator (essentially writing into the `OperatorEntry.dispatchTable_` field through the **Dispatcher**), binding the specific dispatch key, such as `CompositeImplicitAutograd`, `CPU`, `CUDA`, etc. There are some special designs, like **catchall**, will spread and write into all dispatch keys. Using `BackendSelect` to implement `fallback` will redispatch to the next priority dispatch key, etc.

For example:

```c++
// build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
TORCH_LIBRARY_IMPL(aten, CompositeImplicitAutograd, m) {
  // lots of ops
  m.impl("batch_norm", TORCH_FN(wrapper_CompositeImplicitAutograd__batch_norm));
  m.impl("contiguous", TORCH_FN(wrapper_CompositeImplicitAutograd__contiguous));
}
```

Having understood the basic operator registration method, we'll detail the operator registration process:

First, let's expand the macro for `TORCH_LIBRARY_IMPL`.

```c++
// torch/library.h
// `C10_UID` is an unique identifier
#define _TORCH_LIBRARY_IMPL(ns, k, m, uid)  \
  static void C10_CONCATENATE(   \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library&); \
  static const torch::detail::TorchLibraryInit C10_CONCATENATE( \
      TORCH_LIBRARY_IMPL_static_init_##ns##_##k##_, uid)( \
      torch::Library::IMPL, \
      c10::guts::if_constexpr<c10::impl::dispatch_key_allowlist_check( \
          c10::DispatchKey::k)>(\
          []() { return &C10_CONCATENATE( \
                TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid); \
          },  \
          []() { return [](torch::Library&) -> void {}; }), \
      #ns, \
      c10::make_optional(c10::DispatchKey::k), \
      __FILE__,   \
      __LINE__);  \
  void C10_CONCATENATE( \
      TORCH_LIBRARY_IMPL_init_##ns##_##k##_, uid)(torch::Library & m)

static void TORCH_LIBRARY_IMPL_init_aten_CompositeImplicitAutograd_12(torch::Library&);
static const torch::detail::TorchLibraryInit
    TORCH_LIBRARY_IMPL_static_init_aten_CompositeImplicitAutograd_12(
        torch::Library::IMPL,
        c10::guts::if_constexpr<c10::impl::dispatch_key_allowlist_check(
            c10::DispatchKey::CompositeImplicitAutograd)>([]() {
              return &TORCH_LIBRARY_IMPL_init_aten_CompositeImplicitAutograd_12;
            }, []() { return [](torch::Library&) -> void {}; }),
        "aten",
        c10::make_optional(c10::DispatchKey::CompositeImplicitAutograd),
        "pytorch/build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp",
        7156);
void TORCH_LIBRARY_IMPL_init_aten_CompositeImplicitAutograd_12(
    torch::Library& m) {
  m.impl("batch_norm", TORCH_FN(wrapper_CompositeImplicitAutograd__batch_norm));
  m.impl("contiguous", ::c10::CompileTimeFunctionPointer< std::remove_pointer_t<std::remove_reference_t<decltype(wrapper_CompositeImplicitAutograd__contiguous)>>, wrapper_CompositeImplicitAutograd__contiguous>());
}
```

`TORCH_LIBRARY_IMPL_init_aten_CompositeImplicitAutograd_12` will be called by **TorchLibraryInit** when we `import torch`. We won't go into detail here, but let's focus on what happens with `m.impl`.

```c++
// torch/library.h
class TORCH_API Library final {
  template <typename Name, typename Func>
  Library& impl(Name name, Func&& raw_f, _RegisterOrVerify rv = _RegisterOrVerify::REGISTER) & {
#if defined C10_MOBILE
    CppFunction f(std::forward<Func>(raw_f), NoInferSchemaTag());
#else
    CppFunction f(std::forward<Func>(raw_f));
#endif
    return _impl(name, std::move(f), rv);
  }
}

class TORCH_API CppFunction final {
  template <typename Lambda>
  explicit CppFunction(
      Lambda&& f,
      std::enable_if_t<
          c10::guts::is_functor<std::decay_t<Lambda>>::value,
          std::nullptr_t> = nullptr)
      : func_(c10::KernelFunction::makeFromUnboxedLambda(
            std::forward<Lambda>(f))),
        cpp_signature_(c10::impl::CppSignature::make<Lambda>()),
        schema_(c10::detail::inferFunctionSchemaFromFunctor<
                std::decay_t<Lambda>>()),
        debug_() {}
}
```

Here, `func_`, `cpp_signature_`, and `schema_` are initialized using `CppFunction`.

`func_` is a function pointer, which we'll focus on later. `cpp_signature_` is the function signature. If the kernel is created in a way where we can know the function signature (for example, an `unboxed c++ function`), then we store it and use it for checking in later kernel registration and calls.

We'll focus on the construction of `func_`.

```c++
// aten/src/ATen/core/boxing/KernelFunction_impl.h
template<class FuncPtr, bool AllowLegacyTypes>
inline KernelFunction KernelFunction::makeFromUnboxedFunction(FuncPtr func_ptr) {
  // ... c10 mobile alias code
  return makeFromUnboxedFunctor<AllowLegacyTypes, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>(
        guts::make_unique_base<OperatorKernel, typename impl::WrapFunctionIntoFunctor<FuncPtr>::type>()
    );
}

template<bool AllowLegacyTypes, class KernelFunctor>
inline KernelFunction KernelFunction::makeFromUnboxedFunctor(std::unique_ptr<OperatorKernel> kernelFunctor) {

    auto* unboxed_fn = &impl::wrap_kernel_functor_unboxed<KernelFunctor>::call;
    void* void_unboxed_fn = reinterpret_cast<void*>(unboxed_fn);
    bool is_symint = fn_has_symint<decltype(unboxed_fn)>::value;
    return KernelFunction(
        std::move(kernelFunctor),
        &impl::make_boxed_from_unboxed_functor<KernelFunctor, AllowLegacyTypes>::call,
        is_symint ? nullptr : void_unboxed_fn,
        is_symint ? void_unboxed_fn : nullptr
    );
}
```

Eventually, we encapsulate `raw_f` into `KernelFunction`, return it to the outer `CppFunction`, and let it complete the initialization. Then, we call `_impl(name, std::move(f), rv)` for further processing.

```c++
// aten/src/ATen/core/library.cpp
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName name = _parseNameForLib(name_str);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  // here dispatch_key is c10::OptionalBase<c10::DispatchKey> = { init_ = true, storage_ = (dummy_ = '|', value_ = CompositeImplicitAutograd)}
  switch (rv) {
    case _RegisterOrVerify::REGISTER:
      registrars_.emplace_back(
        c10::Dispatcher::singleton().registerImpl(
          std::move(name),
          dispatch_key,
          std::move(f.func_),
          std::move(f.cpp_signature_),
          std::move(f.schema_),
          debugString(std::move(f.debug_), file_, line_)
        )
      );
      break;
    case _RegisterOrVerify::VERIFY:
      c10::Dispatcher::singleton().waitForImpl(name, dispatch_key);
      break;
  }
  return *this;
}
```

We find the familiar object `c10::Dispatcher::singleton()`. Here in registration, we call `c10::Dispatcher::singleton().registerImpl()` to register our encapsulated kernel function (`f.func_`) and signature, schema, etc. into the dispatcher.

```c++
// aten/src/ATen/core/dispatch/Dispatcher.cpp
RegistrationHandleRAII Dispatcher::registerImpl(
  OperatorName op_name,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<impl::CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  std::lock_guard<std::mutex> lock(mutex_);

  // 1. register schema
  auto op = findOrRegisterName_(op_name);

  // 2. register kernel
  auto handle = op.operatorDef_->op.registerKernel(
    *this,
    dispatch_key,
    std::move(kernel),
    std::move(cpp_signature),
    std::move(inferred_function_schema),
    std::move(debug)
  );

  ++op.operatorDef_->def_and_impl_count;

  cond_var_.notify_all();
  // RegistrationHandleRAII automatically recycles resources.
  // This object registers the anonymous function `deregisterImpl_`,
  // which will automatically deregister the kernel function of the operator when the object is destroyed.
  // It's a standard RAII design.
  return RegistrationHandleRAII([this, op, op_name, dispatch_key, handle] {
    deregisterImpl_(op, op_name, dispatch_key, handle);
  });
}

OperatorHandle Dispatcher::findOrRegisterName_(const OperatorName& op_name) {
  const auto found = findOp(op_name);
  if (found != c10::nullopt) {
    return *found;
  }

  operators_.emplace_back(OperatorName(op_name));
  OperatorHandle handle(--operators_.end());
  operatorLookupTable_.write([&] (ska::flat_hash_map<OperatorName, OperatorHandle>& operatorLookupTable) {
    operatorLookupTable.emplace(op_name, handle);
  });

  return handle;
}
```

First, **register schema**: Check if the operator has already been registered in `operatorLookupTable_`. If it has been registered, it returns directly; if not, it writes into the table.

Then, **register kernel**: Call `op.operatorDef_->op.registerKernel()` to register the previously encapsulated kernel function into this `OperatorEntry`.

```c++
// aten/src/ATen/core/dispatch/OperatorEntry.cpp
OperatorEntry::AnnotatedKernelContainerIterator OperatorEntry::registerKernel(
  const c10::Dispatcher& dispatcher,
  c10::optional<DispatchKey> dispatch_key,
  KernelFunction kernel,
  c10::optional<CppSignature> cpp_signature,
  std::unique_ptr<FunctionSchema> inferred_function_schema,
  std::string debug
) {
  // check schema ...

  // Add the kernel to the kernel list. If it's the first kernel, create the list.
  // Redirect catchAll registration to CompositeImplicitAutograd.
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];

  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
  AnnotatedKernelContainerIterator inserted = k.begin();
  // update dispatch table
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;
}
```

Here, it finds `k` (the list of kernels: `(std::list<c10::impl::AnnotatedKernel, std::allocator<c10::impl::AnnotatedKernel> >)`) in `kernels_` through `dispatch_key`, and inserts the kernel at the beginning of the list.

Then it updates the dispatcher's entry. At this point, `registerImpl` has completed the registration of the kernel for the operator.

Finally, it returns the `*this` pointer. `m.impl("contiguous", TORCH_FN(wrapper_CompositeImplicitAutograd__contiguous));` completes.
