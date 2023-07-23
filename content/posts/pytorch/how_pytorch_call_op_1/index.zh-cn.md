---
title: "How Pytorch 2.0 Call Ops(一)"
date: 2023-03-11T09:53:09+08:00
categories: ["pytorch"]
summary: "本文以contiguous调用为例，介绍pytorch 2.0 调用算子的流程，并展开说明具体算子底层实现原理。"
---

## Summary

本文以contiguous调用为例，介绍pytorch 2.0 调用算子的流程，并展开说明具体算子底层实现原理。

## 0. 引入

我们首先看这么一段代码

```py
import torch

N, C, H, W = 1, 64, 5, 4
x = torch.rand(N, C, H, W)
x = x.contiguous(memory_format=torch.channels_last)
print(x.shape)              # torch.Size([1, 64, 5, 4])
print(x.stride())           # (1280, 1, 256, 64)
print(x.is_contiguous())    # False
```

它会将NCHW的内存分布转换为NHWC（channel last）的内存分布，进而在一些特定场景下取得更好的性能提升（如conv2d）

`contiguous`是如何被导出到python层的？其底层实际运行逻辑是怎样的呢？我们将一层层往下走，并最终将调用链路串联起来，揭开pytorch调用算子流程的面纱。

## 1. c++ 到 python：contiguous如何被导出

python层对于contiguous没有额外封装，直接使用c++导出的pyi声明

```py
# torch/_C/__init__.pyi

# Defined in torch/csrc/autograd/python_variable.cpp
class _TensorMeta(type): ...

# Defined in torch/csrc/autograd/python_variable.cpp
class _TensorBase(metaclass=_TensorMeta):
    def contiguous(self, memory_format=torch.contiguous_format) -> Tensor: ...
```

可以看到，`contiguous`是`_TensorBase`的一个类方法。`_TensorBase`使用`_TensorMeta`作为元类（一种python机制，可以动态地修改类内部的属性或方法）。

`_TensorBase`是如何被导出到python层的呢？pytorch使用python自带的**PyModuleDef**机制创建了`torchmodule`，随后调用`THPVariable_initModule`并通过`PyModule_AddObject`导出

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
  // 将`variable_methods`并放到`THPVariableType.tp_methods`中
  THPVariableType.tp_methods = methods.data();
  if (PyType_Ready(&THPVariableType) < 0)
    return false;
  Py_INCREF(&THPVariableType);
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  // ....
  return true;
}
```

我们的`contiguous`方法便位于`variable_methods`中，进而作为`_TensorBase`的成员方法被导出到python层。

## 2. 代码生成简述：`native_functions.yaml`和`variable_methods`

`variable_methods`被定义在`tools/autograd/templates/python_variable_methods.cpp`中。

```c++
// tools/autograd/templates/python_variable_methods.cpp
PyMethodDef variable_methods[] = {
  // ... other functions
  {"contiguous", castPyCFunctionWithKeywords(THPVariable_contiguous), METH_VARARGS | METH_KEYWORDS, NULL},
  ${py_method_defs}
}
```

但注意，此处仅仅是模板，并不是实际被编译运行的代码。实际上，算子开发中有很多函数代码相似，pytorch为了减少重复的工作量，引入了一种**代码生成机制**，简单来说是基于`native.yaml`和模板来生成代码，具体逻辑可见`torchgen/gen.py`，我们不过多展开。

在编译pytorch后，我们可以在generated文件夹下看到更多内容，如新生成的`unsqueeze`

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

`unsqueeze_`来自`native_functions.yaml`中的定义，替换了在模板中的`${py_method_defs}`

```yaml
- func: unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
  variants: method
  device_check: NoCheck
  device_guard: False
  tags: inplace_view
  dispatch:
    CompositeExplicitAutograd: unsqueeze_
```

- `func`：描述函数名称及参数、输出类型等
- `variants`：`method`或`function`，指生成tensor method或单独function
- `device_check`：确保传递给kernel的所有tensor在同一device上
- `device_guard`：确保kernel在指定设备下执行（匹配第一个tensor参数的设备）
- `dispatch`：指定后端与对应的函数。`CompositeExplicitAutograd`指的是显式自动微分dispatch key，需要在`derivative.yaml`写明微分规则。如果是`CompositeImplicitAutograd`则不需要，这是基于该算子底层算子都支持自动微分实现的，如`conv2d`。
- `tags`：算子标签，详见[链接](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/tags.yaml)

值得指出的是，由于`contiguous`代码较为复杂，所以在`tools/autograd/templates/python_variable_methods.cpp`中已经有了完整内容，并不是通过`{py_method_defs}`生成出来的。

## 3. contiguous的调用：在dispatch前

注意：我们调用流程走的是aten算子，而不是`torchprim`的版本算子。笔者是基于cpu编译的pytorch，没有走cuda（cudnn/triton）

如果读者想要gdb调试CPP部分，请设置环境变量`export DEBUG=1`再编译。如果希望运行时看到调用链路，可以设置`export TORCH_SHOW_DISPATCH_TRACE=1`。

由上文可知，我们放到`tensorbase`里的contiguous函数为`THPVariable_contiguous`，这里是直接与python层交互的函数，负责解析参数、执行调用等。

```c++
// torch/csrc/autograd/generated/python_variable_methods.cpp
static PyObject * THPVariable_contiguous(PyObject* self, PyObject* args, PyObject* kwargs)
{
  static PythonArgParser parser({
    "contiguous(*, MemoryFormat memory_format=contiguous_format)",
  });
  ParsedArgs<1> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);
  // 将self参数解析成`at::Tensor`
  auto& self_ = THPVariable_Unpack(self);
  auto memory_format = r.memoryformat(0);
  if (self_.is_contiguous(memory_format)) {
    // jit::tracer does something ...
    return self;
  }
  return THPVariable_Wrap(dispatch_contiguous(self_, memory_format));
}
```

简单而言就是解析python参数，随后判断当前tensor对于所需的`memory_format`是否`contiguous`，如果是的话直接返回，否则调用`dispatch_contiguous`。`is_contiguous()`的具体内容我们下文展开

```c++
// torch/csrc/autograd/generated/python_variable_methods.cpp
static Tensor dispatch_contiguous(const Tensor & self, at::MemoryFormat memory_format) {
  // 释放`Global Interpreter Lock (GIL)`
  pybind11::gil_scoped_release no_gil;
  OptionalDeviceGuard device_guard(device_of(self));
  return self.contiguous(memory_format);
}
```

`pybind11::gil_scoped_release`释放了`Global Interpreter Lock (GIL)`来提高性能（pybind11不会隐式释放，一切由用户操作，如果在释放后还需要访问python object，那么就必须require，详见[pybind11-gil](https://pybind11.readthedocs.io/en/stable/advanced/misc.html)。在此处由于我们已经把参数全部解析成c++参数，所以可以自由释放gil了。

`OptionalDeviceGuard device_guard`是一种**RAII**（Resource Acquisition Is Initialization，资源获取即初始化）的guard，在构造函数中设置为某一设备，在析构函数中取消设置。相对`DeviceGuard`，`OptionalDeviceGuard`允许传一个nullopt，等效于`optional<DeviceGuard>`。这里我们不做展开，有兴趣的读者可以参考`c10/core/DeviceGuard.h`

之后调用`self.contiguous()`

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

细心的读者可能发现，在tensorbase里它再次调用了`is_contiguous`方法，这是否和上面`THPVariable_contiguous`中重复了呢？对于我们例子中从python中调用下来确实是重复了，但contiguous并不是只有python层一个入口，c++层其他tensor也可能调用，所以这里需要加上。

那能不能python层不检查呢，都到此处来检查？理论上也是可以的，但相对而言就会多走一些调用流，降低运行效率。而后文我们会展开`is_contiguous`的判断逻辑，由于其采取了变量形式存储，所以`is_contiguous`运行效率非常高的，因此权衡之下将`is_contiguous`多次调用。

随后调用`TensorBase`的`__dispatch_contiguous()`方法

```c++
// aten/src/ATen/core/Tensor.cpp
TensorBase TensorBase::__dispatch_contiguous(c10::MemoryFormat memory_format) const {
  OptionalTensorRef self(*this);
  return at::_ops::contiguous::call(*self, memory_format);
}
```

注意此处将tensorbase转成了`OptionalTensorRef self`，这将使成员方法调用变成函数方法调用，即self变成了之后调用contiguous算子的**参数**

这也和`native_functions.yaml`中参数声明对应起来了`aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)`

## 4. dispatch contiguous算子：找schema与call kernel

调用`at::_ops::contiguous::call()`来到基于`native_functions.yaml`生成的文件`Operators_4.cpp`中

dispatch分为两步，第一步找到function schema，第二步调用schema中符合条件的kernel（如cpu tensor调度到cpu kernel、cuda tensor到cuda kernel等，该过程后面详细展开）

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

这里的`contiguous::name/overload_name`来自`continuous_ops.h`（生成代码）

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

我们展开说明op的获取流程，首先拿到一个`Dispatcher`的singleton（单例）

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

随后拿着dispatcher的单例去`findSchemaOrThrow()`

```c++
// aten/src/ATen/core/dispatch/Dispatcher.cpp
OperatorHandle Dispatcher::findSchemaOrThrow(const char* name, const char* overload_name) {
  // 这里name = "aten::contiguous", overload_name = ""
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

这里的`operatorLookupTable_`是`Dispatcher.h`中声明的一个私有变量`LeftRight<ska::flat_hash_map<OperatorName, OperatorHandle>> operatorLookupTable_;`，简单来说是一个哈希表，这里传了一个匿名函数进去，在哈希表中查找name，如果有则返回找到的`OperatorHandle`，如果没有则返回`nullopt`

```c++
template <class T>
class LeftRight final {
  template <typename F>
  auto read(F&& readFunc) const -> typename c10::invoke_result_t<F, const T&> {
    // ...

    // _data[_foregroundDataIndex.load()]拿到了所需的 operatorLookupTable
    return readFunc(_data[_foregroundDataIndex.load()]);
  }
}
```

这里我们找到了对应的`c10::OptionalBase<c10::OperatorHandle>`op并返回，随后经过`typed()`最终生成了`c10::TypedOperatorHandle<at::Tensor (const at::Tensor &, c10::MemoryFormat)>`给到外层static变量op。

到这里第一步查找schema步骤完成，我们接着开始查找并调用kernel。

```c++
// build/aten/src/ATen/Operators_4.cpp
at::Tensor contiguous::call(const at::Tensor & self, at::MemoryFormat memory_format) {
    static auto op = create_contiguous_typed_handle();
    return op.call(self, memory_format);
}
```

随后就调用call方法

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
  // 基于tensor等参数算出一个最佳的dispatch key set
  auto dispatchKeySet = op.operatorDef_->op.dispatchKeyExtractor()
    .template getDispatchKeySetUnboxed<Args...>(args...);
  // 根据disptach key set去operatorHandle中找kernel
  const KernelFunction& kernel = op.operatorDef_->op.lookup(dispatchKeySet);
  // ...
  // 最后调用kernel
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

在`call`方法中，首先算出一个`dispatchKeySet`，随后进入到 `op.lookup`中根据`dispatchKeySet`再算出`idx`，随后在`dispatchTable_`中找到最终调度到的kernel function，并调用其模板函数 `call`。

`dispatchKeySet`是一个`uint64_t`位集，每个dispatch key代表一个bit位，越大的bit索引代表优先级越高，例如一个tensor的device指定为`cuda`，disptach key set可能为`{AutogradCUDA | CUDA | ADInplaceOrView}`，那么会先进行dispatch到`AutogradCUDA`上，进行一些自动微分处理，然后再`redispatch`到`CUDA`上。

这里特别指出，`ADInplaceOrView`是一个比较特殊的dispatchkey，专门针对inplace以及view操作时注册，为后续autograd计算提供额外设置。

- 如对inplace操作增加`version counter`，后续autograd engine执行backward的时候会检查version，如果需要执行梯度计算的tensor被inplace操作过，则报错避免不正确的梯度计算。这部分代码在`torch/csrc/autograd/generated/ADInplaceOrViewTypeEverything.cpp`中。
- `view`则同理防止对生成view的tensor做任何修改以确保避免不正确的梯度计算（因为`view`的tensor和原tensor共享存储）。

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

这里如果`unboxed_kernel_func_`非空，就从`boxed_kernel_func_`处拿到`functor`，然后调用`callUnboxedKernelFunction<Return, Args...>`。

`unboxed`指的是未打包的函数，包含完整的签名和参数等，打包的`boxed`函数直观上理解为把所有参数压成一个整体，例如`void conjugateFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack)`中的`stack`，这样不用针对每个参数变体都单独写一个函数签名，可以最大程度复用代码，编译出来的binary占用空间也小一些，方便在移动端部署，但相对解包封包的过程会一定程度上影响效率。

```c++
// aten/src/ATen/core/boxing/KernelFunction_impl.h
template<class Return, class... Args>
inline Return callUnboxedKernelFunction(void* unboxed_kernel_func, OperatorKernel* functor, DispatchKeySet dispatchKeySet, Args&&... args) {
    using ActualSignature = Return (OperatorKernel*, DispatchKeySet, Args...);
    ActualSignature* func = reinterpret_cast<ActualSignature*>(unboxed_kernel_func);
    // 此时functor：&(at::(anonymous namespace)::(anonymous namespace)::wrapper_CompositeImplicitAutograd__contiguous(at::Tensor const&, c10::MemoryFormat))>
    return (*func)(functor, dispatchKeySet, std::forward<Args>(args)...);
}
```

随后来到`wrap_kernel_functor_unboxed_`中调用`call`函数

```c++
// aten/src/ATen/core/boxing/impl/make_boxed_from_unboxed_functor.h
template<class KernelFunctor, class ReturnType, class... ParameterTypes>
struct wrap_kernel_functor_unboxed_<KernelFunctor, ReturnType(ParameterTypes...)> final {
  static ReturnType call(OperatorKernel* functor, DispatchKeySet, ParameterTypes... args) {
    // 注意此处已经不再有dispatch key了
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

随后我们就剥开了层层封装，调度到了实际的functor上（根据编译选项、tensor类型等此处调度到的kernel会有所差异）。

这里调度到了**CompositeImplicitAutograd**上，该dispatch key的含义是组合非显式自动微分，不需要如`ExplicitAutograd`那样单独写微分函数，依赖于底层的其他算子都能实现自动微分来实现

```c++
// build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
at::Tensor wrapper_CompositeImplicitAutograd__contiguous(const at::Tensor & self, at::MemoryFormat memory_format) {
  return at::native::contiguous(self, memory_format);
}
```

最后便调用到了native的contiguous中（`aten/src/ATen/native/TensorProperties.cpp`），至此算子dispatch流程结束

## 5. 为什么能在表里找到contiguous算子：算子register

上文中我们梳理了contiguous的dispatch流程，但有分发就一定有注册，contiguous算子的schema是如何注册到`OperatorHandle`中的，其kernel又是如何注册到`dispatchTable_`中的呢？

在开始说明contiguous算子注册流程前，我们先简单了解一下通用的pytorch算子注册流程，即通过`TORCH_LIBRARY(ns, m)`和`TORCH_LIBRARY_IMPL(ns, k, m)`两个宏进行两步注册。

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

首先，会调用`TORCH_LIBRARY(ns, m)`宏在`ns`namespace下注册schema（本质是通过**Dispatcher**写入`OperatorEntry.schema_`字段），此时只有一个空dispatch table，具体kernel还没有注册。

```c++
// build/aten/src/ATen/RegisterSchema.cpp
TORCH_LIBRARY(aten, m) {
  m.def("batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor", {});
  m.def("contiguous(Tensor(a) self, *, MemoryFormat memory_format=contiguous_format) -> Tensor(a)", {});
}
```

随后，会调用`TORCH_LIBRARY_IMPL(ns, k, m)`注册算子具体实现（本质是通过**Dispatcher**写入`OperatorEntry.dispatchTable_`字段），绑定具体dispatch key，如`CompositeImplicitAutograd`、`CPU`、`CUDA`等。有一些特殊的设计如`catchall`等会扩散写入所有disptachkey，基于`BackendSelect`实现`fallback`会redispatch到下一个优先级的dispatch key等。

例如：

```c++
// build/aten/src/ATen/RegisterCompositeImplicitAutograd.cpp
TORCH_LIBRARY_IMPL(aten, CompositeImplicitAutograd, m) {
  // lots of ops
  m.impl("batch_norm", TORCH_FN(wrapper_CompositeImplicitAutograd__batch_norm));
  m.impl("contiguous", TORCH_FN(wrapper_CompositeImplicitAutograd__contiguous));
}
```

了解了基本算子注册方式后，我们详细展开算子注册流程：

首先对`TORCH_LIBRARY_IMPL`我们进行宏展开

```c++
// torch/library.h
// C10_UID是一个unique identifier，自增 counter
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

`TORCH_LIBRARY_IMPL_init_aten_CompositeImplicitAutograd_12`会在我们`import torch`的时候被**TorchLibraryInit**调用，此处不详细展开，我们重点看`m.impl`发生了什么

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

这里用CppFunction初始化了`func_`, `cpp_signature_`, `schema_`三个变量

`func_`即函数指针，待会我们重点展开，`cpp_signature_`即函数签名，如果kernel是以一种我们可以知道函数签名的方式创建的（例如`unboxed c++ function`），那我们就存储下来并在之后的kernel注册和调用中用于检查。

我们重点看`func_`的构造

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

最终，我们将`raw_f`封装成了`KernelFunction`，返回给了外层的`CppFunction`并让其完成了初始化。随后我们便调用`_impl(name, std::move(f), rv)`进行进一步处理

```c++
// aten/src/ATen/core/library.cpp
Library& Library::_impl(const char* name_str, CppFunction&& f, _RegisterOrVerify rv) & {
  at::OperatorName name = _parseNameForLib(name_str);
  auto dispatch_key = f.dispatch_key_.has_value() ? f.dispatch_key_ : dispatch_key_;
  // 按照contiguous调用到此处：dispatch_key为c10::OptionalBase<c10::DispatchKey> = { init_ = true, storage_ = (dummy_ = '|', value_ = CompositeImplicitAutograd)}
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

我们发现了很熟悉的对象`c10::Dispatcher::singleton()`，在注册这里我们调用了`c10::Dispatcher::singleton().registerImpl()`将我们封装好的kernelfunction（`f.func_`）及signature、schema等信息注册进dispatcher

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

  // 第一步注册schema
  auto op = findOrRegisterName_(op_name);

  // 第二步注册kernel
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

  // RegistrationHandleRAII自动回收机制，该对象注册了匿名函数`deregisterImpl_`，会在对象销毁时自动将op的kernel函数deregister，是很标准的RAII设计
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

首先会查找该op是否已经在`operatorLookupTable_`注册，如果已经注册则直接返回，如果没有则写入table（注意此时还没有注册具体的kernel实现，即第一步schema注册）

随后调用`op.operatorDef_->op.registerKernel()`将之前封装好的kernelfunction注册进该`OperatorEntry`（第二步kernel注册）

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

  // 将kernel加入到kernel list中，如果是第一个kernel则创建list
  // 重定向 catchAll 注册到 CompositeImplicitAutograd.
  auto& k = dispatch_key.has_value() ? kernels_[*dispatch_key] : kernels_[DispatchKey::CompositeImplicitAutograd];

  k.emplace_front(std::move(kernel), std::move(inferred_function_schema), std::move(debug));
  AnnotatedKernelContainerIterator inserted = k.begin();
  // 更新dispatch table
  if (dispatch_key.has_value()) {
    updateDispatchTable_(dispatcher, *dispatch_key);
  } else {
    updateDispatchTableFull_(dispatcher);
  }
  return inserted;
}
```

此处先通过`dispatch_key`找到`kernels_`中找到`k`（kernel的列表：`(std::list<c10::impl::AnnotatedKernel, std::allocator<c10::impl::AnnotatedKernel> >)`），将kernel插入首位

随后更新dispatcher的entry，到这里`registerImpl`就将op的kernel注册完成了

最后，返回`*this`指针，`m.impl("contiguous", TORCH_FN(wrapper_CompositeImplicitAutograd__contiguous));`注册完成

---
*Confused about some of the content? Feel free to report an issue [here](https://github.com/yewentao256/yewentao256.github.io/issues/new).*
