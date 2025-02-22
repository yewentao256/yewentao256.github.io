---
title: "Demystifying Dtype Promotion in PyTorch"
date: 2024-01-06T14:57:58+08:00
categories: ["pytorch"]
summary: "本文深入探讨了 PyTorch 中的数据类型提升(**dtype promotion**)机制，包含 promotion 的基本规则、scalar 如何被 wrapped 成 tensor、**TensorIterator** 在计算数据类型时的作用等细节。"
---

## Summary

本文深入探讨了 PyTorch 中的数据类型提升(**dtype promotion**)机制，包含 promotion 的基本规则、scalar 如何被 wrapped 成 tensor、**TensorIterator** 在计算数据类型时的作用等细节。

>这篇文章使用`O3-mini-high`翻译，如有困惑请参考英文原文

---

## 0. 引言

我们从代码开始：

```py
import torch

float_tensor = torch.ones(1, dtype=torch.float)
double_tensor = torch.ones(1, dtype=torch.double)
complex_float_tensor = torch.ones(1, dtype=torch.complex64)
complex_double_tensor = torch.ones(1, dtype=torch.complex128)
int_tensor = torch.ones(1, dtype=torch.int)
long_tensor = torch.ones(1, dtype=torch.long)
uint_tensor = torch.ones(1, dtype=torch.uint8)
double_tensor = torch.ones(1, dtype=torch.double)
bool_tensor = torch.ones(1, dtype=torch.bool)
long_zerodim = torch.tensor(1, dtype=torch.long)
int_zerodim = torch.tensor(1, dtype=torch.int)

>>> (int_tensor + 5).dtype
>>> (int_tensor + 5.5).dtype
>>> (int_tensor / 5).dtype
>>> (int_tensor + long_zerodim).dtype
>>> (long_tensor + int_tensor).dtype
>>> (bool_tensor + long_tensor).dtype
>>> (bool_tensor + uint_tensor).dtype
>>> (float_tensor + double_tensor).dtype
>>> (complex_float_tensor + complex_double_tensor).dtype
>>> (bool_tensor + int_tensor).dtype
>>> torch.add(long_tensor, float_tensor).dtype
```

你是否曾经好奇，为什么 PyTorch 中算术运算（比如 `add`、`sub` 等）的输出张量会有不同的 `dtype`？本文将探讨这一主题，并在后续部分给出详细的答案。

---

## 1. 数据类型（Dtype）提升的基本规则

在 PyTorch 中，当参与算术运算的输入张量具有不同的 dtype 时，会触发 dtype 提升（promotion）。提升规则主要基于以下几个准则：

- **标量与张量**：如果一个标量的 dtype 所处的类别比张量更高（注意：`complex` > `floating` > `integral` > `boolean`），则最终的 dtype 将被提升到足以容纳所有标量值的类型。

- **零维张量**：如果参与运算的 0 维张量（即标量张量）的类别高于其他有维度的张量，其 dtype 会被提升为能够存储 0 维张量的类型。

- **多个有维度张量**：如果没有更高类别的 0 维张量，则会提升为能够容纳所有有维度张量的 dtype。

- **特殊情况**：例如，对于 `div`（除法）操作，当整数张量除以整数标量时，结果会被提升为 `float` 类型。

---

## 2. PyTorch 的实现细节

接下来，我们深入 PyTorch 源码，看看它是如何实现 dtype 提升的。

### 2.1 包装张量（Wrapped Tensor）

考虑操作 `int_tensor + 5`，这里的 `5` 是一个常量标量。此时，标量 `5` 会被包装成一个 dtype 为 `int64` 的张量。  
这种包装方式使得我们可以复用 `add.Tensor` 运算符，从而不必单独维护 `add.Tensor` 和 `add.Scalar` 两个版本。（需要注意的是，PyTorch 中的 `add.Scalar` 接口没有注册到 dispatcher 上，因此并未实际使用。）

下面展示了标量包装的具体实现：

```c++
// torch/csrc/autograd/generated/python_variable_methods.cpp
static PyObject * THPVariable_add(PyObject* self_, PyObject* args, PyObject* kwargs)
{
  const Tensor& self = THPVariable_Unpack(self_);
  static PythonArgParser parser({
    "add(Scalar alpha, Tensor other)|deprecated",
    "add(Tensor other, *, Scalar alpha=1)",
  }, /*traceable=*/true);

  ParsedArgs<2> parsed_args;
  auto _r = parser.parse(self_, args, kwargs, parsed_args);
  // ...
  switch (_r.idx) {
    case 0: {
      // [已废弃] aten::add(Tensor self, Scalar alpha, Tensor other) -> Tensor
      // ...
    }
    case 1: {
      // aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
      auto dispatch_add = [](const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha) -> at::Tensor {
        pybind11::gil_scoped_release no_gil;
        return self.add(other, alpha);
      };
      return wrap(dispatch_add(self, _r.tensor(0), _r.scalar(1)));
    }
  }
}
```

关键在于调用了 `_r.tensor(0)`，此处将标量转换成了一个 0 维张量。

```c++
// torch/csrc/utils/python_arg_parser.h
inline at::Tensor PythonArgs::tensor(int i) {
  // ...
  return tensor_slow(i);
}

// torch/csrc/utils/python_arg_parser.cpp
at::Tensor PythonArgs::tensor_slow(int i) {
  PyObject* obj = args[i];
  if (!obj) {
    return at::Tensor();
  }
  // ...
  bool save_symint = false;
  at::Scalar scalar;
  if (PyBool_Check(obj)) {
    scalar = at::Scalar(THPUtils_unpackBool(obj));
  } else if (THPUtils_checkLong(obj)) {
    scalar = at::Scalar(THPUtils_unpackLong(obj));
  } else if (THPUtils_checkDouble(obj)) {
    scalar = at::Scalar(THPUtils_unpackDouble(obj));
  } // ... 其他 dtype 的处理 ...
  // ...
  at::Tensor tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  // ...
  return tensor;
}
```

而将标量转换成张量的过程是通过 `fill` 完成的：

```c++
// torch/include/ATen/ScalarOps.h
inline at::Tensor scalar_to_tensor(
    const Scalar& s,
    const Device device = at::kCPU) {
  // 针对 CPU 标量张量的快速路径
  if (device == at::kCPU) {
    return at::detail::scalar_tensor_static(s, s.type(), at::kCPU);
  }
  // ...
}

// aten/src/ATen/ScalarOps.cpp
Tensor scalar_tensor_static(const Scalar& s, c10::optional<ScalarType> dtype_opt, c10::optional<Device> device_opt) {
  // ...
  Tensor result = at::detail::empty_cpu(
      {}, dtype_opt, c10::nullopt, device_opt, c10::nullopt, c10::nullopt);
  scalar_fill(result, s);
  return result;
}
```

在某些 C++ 函数（例如 `at::native::add_(...)`）被调用时，`Scalar` 同样会被包装。

```c++
// aten/src/ATen/native/BinaryOps.cpp
Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.add_(wrapped_scalar_tensor(other), alpha);
}
```

在内核层面（例如 f(a, b) == f(b, a) 的情况），可以通过剥除包装张量和 CPU 标量张量，从而将其视为普通常量，以提升计算效率。示例如下：

```c++
// aten/src/ATen/native/cuda/BinaryBitwiseOpsKernels.cu
void bitwise_and_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, iter.dtype(), "bitwise_and_cuda", [&]() {
    BitwiseAndFunctor<scalar_t> f;
    opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(iter, f);
  });
}

template <typename scalar_t, typename return_t = scalar_t, typename func_t>
void opmath_symmetric_gpu_kernel_with_scalars(TensorIteratorBase& iter, const func_t& f) {
  // ...
  if (iter.is_cpu_scalar(1)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(1);
    iter.remove_operand(1);
    device_guard.reset_device(iter.device(1));
  } else if (iter.is_cpu_scalar(2)) {
    scalar_val = iter.scalar_value<opmath_arg_t>(2);
    iter.remove_operand(2);
  }

  if (iter.ninputs() == 2) {
    gpu_kernel(iter, BinaryFunctor<scalar_t, scalar_t, return_t, func_t>(f));
  } else {
    AUnaryFunctor<scalar_t, scalar_t, return_t, func_t> unary_f(f, scalar_val);
    gpu_kernel(iter, unary_f);
  }
}
```

### 2.2 计算数据类型

数据类型的计算主要发生在 **TensorIterator** 内部。如果你不熟悉 TensorIterator，建议先阅读我之前介绍它的文章 [这里](../structured_kernel_and_iterator/)。

在这篇文章中，我们将重点探讨 dtype 提升的实现过程。

```cpp
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // ...
  // 计算结果张量的 dtype 和设备
  compute_types(config);
  // ...
}

void TensorIteratorBase::compute_types(const TensorIteratorConfig& config) {
  common_dtype_ = ScalarType::Undefined;
  ScalarType output_dtype = ScalarType::Undefined;
  bool has_different_input_dtypes = false;
  bool has_undefined_outputs = false;

  for (auto& op : operands_) {
    if (!op.is_type_defined()) {
      // ...
      if (config.static_dtype_.has_value()) {
        op.target_dtype = config.static_dtype_.value();
      } else {
        has_undefined_outputs = true;
      }
      // ...
    }
    // ... 

    if (!op.is_output) {
      // 判断输入张量是否存在不同的 dtype
      if (op.target_dtype != common_dtype_) {
        if (common_dtype_ == ScalarType::Undefined) {
          common_dtype_ = op.target_dtype;
        } else {
          has_different_input_dtypes = true;
        }
      }
    } else {
      // 判断输出张量是否存在不同的 dtype
      if (op.target_dtype != output_dtype) {
        if (output_dtype == ScalarType::Undefined) {
          output_dtype = op.target_dtype;
        }
        // ...
      }
    }
  }

  // ...

  if (!has_undefined_outputs && !config.check_all_same_device_ &&
      !config.promote_inputs_to_common_dtype_ && !config.cast_common_dtype_to_outputs_ &&
      !config.enforce_safe_casting_to_output_) {
    // 如果无法推断出 common_dtype_ 则置为 Undefined
    common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
    return;
  }

  // 如果需要，计算一个公共 dtype
  if ((has_different_input_dtypes || all_ops_are_scalars_) && config.promote_inputs_to_common_dtype_) {
    common_dtype_ = compute_common_dtype();
  }

  // 对于类似 `div` 操作，将整数输入提升为默认的 float 类型
  if (config.promote_integer_inputs_to_float_ &&
      c10::isIntegralType(common_dtype_, /*includeBool=*/true)) {
    common_dtype_ = c10::typeMetaToScalarType(c10::get_default_dtype());
  }

  // ...
  for (auto& op : operands_) {
    bool is_type_defined = op.is_type_defined();
    if (!is_type_defined) {
      op.target_dtype = common_dtype_;
    }
    // ...
  }
}
```

在 `compute_types` 中，PyTorch 根据输入张量及配置参数（例如 `promote_inputs_to_common_dtype_`）计算出 `common_dtype_`，并将该结果存储到 `op.target_dtype` 中，后续在 `allocate_or_resize_outputs` 中使用。

为了更深入理解 dtype 提升的实现，我们来看看 `compute_common_dtype` 的具体逻辑：

```c++
// aten/src/ATen/TensorIterator.cpp
ScalarType TensorIteratorBase::compute_common_dtype() {
  at::native::ResultTypeState state = {};
  for (const auto& op : operands_) {
    if (op.is_output) {
      continue;
    }
    state = at::native::update_result_type_state(op.tensor(), state);
  }
  common_dtype_ = at::native::result_type(state);
  TORCH_INTERNAL_ASSERT(common_dtype_ != ScalarType::Undefined);
  return common_dtype_;
}
```

在更新每个张量的结果状态时（ResultTypeState），PyTorch 会区分三种情况：
- **dimResult**：用于普通（有维度）张量；
- **zeroResult**：用于未包装的 0 维张量；
- **wrappedResult**：用于包装后的 0 维张量。

`at::native::result_type` 函数会根据这三种结果状态，推导出最终的 `common_dtype_`。

```c++
// aten/src/ATen/native/TypeProperties.cpp
ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state) {
  if (!tensor.defined()) {
    return in_state;
  }
  ResultTypeState new_state = in_state;
  ScalarType current = tensor.scalar_type();
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    // 对于包装张量，使用默认的 complex/float 类型
    if(isComplexType(current)) {
      // 默认：complex<float>
      current = typeMetaToScalarType(at::get_default_complex_dtype());
    }
    else if(isFloatingType(current)) {
      // 默认：float
      current = typeMetaToScalarType(at::get_default_dtype());
    }
  }
  if ( tensor.dim() > 0 ) {
    // 普通张量
    new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
  } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    // 包装张量（标量）
    new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
  } else {
    // 非包装的 0 维张量
    new_state.zeroResult = promote_skip_undefined(in_state.zeroResult, current);
  }
  return new_state;
}

// torch/include/ATen/native/TypeProperties.h
struct ResultTypeState {
  c10::ScalarType dimResult = ScalarType::Undefined;
  c10::ScalarType wrappedResult = ScalarType::Undefined;
  c10::ScalarType zeroResult = ScalarType::Undefined;
};
```

接下来，通过调用 `at::native::result_type` 得到最终的 `common_dtype_`：

```c++
// aten/src/ATen/native/TypeProperties.cpp
ScalarType result_type(const ResultTypeState& in_state) {
  return combine_categories(in_state.dimResult, combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

static inline ScalarType combine_categories(ScalarType higher, ScalarType lower) {
  if(isComplexType(higher)) {
    return higher;
  } else if (isComplexType(lower)) {
    // 如果 higher 为浮点类型，则保持其数值类型
    if (isFloatingType(higher)) {
      return toComplexType(higher);
    }
    // 如果输入为整数，则 lower（complex 类型）优先
    return lower;
  } else if (isFloatingType(higher)) {
    return higher;
  }
  if (higher == ScalarType::Bool || isFloatingType(lower)) {
    return promote_skip_undefined(higher, lower);
  }
  if (higher != ScalarType::Undefined) {
    return higher;
  }
  return lower;
}
```

在大多数情况下，三种结果状态的优先级为：`dimResult` > `zeroResult` > `wrappedResult`。  
当更高的结果 dtype 为 `bool` 或者更低的结果 dtype 为浮点类型时，将调用 `promote_skip_undefined` 进行 dtype 提升：

```c++
// aten/src/ATen/native/TypeProperties.cpp
static inline ScalarType promote_skip_undefined(ScalarType a, ScalarType b) {
  if (a == ScalarType::Undefined) {
    return b;
  }
  if (b == ScalarType::Undefined) {
    return a;
  }
  return promoteTypes(a, b);
}

// c10/core/ScalarType.cpp
constexpr auto u1 = ScalarType::Byte;
constexpr auto i1 = ScalarType::Char;
constexpr auto i2 = ScalarType::Short;
constexpr auto i4 = ScalarType::Int;
constexpr auto i8 = ScalarType::Long;
constexpr auto f2 = ScalarType::Half;
constexpr auto f4 = ScalarType::Float;
constexpr auto f8 = ScalarType::Double;
constexpr auto c2 = ScalarType::ComplexHalf;
constexpr auto c4 = ScalarType::ComplexFloat;
constexpr auto c8 = ScalarType::ComplexDouble;
constexpr auto b1 = ScalarType::Bool;
constexpr auto bf = ScalarType::BFloat16;
constexpr auto ud = ScalarType::Undefined;

ScalarType promoteTypes(ScalarType a, ScalarType b) {
  // 此处依据 NumPy 的 promote_types 生成
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }
  if (a == b) {
    return a;
  }

  // ...

  auto ix_a = dtype2index[static_cast<int64_t>(a)];
  auto ix_b = dtype2index[static_cast<int64_t>(b)];

  // 该查找表与 index2dtype 保持一致
  static constexpr std::array<std::array<ScalarType, index2dtype.size()>, index2dtype.size()>
      _promoteTypesLookup = {{
      /*        u1  i1  i2  i4  i8  f2  f4  f8  c2  c4  c8  b1  bf*/
      /* u1 */ {u1, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, u1, bf},
      /* i1 */ {i2, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, i1, bf},
      /* i2 */ {i2, i2, i2, i4, i8, f2, f4, f8, c2, c4, c8, i2, bf},
      /* i4 */ {i4, i4, i4, i4, i8, f2, f4, f8, c2, c4, c8, i4, bf},
      /* i8 */ {i8, i8, i8, i8, i8, f2, f4, f8, c2, c4, c8, i8, bf},
      /* f2 */ {f2, f2, f2, f2, f2, f2, f4, f8, c2, c4, c8, f2, f4},
      /* f4 */ {f4, f4, f4, f4, f4, f4, f4, f8, c4, c4, c8, f4, f4},
      /* f8 */ {f8, f8, f8, f8, f8, f8, f8, f8, c8, c8, c8, f8, f8},
      /* c2 */ {c2, c2, c2, c2, c2, c2, c4, c8, c2, c4, c8, c2, c4},
      /* c4 */ {c4, c4, c4, c4, c4, c4, c4, c8, c4, c4, c8, c4, c4},
      /* c8 */ {c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8, c8},
      /* b1 */ {u1, i1, i2, i4, i8, f2, f4, f8, c2, c4, c8, b1, bf},
      /* bf */ {bf, bf, bf, bf, bf, f4, f4, f8, c4, c4, c8, bf, bf},
  }};
  // clang-format on
  return _promoteTypesLookup[ix_a][ix_b];
}
```

在 `promote_skip_undefined` 中，PyTorch 借助查找表来高效完成 dtype 的提升。

---

## 3. 回顾与答案

在详细探讨了 PyTorch 的 dtype 提升机制后，我们回顾一下最初提出的各个例子，并给出答案：

```py
>>> (int_tensor + 5).dtype
# 5 被包装成 int64 张量，但其优先级不高于有维度张量，因此结果为 int32
torch.int32

>>> (int_tensor + 5.5).dtype
# 5.5 被包装为 double 张量，经过 `update_result_type_state` 获取默认的 float 类型，
# 最后提升为 float
torch.float32

>>> (int_tensor / 5).dtype
# 5 被包装为 long 张量，初步计算得到 int 类型，但由于 `div` 操作将整数提升为 float，
# 最终输出为 float
torch.float32

>>> (int_tensor + long_zerodim).dtype
# 0 维张量的优先级低于 int_tensor，因此无 dtype 提升
torch.int32

>>> (long_tensor + int_tensor).dtype
# 提升为 long 类型
torch.int64

>>> (bool_tensor + long_tensor).dtype
# 提升为 long 类型
torch.int64

>>> (bool_tensor + uint_tensor).dtype
# 提升为 uint8 类型
torch.uint8

>>> (float_tensor + double_tensor).dtype
# 提升为 double 类型
torch.float64

>>> (complex_float_tensor + complex_double_tensor).dtype
# 提升为 complex128 类型
torch.complex128

>>> (bool_tensor + int_tensor).dtype
# 提升为 int 类型
torch.int32

>>> torch.add(long_tensor, float_tensor).dtype
# 提升为 float 类型
torch.float32 
```

---

## 参考资料

- [PyTorch 源码仓库](https://github.com/pytorch/pytorch)
- [torch.dtype 文档](https://pytorch.org/docs/stable/tensor_attributes.html)
