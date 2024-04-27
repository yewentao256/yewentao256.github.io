---
title: "Demystifying Dtype Promotion in PyTorch"
date: 2024-01-06T14:57:58+08:00
categories: ["pytorch"]
summary: "本文深入探讨了 PyTorch 中的数据类型提升(**dtype promotion**)机制，包含 promotion 的基本规则、scalar 如何被 wrapped 成 tensor、**TensorIterator** 在计算数据类型时的作用等细节。"
---

## Summary

本文深入探讨了 PyTorch 中的数据类型提升(**dtype promotion**)机制，包含 promotion 的基本规则、scalar 如何被 wrapped 成 tensor、**TensorIterator** 在计算数据类型时的作用等细节。

## 等待被翻译

非常抱歉，看起来这篇博文还没有被翻译成中文，请等待一段时间

## 0. Introduction

Let's start with code:

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

Have you ever wondered about the data types (`dtype`) of output tensors in PyTorch? We'll explore this topic and provide answers later in the article.

## 1. Basic Rules of Dtype Promotion

In PyTorch, when the dtypes of inputs in an arithmetic operation (such as `add`, `sub`, etc.) are different, dtype promotion occurs. This is based on the following criteria:

- If the dtype of a scalar is of a higher category than that of a tensor (Note: `complex` > `floating` > `integral` > `boolean`), the dtype is promoted to one that is large enough to contain all scalar values.

- If a zero-dimensional (0-dim) tensor operand has a higher category than dimensioned operands, it is promoted to a dtype that can hold the 0-dim tensor.

- In cases where there are no higher-category 0-dim tensor operands, the dtype is promoted to one that can accommodate all dimensioned operands.

- **Special Cases**: For operations like `div`, dividing an integer tensor by an integer scalar results in a `float` dtype.

## 2. PyTorch Implementation Details

Let's delve into the PyTorch source code to understand how dtype promotion is implemented.

### 2.1 Wrapped Tensor

Consider the operation `int_tensor + 5`, where `5` is a constant scalar. In this scenario, the scalar `5` is wrapped into a tensor with a dtype of `int64`.

This wrapping approach enables the reuse of the `add.Tensor` operator. As a result, there is no need to maintain separate `add.Tensor` and `add.Scalar` operators. (Note: In PyTorch, the `add.Scalar` interface is not registered to the **dispatcher** and is therefore not used.)

Here's how the scalar wrapping occurs:

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
      // [deprecated] aten::add(Tensor self, Scalar alpha, Tensor other) -> Tensor
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

The crucial step is `_r.tensor(0)`, where the scalar is converted into a 0-dim tensor.

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
  } // ... other dtypes ...
  // ...
  at::Tensor tensor = scalar_to_tensor(scalar);
  tensor.unsafeGetTensorImpl()->set_wrapped_number(true);
  // ...
  return tensor;
}
```

And the process of converting scalar to tensor is through `fill`:

```c++
// torch/include/ATen/ScalarOps.h
inline at::Tensor scalar_to_tensor(
    const Scalar& s,
    const Device device = at::kCPU) {
  // This is the fast track we have for CPU scalar tensors.
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

In scenarios where a C++ function (e.g., `at::native::add_(...)`) gets called, the `Scalar` is similarly wrapped.

```c++
// aten/src/ATen/native/BinaryOps.cpp
Tensor& add_(Tensor& self, const Scalar& other, const Scalar& alpha) {
  return self.add_(wrapped_scalar_tensor(other), alpha);
}
```

### 2.2 Computing the dtypes

The computation of dtypes occurs within the **TensorIterator**. If you're unfamiliar with TensorIterator, I recommend reading my article introducing it [here](../structured_kernel_and_iterator/).

And in this article, we will focus on exploring the implementation of dtype promotion.

```cpp
// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::build(TensorIteratorConfig& config) {
  // ...
  // compute the result dtype and device
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
      // Determines if there are varying input dtypes
      if (op.target_dtype != common_dtype_) {
        if (common_dtype_ == ScalarType::Undefined) {
          common_dtype_ = op.target_dtype;
        } else {
          has_different_input_dtypes = true;
        }
      }
    } else {
      // Determines if there are varying output dtypes
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
    // Invalidates common_dtype_ if it could not be inferred
    common_dtype_ = has_different_input_dtypes ? ScalarType::Undefined : common_dtype_;
    return;
  }

  // Computes a common dtype, if needed
  if ((has_different_input_dtypes || all_ops_are_scalars_) && config.promote_inputs_to_common_dtype_) {
    common_dtype_ = compute_common_dtype();
  }

  // Promotes common dtype to the default float scalar type, if needed
  // This is for operators like `div`
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

In `compute_types`, PyTorch calculates the `common_dtype_` based on the input tensors and configuration settings like `promote_inputs_to_common_dtype_`. The resulting dtype is then stored in `op.target_dtype`, which is later used in `allocate_or_resize_outputs`.

To understand how PyTorch implements dtype promotion, let's examine the `compute_common_dtype`.

Note: The `promote_inputs_to_common_dtype_` must be set to `True` to enable this dtype inference mechanism(Typically, the configuration of TensorIterator is determined by macros such as `BINARY_FLOAT_OP_CONFIG`).

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

// aten/src/ATen/native/TypeProperties.cpp
ResultTypeState update_result_type_state(const Tensor& tensor, const ResultTypeState& in_state) {
  if (!tensor.defined()) {
    return in_state;
  }
  ResultTypeState new_state = in_state;
  ScalarType current = tensor.scalar_type();
  if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    // if wrapped tensor, use the default dtype for complex/float
    if(isComplexType(current)) {
      // default: complex<float>
      current = typeMetaToScalarType(at::get_default_complex_dtype());
    }
    else if(isFloatingType(current)) {
      // default: float
      current = typeMetaToScalarType(at::get_default_dtype());
    }
  }
  if ( tensor.dim() > 0 ) {
    // normal tensor
    new_state.dimResult = promote_skip_undefined(in_state.dimResult, current);
  } else if (tensor.unsafeGetTensorImpl()->is_wrapped_number()) {
    // wrapped tensor(scalar)
    new_state.wrappedResult = promote_skip_undefined(in_state.wrappedResult, current);
  } else {
    // zero dim tensor(not wrapped)
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

For each tensor, PyTorch invokes `update_result_type_state` to update the **ResultTypeState**. This state include three types of result dtypes: `dimResult` (for normal tensors), `wrappedResult` (for wrapped tensors), and `zeroResult` (for 0-dim tensors that are not wrapped).

The `at::native::result_type` function is then called to infer the `common_dtype_`:

```c++
// aten/src/ATen/native/TypeProperties.cpp
ScalarType result_type(const ResultTypeState& in_state) {
  return combine_categories(in_state.dimResult, combine_categories(in_state.zeroResult, in_state.wrappedResult));
}

static inline ScalarType combine_categories(ScalarType higher, ScalarType lower) {
  if(isComplexType(higher)) {
    return higher;
  } else if (isComplexType(lower)) {
    // preserve value type of higher if it is floating type.
    if (isFloatingType(higher)) {
      return toComplexType(higher);
    }
    // in case of integral input, lower complex takes precedence.
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

In most cases, the precedence order of the three result types is: `dimResult` > `wrappedResult` > `zeroResult`.

If the higher result dtype is a `bool` or the lower result dtype is a `FloatingType`, the dtype promotion function `promote_skip_undefined` is invoked:

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
  // This is generated according to NumPy's promote_types
  if (a == ud || b == ud) {
    return ScalarType::Undefined;
  }
  if (a == b) {
    return a;
  }

  // ...

  auto ix_a = dtype2index[static_cast<int64_t>(a)];
  auto ix_b = dtype2index[static_cast<int64_t>(b)];

  // This table axes must be consistent with index2dtype
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

In `promote_skip_undefined`, PyTorch employs a lookup table to efficiently execute dtype promotion.

## 3. Review the answer

Having delved into the dtype promotion mechanism of PyTorch, let's revisit and answer the questions posed earlier in the article.

```py
>>> (int_tensor + 5).dtype
# 5 is wrapped to a int64 tensor, but doesn't have higher precedence than
# dim-tensor, so still int32
torch.int32

>>> (int_tensor + 5.5).dtype
# 5.5 is wrapped to a double tensor, then get the default `float` in
# `update_result_type_state`, then dtype promotion to float
torch.float32

>>> (int_tensor / 5).dtype
# 5 is wrapped to a long tensor, we get a `int` after `compute_common_dtype`
# However, since `promote_integer_inputs_to_float` is set for `div` op
# the dtype of output is promoted to float in `compute_types`
torch.float32

>>> (int_tensor + long_zerodim).dtype
# zerodim's precedence is lower than int_tensor, so no dtype promotion here
torch.int32

>>> (long_tensor + int_tensor).dtype
# dtype promotion to long
torch.int64

>>> (bool_tensor + long_tensor).dtype
# dtype promotion to long
torch.int64

>>> (bool_tensor + uint_tensor).dtype
# dtype promotion to uint8
torch.uint8

>>> (float_tensor + double_tensor).dtype
# dtype promotion to double
torch.float64

>>> (complex_float_tensor + complex_double_tensor).dtype
# dtype promotion to complex128
torch.complex128

>>> (bool_tensor + int_tensor).dtype
# dtype promotion to int
torch.int32

>>> torch.add(long_tensor, float_tensor).dtype
# dtype promotion to float
torch.float32 
```

## Referrence

- [pytorch](https://github.com/pytorch/pytorch)
- [torch.dtype](https://pytorch.org/docs/stable/tensor_attributes.html)
