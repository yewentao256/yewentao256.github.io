---
title: "Unraveling PyTorch: Tensor Indexing and Assignment"
date: 2023-08-21T18:06:39+08:00
categories: ["pytorch"]
summary: "This article dissects PyTorch's C++ core to uncover the mechanics of **tensor indexing and assignment**. From translating Python indices to C++ **TensorIndex** to the nuances of `handleDimInMultiDimIndexing`, we explore both basic and advanced tensor operations."
---

## Summary

This article dissects PyTorch's C++ core to uncover the mechanics of **tensor indexing and assignment**. From translating Python indices to C++ **TensorIndex** to the nuances of `handleDimInMultiDimIndexing`, we explore both basic and advanced tensor operations.

## 0. Introduction

Let's start with code:

```py
import torch

t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])
t[1, 2] = 3
```

When we call `t[1, 2] = 3`, we know that the tensor (row 1, column 2) sets to 3, but **what happens in the C++ part**?

Let's find out.

## 1. How to export `set_item` to python layer

```c++
// torch/csrc/autograd/python_variable.cpp

bool THPVariable_initModule(PyObject* module) {
  // ...
  PyModule_AddObject(module, "_TensorBase", (PyObject*)&THPVariableType);
  // ...
}

PyTypeObject THPVariableType = {
    // ...
    &THPVariable_as_mapping, /* tp_as_mapping */
    // ...
};

static PyMappingMethods THPVariable_as_mapping = {
    THPVariable_length,
    THPVariable_getitem,
    THPVariable_setitem,
};
```

It's through the **PyModule_AddObject** and **PyTypeObject** provided by Python, see [python_document](https://docs.python.org/3/c-api/typeobj.html) for more details.

Let's see the function `THPVariable_setitem`:

```c++
// torch/csrc/autograd/python_variable_indexing.cpp
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  // ...
  const auto& self_ = THPVariable_Unpack(self);

  // wrap value to a Tensor
  Variable value;
  if (isQIntType(self_.scalar_type())) {
    // ...
  } else if (self_device.is_cuda()) {
    value = valueToTensor(self_.options(), py_value, at::Device(kCPU));
  } else {
    value = valueToTensor(self_.options(), py_value, self_device);
  }

  // ...
  // wrap index in a tuple if it's not already one
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  // Count the number of indexed dimensions (everything but ellipsis and None)
  int64_t specified_dims = count_specified_dimensions(holder.get());
  // ...
  // get sliced Tensor
  Variable sliced = applySlicing(
      self_,
      holder.get(),
      variableIndices,
      /*is_tracing=*/is_tracing,
      self_device,
      self_.ndimension(),
      specified_dims);
  
  // ... 
  // set value according to type of index, we will talk about this later
}
```

It invokes the `applySlicing` function to obtain the sliced Tensor.

```c++

static inline Variable applySlicing(
    const Variable& self,
    PyObject* index,
    variable_list& outIndices,
    bool is_tracing,
    const at::Device& self_device,
    const c10::optional<int64_t>& self_ndim,
    int64_t specified_dims) {
  int64_t size = PyTuple_GET_SIZE(index);
  int64_t dim = 0;

  // ...
  Variable result = self;
  for (const auto i : c10::irange(size)) {
    PyObject* obj = PyTuple_GET_ITEM(index, i);
    // nested tensor does not have a size (yet) so for now we represent its size
    // as null
    c10::optional<SymIntArrayRef> result_sizes = result.is_nested()
        ? c10::optional<SymIntArrayRef>(c10::nullopt)
        : c10::optional<SymIntArrayRef>(result.sym_sizes());
    result = at::indexing::handleDimInMultiDimIndexing(
        /*prev_dim_result=*/result,
        /*original_tensor=*/self,
        /*index=*/([&]() {
          if (THPUtils_checkLong(obj)) {
            if (is_tracing && THPVariable_Check(obj)) {
              recordSelectTrace(THPVariable_Unpack(obj));
            }
            return at::indexing::TensorIndex(THPUtils_unpackLong(obj));
          } // ...
        })(),
        /*dim_ptr=*/&dim,
        /*specified_dims_ptr=*/&specified_dims,
        /*real_dim=*/i,
        /*outIndices=*/outIndices,
        /*disable_slice_optimization=*/is_tracing,
        /*original_tensor_device=*/self_device,
        /*prev_dim_result_sizes=*/result_sizes);
  }
  return result;
}
```

For each dim, we firstly generate an `at::indexing::TensorIndex(THPUtils_unpackLong(obj))` in the anonymous function, then calls `handleDimInMultiDimIndexing`.

Note on **Nested Tensor**: This is a new feature in PyTorch that acts like a list of tensors. See [pytorch_document](https://pytorch.org/docs/stable/nested.html) for more details.

## 2. `TensorIndex` and `handleDimInMultiDimIndexing`

**TensorIndex** is used for converting C++ tensor indices into `std::vector<TensorIndex>`.

The convert table is:

| Python                   | C++                               |
|--------------------------|-----------------------------------|
| `None`                   | `at::indexing::None`              |
| `Ellipsis`               | `at::indexing::Ellipsis`          |
| `...`                    | `"..."`                           |
| `123`                    | `123`                             |
| `True` / `False`         | `true` / `false`                  |
| `:`                      | `Slice()` / `Slice(None, None)`   |
| `::`                     | `Slice()` / `Slice(None, None, None)`|
| `1:`                     | `Slice(1, None)`                  |
| `1::`                    | `Slice(1, None, None)`            |
| `:3`                     | `Slice(None, 3)`                  |
| `:3:`                    | `Slice(None, 3, None)`            |
| `::2`                    | `Slice(None, None, 2)`            |
| `1:3`                    | `Slice(1, 3)`                     |
| `1::2`                   | `Slice(1, None, 2)`               |
| `:3:2`                   | `Slice(None, 3, 2)`               |
| `1:3:2`                  | `Slice(1, 3, 2)`                  |
| `torch.tensor([1, 2])`   | `torch::tensor({1, 2})`           |

```c++
struct TORCH_API TensorIndex final {
  // Case 1: `at::indexing::None`
  TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}

  // Case 2: "..." / `at::indexing::Ellipsis`
  TensorIndex(at::indexing::EllipsisIndexType)
      : type_(TensorIndexType::Ellipsis) {}
  TensorIndex(const char* str) : TensorIndex(at::indexing::Ellipsis) {
    // ...
  }

  // Case 3: Integer value
  TensorIndex(int64_t integer)
      : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int integer) : TensorIndex((int64_t)integer) {}

  // ...
}
```

In our basic example, during the first iteration, we retrieve **TensorIndex(1)**, followed by **TensorIndex(2)** in the subsequent loop.

Then we call `handleDimInMultiDimIndexing` to get the Tensor slice

```c++
// aten/src/ATen/TensorIndexing.h
static inline Tensor handleDimInMultiDimIndexing(
    const Tensor& prev_dim_result,
    const Tensor& original_tensor,
    const TensorIndex& index,
    int64_t* dim_ptr,
    int64_t* specified_dims_ptr,
    int64_t real_dim,
    std::vector<Tensor>& outIndices,
    bool disable_slice_optimization,
    const at::Device& original_tensor_device,
    const c10::optional<SymIntArrayRef>& prev_dim_result_sizes) {
  if (index.is_integer()) {
    return impl::applySelect(
        prev_dim_result,
        *dim_ptr,
        index.integer(),
        real_dim,
        original_tensor_device,
        prev_dim_result_sizes);
  } else if (index.is_slice()) {
    // ...
  } else if (index.is_ellipsis()) {
    // ...
  } // ...
}


static inline Tensor applySelect(
    const Tensor& self,
    int64_t dim,
    int64_t index,
    int64_t real_dim,
    const at::Device& /*self_device*/,
    const c10::optional<SymIntArrayRef>& self_sizes) {
  // ... some check logic
  // aten::select works on negative indices
  return self.select(dim, index);
}
```

After two iterations, we obtain our desired result: a tensor scalar object. `result.item() = 6`, consistent with accessing the tensor using `tensor[1][2]` in Python.

## 3. Set Value

Since we've got the slice Tensor of our index, we can move on to set value.

```c++
// torch/csrc/autograd/python_variable_indexing.cpp
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  // ... get the tensor sliced and variable Indices

  if (variableIndices.empty()) {
    // set value for some basic type, like integer index
    pybind11::gil_scoped_release no_gil;
    at::indexing::copy_to(sliced, value);
    return 0;
  }

  {
    // set value for types like bool or tensor index(advanced indexing)
    pybind11::gil_scoped_release no_gil;
    SymIntArrayRef valueSizes = value.sym_sizes();
    SymIntArrayRef slicedValueSizes =
        at::indexing::slicePrefix1sSize(valueSizes);
    torch::autograd::Variable valuesSliced;
    if (!valueSizes.equals(slicedValueSizes)) {
      valuesSliced = value.view_symint(slicedValueSizes);
    } else {
      valuesSliced = value;
    }
    at::indexing::dispatch_index_put_(
        sliced, std::move(variableIndices), valuesSliced);
    return 0;
  }
}
```

For our simple case, `variableIndices` is empty and we directly copy our value(also a Tensor) to the slice using `at::indexing::copy_to`

```c++
// aten/src/ATen/TensorIndexing.h
static inline void copy_to(const Tensor& dst, const Tensor& src) {
  // Note: sym_sizes() is for symbolic tracing, if we are not using TorchScript
  // Just consider it as sizes()
  if (dst.sym_sizes().equals(src.sym_sizes())) {
    // case when sizes are the same
    dst.copy_(src);
    return;
  } else if (src.dim() == 0 && src.device().type() == at::kCPU) {
    // case when dst size bigger than src, eg: slice[0,1,2] = 1
    dst.fill_(src);
    return;
  }
  // case when src size is not 0, expand src to the size of dst
  auto src_view = src.view_symint(slicePrefix1sSize(src.sym_sizes()));
  c10::MaybeOwned<Tensor> b_src = expand_inplace(dst, src_view, "setitem");
  dst.copy_(*b_src);
}
```

And we just call `dst.copy_()` to set the value.

Note that this also works for device like **Cuda**, it uses `copy_` operator to set the value to support this feature.

## 4. Expand to advanced indexing

Above we introduce a simple example to show the process of pytorch index, and now let's move on to advanced indexing (tensor index)

```python
import torch

t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

rows = torch.tensor([0, 2])
cols = torch.tensor([1, 1])

t[rows, cols] = 10
```

Question: what's the value of `t` now?

It's easy to know the value now is:

```py
tensor([[ 1, 10,  3],
        [ 4,  5,  6],
        [ 7, 10,  9]])
```

But how?

Similar to the previous section, we call `applySlicing` to obtain our tensor slice. But for this time, our index is Tensor

```c++
// aten/src/ATen/TensorIndexing.h
static inline Tensor handleDimInMultiDimIndexing(
    const Tensor& prev_dim_result,
    const Tensor& original_tensor,
    const TensorIndex& index,
    int64_t* dim_ptr,
    int64_t* specified_dims_ptr,
    int64_t real_dim,
    std::vector<Tensor>& outIndices,
    bool disable_slice_optimization,
    const at::Device& original_tensor_device,
    const c10::optional<SymIntArrayRef>& prev_dim_result_sizes) {
  if (index.is_integer()) {
    // ...
  } // ...
  else if (index.is_tensor()) {
    Tensor result = prev_dim_result;
    const Tensor& tensor = index.tensor();
    auto scalar_type = tensor.scalar_type();
    if (tensor.dim() == 0 &&
        at::isIntegralType(scalar_type, /*includeBool=*/true)) {
      // ...
    } else {
      impl::recordTensorIndex(tensor, outIndices, dim_ptr);
    }
    return result;
  }
  // ...
}
```

we call for `recordTensorIndex` to set `outIndices`

```c++
// aten/src/ATen/TensorIndexing.h
static inline void recordTensorIndex(
    const Tensor& tensor,
    std::vector<Tensor>& outIndices,
    int64_t* dim_ptr) {
  // dim starts from 0, increment with each 'recordTensorIndex' call
  outIndices.resize(*dim_ptr + 1);
  outIndices[*dim_ptr] = tensor;
  (*dim_ptr)++;
};
```

And you can see that we don't change the tensor slice itself when index is Tensor, we just set the outIndices. So after `applySlicing`, we get:

- sliced(original tensor with shape`(3,3)`)
- variableIndices(`outIndices`, `[[0, 2], [1, 1]]`)

Then we use **index_put** (`at::indexing::dispatch_index_put_(sliced, std::move(variableIndices), valuesSliced)`) to set value

```c++
// aten/src/ATen/TensorIndexing.h
static inline Tensor dispatch_index_put_(
    Tensor& self,
    std::vector<Tensor>&& indices,
    const Tensor& value) {
  return self.index_put_(
      impl::typeConvertIndices(self, std::move(indices)), value);
}
```

This is another operator that can supports all kinds of device, including **Cuda**.

Note that this is same with using `index_put_` in Python layer:

```py
t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

rows = torch.tensor([0, 2])
cols = torch.tensor([1, 1])
# t[rows, cols] = 10
t.index_put_((rows, cols), torch.tensor([10, 10]))
```

## 5. Conclusion

1. For Python 1-D setter, we call C++ `at::indexing::set_item` after
converting Python index to C++ TensorIndex. This part is quite easy so we skip it in this article.

2. For Python N-D setter, we call C++ `at::indexing::handleDimInMultiDimIndexing`
for each dim, after converting Python index to C++ TensorIndex. If advanced
indexing is needed, call `index_put_`.
