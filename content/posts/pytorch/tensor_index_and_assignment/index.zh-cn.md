---
title: "Unraveling PyTorch: Tensor Indexing and Assignment"
date: 2023-08-21T18:06:39+08:00
categories: ["pytorch"]
summary: "这篇文章深入探讨了PyTorch**张量索引和赋值**的机制，包含将Python索引转化为C++的**TensorIndex**、`handleDimInMultiDimIndexing`、`index_put`等内容。"
---

## Summary

这篇文章深入探讨了PyTorch**张量索引和赋值**的机制，包含将Python索引转化为C++的**TensorIndex**、`handleDimInMultiDimIndexing`、`index_put`等内容。

>这篇文章使用`O3-mini-high`翻译，如有困惑请参考英文原文

---

## 0. 引言

我们先从代码开始：

```py
import torch

t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])
t[1, 2] = 3
```

当我们调用 `t[1, 2] = 3` 时，我们知道该 tensor（第 1 行第 2 列）的值被设置为 3，但**在 C++ 代码层面究竟发生了什么**？

让我们一探究竟。

---

## 1. 如何将 `set_item` 导出到 Python 层

在 PyTorch 的 C++ 层，通过以下代码将 `set_item` 导出到 Python 层：

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

这里利用 Python 提供的 **PyModule_AddObject** 和 **PyTypeObject** 来将接口导出到 Python 层，详细内容可参见 [Python 官方文档](https://docs.python.org/3/c-api/typeobj.html)。

接下来看看 `THPVariable_setitem` 函数的实现：

```c++
// torch/csrc/autograd/python_variable_indexing.cpp
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  // ...
  const auto& self_ = THPVariable_Unpack(self);

  // 将 py_value 包装为一个 Tensor
  Variable value;
  if (isQIntType(self_.scalar_type())) {
    // ...
  } else if (self_device.is_cuda()) {
    value = valueToTensor(self_.options(), py_value, at::Device(kCPU));
  } else {
    value = valueToTensor(self_.options(), py_value, self_device);
  }

  // ...
  // 如果 index 不是 tuple，则将其包装为 tuple
  THPObjectPtr holder = wrapTuple(index);

  variable_list variableIndices;
  // 统计被索引的维度数量（不包括 ellipsis 和 None）
  int64_t specified_dims = count_specified_dimensions(holder.get());
  // ...
  // 获取切片 Tensor
  Variable sliced = applySlicing(
      self_,
      holder.get(),
      variableIndices,
      /*is_tracing=*/is_tracing,
      self_device,
      self_.ndimension(),
      specified_dims);
  
  // ... 
  // 根据 index 类型设置值，后面会详细讨论这一部分
}
```

该函数调用了 `applySlicing` 来获得切片后的 Tensor。

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
    // 对于 nested tensor，目前还没有 size，所以用 null 表示
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
          } // ... 其他情况的处理
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

对每个维度，首先通过匿名函数生成一个 `at::indexing::TensorIndex(THPUtils_unpackLong(obj))`，随后调用 `handleDimInMultiDimIndexing` 来处理该维度的索引。

> **注意：** 关于 **Nested Tensor**：这是 PyTorch 的一项新特性，其行为类似于 tensor 的列表。详情请参阅 [PyTorch Nested Tensor 文档](https://pytorch.org/docs/stable/nested.html)。

---

## 2. `TensorIndex` 与 `handleDimInMultiDimIndexing`

**TensorIndex** 用于将 C++ 层的 tensor 索引转换为 `std::vector<TensorIndex>`。

下面是索引类型的转换表：

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
  // 情况 1：`at::indexing::None`
  TensorIndex(c10::nullopt_t) : type_(TensorIndexType::None) {}

  // 情况 2： "..." / `at::indexing::Ellipsis`
  TensorIndex(at::indexing::EllipsisIndexType)
      : type_(TensorIndexType::Ellipsis) {}
  TensorIndex(const char* str) : TensorIndex(at::indexing::Ellipsis) {
    // ...
  }

  // 情况 3： 整数值
  TensorIndex(int64_t integer)
      : integer_(integer), type_(TensorIndexType::Integer) {}
  TensorIndex(int integer) : TensorIndex((int64_t)integer) {}

  // ...
}
```

在我们的示例中，在第一次迭代中会获得 **TensorIndex(1)**，而在下一次迭代中获得 **TensorIndex(2)**。

接下来调用 `handleDimInMultiDimIndexing` 来得到相应的 tensor 切片：

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
```

对于整型索引的情况，会调用 `applySelect`：

```c++
static inline Tensor applySelect(
    const Tensor& self,
    int64_t dim,
    int64_t index,
    int64_t real_dim,
    const at::Device& /*self_device*/,
    const c10::optional<SymIntArrayRef>& self_sizes) {
  // ... 一些检查逻辑
  // aten::select 支持负索引
  return self.select(dim, index);
}
```

经过两次迭代后，我们得到了期望的结果：一个标量 Tensor，其值为 6，与 Python 层使用 `tensor[1][2]` 访问一致。

---

## 3. 设置值

既然我们已经根据索引获得了切片后的 Tensor，就可以进行赋值操作了。

```c++
// torch/csrc/autograd/python_variable_indexing.cpp
int THPVariable_setitem(PyObject* self, PyObject* index, PyObject* py_value) {
  // ... 获取切片后的 Tensor 以及 variableIndices

  if (variableIndices.empty()) {
    // 针对简单的基本类型（如整数索引）的赋值
    pybind11::gil_scoped_release no_gil;
    at::indexing::copy_to(sliced, value);
    return 0;
  }

  {
    // 针对 bool 或 tensor 索引（高级索引）的赋值
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

对于我们的简单例子，`variableIndices` 为空，因此直接使用 `at::indexing::copy_to` 将值（同样是一个 Tensor）复制到切片中。

```c++
// aten/src/ATen/TensorIndexing.h
static inline void copy_to(const Tensor& dst, const Tensor& src) {
  // 注意：sym_sizes() 用于符号跟踪，如果不使用 TorchScript，可视为 sizes()
  if (dst.sym_sizes().equals(src.sym_sizes())) {
    // 当尺寸完全相同时
    dst.copy_(src);
    return;
  } else if (src.dim() == 0 && src.device().type() == at::kCPU) {
    // 当 dst 尺寸大于 src，例如：slice[0,1,2] = 1
    dst.fill_(src);
    return;
  }
  // 当 src 的尺寸不为 0 时，将 src 扩展到与 dst 相同的尺寸
  auto src_view = src.view_symint(slicePrefix1sSize(src.sym_sizes()));
  c10::MaybeOwned<Tensor> b_src = expand_inplace(dst, src_view, "setitem");
  dst.copy_(*b_src);
}
```

最后调用 `dst.copy_()` 完成赋值。需要注意的是，对于 Cuda 等设备，同样使用 `copy_` 操作符来支持跨设备赋值。

---

## 4. 扩展到高级索引

前面介绍了简单的索引例子，现在让我们来看下高级索引（tensor 索引）的情况。

```python
import torch

t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

rows = torch.tensor([0, 2])
cols = torch.tensor([1, 1])

t[rows, cols] = 10
```

问题：此时 `t` 的值是多少？

显然结果为：

```py
tensor([[ 1, 10,  3],
        [ 4,  5,  6],
        [ 7, 10,  9]])
```

那么这一过程是如何实现的呢？

类似于前面的过程，我们依然调用 `applySlicing` 获取切片，但这一次索引的元素是 Tensor。

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

此处会调用 `recordTensorIndex` 将索引 Tensor 记录到 `outIndices` 中：

```c++
// aten/src/ATen/TensorIndexing.h
static inline void recordTensorIndex(
    const Tensor& tensor,
    std::vector<Tensor>& outIndices,
    int64_t* dim_ptr) {
  // 索引的维度从 0 开始，每调用一次 recordTensorIndex，dim 自增
  outIndices.resize(*dim_ptr + 1);
  outIndices[*dim_ptr] = tensor;
  (*dim_ptr)++;
};
```

可以看到，当索引为 Tensor 时，并不会修改 tensor 切片本身，只是将对应的索引保存到 outIndices 中。因此，经过 `applySlicing` 后，我们得到了：
- 切片（sliced）：原始 tensor（形状为 (3,3)）
- variableIndices（即 outIndices）：包含 `[[0, 2], [1, 1]]`

随后，通过 **index_put**（调用 `at::indexing::dispatch_index_put_(sliced, std::move(variableIndices), valuesSliced)`）将值设置到对应位置。

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

这正是 Python 层调用 `index_put_` 的底层实现：

```py
t = torch.tensor([[1, 2, 3], 
                  [4, 5, 6], 
                  [7, 8, 9]])

rows = torch.tensor([0, 2])
cols = torch.tensor([1, 1])
# t[rows, cols] = 10
t.index_put_((rows, cols), torch.tensor([10, 10]))
```

---

## 5. 结论

1. 对于 Python 一维 setter，我们先将 Python 索引转换为 C++ 层的 TensorIndex，然后调用 C++ 中的 `at::indexing::set_item`。这一部分较为简单，本文略过详细描述。

2. 对于 Python 多维 setter，我们对每个维度调用 `at::indexing::handleDimInMultiDimIndexing` 将 Python 索引转换为 C++ 的 TensorIndex。如果需要高级索引，则调用 `index_put_`。
