---
title: "Deep Dive to Pytorch AutoGrad(1)"
date: 2023-07-04T10:48:42+08:00
categories: ["pytorch"]
summary: "This article introduces the implementation details of pytorch autograd mechanism."
---

## Summary

This article introduces the implementation details of pytorch autograd mechanism.

## Introduction

Version of pytorch：`2.1.0a0+gita3dddae`

```py
import torch

x = torch.tensor([3.], requires_grad=True)
y = x * x
print(y)       # tensor([9.], grad_fn=<MulBackward0>)
y.backward()
print(x.grad)  # tensor([6.]), y = x^2, dy/dx = 2x
```

This is a basic autograd operation. We create a new leaf node `x`, then construct a computation graph when `y = x * x`, and finally call `y.backward()` for the backward operation.

We will delve into the C++ level, step by step to analyze how PyTorch constructs `requires_grad` tensors, how to construct a computation graph during forward computation, and how to implement automatic differentiation and gradient accumulation based on the computation graph during backward derivation.

## Creation of tensor `x`: `x = torch.tensor([3.], requires_grad=True)`

### Construction of tensor through `tensor_ctor`

When executing `x = torch.tensor([3.], requires_grad=True)`, the tensor creator is first called:

```c++
// torch/csrc/autograd/python_torch_functions_manual.cpp
static PyObject* THPVariable_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS      // try-catch micro to handle errors
  static PythonArgParser parser({
      "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  constexpr int ctor_num_args = 6;
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    // check overwrite, see [document](https://pytorch.org/docs/stable/notes/extending.html#extending-torch)
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // ...
  return THPVariable_Wrap(torch::utils::tensor_ctor(
      torch::tensors::get_default_dispatch_key(),   // default: CPU
      torch::tensors::get_default_scalar_type(),    // deafult: float
      r));
  END_HANDLE_TH_ERRORS
}
```

Then calls for `tensor_ctor`

```c++
// torch/csrc/utils/tensor_new.cpp
Tensor tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  // ...
  PyObject* data = r.pyobject(0);
  // if dtype not passed by user, `internal_new_from_data` will infer
  bool type_inference = r.isNone(1);
  bool pin_memory = r.toBool(3);
  bool args_requires_grad = r.toBool(4);
  auto new_tensor = internal_new_from_data(
      typeIdWithDefault(r, 2, dispatch_key),
      r.scalartypeWithDefault(1, scalar_type),
      r.deviceOptional(2),
      data,
      /*copy_variables=*/true,
      /*copy_numpy=*/true,
      /*type_inference=*/type_inference,
      pin_memory);
  // ...
  new_tensor.detach_();   // ensure `new_tensor` is a leaf node
  new_tensor.set_requires_grad(args_requires_grad);
  return new_tensor;
  // ...
}
```

The key functions are: `new_tensor.detach_()` and `new_tensor.set_requires_grad(args_requires_grad);`

### `detach` and `AutogradMeta`

`new_tensor.detach_()` calls `inline at::Tensor & Tensor::detach_()`

```c++
// torch/include/ATen/core/TensorBody.h
inline at::Tensor & Tensor::detach_() const {
    return at::_ops::detach_::call(const_cast<Tensor&>(*this));
}
```

Through dispatch, it calls into `VariableTypeManual.cpp`. We won't delve into the dispatch process here. Those interested can read my previous document [deep_dive_into_contiguous(1)](../deep_dive_into_contiguous_1).

```c++
// torch/csrc/autograd/VariableTypeManual.cpp
Tensor& detach_(c10::DispatchKeySet ks, Tensor& self) {
  RECORD_FUNCTION("detach_", std::vector<c10::IValue>({self}));
  // ...
  auto autograd_meta = impl::materialize_autograd_meta(self);
  autograd_meta->set_requires_grad(false, self.unsafeGetTensorImpl());
  autograd_meta->grad_fn_.reset();
  autograd_meta->output_nr_ = 0;
  autograd_meta->fw_grad_.reset();

  return self;
}
```

`detach` first calls `materialize_autograd_meta` to get `autograd_meta_` (if it doesn't exist, it will initialize the tensor with `std::make_unique<AutogradMeta>()`), and then clears `requires_grad`, `grad_fn` and `output_nr` to detach the node from the computation graph. However, there is no computation graph at this point, so this serves an initialization role.

Let's take a look at the data structure of `AutogradMeta`:

```c++
// torch/csrc/autograd/variable.h
struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;                           // grad_, a tensor
  std::shared_ptr<Node> grad_fn_;           // grad function (node)
  // to accumulate grad, used by leaf node
  std::weak_ptr<Node> grad_accumulator_;

  // Used to compute higher-order derivatives, stores forward gradients.
  std::shared_ptr<ForwardGrad> fw_grad_;
  std::vector<std::unique_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list_;

  bool requires_grad_{false};
  bool retains_grad_{false};
  bool is_view_{false};

  // The output index, for example, if this variable is the second output of a function, then output_nr = 1.
  uint32_t output_nr_;

  // ...

  AutogradMeta(
      at::TensorImpl* self_impl = nullptr,
      bool requires_grad = false,
      Edge gradient_edge = Edge())  // 创建autograd_meta_需要一个（默认的）Edge
      : grad_fn_(std::move(gradient_edge.function)),
        output_nr_(gradient_edge.input_nr) {
    if (requires_grad) {
      // ...
      set_requires_grad(requires_grad, self_impl);
    }
    // ...
  }
};
```

Each tensor (or **Variable**) has a **unique** `autograd_meta`, which is used to store data required for automatic differentiation (such as the derivative function `grad_fn_`, gradient value `grad_`, etc.). It's worth noting that the tensor itself does not initialize AutogradMeta when it is declared. It is a `nullptr` to minimize overhead as much as possible. All `autograd_meta` need to be explicitly set through the set method, or initialized through `materialize_autograd_meta` as mentioned above.

### `set_requires_grad`, tensor construction completed

After `detach_` is executed, we perform `set_requires_grad` to set attributes for the tensor.

```c++
// torch/include/ATen/core/TensorBody.h
class TORCH_API Tensor: public TensorBase {
  // ...
  const Tensor& set_requires_grad(bool requires_grad) const {
    TensorBase::set_requires_grad(requires_grad);
    return *this;
  }
}

// aten/src/ATen/core/TensorBase.h
class TORCH_API TensorBase {
  const TensorBase& set_requires_grad(bool requires_grad) const {
    impl_->set_requires_grad(requires_grad);
    return *this;
  }
}

// c10/core/TensorImpl.cpp
void TensorImpl::set_requires_grad(bool requires_grad) {
  // ...
  if (!requires_grad && !autograd_meta_)
    return;
  if (!autograd_meta_)  // after detach_, autograd_meta_ has been initialized
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  autograd_meta_->set_requires_grad(requires_grad, this);
}
```

Setting the `autograd_meta` parameters through the `Tensor(TensorBody)` -> `TensorBase` -> `TensorImpl` -> `autograd_meta_->set_requires_grad()` path, which is fairly straightforward.

Up to this point, our tensor has been created, and the `autograd_meta_` with `requires_grad` set to `True` has been successfully initialized in `TensorImpl`.

## Forward computation `y = x * x`, constructing the computation graph

### Dispatch to AutogradCPU: `mul_Tensor`

Let's look at the next statement in the example, `y = x * x`

After the dispatch of the mul operator (the dispatch key is `AutogradCPU`), it goes to `torch/csrc/autograd/generated/VariableType_0.cpp` (generated code, you need to compile pytorch to get it).

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
at::Tensor mul_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, other );
  [[maybe_unused]] auto _any_has_forward_grad_result = (isFwGradDefined(self) || isFwGradDefined(other));
  
  std::shared_ptr<MulBackward0> grad_fn;
  if (_any_requires_grad) {
    // generate grad_fn
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
    grad_fn->other_scalar_type = other.scalar_type();
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->self_scalar_type = self.scalar_type();
  }
  // ...
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mul(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  // ...
  if (grad_fn) {
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  // ...
  return result;
}
```

### Construct `grad_fn` (`Node`)

Note `grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);`. What is `MulBackward0` here?

```c++
// torch/include/torch/csrc/autograd/generated/Functions.h
struct TORCH_API MulBackward0 : public TraceableFunction {
  using TraceableFunction::TraceableFunction;
  variable_list apply(variable_list&& grads) override;
  std::string name() const override { return "MulBackward0"; }
  void release_variables() override {
    std::lock_guard<std::mutex> lock(mutex_);
    other_.reset_data();
    self_.reset_data();
  }
  SavedVariable other_;             // tensor
  at::ScalarType other_scalar_type;
  SavedVariable self_;
  at::ScalarType self_scalar_type;
};

// torch/include/torch/csrc/autograd/function.h
struct TraceableFunction : public Node {
  using Node::Node;
  bool is_traceable() final {
    return true;
  }
};

struct TORCH_API Node : std::enable_shared_from_this<Node> {
 public:
    // A node is created based on next_edges, and the next constructor calls the constructor above.
    explicit Node(uint64_t sequence_nr, edge_list&& next_edges = edge_list())
      : sequence_nr_(sequence_nr), next_edges_(std::move(next_edges)) {
    for (const Edge& edge : next_edges_) {
      update_topological_nr(edge);
    }
    // ...
  }
  explicit Node(edge_list&& next_edges = edge_list())
      : Node(/*sequence_nr=*/at::sequence_number::get_and_increment(),
            std::move(next_edges)) {}
}
```

Here, `MulBackward0` is a **Node**.

Let's focus on the **Node**:

- `Node` is an abstract base class, all **functions** in the PyTorch autograd mechanism (like `MulBackward0` above) inherit this class and override its **apply** method.

- In the computation graph, `Node` connects with other `Nodes` through **Edges** (represented by <`Node`, `input_nr`> pairs). `Variable` is propagated between nodes through `Edge`. It's worth noting that when two or more edges from different `Nodes` point to the same `Node` as inputs, all gradients generated on these edges will be implicitly summed by the `Node`.

- `Node` can support arbitrary inputs and outputs. For example, `AccumulateGrad` has multiple inputs but no outputs, while `GraphRoot` has no inputs but multiple outputs. The number of inputs and outputs can be determined by `num_inputs()` and `num_outputs()`.

- `Node` can use the `next_edge()` method to get all output edges, or call `next_edge(index)` to get a specific edge. It can also use `add_next_edge()` to set edges, etc. These methods are often used in **JIT**.

- Each `Node` has a **sequence number** that monotonically increases according to the order in which the `Node` is constructed. However, this monotonic increase **only takes effect in the current thread**. If `Node` A and B are created in thread 1, and `Node` C is created in thread 2, then the sequence number of C is completely unrelated to A and B.

- `Node` has the following data members:

```c++
struct TORCH_API Node : std::enable_shared_from_this<Node> {
 protected:
  const uint64_t sequence_nr_;
  uint64_t topological_nr_ = 0;
  mutable bool has_parent_ = false;

  // The Node in autograd is not thread-safe, users need to consider 
  // using locks when calling `release_variables()`, `apply()`.
  // Note that this cannot ensure that hooks are thread-safe, 
  // PyTorch requires users to register thread-safe hook code themselves
  // if they want the hooks to get correct results in a multi-threaded environment.
  uint64_t thread_id_ = 0;
  std::mutex mutex_;
  
  edge_list next_edges_;

  // Store a weak reference to a Python object, so we can call autograd operations defined in Python through this.
  PyObject* pyobj_ = nullptr;
  // Exception metadata, store additional information related to the exception.
  std::unique_ptr<AnomalyMetadata> anomaly_metadata_ = nullptr;

  // Hooks that are called when the Node is executed.
  std::vector<std::unique_ptr<FunctionPreHook>> pre_hooks_;
  // Will be called even if the Node is not executed, as long as the flow reaches here.
  std::vector<std::unique_ptr<FunctionPreHook>> tensor_pre_hooks_;
  // Similar to tensor_pre_hooks_, but will be called after all tensor_pre_hooks_ have been called.
  std::unordered_map<int, std::unique_ptr<FunctionPreHook>> retains_grad_hooks_;
  // Called after the Node is executed.
  std::vector<std::unique_ptr<FunctionPostHook>> post_hooks_;
  at::SmallVector<InputMetadata, 2> input_metadata_;
};
```

Among them, **sequence_nr_** is used to determine the priority of the backward task. The later the Node is created, the larger the `sequence_nr_` is, meaning that it has a higher priority in reverse execution (this can be seen together with the priority queue below).

It's worth pointing out that the `sequence_nr_` of `AccumulateGrad` is explicitly set to **UINT64_MAX**, which means that as long as there is `AccumulateGrad` in the queue (and other conditions are the same), the gradient is calculated for AccumulateGrad first, which can quickly clear the queue and improve running efficiency.

In addition, **topological_nr_** represents the longest path length from this node to any leaf node, for example, the value for AccumulateGrad is 0.

For any nodes X and Y in the graph, if there is a path from X to Y, `then topo_nr(X) > topo_nr(Y)`, but the reverse is not true. In other words, we can directly determine that there is no path from X to Y through `topo_nr(X) <= topo_nr(Y)`.

But note that using `topological_nr` has an assumption, that is, once a node is used (there is a parent node), its `topological_nr` cannot change. PyTorch uses `has_parent_` to enforce this point. Why can't it change? For example:

```c++
//   1) 2 -> 1 -> 0
//   2)        2 -> 1 -> 0
//            /
//      2 -> 1 -> 0        Here an additional 2 is added as the next_edge of 1, although 1 already has a parent
//   3)        2 -> 1 -> 0
//            /
//      2 -> 3 -> 0        Here 2 < 3, but there is obviously a path from 2 to 3
```

After the `Node` is created, we go back to `grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);` and continue.

### Setting `Edge` for `grad_fn`

First, call `collect_next_edges`, return all `next_edges` of variables. Note that here the template programming of `...` is used to encapsulate `(self, other)` into a `variables` parameter, and then call the apply method of `MakeNextFunctionList` (inherited from **IterArgs**) to recursively iterate the parameter package `variables`.

```c++
// torch/include/torch/csrc/autograd/function.h
template <typename... Variables>
edge_list collect_next_edges(Variables&&... variables) {
  detail::MakeNextFunctionList make;
  make.apply(std::forward<Variables>(variables)...);
  // Here, move transfers ownership, using move semantics without the need for a copy
  return std::move(make.next_edges);
}

// aten/src/ATen/core/Variadic.h
struct IterArgs {
  template <typename T, typename... Args>
  inline F& apply(T&& arg, Args&&... args) {
    // Parse an arg first, then recursively parse all args
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }
}
```

Here, after template and perfect forwarding, we arrive at the `()` overload symbol in `MakeNextFunctionList`.

```c++
// torch/include/torch/csrc/autograd/function.h
// The actual function body of `collect_next_edges`, after unpacking above,
// the parameter that comes in here is already a single variable
struct MakeNextFunctionList : IterArgs<MakeNextFunctionList> {
  edge_list next_edges;
  using IterArgs<MakeNextFunctionList>::operator();
  void operator()(const Variable& variable) {
    if (variable.defined()) {
      next_edges.emplace_back(impl::gradient_edge(variable));
    } else {
      next_edges.emplace_back();
    }
  }
}

// torch/csrc/autograd/variable.cpp
Edge gradient_edge(const Variable& self) {
  // If the obtained grad_fn is nullptr (such as in the case of leaf nodes),
  // we return an edge of `grad_accumulator` (AccumulateGrad), 
  // which will sum all the gradients of the incoming edges and accumulate them into the grad attribute of the variable.
  // Note that only leaf nodes with `requires_grad = True` will have AccumulateGrad
  if (const auto& gradient = self.grad_fn()) {
    return Edge(gradient, self.output_nr());
  } else {
    return Edge(grad_accumulator(self), 0);
  }
}
```

At this point, we will go down the else branch to create and set `grad_accumulator` for the two tensor parameters `self` and `other` of the mul operation.

```c++
// torch/csrc/autograd/variable.cpp
std::shared_ptr<Node> grad_accumulator(const Variable& self) {
  auto autograd_meta = get_autograd_meta(self);
  // ...

  // intrusive_ptr is a kind of smart pointer for PyTorch reference counting
  c10::raw::intrusive_ptr::incref(self.unsafeGetTensorImpl());
  auto intrusive_from_this =
      c10::intrusive_ptr<at::TensorImpl>::reclaim(self.unsafeGetTensorImpl());
  result = std::make_shared<AccumulateGrad>(
      Variable(std::move(intrusive_from_this)));
  autograd_meta->grad_accumulator_ = result;
  return result;
}
```

Here, we also introduce the `Edge` data structure, which has two important data members, one is `function`: a `std::shared_ptr` pointing to a `Node`, and the other is `input_nr`: used to identify the position of the input corresponding to this edge among all inputs in the function node. For example, if the `function` node has 3 inputs (i.e., there are three edges pointing to this node) (here the input refers to the gradient passed back from the previous node), `input_nr` could be 0, 1, or 2. If it is 1, it means that this edge is the second of the three incoming edges to this node.

```c++
// torch/include/torch/csrc/autograd/edge.h
struct Edge {
  Edge() noexcept : function(nullptr), input_nr(0) {}

  Edge(std::shared_ptr<Node> function_, uint32_t input_nr_) noexcept
      : function(std::move(function_)), input_nr(input_nr_) {}

  std::shared_ptr<Node> function;
  uint32_t input_nr;
};
```

After the Edge of `grad_accumulator` is created, then the second variable is parsed. After the same process, it is returned to `collect_next_edges`, and two edges are collected in total.

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
```

Then, call `grad_fn->set_next_edges` to set these collected edges for `grad_fn` (mulbackward). Note `update_topological_nr`, the principle of which we have already introduced when introducing the data members of `Node`.

```c++
// torch/include/torch/csrc/autograd/function.h
struct TORCH_API Node : std::enable_shared_from_this<Node> {
  void set_next_edges(edge_list&& next_edges) {
    next_edges_ = std::move(next_edges);
    for (const auto& next_edge : next_edges_) {
      update_topological_nr(next_edge);
    }
  }

  void update_topological_nr(const Edge& edge) {
    TORCH_INTERNAL_ASSERT(
        !has_parent_,
        "Cannot update a node's topological_nr after it already has a parent."
        " If we allow this, we can no longer guarantee that a parent's"
        " topo_nr is always greater than those of all its children")
    Node* node = edge.function.get();
    if (node) {
      auto topo_nr = node->topological_nr();
      if (topological_nr_ <= topo_nr) {
        topological_nr_ = topo_nr + 1;
      }
    }
  }

  uint64_t topological_nr() const noexcept {
    // Called when setting an edge. Since it is called, it must have a parent,
    // so the variable `has_parent_` can be directly set.
    has_parent_ = true;
    return topological_nr_;
  }
}
```

After `set_next_edges`, we call `should_compute_output` to determine whether the edge has a function (needs to compute gradient). Particularly note that when `should_compute_output(0)` (corresponding to the self edge) we save `grad_fn->other_`, and when `should_compute_output(1)` (corresponding to the other edge) we save `grad_fn->self_`. This is because the essence of mul backward is gradient exchange.

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
  if (_any_requires_grad) {
    // ...
    // y = a * b
    // y'(a) = b * grad 
    // y'(b) = a * grad
    if (grad_fn->should_compute_output(0)) {
      grad_fn->other_ = SavedVariable(other, false);
    }
    grad_fn->other_scalar_type = other.scalar_type();
    if (grad_fn->should_compute_output(1)) {
      grad_fn->self_ = SavedVariable(self, false);
    }
    grad_fn->self_scalar_type = self.scalar_type();
  }
```

### Redispatch and **Guard**

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard; 
    return at::redispatch::mul(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
```

Next, we call the anonymous function `tmp` to redispatch the mul op. We first declare the RAII `at::AutoDispatchBelowADInplaceOrView guard` to add `autograd_dispatch_keyset_with_ADInplaceOrView` (which includes AutogradFunctionality, AutogradOther, AutogradNestedTensor, ADInplaceOrView four types of dispatch keys) to the exclude list of the local thread, ensuring that this call will not be dispatched to these four types of keys throughout the entire link.

Let's give an example. We modify `tools/autograd/gen_variable_type.py` to turn off guard and recompile PyTorch, execute `export TORCH_SHOW_DISPATCH_TRACE=1` to print the dispatch trace, and re-execute the example code above to get one part of the output.

```bash
# ...
 [call] op=[aten::isfinite], key=[AutogradCPU]
  [call] op=[aten::eq.Tensor], key=[AutogradCPU]
   [redispatch] op=[aten::eq.Tensor], key=[CPU]
  [call] op=[aten::abs], key=[AutogradCPU]
   [redispatch] op=[aten::abs], key=[CPU]
    [call] op=[aten::empty.memory_format], key=[BackendSelect]
     [redispatch] op=[aten::empty.memory_format], key=[CPU]
    [call] op=[aten::abs.out], key=[AutogradCPU]
     [redispatch] op=[aten::abs.out], key=[ADInplaceOrView]
      [redispatch] op=[aten::abs.out], key=[CPU]
# ...
```

However, if we do not turn off the guard, executing the original PyTorch will yield the output.

```bash
# ...
 [call] op=[aten::isfinite], key=[AutogradCPU]
  [call] op=[aten::eq.Tensor], key=[AutogradCPU]
   [redispatch] op=[aten::eq.Tensor], key=[CPU]
  [call] op=[aten::abs], key=[AutogradCPU]
   [redispatch] op=[aten::abs], key=[CPU]
    [call] op=[aten::empty.memory_format], key=[BackendSelect]
     [redispatch] op=[aten::empty.memory_format], key=[CPU]
    [call] op=[aten::abs.out], key=[CPU]
# ...
```

We can clearly see that after turning off the guard, the `abs` operation will be redispatched to `AutogradCPU` and `ADInplaceOrView`, generating unnecessary operations.

Then we compute the new dispatch key. Note that here ks is `DispatchKeySet(CPU, AutogradCPU)`, `c10::after_autograd_keyset` is `"DispatchKeySet(CPU, CUDA, HIP, XLA, ...`, and the bitwise AND of the two yields `DispatchKeySet(CPU)`. That is, our next mul will be dispatched to the CPU operator, i.e., `wrapper_CPU_mul_Tensor`.

### Structured kernel and Stub

The `wrapper_CPU_mul_Tensor` is essentially a **structured kernel**, with its key methods `.meta` and `.impl`.

```c++
// build/aten/src/ATen/RegisterCPU.cpp
at::Tensor wrapper_CPU_mul_Tensor(const at::Tensor & self, const at::Tensor & other) {
  structured_mul_out_functional op;
  op.meta(self, other);
  op.impl(self, other, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

// torch/include/ATen/ops/mul_native.h
struct TORCH_API structured_mul_out : public at::meta::structured_mul_Tensor {
    void impl(const at::Tensor & self, const at::Tensor & other, const at::Tensor & out);
};

// torch/include/ATen/ops/mul_meta.h
struct TORCH_API structured_mul_Tensor : public TensorIteratorBase {
    void meta(const at::Tensor & self, const at::Tensor & other);
};
```

The `structured_mul_out_functional` is essentially an inheritance from **TensorIterator**. Its two functions, `meta` and `impl`, are overridden via macro definitions.

```c++
// aten/src/ATen/native/BinaryOps.cpp
TORCH_META_FUNC2(mul, Tensor) (
// void structured_mul_Tensor::meta (
  const Tensor& self, const Tensor& other
) {
  // maybe_get_output() retrieves the default output_ (undefined) of the structured_mul_out_functional op
  // then follows `build_borrowing_binary_op`, i.e., tensor iterator infer, to make it defined
  build_borrowing_binary_op(maybe_get_output(), self, other);
}

// aten/src/ATen/TensorIterator.cpp
void TensorIteratorBase::build_borrowing_binary_op(
    const TensorBase& out, const TensorBase& a, const TensorBase& b) {
  build(BINARY_OP_CONFIG()
      .add_output(out)
      .add_input(a)
      .add_input(b));
}
```

After calling `meta`, a **TensorIterator** is created. This is also a very important class, which we will not go into detail here. Those interested can refer to my previous article [deep_dive_into_contiguous(3)](../deep_dive_into_contiguous_3).

Then call the `impl` method. Here, `mul_stub` is a `struct` that inherits from **DispatchStub** and has the corresponding template filled in.

```c++
// aten/src/ATen/native/BinaryOps.cpp
DEFINE_DISPATCH(mul_stub);      // struct mul_stub mul_stub

TORCH_IMPL_FUNC(mul_out) (
  // void structured_mul_out::impl(
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  // device_type() directly retrieves the inferred device from within the tensor iterator
  // `this` is an instance of structured_mul_out
  mul_stub(device_type(), *this);
}

// torch/include/ATen/native/BinaryOps.h
DECLARE_DISPATCH(structured_binary_fn, mul_stub);
/* struct mul_stub : DispatchStub<structured_binary_fn, mul_stub> {
  mul_stub() = default;
  mul_stub(const mul_stub&) = delete;
  mul_stub& operator=(const mul_stub&) = delete;
};
extern __attribute__((__visibility__("default"))) struct mul_stub mul_stub */
```

For all kinds of **stub** in PyTorch, they will uniformly go to `aten/src/ATen/native/DispatchStub.h`, and choose the appropriate call method based on the device.

```c++
template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

private:
  FnPtr get_call_ptr(DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
// Select the instruction set. CPU, AVX2, AVX512, etc. are all Intel's instruction sets.
// PyTorch will automatically select the more optimized instruction set.
#ifdef HAVE_AVX2_CPU_DEFINITION
      , reinterpret_cast<void*>(AVX2)
#endif
// ...
      )
    );
  }

public:
  template <typename... ArgTypes>
  rT operator()(DeviceType device_type, ArgTypes&&... args) {
    FnPtr call_ptr = get_call_ptr(device_type);
    return (*call_ptr)(std::forward<ArgTypes>(args)...);
  }
private:
  DispatchStubImpl impl;
}
```

Through `get_call_ptr`, we get the pointer to `mul_kernel`, and then call it.

```c++
// aten/src/ATen/native/cpu/BinaryOpsKernel.cpp
void mul_kernel(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (dtype == ScalarType::Bool) {
    cpu_kernel(iter, [=](bool a, bool b) -> bool { return a && b; });
  } else if (dtype == kComplexHalf) {
    // ...
  } else if (iter.is_scalar(2) && at::isReducedFloatingType(dtype)) {
    // ...
  } else {
    AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(kBFloat16, kHalf, dtype, "mul_cpu", [&]() {
      cpu_kernel_vec(iter,
        [=](scalar_t a, scalar_t b) __ubsan_ignore_undefined__ -> scalar_t { return a * b; },
        [=](Vectorized<scalar_t> a, Vectorized<scalar_t> b) __ubsan_ignore_undefined__ {
          return a * b;
        });
    });
  }
}
```

Here, `tensor iterator` and several anonymous functions are passed to the `cpu_kernel_vec` vectorization method, and then the loop is split and vectorized calls are made. This part of the content has been detailed in my previous articles, so we will not expand on it here.

### `set_history`, Completion of Calculation Graph Construction

After executing the `_tmp` method, we get the forward result.

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
at::Tensor mul_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  // ...
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard;
    return at::redispatch::mul(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
  // ...
  if (grad_fn) {
      // Encapsulate the tensor into a variable list
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  // ...
  return result;
}
```

Then we call `set_history`, which is a key step in building the computation graph. It sets `grad_fn` of `result` for retrieval.

```c++
// torch/csrc/autograd/functions/utils.h
inline void set_history(
    at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  AT_ASSERT(grad_fn);
  if (variable.defined()) {
    // Check if the tensor has a differentiable dtype
    TORCH_INTERNAL_ASSERT(isDifferentiableType(variable.scalar_type()));
    // Store the variable in the input_metadata_ of grad_fn and return its index
    // Here we get 0, which means that variable is the first input of this node
    auto output_nr = grad_fn->add_input_metadata(variable);
    // {grad_fn, output_nr} constructs a new edge
    // Then use this edge to set autograd_meta for the variable
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

// torch/csrc/autograd/variable.cpp
void set_gradient_edge(const Variable& self, Edge edge) {
  // If the variable self has not defined autograd meta, then set it here
  auto* meta = materialize_autograd_meta(self);
  meta->grad_fn_ = std::move(edge.function);
  meta->output_nr_ = edge.input_nr;
  // ...
}
```

Up to this point, the forward computation of the multiplication operator has been fully executed, including:

1. The `next_edges` of `grad_fn` point to the forward inputs `self` and `other` (`AccumulateGrad`)
2. The `input_metadata_` of `grad_fn` stores the forward `result`
3. `grad_fn` is bound to the **autograd_meta_** of the forward `result` Tensor

After these operations are completed, we can call `result.backward()`, i.e., find `grad_fn` through the `autograd_meta_` of the tensor and perform backward computation.
