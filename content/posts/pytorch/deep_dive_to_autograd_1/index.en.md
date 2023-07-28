---
title: "Deep Dive to Pytorch AutoGrad(1)"
date: 2023-07-04T10:49:42+08:00
categories: ["pytorch"]
summary: "This article introduces the implementation details of pytorch autograd mechanism."
---

## Summary

This article introduces the implementation details of pytorch autograd mechanism.

## To be translated

Oh Sorry!

This blog has't been translated to English, please wait for a little while...

## 引入

笔者使用pytorch版本：`2.1.0a0+gita3dddae`

```py
import torch

x = torch.tensor([3.], requires_grad=True)
y = x * x
print(y)       # tensor([9.], grad_fn=<MulBackward0>)
y.backward()
print(x.grad)  # tensor([6.])，y = x^2，dy/dx = 2x
```

这是一个基本的autograd运算，新建叶子结点`x`，然后`y = x * x`时构建计算图，最后调用`y.backward()`进行反向运算。

我们将深入c++层面，逐步分析pytorch是如何构建`requires_grad`的tensor，前向运算中如何构建计算图，反向求导时如何根据计算图实现自动微分与梯度累加。

## tensor `x`的创建：`x = torch.tensor([3.], requires_grad=True)`

### `tensor_ctor`构造tensor

执行`x = torch.tensor([3.], requires_grad=True)`时，首先调用到python层与c++层的tensor creator处：

```c++
// torch/csrc/autograd/python_torch_functions_manual.cpp
static PyObject* THPVariable_tensor(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS      // try-catch宏，处理错误
  static PythonArgParser parser({
      "tensor(PyObject* data, *, ScalarType dtype=None, Device? device=None, bool pin_memory=False, bool requires_grad=False, DimnameList? names=None)",
  });

  constexpr int ctor_num_args = 6;
  ParsedArgs<ctor_num_args> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  if (r.has_torch_function()) {
    // 检查有没有用户重载，详见[document](https://pytorch.org/docs/stable/notes/extending.html#extending-torch)
    return handle_torch_function(
        r, nullptr, args, kwargs, THPVariableFunctionsModule, "torch");
  }
  // ...
  return THPVariable_Wrap(torch::utils::tensor_ctor(
      torch::tensors::get_default_dispatch_key(),   // 默认CPU
      torch::tensors::get_default_scalar_type(),    // 默认float
      r));
  END_HANDLE_TH_ERRORS
}
```

随后调用`tensor_ctor`方法，以构造函数的方式创建tensor

```c++
// torch/csrc/utils/tensor_new.cpp
Tensor tensor_ctor(
    c10::DispatchKey dispatch_key,
    at::ScalarType scalar_type,
    PythonArgs& r) {
  // ...
  PyObject* data = r.pyobject(0);
  // 如果用户没传dtype就是true，然后之后internal_new_from_data时会infer tensor的dtype
  bool type_inference = r.isNone(1);
  bool pin_memory = r.toBool(3);
  bool args_requires_grad = r.toBool(4);
  // 根据参数创建tensor，这里我们不详细展开
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
  new_tensor.detach_();   // 确保new_tensor是一个叶子结点
  new_tensor.set_requires_grad(args_requires_grad);
  return new_tensor;
  // ...
}
```

上面最关键的就是`new_tensor.detach_()`和`new_tensor.set_requires_grad(args_requires_grad);`

### `detach`与创建`AutogradMeta`

`new_tensor.detach_()`调用到

```c++
// torch/include/ATen/core/TensorBody.h
inline at::Tensor & Tensor::detach_() const {
    return at::_ops::detach_::call(const_cast<Tensor&>(*this));
}
```

通过dispatch调用到`VariableTypeManual.cpp`，这里我们不详细展开dispatch的流程，有兴趣的小伙伴可以阅读笔者之前的文档[How_pytorch_call_ops](../how_pytorch_call_op_1/index.en.md)

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

`detach`首先调用`materialize_autograd_meta`拿到`autograd_meta_`（如果不存在，则会给tensor初始化一个`std::make_unique<AutogradMeta>()`），然后将`requires_grad`, `grad_fn` 和 `output_nr`清空，实现将节点从计算图中detach出来。但此时还没有计算图，所以起到的是一个初始化的作用。

我们看一下`AutogradMeta`的数据结构：

```c++
// torch/csrc/autograd/variable.h
struct TORCH_API AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  Variable grad_;                           // grad_，本质是一个 tensor
  std::shared_ptr<Node> grad_fn_;           // 求导函数（节点）
  std::weak_ptr<Node> grad_accumulator_;    // 梯度累加器，叶子结点使用

  std::shared_ptr<ForwardGrad> fw_grad_;    // 计算高阶导数时使用，存储前向梯度
  std::vector<std::unique_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list_;

  bool requires_grad_{false};
  bool retains_grad_{false};
  bool is_view_{false};

  // output索引，如这个variable是一个function的第二个输出，那么output_nr = 1
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

每个tensor（或者说**Variable**）都有一个**unique**的`autograd_meta`，用其存储自动求导所需的数据（如求导函数`grad_fn_`，梯度值`grad_`等）。值得指出的是，tensor本身声明时并不会初始化AutogradMeta，它是一个`nullptr`以尽可能减少开销，所有的`autograd_meta`都需要通过set方法显式设定，或者像上文那样通过`materialize_autograd_meta`初始化。

### `set_requires_grad`，tensor构造完成

`detach_`执行完后，我们执行`set_requires_grad`给tensor设置属性

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
  if (!autograd_meta_)  // 上面detach_执行完后，autograd_meta_已经存在
    autograd_meta_ = impl::GetAutogradMetaFactory()->make();
  autograd_meta_->set_requires_grad(requires_grad, this);
}
```

即通过`Tensor(TensorBody)` -> `TensorBase` -> `TensorImpl` -> `autograd_meta_->set_requires_grad()`这一路径设置`autograd_meta`的参数，并没有什么特别的。

到目前为止，我们的tensor就已经创建好了，并成功在`TensorImpl`里初始化了`requires_grad`为True的`autograd_meta_`。

## 前向计算`y = x * x`，构建计算图

### dispatch到AutogradCPU：`mul_Tensor`

我们看引入例子中的下一条语句`y = x * x`

经过mul算子的dispatch(dispatch key为`AutogradCPU`)到`torch/csrc/autograd/generated/VariableType_0.cpp`（生成的代码，需要编译才能得到）

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
at::Tensor mul_Tensor(c10::DispatchKeySet ks, const at::Tensor & self, const at::Tensor & other) {
  auto& self_ = unpack(self, "self", 0);
  auto& other_ = unpack(other, "other", 1);
  [[maybe_unused]] auto _any_requires_grad = compute_requires_grad( self, other );
  [[maybe_unused]] auto _any_has_forward_grad_result = (isFwGradDefined(self) || isFwGradDefined(other));
  
  std::shared_ptr<MulBackward0> grad_fn;
  if (_any_requires_grad) {
    // 如果有tensor requires_grad，生成grad fn
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

### 构建`grad_fn`（`Node`）

注意`grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);`，这里`MulBackward0`是什么呢？

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
  /// 基于next_edges创建node，下一个构造函数调用了上面这个构造函数
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

这里`MulBackward0`是一个**Node**

我们重点介绍一下**Node**：

- `Node`是一个抽象基类，所有的pytorch autograd机制中的**function**（如上文的`MulBackward0`）都继承这个类并重写其**apply**方法。

- 在计算图中，`Node`通过**Edge**（<`Node`, `input_nr`>对表示）与其他`Node`连接。`Variable`通过`Edge`在节点间传导。值得指出的是，当有两条或更多来自不同`Node`的`Edge`作为输入指向同一个`Node`时，所有在这些`Edge`上产生的梯度会被`Node`隐式求和。

- `Node`可支持任意输入输出，如`AccumulateGrad`有多个输入但没有输出，如`GraphRoot`没有输入但有多个输出。输入输出的数量可以通过`num_inputs()`和`num_outputs()`来确定

- `Node`可以用`next_edge()`方法获取所有的输出边，或调用`next_edge(index)`来获取具体的边，可以用`add_next_edge()`设置边等，这些方法常在**JIT**中使用。

- 每个`Node`都有一个按照`Node`构造顺序单调递增的**sequence number**，但这个单调递增**只在当前线程生效**，如果`Node` A、B 在线程1创建，`Node` C在线程2创建，那么C的sequence number与A、B的完全无关。

- `Node`有以下数据成员：

```c++
struct TORCH_API Node : std::enable_shared_from_this<Node> {
 protected:
  const uint64_t sequence_nr_;
  uint64_t topological_nr_ = 0;
  mutable bool has_parent_ = false;

  // autograd的Node并非线程安全，用户需要在调用`release_variables()`、`apply()`时考虑
  // 使用锁，注意这不能保证hook是线程安全的，pytorch需要用户自己注册线程安全的hook代码如果
  // 用户希望hook在多线程环境中能得到正确的结果。
  uint64_t thread_id_ = 0;
  std::mutex mutex_;
  
  edge_list next_edges_;

  // 存储一个python对象弱引用，通过此调用在python中定义的自动微分操作
  PyObject* pyobj_ = nullptr;
  // 异常元数据，存储异常相关的额外信息
  std::unique_ptr<AnomalyMetadata> anomaly_metadata_ = nullptr;

  // Node被执行时才调用的hook
  std::vector<std::unique_ptr<FunctionPreHook>> pre_hooks_;
  // 即使Node没有执行，只要流到此处也会调用
  std::vector<std::unique_ptr<FunctionPreHook>> tensor_pre_hooks_;
  // 类似tensor_pre_hooks_，但在所有tensor_pre_hooks_调用后才会调用
  std::unordered_map<int, std::unique_ptr<FunctionPreHook>> retains_grad_hooks_;
  // Node执行完后调用
  std::vector<std::unique_ptr<FunctionPostHook>> post_hooks_;
  at::SmallVector<InputMetadata, 2> input_metadata_;
};
```

其中，**sequence_nr_**用于确定backward task的优先级，越晚创建的Node `sequence_nr_`越大，意味着在反向执行优先级越高（这一点可以和下文优先队列一起看）。

值得指出的是，`AccumulateGrad`的`sequence_nr_`被显式设定为**UINT64_MAX**，这意味着只要队列中有`AccumulateGrad`（且其他条件相同），那么优先给AccumulateGrad计算梯度，这样可以快速清空队列，提高运行效率。

此外，**topological_nr_**表示这个节点到任何叶节点最长的路径长度，如AccumulateGrad该值就为0。

对于图中任意节点X、Y，如果存在X到Y的路径则有`topo_nr(X) > topo_nr(Y)`，但反过来不成立。换句话说，我们可以通过`topo_nr(X) <= topo_nr(Y)`来直接判断不存在X到Y的路径

但注意，使用`topological_nr`有一个假设，即一旦一个节点被使用过（存在父节点），那么它的`topological_nr`不能改变，pytorch中使用`has_parent_`来强制校验这一点。为什么不能被改变呢？例如：

```c++
//   1) 2 -> 1 -> 0
//   2)        2 -> 1 -> 0
//            /
//      2 -> 1 -> 0        这里添加了一个2作为1的next_edge，尽管1已经有parent了 
//   3)        2 -> 1 -> 0
//            /
//      2 -> 3 -> 0        这里2 < 3，但显然有一条2到3的路径
```

创建好`Node`后，我们回到`grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);`继续往下走

### 给`grad_fn`设置`Edge`

首先调用`collect_next_edges`，返回variables的所有`next_edges`，注意这里使用了`...`的模板编程，把`(self, other)`封成了一个`variables`传参，随后调用`MakeNextFunctionList`（继承自**IterArgs**）的apply方法递归迭代参数包`variables`。

```c++
// torch/include/torch/csrc/autograd/function.h
template <typename... Variables>
edge_list collect_next_edges(Variables&&... variables) {
  detail::MakeNextFunctionList make;
  make.apply(std::forward<Variables>(variables)...);
  return std::move(make.next_edges);    // 这里move转移所有权，使用移动语义而无需copy
}

// aten/src/ATen/core/Variadic.h
struct IterArgs {
  template <typename T, typename... Args>
  inline F& apply(T&& arg, Args&&... args) {
    // 先解析一个arg，然后递归解析所有args
    self()(std::forward<T>(arg));
    if (self().short_circuit()) {
      return self();
    } else {
      return apply(std::forward<Args>(args)...);
    }
  }
}
```

这里经过模板及完美转发后来到`MakeNextFunctionList`的`()`重载符中。

```c++
// torch/include/torch/csrc/autograd/function.h
// `collect_next_edges`的实际函数体，经过上面拆包，这里进来的参数已经是单个variable了
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
  // 如果拿到的grad_fn为nullptr（如叶子结点的情况），我们就返回一个`grad_accumulator`
  // （AccumulateGrad）的edge，它会将所有入边的梯度求和并累积到变量的grad属性中。
  // 注意只有`requires_grad = True`的叶子节点才会有AccumulateGrad
  if (const auto& gradient = self.grad_fn()) {
    return Edge(gradient, self.output_nr());
  } else {
    return Edge(grad_accumulator(self), 0);
  }
}
```

此处我们会走下面的分支创建给mul的两个tensor参数self和other设置`grad_accumulator`

```c++
// torch/csrc/autograd/variable.cpp
std::shared_ptr<Node> grad_accumulator(const Variable& self) {
  auto autograd_meta = get_autograd_meta(self);
  // ...

  // intrusive_ptr是一种pytorch引用计数的智能指针
  c10::raw::intrusive_ptr::incref(self.unsafeGetTensorImpl());
  auto intrusive_from_this =
      c10::intrusive_ptr<at::TensorImpl>::reclaim(self.unsafeGetTensorImpl());
  result = std::make_shared<AccumulateGrad>(
      Variable(std::move(intrusive_from_this)));
  autograd_meta->grad_accumulator_ = result;
  return result;
}
```

我们这里也介绍一下`Edge`数据结构，其有两个重要数据成员，一个是`function`：指向一个`Node`的智能指针，一个是`input_nr`：用于标识这条边对应的输入在function节点中所有输入中的位置。例如，`function`节点有3个输入（即有三条edge指向这个节点）（这里的输入指的是grad_output，即上一个节点反向传进来的梯度），`input_nr`可能是0、1或2，假设其为1，则表示这条边是这个节点三条入边中的第二条。

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

`grad_accumulator`的Edge创建好后返回，然后解析第二个variable，同样过程后返回到`collect_next_edges`，一共收集到两条edge。

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
  if (_any_requires_grad) {
    grad_fn = std::shared_ptr<MulBackward0>(new MulBackward0(), deleteNode);
    grad_fn->set_next_edges(collect_next_edges( self, other ));
  }
```

随后调用`grad_fn->set_next_edges`给`grad_fn`（mulbackward）设置上这些收集到的edge。注意`update_topological_nr`，其原理我们在上面介绍`Node`数据成员已经介绍。

```c++
// torch/include/torch/csrc/autograd/function.h
struct TORCH_API Node : std::enable_shared_from_this<Node> {
  void set_next_edges(edge_list&& next_edges) {
    next_edges_ = std::move(next_edges);    // 仍然是移动语义并转移所有权，避免复制
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
    // 在设置edge时调用，被调用说明一定有parent，可直接设置变量`has_parent_`
    has_parent_ = true;
    return topological_nr_;
  }
}
```

`set_next_edges`后，我们调用`should_compute_output`判断edge是否有function（需要计算梯度），这里特别注意，`should_compute_output(0)`（对应self edge）时我们保存`grad_fn->other_`，`should_compute_output(1)`（对应other edge）时我们保存`grad_fn->self_`，这是因为mul backward本质是梯度互换。

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

### redispatch与**guard**

```c++
// torch/csrc/autograd/generated/VariableType_0.cpp
  auto _tmp = ([&]() {
    at::AutoDispatchBelowADInplaceOrView guard; 
    return at::redispatch::mul(ks & c10::after_autograd_keyset, self_, other_);
  })();
  auto result = std::move(_tmp);
```

随后我们调用匿名函数`tmp`来redispatch mul op，首先声明RAII的`at::AutoDispatchBelowADInplaceOrView guard`将`autograd_dispatch_keyset_with_ADInplaceOrView`（包含AutogradFunctionality、AutogradOther、AutogradNestedTensor、ADInplaceOrView四种dispatch key）加入到local thread的exclude list中，确保本次调用全链路都不会再dispatch到这四种key上。

我们举个例子，修改`tools/autograd/gen_variable_type.py`关闭guard后重新编译pytorch，执行`export TORCH_SHOW_DISPATCH_TRACE=1`打印dispatch trace，重新执行我们上面引入的代码，得到其中一段输出

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

而如果没有关闭guard，原生pytorch执行会得到输出

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

我们可以很清晰地看出，关闭guard后对于abs操作会重新dispatch到`AutogradCPU`和`ADInplaceOrView`上，产生了不必要的操作。

然后计算新的dispatch key注意此处ks为`DispatchKeySet(CPU, AutogradCPU)`，`c10::after_autograd_keyset`为`"DispatchKeySet(CPU, CUDA, HIP, XLA, ...`，两者相&后得到`DispatchKeySet(CPU)`，即我们下一个mul会调度到cpu的算子上，即`wrapper_CPU_mul_Tensor`

### Structured kernel与stub

`wrapper_CPU_mul_Tensor`本质是一个**structured kernel**，其关键方法为`.meta`和`.impl`

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

`structured_mul_out_functional`本质是继承**TensorIterator**，它的两个function`meta`和`impl`通过宏定义重写。

```c++
// aten/src/ATen/native/BinaryOps.cpp
TORCH_META_FUNC2(mul, Tensor) (
// 本质是void structured_mul_Tensor::meta (
  const Tensor& self, const Tensor& other
) {
  // maybe_get_output()拿到structured_mul_out_functional op默认的output_(undefined), 然后走`build_borrowing_binary_op`即tensor iterator infer使其defined
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

`meta`调用后创建好了**TensorIterator**，这也是个很重要的类，我们这里不具体展开，有兴趣的同学可以参考笔者之前的文章[How_pytorch_call_ops](../how_pytorch_call_op_3/index.en.md)

随后调用`impl`方法，这里`mul_stub`是一个`struct`，继承自**DispatchStub**并填写好了对应模板。

```c++
// aten/src/ATen/native/BinaryOps.cpp
DEFINE_DISPATCH(mul_stub);      // 宏展开：struct mul_stub mul_stub

TORCH_IMPL_FUNC(mul_out) (
  // 宏展开：void structured_mul_out::impl(
  const Tensor& self, const Tensor& other, const Tensor& result
) {
  // device_type()直接取tensor iterator里拿infer好的device
  // this是structured_mul_out实例
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

对于pytorch的各种stub，统一会走到`aten/src/ATen/native/DispatchStub.h`，根据设备选择合适的调用方法。

```c++
template <typename rT, typename T, typename... Args>
struct DispatchStub<rT (*)(Args...), T> {
  using FnPtr = rT (*) (Args...);

private:
  FnPtr get_call_ptr(DeviceType device_type) {
    return reinterpret_cast<FnPtr>(
      impl.get_call_ptr(device_type
      , reinterpret_cast<void*>(DEFAULT)
// 选择指令集，对于CPU，AVX2、AVX512这种都是intel的指令集，pytorch会自动选择更优化的指令集
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

通过`get_call_ptr`我们拿到了`mul_kernel`的指针，随后调用

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

这里传了`tensor iterator`和若干匿名函数给到`cpu_kernel_vec`向量化方法，然后拆分loop进行循环vec调用，这一部分内容在笔者之前的文章中有详细介绍，我们这里也不进行展开。

### `set_history`，计算图构建完成

我们这么多调用栈下来，外围还在`_tmp`方法中，执行完`_tmp`方法后，我们拿到了前向result

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
      // 将tensor封成一个variable list
      set_history(flatten_tensor_args( result ), grad_fn);
  }
  // ...
  return result;
}
```

随后我们调用`set_history`，这一步在构建计算图中很关键，它给`result`设置了`autograd_meta`存储`grad_fn`用于检索。

```c++
// torch/csrc/autograd/functions/utils.h
inline void set_history(
    at::Tensor& variable,
    const std::shared_ptr<Node>& grad_fn) {
  AT_ASSERT(grad_fn);
  if (variable.defined()) {
    // 检查tensor是可求导的dtype
    TORCH_INTERNAL_ASSERT(isDifferentiableType(variable.scalar_type()));
    // 将variable存储进grad_fn(Node)的input_metadata_，返回其索引
    // 我们这里得到0，即意味着variable是该node的第一个输入
    auto output_nr = grad_fn->add_input_metadata(variable);
    // {grad_fn, output_nr}构造了一个新edge
    // 然后用这个edge给variable设置 autograd_meta
    impl::set_gradient_edge(variable, {grad_fn, output_nr});
  } else {
    grad_fn->add_input_metadata(Node::undefined_input());
  }
}

// torch/csrc/autograd/variable.cpp
void set_gradient_edge(const Variable& self, Edge edge) {
  // 如果variale self没有定义autograd meta，那么这里进行设置
  auto* meta = materialize_autograd_meta(self);
  meta->grad_fn_ = std::move(edge.function);
  meta->output_nr_ = edge.input_nr;
  // ...
}
```

到此为止，乘法算子的前向运算就全部执行完成了，包括

1. `grad_fn`的`next_edges`指向前向输入`self`和`other`（`AccumulateGrad`）
2. `grad_fn`的`input_metadata_`存储了前向`result`
3. `grad_fn`绑定到前向`result` Tensor的**autograd_meta_**上

完成这些操作后，我们就可以调用`result.backward()`，即通过tensor的`autograd_meta_`找到`grad_fn`并进行反向运算了。
