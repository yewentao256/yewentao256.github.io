---
title: "Distributed Training Strategy Introduction"
date: 2024-04-13T19:20:48+08:00
categories: ["pytorch"]
summary: "这篇博客介绍了分布式并行训练策略，包括 Data Parallelism (DP), Zero Redundancy Optimizer (Zero), Pipeline Parallelism (PP) 和 Tensor Parallelism (TP)。"
---

## Summary

这篇博客介绍了分布式并行训练策略，包括 Data Parallelism (DP), Zero Redundancy Optimizer (Zero), Pipeline Parallelism (PP) 和 Tensor Parallelism (TP)。

---

![image](resources/distribution_strategy.png)

---

## Data Parallelism

最简单的并行策略，每个 GPU 上都持有一份完整模型，并行地使用不同的训练数据输入模型：前向计算时每个 GPU 对分配的数据独立执行并计算 loss，再独立地进行反向传播计算梯度，随后通过通信 reduce（avg） 梯度，再 broadcast 到所有GPU上，最后使用平均梯度更新模型权重。

由于 DP 每个 GPU 上都持有完整模型参数、优化器状态，所以**对显存并不友好**

此外，DP 在每个 GPU 前向反向计算完后有一次通信聚合梯度

Pytorch 原生支持了这一功能，使用`DataParallel（DP）`或`DistributedDataParallel（DDP）`即可快速上手。我们更建议使用 DDP 而不是 DP，这是因为：

Pytorch DDP：

1. 初始化后，每个 GPU 都有一份完整的模型。
2. 数据划分成多个 **micro batch**，每个 GPU 基于自己的一部分数据进行前向计算，得到 loss，反向传播计算梯度
3. 所有 GPU 通信，使用 **all-reduce** 让每个GPU得到加和后平均的梯度。
4. 每个 GPU 基于平均梯度更新自己的模型

Pytorch DP：

1. GPU0 加载一个 batch 的数据到内存中，然后划分并发送给其他 GPU
2. GPU0 复制并发送整个模型 给其他 GPU，**注意每个batch都需要执行一次这样的操作**
3. 其他 GPU 进行前向计算，得到的结果发送给 GPU0 计算 loss
4. loss 计算完成后，GPU0 将其广播给其他GPU，进行反向传播
5. 其他 GPU 再把反向传播得到的梯度发送给 GPU0 进行加和平均。
6. GPU0 更新自己的模型，在下一个 batch 再发送整个模型给其他 GPU

我们可以很清晰地看出：

- 一个 batch Pytorch DDP 只需 all-reduce 通信一次梯度，而DP需要进行五次数据交换（分发数据、模型复制、前向结果、loss广播、梯度聚合）。
- DP 引入了主节点，主节点负担过大导致整体利用率无法提升
- DP 使用 `threads` 通信（ GIL 限制）而 DDP 使用 `torch.distributed`通信。

此外，Pytorch 也开发了 **Fully Sharded Data Parallelism (FSDP)**，在 DDP 的基础上引入模型参数的分片，不过由于使用成本较高，业界现在更多使用下文介绍的 **Zero**

## Zero Redundancy Optimizer：Zero

Zero将Optimizer State（优化器状态），梯度和模型参数划分到各个数据并行进程中，消除了大量内存冗余占用，并通过一种动态的通信机制在各设备间共享必要的状态。

### Zero的三个阶段

![image](resources/DeepSpeed-Image-1.png)

在图中，`Ψ` 表示模型参数数，`K` 表示优化器特定的常量，`Nd` 指 GPU 数量，优化对应三个阶段。

**Stage1**：优化器状态分割，在图示情况下减少四倍内存占用，与 DP（data parallelism）相同的通信量。例如：Adam 有 FP32的 params、momentum和variance

**Stage2**：增加梯度分割（FP32 gradient），减少八倍内存占用，与 DP 相同通信量。

**Stage3**：增加模型参数分割（运行时FP 16），随 GPU 增加线性倍数减少内存占用（如64个 GPU 就减少64倍），大约增加50%通信量。

### 训练流程实例

我们开启 Zero 三个 stage，假设有四块卡进行训练（一次前向反向和参数更新）为例，说明训练流程.

![image](resources/zero-0.png)

首先进行前向，数据会划分为4份准备输入各个 GPU，GPU 上有 FP16 的参数和梯度（用于实际计算）和存储的 FP32 优化器状态（这部分会在 FP16 梯度计算完成后使用）。

![image](resources/zero-1.png)

准备开始进行前向运算，每个 GPU 持有一部份模型参数。

那对于 GPU 没有的参数，如何进行前向运算呢？答案是通信

![image](resources/zero-2.png)

如图所示，GPU0 广播了 M0 的前向参数（FP16 Params）给其他 GPU，然后每个 GPU 这部分参数，以自己的数据进行前向运算。前向运算结束后，其他GPU上这部分共享的参数会被删除。

注意前向运算得到的激活值只会保存一部分，以尽可能节约显存占用。被丢弃的激活值会在反向传播时通过**重计算 ，即checkpoint** 得到。

随后是 GPU1 广播 M1 的前向参数，以此类推，直到完成整个前向过程，计算出 loss（每个 GPU 基于自己数据集的loss）。

![image](resources/zero-3.png)

随后反向传播开始：

![image](resources/zero-4.png)

此时 M3 的模型参数都还在，缺失的激活值会通过之前保留的部分激活值**重计算**得到（如我们有十层，保留0、2、4、6、8，通过模型参数和保留的激活值 计算得到1、3、5、7、9的前向输出）

每个 GPU 根据 loss、激活值和模型参数 进行反向传播计算出自己的梯度后，其他 GPU 将梯度通信给 GPU3 ，GPU3 进行梯度累加并平均，这样 GPU3 上的梯度就是综合考虑所有数据的完整的梯度。

![image](resources/zero-5.png)

梯度累加计算完成后，其他 GPU 删除 M3 的参数、梯度以尽可能节约显存，同时所有 GPU 都删除了所有保存的前向激活值。反向传播继续进行：

![image](resources/zero-6.png)

我们现在需要计算 M2 的梯度，此时 GPU2 会 broadcast 这部分参数给其他 GPU，随后同样通过参数 + 保存的部分激活值重计算 + loss 得到 M2 的梯度。

以此类推，直到所有梯度都计算完成

![image](resources/zero-7.png)

此时，所有GPU都有自己的梯度（来自自所有数据集）。

随后优化器利用这些梯度（FP16）更新自己的模型参数（FP32）（运算会使用FP32梯度、FP32 momentum、FP32 variance和 FP32 Params）

![image](resources/zero-8.png)

最后，这些 FP32 的模型参数会被 **cast** 到 FP16，用于下一个iter的模型训练（混合精度训练）。

## Pipeline Parallelism

PP 的核心思想是把一个模型的不同层划分到不同 GPU 上，每个 GPU 只需要负责一部分模型

例如，一个六层的模型划分到两个GPU上：

```bash
=============  =============
|  0 | 1 | 2   |  3 | 4 | 5 
=============  =============
     gpu0           gpu1
```

当一批数据输入时，先进入 gpu0 经由 layer0~2 计算，随后`.to()`到 gpu1 上经由 layer3~5 计算

通常 **labels** 会被送到模型最后一层所在的 GPU 上直接计算 loss，然后反向传播、更新模型参数。

这种设计有什么问题呢？当一批数据在 GPU1 计算的时候，GPU0 处于空闲状态！因此，我们引入了流水线并行，尽可能减少 GPU 空闲的时间。

一种实现如 **GPipe**：

![image](resources/pp.png)

其核心思想是将输入数据划分为多个 **micro-batch**，让 GPU 可以并行处理不同的 micro-batch 以尽可能提高利用率，减少 GPU 空闲时间（气泡，**bubble**）。在上面的图中，4个 GPU 以 **`chunk=4`**（即在一个 pipeline stage同时中处理 4 批次的数据）训练，以 GPU0 为例，从 F0~F3 进行四个 forward 计算然后转交给下一个 GPU ，随后执行 B3~B0 四个 backward 计算。

如果在训练中同时开启了 DP 和 PP，我们有`global_batch_size = micro_batch_size * dp_size * chunk`：如 `global_batch_size=512`、`DP=4`、`chunk=4`，则最终得到 `micro_batch_size=32`，即每个 `micro-bacth` 有32条数据。

另外一种 PP 并行策略的实现如 **Interleaved Pipeline**：

![image](resources/parallelism-sagemaker-interleaved-pipeline.png)

如图，通过设计流水并行策略，我们可以减少 bubble 并提高 GPU 利用率。

值得指出的是，PP 只需要在相邻模型层传递计算好的批次数据结果，且我们可以通过重叠计算和通信来进一步减少开销，因此 PP 所需通信量是最小的，**在实际训练中一般会把 PP 层跨节点使用，把节点内高带宽通信让给 TP**

## Tensor Parallel（Model Parallel）

Tensor Parallelism 有时也称为 Model Parallel，属于计算层面的优化，核心思想是每个 GPU 只处理 tensor 的一部分（与PP 处理模型不同层不同，这里 TP 将模型的一层拆分），只在需要时聚合在一起。

例如：`Y = GELU(XA)`，这里 Y 是输出，X 是输入，A 是模型权重，GELU 是非线性激活

![image](resources/parallelism-tp-parallel_gemm.png)

我们可以把 A1、A2、A3 放到不同的 GPU 上计算以节约显存。

TP 有两种切分方式，一种是行并行（**row Parallelism**），一种是列并行（**column Parallelism**）：

![image](resources/parallelism-tp-parallel_mode.png)

column Parallelism 不处理输入数据（`2*4`），按 column 划分模型参数（`4*1`），最后结果（`2*1`）cat起来得到结果（`2*2`）

row Parallelism 将输入数据按 column 划分（`2*2`），将模型参数按 row 划分（`2*2`），点积后再相加得到结果（`2*2`）。

我们可以在模型设计上直接利用 TP，如 多层感知机 MLP（**Multi-Layer Perceptron**）：

![image](resources/parallelism-tp-parallel_shard_processing.png)

再比如 MHA（**Multi Head Attention**），由于设计上 MHA 就是并行执行的，引入 TP 更加简单：

![image](resources/parallelism-tp-parallel_self_attention.png)

**注意**：由于切分计算后往往要 gather 通信，TP 总通信量较大，不建议跨节点使用 TP（`tp_size < GPUs per node`）。此外，TP 切分后，每个 GPU 需要存储的前向激活值也对应切分，可以较好地降低显存占用。

## 实例介绍

### DP + PP

![image](resources/parallelism-zero-dp-pp.png)

如图所示，我们有 4 个 GPU，`2 DP + 2 PP`。`GPU[0, 2]`和`GPU[1, 3]` 构成 PP 通信组，`GPU[0, 1]`构成 DP 通信组。

对 DP 而言，只能看见 GPU0 和 GPU1 两个实例，“看不见” GPU2 和 GPU3。在训练时，DP 会将数据划分给两个看得见的GPU，然后 GPU0 和 GPU1 利用 PP “偷偷”把任务转交一部分给 GPU2 和 GPU3。

### Zero（DP） + PP + TP

![image](resources/Blog_DeepSpeed3_Figure2_highres.png)

如图所示，我们有 八个节点、32 个 GPU，每个节点四个GPU。我们将 TP 通信组放在同一个节点上，然后 PP = 4，DP = 2，三维并行训练。

Zero可以理解为一种高度扩展的DP，可以与PP和TP共同使用，但一般只适用于 Zero Stage1

这是因为如果引入 Zero Stage2 梯度分割的话，每个 micro batch 执行都需要引入一次通信聚合梯度，而使用 PP 天然就会使用更多的micro batch，因此这部分通信开销会相对更大。此外，引入 PP 后模型分层，梯度本身大小也变为了原来的`1 / PP_size`，节约显存效果也没那么明显。因此我们一般不会让 Zero Stage2 与 PP 共同使用。

此外，使用 Zero 后，**Zero-Offload** 可以让我们把一部分优化器状态存储到CPU上，以进一步节约显存。

## Referrence

- Microsoft Research Blog: "[Zero DeepSpeed: New System Optimizations Enable Training Models with Over 100 Billion Parameters](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)"
- Hugging Face Transformers Documentation: "[Parallelism Techniques](https://huggingface.co/transformers/v4.11.3/parallelism.html?from_wecom=1)"
- Google AI Blog: "[Introducing GPipe: An Open Source Library for Efficiently Training Large-scale Neural Network Models](https://ai.googleblog.com/2019/03/introducing-gpipe-open-source-library.html)"
- AWS SageMaker Documentation: "[Model Parallel Core Features](https://docs.aws.amazon.com/sagemaker/latest/dg/model-parallel-core-features.html)"
- Arxiv Preprint: "[An Analysis of Model Parallelism in Training Large-scale Neural Networks](https://arxiv.org/abs/2104.04473)"
- DeepSpeed Tutorial: "[Pipeline Parallelism Tutorial](https://www.deepspeed.ai/tutorials/pipeline/)"
- Microsoft Research Blog: "[DeepSpeed: Extreme-Scale Model Training for Everyone](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)"
- Hugging Face Documentation: "[Training Large Models: A Deep Dive on GPU Performance](https://huggingface.co/docs/transformers/perf_train_gpu_many)"
