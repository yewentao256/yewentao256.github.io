---
title: "2025 Technical Notes（3）"
date: 2025-05-24T08:50:56+08:00
categories: ["technical_notes"]
summary: "2025年技术积累笔记（三）"
---

## 基础知识

**Sequence Parallel** 线性减少logits（最终词表概率分布）的memory footprint，所以 Sequence Parallel 也是有用的。

**NVLink**是一种高速点对点传输技术，GPU间直接通信（共享显存池），比PCIe带宽高得多；**NVswitch**是一种专用的NVlink交换ASIC，实现多GPU全连接非阻塞拓扑。将多个Nvlink GPU连接到一个fabric网络中解决了通信效率问题。

---

PPO（Proximal Policy Optimization） 本质是一种RL算法， 训练Agent在每个状态下根据策略做出最优动作，它主要评估：

1. 状态值函数（State Value Function）：某一状态下当前策略的预期累计奖励（return value）
2. 优势函数（advantage function）：采取某一动作相较于其他动作**平均水平**的优势
3. 策略分布（Policy Distribution）：输出给定状态下各个动作的概率

---

Pytorch Reduce Scatter算子:

- 1D tensor：

```py
dst = torch.zeros(5, dtype=torch.float32)
index = torch.tensor([0, 1, 1, 2, 3])
src = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
result_pytorch = dst.scatter_add(0, index, src)
print(result_pytorch)   # ([1., 5., 4., 5., 0.]) 很好理解，按位置加数即可
```

- 2D tensor （dim=0）:

```py
# 元素加到dst[target_row, same_col]中
dst = torch.zeros(3, 4, dtype=torch.float32)
index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 1]])
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
result_pytorch = dst.scatter_add(0, index, src)
print(result_pytorch)
# tensor([[1., 0., 7., 4.],
#        [5., 2., 0., 8.],
#       [0., 6., 3., 0.]])
```

- 2D tensor （dim=1）:

```py
# 元素加到dst[same_row, target_col]中
dst = torch.zeros(3, 4, dtype=torch.float32)
index = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
src = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.float32)
result_pytorch = dst.scatter_add(1, index, src)
tensor([[1., 2., 3., 4.],
        [8., 7., 6., 5.],
        [0., 0., 0., 0.]])
```

scatter add的一个很有意义的用法是batched bin count，在nlp中可以用于统计词频
例如

```py
import torch

def batched_bincount(x, dim, vocab_size):
    target  = torch.zeros(x.shape[0], vocab_size,
                          dtype=x.dtype, device=x.device)
    values  = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target

x = torch.tensor([[0, 1, 2, 3],      # batch-0
                  [2, 2, 0, 3]])     # batch-1
vocab_size = 5

hist = batched_bincount(x, dim=1, vocab_size=vocab_size)
print(hist)
# tensor([[1, 1, 1, 1, 0], [1, 0, 2, 1, 0]])
```

## TVM

Operator level optimization：很重要的一点是把计算（比如乘法）和schedule（比如tile looping）分离。重点是schedule优化

简单的schedule优化就能做到60%的mkl性能，但要达到100%还需要时间。所以TVM也支持ML propose schedule（autoTVM）。基于template fill和自动生成。

还有图级别的优化，有两层IR，一层是Relay——用于表示nn计算图。一层是TIR（tensor IR），这是算子内部贴近硬件的IR。一个典型的图优化就是算子融合。

summary：把model转成relay，应用图优化（特别是fused layer）。然后对每个fused layer，定义搜索空间然后执行autoTVM生成具体硬件的优化二进制，然后测试

## NLP

### 基础常识

prefill：处理所有input token，计算qkv和attention分数，缓存KV。（计算密集型）

decode：一次生成一个token，直到满足停止条件。（memory 密集型）

现在一般强调prefill和decode分离来更好利用资源，或者分块prefill等方式提高效率

---

Speculative Decoding（推测性编码）是一种加速LLM infer的技术，提高推理速度并保持质量。

核心思想：用一个较小较快的draft模型来推测可能的token序列，然后用更大更准确的目标模型来验证，如果预测正确，则可并行生成多个token而不是一次一个从而节省时间

例如：上下文“人工智能”，小模型预测“正在改变我们的生活”。大模型正确答案是“正在改变人类的生活”，验证“正在改变人类”，那么一次接受“正在改变”的四个token，“我们”被丢弃。（从错误的地方开始全部丢弃）

推测性编码可以大幅减少计算量，提高性能数倍（并行接受）

---

BERT: CLS放在最开始的位置，聚合整个序列的信息，特别用于分类任务。在分类任务的微调过程中，CLS通常用于表示整个序列的表示向量。SEP用于分隔不同句子或文本段落。next sentence prediction是一个分类任务，主要参考CLS

---

NLL loss 和cross entropy的关系：
如果输入是one-hot编码，那么cross entropy退化为nll loss（只有一项在求和）

`NLL=−logpθ​(w∗∣context)`。我们希望预测到的`w*`就是原本mask掉的词，`p(w*)`越高（越接近1），loss越接近-0，加个负号越接近+0

---

BLEU score: 用于机器翻译评估，实践效果和人类评估分数有较高相关度

1. 检查n-gram的重合情况，越长片段同时匹配到，说明译文效果好

2. 对于每个n-gram，计算`分数 = 命中片段数 / 机器译文片段总数`。（惩罚过长译文）

3. BP：brevity penalty 简洁惩罚（惩罚过短译文）

    `BP = 1` if 译文较长。  `BP = e^(1 - ref/out)` if 译文较短

4. `BELU = BP * e^(四个ngram取ln加权平均)`

### Transformer的时间空间复杂度分析

矩阵乘的时间空间复杂度：

假设`A（M，N）` @ `B（N，P）`得到`C（M，P）`。那么时间复杂度为`O（M*N*P）`（对于C的每个元素，都要进行N个元素的乘积和求和）。空间复杂度为`O（1）`（如果不计算输入，输出矩阵大小，没有中间变量）；或者可以看做`O（M*N+N*P+M*P）`

现在我们看attention

训练和推理的复杂度并不相同，我们先看蓄念：

输入`（N，D）`

1. QKV projection计算：`QKV = X@Wqkv（D，D）` 得到`（N，D）`。时间复杂度`O（N*D^2）`
2. 注意力得分计算：`Q（N，D）@K.T（D,N）`，得到结果`(N,N)`，时间复杂度`O（N^2*D）`
3. 乘以v计算：权重`（N,N）@V（N,D）`得到结果（N，D），时间复杂度`O（N^2*D）`

所以综合下来时间复杂度为`O（N^2 * D）`，有时候为方便简化为`O（N^2）`

空间复杂度就考虑额外的QKV和activation的存储，即`O（N^2+3*ND）`，一般简化为`O（N^2）`

注意上面是训练时候的时间空间复杂度。推理的时候，我们是逐个token生成（假设没有context，直接生成）。

- 第一个token：`O（D^2）`（QKV投影），然后把KV cache了
- 第二个token：`O（D^2 + 1*D）`（新的QKV投影与一个历史token做attention，`（1,D）@（D，1）`，复杂度O（D））
- 第三个token：`O（D^2 + 2*D）`
- 以此类推，第N个token：`O（D^2 + (N-1)D）`

总的时间复杂度：`O（ND^2 + D(1+2+..+N-1)）= O(ND^2+DN(N-1)/2`） 近似于`O（N^2 * D）`再近似于`O（N^2）`

关于memory复杂度，我们从上面过程中看到每次都多cache一份`KV（1，D）`，因此线性增长`（O（N））`

### Loss计算

我们使用cross entropy loss计算，例如："I love deep learning"

假设词汇表`V`：`{"I": 0, "love": 1, "deep": 2, "learning": 3, "and": 4}`

输入序列`input_t = [0, 1, 2]`（"I love deep"）；目标序列`target_t = [1, 2, 3]`（"love deep learning"）

模型前向传播生成logits（对于每一个词表中的每个词的可能性）：

- 对于"I"预测"love"的logits: [0.1, 2.5, 0.8, 0.3, 0.2]
- 对于"I love"预测"deep"的logits: [0.2, 0.3, 2.1, 0.5, 0.1]
- 对于"I love deep"预测"learning"的logits: [0.1, 0.2, 0.3, 1.9, 0.4]

然后过softmax得到

- [0.05, 0.55, 0.25, 0.10, 0.05]  # "I"后的预测概率
- [0.08, 0.09, 0.55, 0.18, 0.10]  # "love"后的预测概率
- [0.07, 0.08, 0.10, 0.60, 0.15]  # "deep"后的预测概率

最终 `loss = （-log(0.55) + -log(0.55) +  -log(0.60)  / 3） = (0.60 + 0.60 + 0.51) / 3 = 0.57`

### Expert Parallel

Expert Parallel用于MOE架构，将不同专家网络分布到多个设备上。

有一个路由网络决定token给哪个专家处理，与TP不同的是，它没有分割单个专家的权重，而是将完整的专家分布到不同设备上（基础EP）。

如果GPU高于专家数量，则会专家切片（expert-slicing），GPU间垂直/水平切割专家

专家处理完各自分配的任务后，通过all-to-all通信组合结果

### 并行推理如何padding？

1. pad to max：一个batch 全部pad到最长长度——浪费资源
2. bucket：长度相似请求分桶，类似大小一次请求——算法复杂，有时候等待时间长
3. Token-Level Scheduling：以token为粒度刷新batch，利用paged attention的kv cache，能快速找到context。注意：并非是对batch里的每个sequence单独调用infer得到下一个token（这样就没有batch的意义了），而是把batch里每个sequence的最后一个token 一起做infer（shape `[B,1]`）

注意：除了最后一个token，还有KV cache的 block location也一起送入了kernel

补充：**Selective Batching** by ORCA

把一个batch的sequence拼成一个大sequence，然后分类：

- attention路径：对于每个序列单独attention
- 非attention路径：一次做完linear相关操作

然后两个路径拼到一起，进入下一层。

## 系统设计例子

Global System Design

Key Question: From a global system perspective i.e. across multiple instances of the inference serving engine, how should a unified orchestrate optimally route and load balance across LoRA requests to minimize both TTFT and TPOT metrics?

一、调度目标

1. 高命中率：请求发到已有实例的结点
2. 负载均衡：热门Lora不会打到一个单机，冷门Lora也不长时间排队
3. 扩容缩容的时候仅迁移最少的Lora
4. 简单可落地：实现简单效果好

二、核心思路

两层 路由 + 两个后台线程

1、**Consistent Hash with Bounded-Load**  (CH-BL)

在介绍这个之前，我们先介绍一下consistent hash（一致性hash）：可以理解为每个服务器结点先算一次hash，就在环形上注册了旗子。然后请求来的时候，请求也hash一次，得到一个地址，然后顺时针移动，在遇到的第一个旗子处结算。

如果增加结点，那么只有一小段弧的结点会受影响。例如A在20度，C在22度，D在24度，现在加一个B到21度，那么ACD本身都不变，只有原本要落在20~22度的请求会一分为二。我们一般会让N很大（虚拟结点）实现每个弧受到的影响最小。

这样的好处是：如果增加或者删除结点，那么只需要修改一小部分请求

Consistent Hash with Bounded-Load（CH-BL）又做了什么呢？它在少迁移的基础上，加了一个配额的概念，即每个结点可以乘载键的量为（比如`1+ε * 全局平均`），这样请求移动找结点的时候，如果发现当前结点已经爆满了，就继续往前走圆弧找下一个结点。数学上论文证明了只要多留一点ε，一般多走一两个结点即可找到，转圈代价仍然是`O（1）`

介绍结束，我们回来设计。

我们用(adapter-id, base-model-id)做键，加CH-BL做负载均衡。为什么是base-model-id也要加入键呢，因为LORA只有和自己具体的model绑定才有意义。

2、**Power-of-Two Choices** (P2C)

我们用CH-BL，很自然地会在实践中引入replica的概念，继续找下一个结点直到满足k分replica，有多个replica的话如何选择呢？

简单地从拥有replica的集合里随机找两个，计算分数`score = queue_len × α + gpu_util × β`
分数低的队列少且gpu使用率低，用这台来做更快。

为什么P2C？`O（1）`快速实现，实践效果也好

3、**后台线程1：动态复制/回收**

每隔30s检查一下各LORA的hot score，如果升温，那么选额外`k_new - k_old`台空闲的机器结点执行prefetch然后加到replica table里。如果降温那么标记为draining，队列清空后自动卸载

4、**后台线程2：版本一致性**

如果发现HUB上LORA的revision hash变化，那么自动将新版本推送到持有旧hash的结点。

实例收到后把旧 ΔW 标记 stale，在无排队或 GPU 空闲片段时热替换，Router 在全部副本完成替换前仍接受旧 hash 请求，保证 0 停机。
