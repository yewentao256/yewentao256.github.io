---
title: "2025 一月 Technical Notes"
date: 2025-01-31T14:02:12+08:00
categories: ["technical_notes"]
summary: "2025年一月技术积累笔记"
---

## Robotic

### RL

Reinforcement Learning：强化学习

Proximal Policy Optimization (PPO) 是一种深度强化学习算法（基于策略优化）

1、基于值的算法

评估每个动作的好坏，选择最好的动作改进。例如迷宫，当前往左一步值高，往右是陷阱值低。如Q-Learning，记录每个位置某个动作的好坏，选择最高得分的动作。

简单，适合小型离散问题。但贪心可能错过最优解，且连续动作（如机器人关节角度）难以解决

注意：这不是imitate learning，它不需要示范动作，而是自己与环境交互来评估动作好坏

2、基于策略的算法（policy-based methods）：

指定一个policy，agent自己执行action来获得reward，使长期return最大化。对于机器人场景，纯RL成本和风险非常高

如，“看到陷阱我倾向绕开”，“遇到宝箱我倾向打开”

适合连续动作，更灵活；但学习效率较低，不断试错，同时试错波动很大（方差很大）。

PPO就是这种基于策略的方法

两者结合：Actor-Critic

critic评估值，判断当前动作好不好；actor负责生成动作并改进策略

Pretraining in RL：经典RL中，往往是从初始化策略开始，但这样非常耗时，需要大量交互甚至有安全风险（机器人场景），所以引入了pretrain。即先利用离线数据（offline）或者无监督方式预训练，来加速RL的拟合

### IL

IL（Imitation Learning）：模仿学习（学习示范）

根据专家示范数据，直接模仿专家选择。用于比如机器臂操作，服务机器人，自动驾驶等。对于机器人场景，纯IL泛化能力较差，无法举一反三。

两者结合：先模仿后RL，或者一边RL一边给专家意见等。

Behavior Cloning是模仿学习的一个子类，核心是监督学习。给定一个状态，预测一个动作。就和传统NN的训练方式类似，但data非常重要

### RLHF

这个不仅仅是机器人，也是DL的概念

RLHF（Reinforcement Learning from Human Feedback）

1、Supervised Pretraining实现model

2、收集人类反馈（data）

3、训练奖励模型（reward model）

这个奖励模型输入是LLM输出，输出是人类打分

4、RL Fine Tune

PPO (Proximal Policy Optimization) 等算法，利用奖励模型作为奖励函数，来微调LLM

5、持续收集数据和迭代

## 线性代数

矩阵特征值：矩阵在特征向量方向上对该向量进行拉伸或压缩的倍数。

例如：A = [[2,0] [0,3]]，特征向量为[1, 0]和[0,1]，对于x轴，特征值为2（放大两倍），对于y轴，特征值为3（放大3倍）

求根公式：`discriminant = b^2 - 4ac`

二元一次方程 x = (-b +- discriminant ^0.5) / 2a（初中知识，怎么来的？直接将二元一次方程手解，凑平方移项得到）

假设A = `[[a,b],[c,d]]`

矩阵行列式：`det(A) = ad - bc`；A的trace：`tr(A) = a + d`

矩阵的特征值求解方式：`det(A-λI) = 0`

即 det([[a - λ, b], [c, d-λ]]) = λ^2 - (a+d) λ  + ad-bc = 0

## 分布式训练

### Strategy

传统DP：

- 优点：简单易于实现；社区支持
- 缺点：每个device maintain一份模型权重和优化器状态

所以why Zero（only stage1 — optimizer state）

- 优点：对上层来说感官就像DP一样，仍然是把数据切成N份给worker，但此时底层的device optimizer state已经被切分了
- 缺点：需要框架支持（deep speed）

tensor parallelism：

- 优点：单层算子吞吐量提升
- 缺点：需要对模型做一些修改（但multi-head Attention天然分层）；通信量大

PP：

- 优点：节约显存
- 缺点：处理bubble，难以debug

DeepSpeed使用例子：

```python
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)

# torch.distributed.init_process_group(...) -> deepspeed.init_distributed()
```

默认 NCCL backend

之后就像平常训练一样

```py
for step, batch in enumerate(data_loader):
    #forward() method
    loss = model_engine(batch)
    #runs backpropagation
    model_engine.backward(loss)
    #weight update
    model_engine.step()
```

自动做了并行。如果多节点，默认用Open MPI：A High Performance Message Passing Library

### RDMA

RDMA (Remote Direct Memory Access)：直接内存读写（网络层）

- InfiniBand (IB)
- RoCE（RDMA over Converged Ethernet）

性能极高，RDMA绕过了传统的TCP/IP协议栈，减少了CPU干预。适用于分布式计算

NVlink常用于单台服务器内部，GPU、GPU高速互联。多节点间常用RDMA（GPUdirect RDMA）

## Deep Learning

### 基础常识

剪枝 (Pruning)：减少冗余权重，甚至整个层丢弃。

- 基于权重大小，即权重值很小认为不重要，设定阈值剪去
- 基于梯度大小，权重梯度很小表示它对损失函数影响较小，可能不会显著影响训练
- 基于activation大小，激活值很小表示对输出影响较小。

剪枝会将对应值设为0，然后矩阵变稀疏可以专门优化减少存储和计算开销。

剪枝后模型性能下降，需要fine-tune来恢复模型性能。这个过程可能迭代多次

蒸馏 Distillation（教师 软标签 给 学生（小模型））

ONNX Runtime：多后端支持（CPU、GPU、TensorRT）等，将不同模型统一到ONNX处理

RCNN传统算法（Selective Search）候选框然后目标检测；Fast-RCNN 全图扫描出特征图然后再扫出候选框；Faster-RCNN直接用NN来候选框

3D 检测（3D Object Detection）比如摄像头和lidar的融合；nerf并不做3d边界框的检测，之后是3d重建

Finetune 会冻结模型某些层，然后用新数据训练，只做小更改。

CrossEntropyLoss: `Loss = -∑ yi * log(y_softmax)`，这里yi是正确标签的one hot，y_softmax是过了softmax的值。如：`y=[0,1,0]`，y_softmax=`[0.1, 0.8, 0.1]`，然后`loss = -[0 + log 0.8 + 0] = 0.223`

BCE: `Loss = - [yi * log(y_p) + (1-yi) * log(1-y_p)]`，y_p是如过了sigmoid的正值

低秩分解（Low-Rank Decomposition）：通过奇异值分解或者其他矩阵分解方法，将weights分解为多个较小矩阵来近似原始矩阵

out算子快是因为：

- 减少了一次运行时malloc
- 减少了一次extra copy（主要）

malloc由于pytorch有一套可复用的显存管理，实际开销不大，比如100s的调用，可能占用就2~3s malloc，但如果调了out算子这malloc可能就可以提前被handle（比如等待通信时分配）

注意：cuda malloc开销是很大的。

### 量化（TensorRT）

Post-Training Quantization（PTQ）：先浮点训练完成后，离线量化。

Quantization-Aware Training（QAT）：训练中就量化

PTQ流程：

1. 导出model为ONNX格式（注意：需要确保所有算子都支持，否则需要手动写插件，比如flash attention、RMSnorm这种LLM算子还没及时支持的）

2. 构建tensorRT引擎（engine），序列化为.plan文件，或者直接加载到runtime里推理

为什么需要校准？因为int8的数值范围有限（-128~127），而浮点数通常到几千几万所以我们需要校准放缩范围。例如：`int8_value = round(float_value / scale) + zero_point`

这个scale需要尽量覆盖大部分实际出现的activation以使误差最小。

校准的方法：

1. MinMax校准：直接统计最小最大值作为量化边界，简单但对异常值敏感，如果极端值则可能让量化区间太大。
2. 熵校准：Entropy / KL 散度校准，通过对比浮点分布与量化后分布的差异（如使用KL散度），搜索一个最佳截断阈值，让量化前后与原始分布尽可能接近。（KL散度：Kullback-Leibler Divergence，相对熵，用于衡量两个概率分布之间差异的一种度量。）
3. 百分比策略，直接舍弃极端值如99.9%分位数作为量化边界然后minMax

具体执行：

1. 准备校准数据：一批具有代表性的样本数据，覆盖infer时的输入分布。
2. 构建校准器（Calibrator）
3. 构建engine并用校准器校准数据，获得各激活的分布

QAT和混合精度训练不是一种技术。

QAT是在训练时就引入量化误差的模拟，让它适应低精度（如int8）的计算。即在前向时模拟量化操作，反向时通过一些特殊策略如伪量化（fake quantization）和直通估计（Straight-Through Estimator, STE）来处理不可微分的问题（两个一起用）

伪量化：

1. 量化：`q = clip（round（x/S）+ zero_point,-128,127）` 但此时还是float
2. 反量化`x_new = (q-zero_point) * S`

注意：伪量化是一个算子模块，算子/层后进行的

- activation伪量化：在层（比如CNN）后插入一个伪量化模块
- 权重伪量化：前向计算时模拟低精度表示，反向传播时传给真实权重

STE：round和clip等函数直接假设导数为1传递下去

例如：

原始激活：`x=[0.45,−1.23,2.99,0.02]` 然后进伪量化层变成`[5,-12,30,0]`。反量化后的激活：`x=[0.5,−1.2, 3.0, 0.0]`，这样就模拟了量化带来的精度误差。注意这些全部都在伪量化模块里面做。

### 常见问题

为什么会梯度消失？

- 层数过深：合理设计网络  +  残差连接
- 激活函数吞梯度（RELU 小于0或sigmoid输入值较大或较小）：
  - RELU -> Leaky RELU等；
  - 使用Normalization（BN等）
  - 合适的初始化方法
    - Xavier：用于sigmoid / tanh，通过神经元个数调整初始值范围，让输入输出尽量均衡
    - He：Relu，因为RELU会截断一部分输入，所以He初始化会更大一点

没有标签怎么办？（例如判断用户上传的video是否与restaurant相关）

- 简单规则提取作为弱标签（restaurant、menu、food等标题）
- 特征提取：
  - 预训练分类模型处理视频frame：如果有食物、菜单、餐桌等也作为弱标签
  - 多模态模型CLIP等对视频帧提取，然后restaurant scene查看分数
- 无监督方法（如聚类），用帧内容、音频识别等方式提取特征然后分组，手动给类别赋标签

然后有初步数据训练出初步模型，之后就可以迭代Pseudo-label了，不断喂新数据查看效果等

### logSoftmax的前向反向

```py
import numpy as np
class LogSoftmax:

  def __call__(self, x: np.ndarray) -> np.ndarray:
    assert len(x.shape) == 2, "x is shape (batch_size, in_dim)"
    # log_softmax = x - log(∑exp(x)) = x - (m + log(∑exp(x-m))) where m = max(x, dim)
    # here we default use dim=1 (in_dim)
    max_val = np.max(x, axis=1, keepdims=True)   # (batch_size, 1)
    self.exp = np.exp(x - max_val)     # (batch_size, in_dim)
    self.sum_exp = np.sum(self.exp, axis = 1,  keepdims=True)
    log_value = np.log(self.sum_exp)   # (batch_size, 1)
    return x - (max_val + log_value)

  def bprop(self, x: np.ndarray, dedy: np.ndarray) -> np.ndarray:
    """
    e: error (or loss)
    y: fprop output
    dedy: de / dy (derivative)
    x: input to this layer
    """
    # Normally, this will be handled by auto differentiation
    # f(x) = x - log(∑exp(x))   # don't consider m here, since it is the same
    # f'(x)_1 = 1 - (log(∑exp(x)))' = 1 - 1 / ∑exp(x) * (∑exp(x))' = 1 - exp(x) / ∑exp(x) = 1 - softmax(x)  (i=j)
    # f'(x)_0 = 0 - softmax(x-m) = -softmax(x) (i!=j)
    # f'(x) = f'(x)_1 + f'(x)_0 
    #       = dedy * f'(x)_1 + ∑dedy * f'(x)_0
    #       = dedy - softmax(x)(i=j) - ∑dedy * softmax(x) (i!=j)
    #       = dedy - ∑dedy * softmax(x)

    # why "∑" in ∑dedy * f'(x)_0? Because i!=j should consider all of the components
    # Let's use an example of scalar
    # x=(x1, x2) -> y= (y1, y2) = (x1- log(S), x2-log(S)), S = e^x1 + e^x2
    # y'|x1 = dedy1 * (1 - softmax(x1)) + dedy2 * (0 - softmax(x1))) = dedy1 - dedy1 * softmax(x1) - dedy2 * softmax(x1) = dedy1 - ∑dedy * softmax(x1)
    # y'|x2 = dedy1 * (0 - softmax(x2) + dedy2 * (1 - softmax(x2))) =  dedy2 - ∑dedy * softmax(x2)
    # y'|x = dedy - ∑dedy * softmax(x)
    softmax = self.exp / self.sum_exp
    sum_dedy = np.sum(dedy, axis=1, keepdims=True)
    dedx = dedy - softmax * sum_dedy
    return dedx
```

## CV

### Diffusion

#### DDPM（Denoising Diffusion Probabilistic Models）

2020奠基之作，标准前向逐渐添加噪声，反向NN学习去噪，使用MSE loss

前向: `Xt = 根号(1-βt) X_(t-1) + 根号(βt) ϵ`

反向：X_(t-1) 粗略等于  X_t - 根号(βt)ϵ0  + 随机补偿 根号(βt) z

注：通常噪声量由一个时间调度参数βt控制，DDPM中是一个线性增长的βt。

问题：反向采样数百步太慢

#### DDIM（Denoising Diffusion Implicit Models）

原团队2021年的改进

- 采用确定性反向采样，而不是依赖随机（即把随机删掉了）
- 跳步采样（如1000步减少到50~100步），大幅加速且保证了质量

如何确定性反向采样？引入了一个隐式 ODE（常微分方程，ordinary differential equation）框架，使得扩散过程在确定性路径上运行。

#### Latent Diffusion Models (LDM)

进一步优化，将diffusion process从高维空间移入低维空间（latent space）

1. 使用预训练的VAE（Variational Autoencoder）将高维压缩为低维表示
2. 在latent space中执行diffusion降低计算成本。
3. 反向denoising的时候，可以+一个conditioning（semantic map、text、image）等embedding
4. 最后过一层decoder还原为图像

快，但依赖VAE的质量，生成细节可能有所损失

注：VAE——训练一个nn，将输入encode到latent space，然后decoder还原，让输入和输出尽可能接近

#### Stable Diffusion

- 基于latent diffusion model的具体实现，by Stability AI
- 使用CLIP（Contrastive Language–Image Pre-training）提取文本条件，控制生成内容——>强大多模态能力

CLIP（by openai）即image和caption训练到同一embedding的那个方式得到的预训练模型

## NLP

### 基础的常识

1、GPT（Generative Pre-trained Transformer）

从左到右依次预测下一个token的概率分布，仅适用transformer的decoder部分，不会看到未来的词，所以生成任务表现出色。

预训练时给定出现的文本预测下一个词（Language Modeling）

2、Bert（Bidirectional Encoder Representations from Transformers）

双向编码（可以在attention时关注到右侧未来的词），仅仅使用transformer的encoder部分，所以适合理解类任务。

预训练任务有MLM（Masked Language Modeling）即随机mask部分词，根据上下文预测。和NSP（Next Sentence Prediction）让模型判断两个句子是否连贯

3、T5（Text-to-Text Transfer Transformer）

完整保留了transformer的encoder和decoder。所有任务都被转换为“输入文本-> 输出文本”。所以比较适合翻译等

预训练任务为Span Corruption（遮盖片段），随机选择文本片段，让模型decoder输出时恢复被遮盖的文本

### LLM 算子

1、Fused Multi-Head Attention 算子

把`Softmax(Q * K^T / 根号d ) * V` 这些操作进行fuse单独写kernel实现

注意：Q、K、V的shape为`（N，d）`。N表示序列长度，d表示（hidden dimension），如果multi-head就是hidden size / head_num

2、**Flash attention**

常规attention会存储`N*N`中间activations，让内存使用量`O（N^2）`增长
所以引入了flash attention优化，即通过块（chunk）来完成softmax和乘法运算

做法：

- 分块，将QKV对应划分为较小的块（如`block_size = 128`）
- 逐块计算，局部的matrix mul、softmax（max-shift）得到 局部的attention分数和softmax统计量。然后
- 这些操作fuse为整个GPU kernel大幅提升效率

这样显存占用就降低为了`O（N*d）`而不是平方了。N越大收益越高，速度也越快

例如：
QKV的shape都是(4*2)
标准做法要`softmax_mask(QK^T / 根号2(d=2)) @ V` 需要存储activation=`(4, 4)`矩阵

flash attention分成Q1 Q2，K1、K2 V1 V2，shape都是`（2, 2）`

- Q1 K1计算 然后softmax 和V1计算
- Q1 K2计算 然后softmax 和V2计算，结果加和就得到了Q1与所有K的计算（2*2）

注意：这里为了维系softmax值做了一些努力
如K1时得到临时最大值`M1` 先算局部累加和`σ1=∑e^(xi-M1+M1) = e^M1 * ∑e^(xi-M1) = e^M1 * S1`，然后K2时拿到局部最大值`M2`，然后拿到局部累加和`σ2=∑e^(xi-M2+M2) = e^M2 * ∑e^(xi-M2) = e^M2 * S2`。（这里是标准max-shift）

都计算完后我们知道了全局最大值`M0`

但σ1还是用M1算的，怎么办呢？（ log-sum-exp ）的思路

`e^M1 = e^M0 * e^(M1-M0）`

所以`σ1 = e^M0 * e^(M1-M0）* S1`

类似的`σ2 = e^M0 * e^(M2-M0）* S2`

所以全局累加和`σ`就可以轻松得到了（M0、M1、M2、S1、S2全部已知），加起来即可。

之后对于某个`softmax(x_i)`只需要 `exp(xi - M0) / σ`即可（注意这里换成了全局`M0`，不再需要M1、M2）

flash attention不同版本的区别？核心思路都一样，就是加上了分块大小可调，kernel层面优化，支持变长等。

3、RMSNorm

一种代替LayerNorm的轻量normalization，少了一步减均值的操作。更快，且收敛性实际表现较好

`RMSNorm(x) = γ * x / RMS(x)`; `RMS(x) = 根号((1/d) * ∑x^2)`

这里γ是可学习的参数

4、Masked Softmax

其实就是softmax 加一个mask的融合算子，加一个极大负值减少影响

因为在attention中需要对未来位置进行屏蔽（特别是GPT）

5、Rotary Position Embedding(RoPE)
在传统固定位置编码（sin/cos）和可学习编码外，引入了一种二维旋转矩阵加入位置信息的方式。

何为旋转？x=(x1,x2)，给定一个旋转角θ，那么2D平面对x线性变换是

`RoPE(x) = (cosθ, -sinθ)`
          `(sinθ, consθ)   * x`

在高维向量（d）里，按偶数对分成`d/2`组，即`d/2`个平面，然后每个平面旋转θ

θ可能有一个线性对数或者其他函数

注意：cos(a)cos(b) + sin(a)sin(b) = cos(a-b)，这样利于模型学到相对位置信息

### Transformer

其他细节请review 2024技术积累笔记

从网络结构层面看：

Encoder：

```bash
(words -> embedding) + positional encoding ->
[Multihead self attention -> Norm -> FNN(RELU) (Residual Connection) -> Norm] * N
```

Decoder:

```bash
(words -> embedding) + positional encoding -> 
[Multihead self attention（会添加一个mask屏蔽未来单词） + Norm ->  Cross attention + Norm -> FNN -> Norm] * N -> Linear -> Softmax
```

为什么 self attention？

- RNN每一步都依赖之前的输出，无法并行
- CNN可以在kernel层级并行，但如果要覆盖整个序列的依赖，需要多层或者大卷积核，这样计算复杂度就会极高
- 而自注意力每个位置都可以直接与其他位置直接建立联系，multi-head高效并行

### LLama2 70B 训练

数据集？两万亿token （2 trillion）

注意：LLM 一般就是1个epoch（扫一次所有数据）

sequence length：最大支持4096 tokens

`global_batch_size = micro_batch_size × dp_size × chunk（PP micro batch）`

然后一个iter的token数就是`global_batch_size * seq_length`

训练时间例子：210TGS，1024卡，215040 tokens/s，需要差不多108天

一个训练配置：
DP_size(zero stage1)=8, TP=16, PP(chunk)=8, global_batch_size = 2048 ==> micro_batch_size = 16

## Recommendation

Job 推荐流程 in JobRight：

我们现在只有C端，正在做B端（引入B端后策略会复杂一些），只有C端优化的目标就非常直观：用户的申请量（多用户平均的申请量，申请量和用户留存是正相关的，申请量多留存率大）

精排模型也都是以apply rate来作为metric优化，发到前端记录哪些点了哪些没点，然后存下来数据

天级重新训练模型是为了适应job和model，为了catch up最新的数据（不然落后与现在的job与user）（如果是新闻那种，可能半小时就重新推上线）；如果要优化模型的话，都是离线看metric  

1、召回 10000条

非常依赖用户输入title的系统，基于title做search，search的时候用embedding，比如JavaScript 和 react developer，虽然词不一样，但我们知道是一类型的job
fine tune的数据源是把不同的job 划分为category，text taxonomy（分类体系，如游戏-MOBA等）
然后embedding search（cos相似度）；以及给用户的选项进行filter

2、粗排，得删除，无法处理那么多 10000 -> 500

这里用一些人工硬性策略，比如比较新的job偏好，title比较像的
人工规则打分（比如最近活跃度，热门度等，发布时间等一套加权评分体系），然后截断至500。recall的分数也可以做一个加权

3、精排： 不删除

item侧的feature，job比如category，skills；user侧的feature，看用户点过哪些job，是否新老用户，用户本身的属性等等

这个模型不是很深，比较依赖cross-feature，两边skill的match，丢给我们的推荐模型（GBDT、lightgbm）然后做推荐
（这里不是LLM，LLM太慢了，B端可能可以用LLM，离线可以接受更多延迟）

4、重排（rerank）的策略

flashness，job有多新，job company有好有坏（glassdoor评分等）

粗排和重排可能有一点类似，重排很多的策略级别都移到粗排

这一套系统可以在s级别响应

## 常见开源框架

### Megatron

megatron是NVIDIA推出的LLM框架，调用pytorch，支持多种transformer结构（GPT、BERT、T5等）

megatron聚焦于 **Tensor parallel(TP)**和**Pipeline parallel(PP)**，特别是TP对multi-head attention天然友好。同时深度结合CUDA自家GPU，通信到算子，都最大程度优化。

DP用的是Torch的DP，TP用`--tensor-model-parallel-size`控制，PP用`--pipeline-model-parallel-size`控制

Deepspeed则更多强调Zero，也支持TP和PP。

业界有组合使用Megatron-DeepSpeed（也是internlm的方式），Zero+TP+PP

ColossalAI：HPC-AI开发，社区相对更活跃，也更容易上手入门

### ML Flow

Databricks 开源的ML项目全生命周期平台，包括实验追踪、项目打包、模型管理和部署等

1、ML tracking：记录和查询实验过程和结果

2、MLflow Projects：标准化描述ML项目，便于重现和运行

3、MLflow Models：通用模型打包方式

4、MLflow Model Registry：模型版本、评审管理等

### vLLM

LLM Inference Systems：部署和运行LLM的基础设施，能处理负载均衡、高效利用硬件资源、支持流式输出等

如Hugging Face Text Generation Inference (TGI), DeepSpeed-Inference, TensorRT/ONNX Runtime, FasterTransformer 等

vLLM：对LLM推理做高吞吐量和低延迟优化，特别适用于流式场景。创新点在于对KV缓存管理。

传统LLM中，每一步生成都要读取上下文的注意力信息（查表），如果每个请求都独立的KV内存，那么高并发就显存压力太大了。vLLM引入了PagedAttention ，对k/v 向量进行了分页存储，通过灵活的“分配表”和“索引表”来管理各请求的 KV 分块实现了复用。

vLLM也自带批处理机制，可以将多个并发的推理合并成一个批次提高效率。

## System Design

ML system design技巧：

- 明确需求，定义好输入输出（基本需求），以及其他需求：延迟、可用性、是否需要scale（尽量往scale和分布式上靠）等
- 画图或文字记录（推荐文字，不需要额外开界面或共享屏幕）
- 三段式逐渐展开
  - data
    - gather data
    - preprocessing
    - 可选：scale
  - model
    - train（k-fold validation）
    - evaluate
    - 可选：scale
  - deploy
    - env（dev、test、prod）
    - online test（A/B，遇到问题怎么办）
    - 可选：scale、优化（量化quantization）
- 停顿问反馈，和面试官多交互

### 推理加速系统

问题：设计一个机器学习推理加速系统

背景：你需要为一个应用设计一个推理加速系统。目标是确保模型推理过程能够高效、低延迟地运行，特别是在计算资源有限的情况下，如移动设备、边缘设备或者负载较高的服务器环境。

要求：

- 输入：
  - 模型：深度学习模型（例如，卷积神经网络（CNN）、Transformer、推荐模型等）。
  - 输入数据：模型的推理输入数据，可能是用户行为数据、图像数据或文本数据。
  - 计算资源：可用的硬件资源，包括CPU、GPU、TPU等。

- 输出：
  - 推理结果：根据输入数据生成的预测输出（例如，推荐的视频列表、图片分类结果、自然语言处理的回答等）。
  - 性能指标：包括推理的延迟（响应时间）、吞吐量（每秒处理的请求数）、模型的大小和精度。

回答：

你会如何设计系统架构以支持低延迟和高吞吐量的需求？

- 要求低延迟，那么模型就必须小。所以我们要通过模型压缩、量化的方式提升模型推理效率

- 模型压缩：主要通过剪枝（基于权重大小、基于梯度大小、基于activation大小）实现。剪枝会将对应值设为0，然后矩阵变稀疏可以专门优化减少存储和计算开销。剪枝后模型性能下降，需要fine-tune来恢复模型性能。这个过程可能迭代多次

- 量化：我们可以考虑两种量化方式，1、post training quantilization（PTQ），即训练完成后量化，这里我们可以使用导出model为ONNX格式（注意：需要确保所有算子都支持，否则需要手动写插件，比如flash attention、RMSnorm这种LLM算子还没及时支持的）然后使用tensorRT序列化为.plan文件推理。注意进行校准（99分位后minmax校准或熵校准），以及校准后验证。2、QAT，quantization aware training，但题目假定了模型就是输入，那么我们不能使用这一点。

- 关于高吞吐量，我们可以通过scale的方式加服务器结点，来同时接受更多请求。这一点可以用kubernetes很容易地实现。

- **补充**：除了这些策略外，针对大量请求本身，也可以使用缓存（对于常见请求）（redis），vLLM对于LLM也有paged attention来专门做KV缓存优化。以及消息优先队列（kafka）等来管理推送请求，并加负载均衡load balancer打到各个结点上。

你会如何利用硬件加速来优化推理过程？具体哪些硬件和技术（如GPU、TPU等）适合在这个系统中使用？
  
- 假设我们有GPU集群，如果模型通过上面的剪枝量化、以及可能的其他压缩方式（比如低秩压缩）后不大，那么直接部署单个GPU上即可。如果模型很大（比如LLM），可能需要引入分布式策略，由于是推理场景所以没有DP，通过TP、PP对模型分层切片，实现分布式推理。在这一过程中NVlink等GPU通信方式可以较大幅度提升推理性能。

- 如果我们可以用开源框架，那么vLLM、DeepSpeed-inference等都是可以选择的对象

- **补充**：还有知识蒸馏也可以

如果系统需要扩展，你会如何设计一个高可扩展性的解决方案？

- kubernetes的特性，上面已经提到了，本身便于扩展和加节点。在容器化服务后，
- Horizontal Pod Autoscaling (HPA)和Cluster Autoscaler自动加结点。

你会如何确保系统在面对不同类型的输入数据（如图像、文本、用户行为数据等）时，仍然能保持高效？

- 这里有两种数据，一种是要求立刻回复的数据，那我们需要单独留出部署好的服务器结点（服务一体化不走微服务）来对接这种请求。
- 另一种是不着急处理的数据，我们可以在系统结构上加一层，用户输入数据会统一经过一个预处理层，比如图像提前处理为embedding，文本提前处理为embedding，用户行为数据提前清理制表等，然后再queue喂给模型。

整体补充：整个系统上线前需要压测，并用prometheus等进行性能监控和调优。也可以通过A/B test部分上线来预测试，确保整个系统上线完备。此外也要有版本控制，布置好CICD，方便之后上线。如果万一有问题，还可以回滚。

### Robot夹爪系统

一、整体设计

- 感知层（Perception Layer）：视觉传感器、力矩传感器（torque sensors）、位姿传感器等
- 规划与决策层（Planning & Decision Layer）：根据感知层数据，进行object detection、路径规划、抓取策略等
- 执行层（execution layer）：包括机器臂本身、关节电机、夹爪等

注意：力的作用是相互的，所以你不需要被夹物体显示数据，夹爪可以自己读数

二、关键数据

- 机器臂自身数据（角度、速度、加速度等，机械臂自己的结构参数如质量、DH参数等）
- 环境与目标物体数据：视觉传感器、深度传感器获得物体的分类、状态、位姿、尺寸等
- 夹爪数据：夹爪开合范围、力矩反馈等。如果有触觉和滑动传感器也可以收集
控制和安全数据：关节指令、执行结果；关节限位、过载、电机温度等

三、总结：感知层获取机器臂、环境和夹爪信息，通过决策层进行路径规划和抓取策略决定，然后执行层对关节和夹爪命令控制，同时监视安全和故障信息来实现可靠的抓取。

## CUDA

CUTLASS（CUDA Templates for Linear Algebra Subroutines）

NVIDIA用来简化并加速GEMM（general matrix mul）的C++模板库，提供了可复用可组合的模板组件

## 编译器

为什么unrolling能加速？

1、减少循环控制开销（i跳转等）（单核CPU也是）

2、降低循环分支预测失败的概率（单核CPU也是）
分支预测：现代CPU利用分支预测来实现让流水线无缝执行

3、指令级并行，给编译器和CPU更好地调度和优化

但注意：它可能导致代码体积增大，在特殊情况下比如指令缓存较小反而影响性能
此外，现代编译器一般已经自动做了循环展开的优化，影响几乎微乎其微
