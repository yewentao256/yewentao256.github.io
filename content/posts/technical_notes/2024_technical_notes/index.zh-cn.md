---
title: "2024 Technical Notes"
date: 2024-12-31T14:02:12+08:00
categories: ["technical_notes"]
summary: "2024技术积累笔记"
---

## Deep Learning基础

### 常见问题

如何减少过拟合？

- dropout
- normalization
- Regularization（训练时加入惩罚项）：
  - 1、直接加到loss函数里 （L2就`λθ^2`， L1就`λθ`），这里`θ`是参数
  - 2、直接`weight * λ` （也叫**weight decay**）
- 增加数据量、早停等

### Zero

zero都不能16位半精度训练更新参数的原因是梯度更新需要高精度，不然小梯度会消失

### Activation

Leaky relu：引入小的负斜率导致不会有0导数

x > 0 时 为x

x <=0 时 为αx （α为0.01等参数）

### 反向传播

#### 矩阵乘

我们有`Out = t1 @ t2`

`grad_t1 = grad_output @ t2.T`，`grad_t2 = t1.T @ grad_output`

帮助记忆：假设grad_output = 1, 那么`grad_t1 = t2.T`, `grad_t2 = t1.T`，然后按shape推一下grad_out放哪边即可

为什么？可以举一个例子`[[1,2,3][4,5,6]] (2*3) @ [[1,2][3,4][5,6] (3*2)`

result shape (2,2)

```bash
r[0,0] = a[0,0] * b[0,0] + a[0,1] * b[1,0] + a[0,2] * b[2,0]
r[0,1] = a[0,0] * b[0,1] + ...
```

反向传播时，我们有`r' | a[0,0] = b[0,0] * d'[0,0] + b[0,1]* d'[0,1]`

这恰好是 用`r' @ b.t` 对应第a的位置值，以此类推grad_a的每个位置，所以`grad_a = r' @ b.T`

核心思想：当对梯度传播不理解的时候，举小例子画图，以scalar的维度思考就很容易理解了

#### Conv

同理，conv的反向，关于input的导数其实就是对d reverse conv。推导类似矩阵乘，用scalar主体看

### LoRA

在不改变原始权重矩阵的基础上，LoRA 将权重矩阵的更新拆分为两个小矩阵的乘积

公式上可以表示为：

`𝑊′=𝑊+Δ𝑊=𝑊+𝐴⋅𝐵`

其中，`$A$` 和 `$B$` 是低秩矩阵，LoRA 通过优化 `$A$` 和 `$B$` 来实现对模型的微调，`$W$` 保持不变。

可以在不牺牲原始模型性能的前提下实现特定任务的微调，并且速度快

### Normalization

为什么要Normalization？让每一层的输出调整到一个相对稳定的分布（均值0标准差1），再走激活就有意义,

如resnet 50 BN完后RELU，归一化后再RELU就可以避免大量负值无意义。实践效果可以加速收敛和提升最终效果，也能防止Over fit

假设一个NCHW

BN：对一个batch的norm（N）结合HW，减均值除标准差让数据无shift

Layer Norm：对单个样本的特征（C）结合HW，减均值除标准差

Group Norm：将C分成G组后，对每组（C/G）减均值除标准差，然后再拼起来（当前批值不稳定的时候好用）

Instance Norm：只考虑H和W做归一化

### Optimizer

最原始的公式：`θ -= γ * G`

带momentum的SGD： `θ -= γ * v`, `v = β * v_(t-1) + (1-β) * G`

#### Adam

核心公式：`θ -= (γ * m_hat) / (v_hat ^ 0.5 + ϵ)`

其中 `m`为一阶矩（滑动平均值） `m_t = β1 * m_(t-1) + (1-β1) * G`

`v`为二阶矩（滑动平均方差） `v_t = β2 * v_(t-1) + (1-β2) * G^2`

偏差校正：`m_hat = m_t / (1 - β1^t)`，`v_hat = v_t / 1 - β2^t`

为什么要有这个偏差校正？一开始m和v初始化为0，`β`的值很大，m和v的更新会很慢，加入这个偏差校正让它在一开始就有有效的值。后期随着迭代轮数t的增大，`β^t`会逐渐减小到0，`m_hat`和`v_hat`也对应趋近于`m`和`v`的值。

一个代码：

```py
def adam_optimizer(grad, params, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    m = np.zeros_like(params)
    v = np.zeros_like(params)

    for t in range(1, num_iterations + 1):
        g = grad(params)
        m = beta1 * m + (1-beta1) * g
        v = beta2 * v + (1-beta2) * g * g

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
    return params
```

### Loss

L2 loss：MSE，mean square error，1/n  ∑ (yi - y)^2

L1 loss: Mean Absolute Error，MAE，直接是绝对值误差

Huber loss：大误差用L1损失（求导为1，对大误差不敏感），小误差用L2损失

binary cross entropy loss：`L = - ∑ ( yi log(pi) + (1 - yi) log(1 - pi))`

例如：

```bash
样本    真实标签 (y)    模型预测概率 (p)
1        1                0.9
2        0                0.2
3        1                0.6
```

loss = - (log(0.9) + log(1-0.2) + log(0.6)) = 0.28左右

如果是多分类的cross entropy：`Loss = -∑ yi * log(pi)`

这里pi必须是总和为1的概率，如多分类过完softmax

二分类可以用sigmoid z = 1 / (1 + e^(-x)) （工程中，如果x<0会整体乘一个e^x）降低数值

多分类就是softmax z = e^x / ∑ e^x

## NLP

### Perplexity

困惑度

`P = exp(-1/N * ∑logP(wi | w1, w2, ..., wi-1))`

表示模型预测下一个词的不确定性，假设百分百正确预测，那么困惑度为1

困惑度越高表示下一个词的不确定性越大。

### Transformer

Attention本质：查表找到最有利于你预测的上下文

Transformer 的架构分为**编码器（Encoder）和解码器（Decoder）**

Encoder 处理输入数据并生成hidden state；Decoder 将hidden state转成目标序列

位置编码：通过sin或cos给位置一个编码，如`sin(pos/10000^(2i/d))`。还有一种是位置embedding

Encoder和Decoder中，就有multi head self attention，FFN（两个线性层+RELU/GELU-x*标准正态分布累积分布函数）和Layer
Normalization

Self attention：生成查询（Query）、键（Key）和值（Value）向量：

X为输入（n*d）

`Q=X @ Wq，K=X @ Wk，V=X @ Wv`

QKV的shape都是（n*d2），其中d2为d/head数。即每个head关注自己的QKV，最后会拼接起来，走一层线性层生成最终attention输出。（这种multi head特别合适tensor parallelism）

然后attention = `softmax(Q @ K转置 / 根号d2) @ V`，（根号d2位缩放因子减少softmax尖锐的问题）

cross attention（位于decoder）不同于self attention只看自己的信息，它的Q来自hidden state，但K和V来自encoder。其他公式与上文一样。通过看encoder的信息能够生成与输入序列相关的输出，从而在机器翻译中这种任务更好地发挥作用

decoder还有一个masked self attention遮蔽未来的位置数据（注意力分数为负无穷大）

此外，transformer有重要的一点即残差连接，输入直接加到attention和FFN后然后再layer normalization，缓解梯度消失问题

transformer本身没有loss，loss取决于下游架构。如bert的两个loss 遮掩语言模型（Masked Language Model, MLM）损失 （BCE二分类） 与 下一句预测（Next Sentence Prediction, NSP）损失（BCE 二分类 is_next）

### 如何 embedding？

如何得到embedding预训练向量？之前比较多的是Word2Vec: 使用基于词的连续词袋模型（CBOW，上下文词预测目标词）或跳跃模型（Skip-gram，给定目标预测上下文），但这样问题很多——太多词了，没见过的也没办法处理

然后bert出来后就都用bert了：wordpiece分词（un”、“##happi”、“##ness），可以有unknown词，然后词元同样@ token embedding矩阵得到向量（30,000词元），再加上positional encoding 和 Segment Embedding（区分句子）相加得到input representation，然后输入transformer得到最终的上下文相关的句向量

embedding矩阵如何得到？海量数据用Masked Language Modeling, MLM和Next Sentence Prediction, NSP来预训练。预训练不仅仅训练token embedding（positional embedding在bert中是embedding不是固定的cos和sin）和segment embedding（其实就2*d，判断是否是句子1还是句子2，然后d vector会被加到input上），也预训练了transformer的各种weights

transformer各种weights包括（Wq Wk Wv，multi head attention还有一个Wo负责把concat的权重再均匀整合一下等），FNN W和B，layer norm 的 γ和β（对CHW减均值除标准差后再*γ + β）

后面出来了GPT，GPT不像bert是双向的，它只考虑左侧上下文，训练时预测下一个词元（所以为什么更合适生成任务）

分词也可以用Byte-Pair Encoding (BPE)，一开始lower拆成 l o w e r <\w>然后训练时逐渐合并形成词元

### Transformer相关问题

transformer中为什么用layer norm？因为BN要求N是稳定的，transformer 每次序列N不一样，Layer norm不依赖N的大小，对每个样本所有特征维度上处理

为什么transformer公式那里有个根号d？

可以理解为缩放因子，不然直接进softmax的话值会很尖锐。为什么是根号d呢？可以理解为原来QK假设均值0标准差1，点积后放大了这个差值（到d，所以标准差要根号d）

Hugging Face: 一套统一的API——Tokenizer、Model

T5（Text-to-Text Transfer Transformer）：通用文本生成模型，旨在将所有 NLP 任务统一成一个“文本到文本”格式。输入文本 -> 输出文本

RoBERTa (Robustly Optimized BERT Approach)：干掉了Next Sentence Prediction (NSP) ，专注于Masked Language Model（MLM）

## CV

### Resnet50

Resnet50：Residual Network with 50 layers

图像分类任务，在给定数据集中对类别划分

特点：
1、residual block：通过短路+缓解梯度消失
2、每个residual block包含CNN + BN + RELU，输入固定大小image（224*224），输出类别概率分布

架构：

Image （3*224*224） -> Conv1 -> 64*112*112 -> Conv2 -> 256*56*56 -> Conv3 -> 512*28*28 -> Conv4 -> 1024*14 * 14
-> Conv5 -> 2048*7*7 -> Avg Pooling -> 2048*1*1 -> Fnn + Softmax -> 1000

### Diffusion Model

前向：`Xt = 根号(1-βt) X_(t-1) + 根号(βt) ϵ`

其中ϵ∼N(0,I)，表示均值为0 标准差1的高斯分布

反向：X_(t-1) 粗略等于  X_t - 根号(βt)ϵ0  + 随机补偿 根号(βt) z   其中z∼N(0,I)

ϵ0是NN预测的结果

loss：预测噪声ϵ0和真实噪声ϵ的均方误差（MSE：差值的平方）

### GAN

#### 常用指标

FID (Frechet Inception Distance)：常用于GAN，检查生成图像和真实图像**分布**差异，分数越低表示越接近真实，考虑了视觉感受

PSNR (Peak Signal-to-Noise Ratio)：更关注**像素**差异，差异越小，分数越高，表示与原图越接近，就像复制品一样。但它不太注意人的视觉感受，可能细节模糊的图PSNR也会很高

SSIM (Structural Similarity Index Measure)：查看生成图片与原图“结构”上的差异，不仅仅是像素，还有“亮度”“对比度”等，越高表示感官和结构上越相近

IS（Inception Score）：综合考虑真实性（用一个预训练的分类模型，通常是inception来分辨），和多样性（不应该千篇一律）。IS分数越高越好

#### 具体模型

UnetGenerator：卷积 batch norm + RELU 不断 encode到1维，这一层主要负责捕捉图像的全局语义和上下文信息。会不会丢失大量信息呢？会，所以为什么有Skip Connections把encode层对应输出concat进去

PatchDiscriminator：如我们代码中最终输出1*30*30的概率分布，对于256*256的原图 类似于9*9的滑动窗口，patch去给9*9的窗口打分，关注局部细节并减少参数量。

### Swin Transformer

将transformer 与 CV 结合。

如果对每一个像素都attention计算，那么复杂度过高了

Swin transformer将图像划分为多个固定大小的窗口，每个窗口独立self attention计算

为了解决分区窗口的注意力局部性限制，Swin transformer还引入了shifted window，在连续transformer层中window会稍微移动以让特征互相联系

## 多模态

### CLIP

CLIP：Contrastive Language–Image Pre-training 对比语言-图像预训练的多模态模型

将图像和文本嵌入到一个语义空间中

通过对比学习训练（每一个图像都有文本描述caption），尽可能将image encoder和text encoder的输出对齐

## Python相关

python同时赋值语句，先计算好右侧元组，然后依次赋值。所以pre, cur, cur.next = cur, cur.next, pre会有问题（因为cur已经是新的了）

### asyncio

asyncio - coroutines（协程）

协程是单线程并发模型，通过loop来调度多个coroutine，在IO等待时切换到其他协程

非常适合IO密集型任务，但由于只有单线程并不适合CPU密集型任务

但对于IO密集型任务，也分为网络IO和文件IO，coroutines擅长internetIO，因为大多数操作系统原生支持异步网络操作，（如non-blocking sockets和事件通知机制）

但coroutines对于File IO效果就比较差，因为很多操作系统文件IO是阻塞型的，如linux（传统posix标准设计）。当然现在新版linux也引入了原生异步IO如io_uring和AIO来逐步解决这一问题

## RAG

树的特殊结构为什么比llama index好？HierarchicalNodeParser 切分后它返回列表，需要get_deeper_nodes 来拿指定层级的nodes，一看源码其实就是遍历列表（而且还要先拿root再多次get children node）

我们检索指定层级nodes直接O（1）拿所有文档下同一层的nodes 如sentences（dict列表），然后也是O（1）拿父子结点

RAG的一些常用评价指标：

常见的如accuracy、precision（TP/（TP+FP））、recall（TP/（TP + FN））、F1等直接判断找到的文档数

平均倒数排名（Mean Reciprocal Rank, MRR）——第一相关结果（数据集标注）与实际相关结果在列表中位置的关系。越接近1越好

归一化折损累积增益（Normalized Discounted Cumulative Gain, nDCG），考虑了检索的返回排序与检索相关性等级，公式比较复杂。越接近1越表示返回的排序和相关性等级理想。

## CUDA

shared memory不会溢出，会报错

现代计算机主要速度和memory是一致的，省内存就是省速度

cuda——让一个thread只做一个计算，可以有很多block然后交给硬件调度，而不是在一个线程里for循环

## 机器学习基础

- 线性回归（Linear Regression）：找到一条线获取输入输出关系
- 逻辑回归（Logistic Regression）：通过线性模型将输出映射到0/1
- 支持向量机（Support Vector Machine, SVM）：通过不同核函数，如线性核，多项式核，高斯核等，将数据映射到一个高维空间，找到一个超平面进行线性分类

- K-最近邻（K-Nearest Neighbors, KNN）：计算距离选取k个邻居，利用邻居的多数类别或平均值预测
- 决策树（Decision Tree）：逐渐对特征条件判断
- 随机森林（Random Forest）：多个决策树
- 朴素贝叶斯（Naive Bayes）：假设特征独立，意义不是太大
- K-均值聚类（K-Means Clustering）：无监督方法，将数据集划分为K个簇，目标是最小化簇内平方误差
- 梯度提升树（GBDT）：逐步添加决策树，每个新树都是为了减少前面模型的残差，而残差恰好就是负梯度。让梯度下降，残差减少。XGBoost是基础实现。lightgbm连续特征分箱直方建图加快效率，优先叶子生长策略（容易过拟合所以要depth限制）和引入goss算法（重心放在大梯度上）。catboost对类别特征有内置支持
