---
title: "Technical Notes for vLLM"
date: 2025-06-08T11:27:56+08:00
draft: true
categories: ["vllm"]
summary: "vLLM 技术笔记"
---

## 基础知识

Multi-Head Latent Attention：deepseek团队的高效attention机制，通过对key value做Lora，压缩kv cache成更小的latent attention

Microscaling FP8: MXfp8将tensor拆成32元素一个block，每个block有一个E8M0的纯指数缩放因子，量化时先把数据压成E4M3或E5M2的范围，然后再存8bit浮点。这样缩放粒度更细，对精度影响更小。

Multi step scheduling: 生成一次token调度一次改成生成多个token调度一次。注意：并不是一次性输出多个token，而是把n次前向-采样-KV写入的小循环打包给GPU做，减少和CPU的交互

GPTQ（命名取自GPT + PTQ）：一次性"权重"量化方法，用一小批校准数据把权重压成8/4 bit，同时尽量保持精度；核心思想是把每个线性层 y = Wx的 W分块（按列/通道），按块选择scale，并利用输入的二阶统计（hessian）来最小化输出误差。注意：和vllm里面的per token group quant不是一个东西（activation运行时量化）

对于layer norm这种memory bound：当num tokens很多的时候，对应block总数也多（一个token一个block），此时block_dim减少能让SM有更多的块，这样能利用好warp读取global memory（memory bound）。当num tokens很少的话，block数量也少（比如一个SM上只有一个block），此时我们增大block dim让一个block的warp更多，减少没事干的warp数量。

TP与行列切分（M、K、N）：

```bash
2 * 3 @ 3 * 2 -> 2 * 2
按N切：2 * 3 @ 3 * 1 -> 2 * 1，之后要all gather拼接
按M切：2 * 1.5 @ 1.5 * 2 -> 2 * 2，之后要all reduce加和
```

Prefix caching: 即“把一段固定开头算过一次，以后遇到同样的开头就不再重算”

很多LLM system里都是system prompt+历史对话+本轮用户新问题；system prompt基本总是重复，所以第一次遇到长输入的时候我们把这段prefix计算一遍，存储kv cache等然后之后就可以复用这段缓存。注意和普通http缓存不同的是，Prefix caching 不缓存“最终答案”，而是缓存输入前缀的中间计算结果

## CPU 问题

chrome切换到vscode发生什么，为什么要等一段时间？

1、内存不足导致的压力

chrome占用大量内存和现有页表，vscode也要占用。如果内存不足，就会触发大量Page Fault反复读写内存（为什么写？脏页（已经改过数据的chrome页）需要写回，不一定发生，但读一定发生）

2、进程上下文切换cpu开销

- 保存寄存器到当前进程控制块（process control block）
- TLB（Translation Lookaside Buffer）是页表缓存，（快表），要切换
- CPU cache （L1/L2）被冲刷，这里损失一些CPU周期
- 影响不大但有：branch prediction miss

## flash attention

为什么FA能加速？

1. 并不是parallel，常规attention的matrix mul也做的很好。事实上，直到flash attention2/3对warp重新划分后才显著提升，flash attention 1这一块做的还不够好。
2. save memory to increase larger block
3. hit cache，这是最核心的点，memory bound改成了compute bound。self attention反复把n*n的 attention 矩阵从HBM（global memory）读到GPU SRAM（shared memory），memory bound严重。而flash attention用tiling一次读取所有所需数据，这样memory bound情况大幅减轻。
4. 此外还有kernel fusion（进一步减少了memory access）

## DeepseekV3.2

[Release Doc](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)

最核心的点在于DSA（deepseek sparse attention）

DSA=两阶段，逐token的动态稀疏注意力
每来一个查询token，先过一个很轻的lightning indexer给整段历史打分，拿得分最高的2048个历史token做注意力计算。

例子：把一份 100,000 token 的合同+往来邮件塞进上下文里问第 7.3 节里的违约金和‘不可抗力’是否冲突？

传统dense对每个查询token要对100k历史位置全部attention；现在只需要top-2048做attention了，计算量由O（L^2） 变为 O（2048*L）

注意indexer有自己的fp8 kv cache，分为非paged和paged（类似vllm paged 分页管理）两种

## DCP

Decode Context Parallel（DCP，解码上下文并行）

一句话理解：每次decode都要对很长的kv cache做attention，DCP核心就是把kv cache按照时间/token维度分片到多张卡上，需要用时一次通信拼接

现在每张卡只存和处理自己的KV，单卡KV显存/n，Decode的时候，每张卡对自己的那份KV做attention并行，然后合并结果得到与全局一致的输出。

例子：你需要根据一万字的上下文续写一段话，你之前要把一万字都背在脑子里，现在你找了三个朋友每人记四分之一，每人给出自己的局部建议，最后参考每个人的建议续写。

优点是显存友好，代价是多一次通信。

怎么通信：`all gather + reduce scatter`：先把Q在DCP组内做allgather，让每个rank得到完整Q，然后用完整Q+本地kv 分片计算得到local_out和local_lse。（注意：这其实就是flash attention的思路，我们只需要算局部值利用log-sum-exp得到全局softmax），然后全局gather `local_lse`并计算`log-sum-exp`得到`global_lse`，随后用它把各rank的local_out缩放，求和得到`global_out`（矩阵乘每块都有贡献），随后再reduce scatter切回各个rank各自需要的那部分。

注意：切的不是模型，切的是“历史上下文”KV。常常和TP一起用（切矩阵/通道）

上下文特别长的时候特别有优势。

1、为什么会有局部Q？局部Q来自TP的head 分片，每个rank只有一部分`query heads（w_q）`,对于输入`X`要矩阵乘`W_q`得到局部`Q`。

2、最后为什么需要`reduce scatter`再次切分？因为TP rank每个GPU只需要自己的那部分

## Batch Invariant

为什么对RL有帮助？最核心原因是sample/argmax，logits不一样，导致同样代码同样seed，训练曲线不一样，可能这次能收敛，下一次直接无法收敛甚至崩溃。

如果不同的训练和推理框架（pytorch + vllm）policy会有一个隐形的off-policy（比如kernel 不同），而RL要求policy一致

此外则因为RL的policy用到的状态V/Q是基于模型算出来的，简单来说是“模型自己的预测作为下一步的label”会进一步放大误差

## Async Scheduling

1、gpu 到 cpu copy异步

单独stream + event处理，最终在`get_output` sync

`AsyncModelRunnerOutput` 不会在worker主线程里sync，而是会放到aysnc output queue，让async_output_busy_loop去做

2、允许多个in-flight batch

对于一个request来说，上一个token出来了下一个token才能走，但我们一个batch可能有很多requests，或者prefill case有很多prompt token，又或者spec decode可能有多个token要处理。

此外vllm有num_output_placeholders可以支持结果token 还没有回cpu，但认为它先有,在runner那边最后送上gpu的时候一定会检查确保依赖的前置项没有placeholder（如果没有硬依赖不检查）

主要函数：`step_with_batch_queue`，优先填batch queue，满了才block

Batch queue: 长度为2，优先放batch queue，满了block 弹出先入队的

这里两个batch 基于continuous batching策略分布

B中可能有同时有A中要预测的下下一个token（利用placeholder机制）

## Pipeline Parallel

1. pp 间传输使用`tensor_dict`，如果有TP会尝试使用all gather给对应tp传slice，用的时候再统合，详情看 `send_tensor_dict` 和 `recv_tensor_dict`。
（例子：TP=8，PP=2，那么rank0和rank8形成PP组，rank0只传部分slice给rank8；对于所有stage1（rank8~rank15）的tp，内部做一个all gather所以有完整tensor）
这里的tensor是hidden state（activation），PP rank0产出，作为pp rank1的输入

2. `get_pp_indices`用于切层，一般均匀分，实在不能整除从倒数第二个stage开始往前分（最后一个stage要采样更多任务所以不分）。对于不属于当前rank的层，使用PPMissingLayer占位

3. forward的时候如何传递？基于`IntermediateTensors`的构造，上游产出`IntermediateTensors`（hidden states）给下游stage，然后最后一个rank在gpu model runner里得到`compute_logits`得到`logits`（过一个linear得到词表分数）然后过softmax sample的到token

4. IntermediateTensors基于`isend`/`irecv`传递，需要用到实际数据的时候wait synchronization

5. 和**async scheduling**结合：最后stage直接用 gpu通信 把sampled token ids给broadcast给之前的rank。（首rank需要很容易理解，其他rank也需要用此更新自己的request状态/长度）

问题：是否`GroupCoordinator`每个rank持有一个实例？是的，每个PP rank 调用`init_model_parallel_group` 初始化 然后走自己的一套

问题：PP如何与pd分离结合？可以共存但彼此互相无感知，pd是不同vllm实例，实例间用kv connector传输和接收，实例内pp处理自己的请求

## MOE

token 如何跑 MoE：router 选 top‑k experts 后，如果命中的 expert 不在本卡，就会通过 all2all 把该 token 的 hidden state 发到对应 ep_rank 的卡上算，再 all2all 把结果发回、在原位置做加权合并。

### MOE的shape

`A(M * K) @ W1 (K * 2N) -> O1(M * 2N)` ，这里W1把up和gate放在一起了所以是2N

然后O1 切成gate和up，gate过一个f（sigmoid或者silu、gelu等）然后与up按位乘

例如，`gate = [-3,0,2]` `sigmoid(gate) = [0.047, 0.5, 0.881]`, 设up为`[2, -1, 0.5]`

点积后得到`A2 = [0.094, -0.5, 0.441]`，通俗来讲就是门控让特征能不能过

`A2(M * N)` 再矩阵乘 `W2(N*K)` 得到 `M*K` output

### TP/DP with/without EP

```py
    # DP + EP / TP + EP / DP + TP + EP
    # In EP, each device owns a set of experts fully. There is no tensor
    # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
    ep_size = tp_size
    ep_rank = tp_rank
    return FusedMoEParallelConfig(
        tp_size=1,
```

所以对于 TP + EP (TEP) 时的MOE layer, 我们让EP=TP size并且设置TP size为 1

而对Non-MOE layer或没有启动EP的case来说, 我们使用纯TP

例子: tp = 8, gpus=8, experts=64

- EP未启用. MOE layer 每张卡都有64个experts，但每个expert只有1/8
- EP启用, 每张卡有8个完整的experts
- 对DP 同样适用

一句话总结：MoE 里 dp/tp 都“凑数”进入 EP 组来拆 experts；但 MoE 之外，tp 是切分，dp 是复制/副本语义。

## Sequence Parallel

### 简介

常规 TP 要`all reduce`拿完整的hidden state来去做接下来的rms norm、MOE路由等操作（`all_reduce -> rmsnorm -> MOE -> ...`）

SP会将其变成 `reduce_scatter -> (local) rmsnorm -> MOE 路由 -> ... -> rms_norm （下一层的input） -> all_gather`

简单理解即我们将all gather尽量延后了，计算在local进行而非全量计算。

一个简单例子：`[8, hidden]` 的hidden state我们tp=2拆成 `[4, hidden]`走local计算，最后all gather成`[8, hidden]`

为什么值是正确的？RMS norm等算子是按token独立的，token N数量不影响。MOE router是对每个token算应该去的expert topk，这里也是token独立

SP省计算很明确，但SP关于通信有点复杂，我们展开讲

### 通信

我们将通信简化为dp=1的case

非SP流程

```bash
Attention partial output
→ TP all-reduce ①
→ 每个 rank 拥有全部 token
→ 各 rank 计算本地 experts 的贡献
→ TP all-reduce ②，汇总所有 expert 贡献
```

SP流程

```bash
Attention partial output
→ TP reduce-scatter
→ 每个 rank 只拥有部分 token
→ dispatch 到 expert rank
→ expert compute
→ combine 回 token-owner
→ TP all-gather（通常在下一次 attention 前）
```

1、增加通信的地方：如果在attention后直接all reduce 每个rank上都有所有token，那可以直接用。但如果是SP，token可能不在对应gpu上，那么还需要额外dispatch（一对多）到对应gpu，然后combine（多对一）回原gpu。

2、减少通信的地方：注意到我们省掉了all reduce②。为什么combine可以代替第二个all reduce？非SP没有dispatch时，我们有rank 0：w1 · y1，rank 1：w3 · y3，这里需要all reduce得到sum。但SP我们dispatch完再combine就自动是w1 · y1 + w3 · y3

此外，SP会将token padding到tp size，小batch/短序列不一定赚，所以当前只在特殊条件下开（EP + TP > 1 + DP > 1）

在DP>1的case下，非SP需要在DP-group多dispatch/combine一次，SP则是在转换成在EP-group dispatch/combine。所以DP下会更赚性能

注意和TP的区别，TP切的是W_QKV heads维度，每个rank都是local attention最后要all reduce统合。SP切的是token维度。

- 知识补充：All reduce=reduce scatter + all gather；
- 知识补充2：注意TP如果用row parallel，output shape每个rank和最终output rank是相同的（`4*2 @ 2*4 = 4*4`，拆成`2*2 @ 2*2 + 2*2 @ 2*2`，然后直接加起来）；column parallel才要cat。
- 知识补充3：router给每个token选完topk后，会把该token的hidden state发送给experts所在gpu（dispatch，也叫all to all），experts一套独立参数的FFN计算后将token按原始位置送回，并按topk weights加权合并（combine，反向all to all）
- 知识补充4：topk即`hidden states @ gate_weight`——得到当前token对每个专家的原始logits（合适程度）。然后topk选择最佳前N个专家，然后加权求和得到每个专家的权重（总合为1）

## DP mode

- internal LB：一个vllm serve起多个DP rank（可能多个node），多机时对外head node暴露endpoint，其他node headless，vLLM自己在内部根据负载情况选rank
- hybrid LB：每个node一个vllm serve，对应一个endpoint，比如node-a:8000，node-b:8000，打到某个node后vLLM会再根据负载选rank
- external LB：每个DP rank 一个vllm serve对应一个endpoint
- 新增的dp supervisor：在external LB的基础上加一个一键启停服务+health check的结点。

## DSv4 compressor

Dsv4 compressor和indexer
attention类（c4）：有主mla kv cache，swa kv cache，c4 compressor 和 indexer
（c128）有主mla kv cache、swa kv cache，c128 compressor，没有indexer（因为128压缩已经很少了，直接全部参与attention）
C4 indexer有自己的compressor
swa kv cache 保留最近的未压缩信息，最终attention使用Top-K选中的 Main compressed slots + SWA KV slots
Compressor 4可以理解为一个长度为8，步进为4的sliding window
例如：A B C D E F G H I J K L
slot 0: A B C D
slot 1: A B C D E F G H
slot 2: E F G H I J K L
它不是直接压缩hidden states，它会经过wkv（写入内容）和wgate（写入权重）专门的projection，（一个大矩阵乘优化然后切开），加权融合
然后main compressor写自己的cache，indexer compressor也写自己的cache，slot编号一一对应
例如：Slot编号：    0  1  2  3  4  5
Indexer cache： I0  I1  I2  I3  I4  I5  每个128维
Main cache：  M0  M1  M2  M3  M4  M5  每个512维

Topk是怎么做的？
对每个query token先w index投影，然后对所有indexer slot比较得到分数，选择topk的indexer slot id去main cache里查做final attention

Dsv32相比于此，主要区别在于没有compressor，kv cache存slots数量会翻4倍/128倍，也没有SWA。但dsv32能跨层复用，避免反复indexer和topk