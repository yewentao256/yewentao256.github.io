---
title: "Summary: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
date: 2025-04-02T15:06:56+08:00
categories: ["paper_summary"]
summary: "论文速览：'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models'"
---

> 本博客使用`claude-3.7`翻译，如有冲突请优先参考英文原文

## 下载论文

[论文链接](https://arxiv.org/pdf/1910.02054)

> 注意：如果你对这个主题感兴趣，也可以阅读[分布式训练策略](../../zh-cn/distribution_training_strategy/)。

## 1. 这篇论文讲了什么？

- 它介绍了**ZeRO（零冗余优化器）**，一种优化内存使用的方法，用于超大规模神经网络训练（潜在可扩展至万亿参数）。

- ZeRO-DP将**优化器状态**、**梯度**和**参数**分散到不同设备上，从而减少冗余内存使用，实现更大的批处理大小和更快的训练吞吐量。

- 它还提出了ZeRO-R来处理**激活内存**、**临时缓冲区**和**内存碎片**，旨在进一步减少内存占用，同时保持高效计算。

## 2. 与先前工作相比，这篇论文有什么创新？

- 不同于**传统数据并行(DP)**或**张量并行(TP)**（前者在所有设备上复制模型，后者垂直分割层），ZeRO对模型状态进行分区，仅在需要时重新物化它们。

- ZeRO的方法保持了DP的简单性和低通信开销，同时大幅减少内存开销——这是之前的TP或**流水线并行(PP)**技术所未能实现的。

- 它通过分析和实验证明，利用足够数量的GPU并采用其分区策略，ZeRO可以实现万亿参数模型的训练。

- ZeRO不需要复杂的模型重构，而之前的解决方案如**Megatron-LM**或**G-Pipe**通常需要对模型架构或训练循环进行重大改变。

## 3. 进行了哪些实验来支持论文的论点？

- 作者在数百个V100 GPU上训练了从1.5B到170B参数不等的GPT-2风格transformer模型。

- 他们将ZeRO与基准DP（**PyTorch DDP**）和最先进的TP系统（Megatron-LM）进行比较，展示了吞吐量（TFLOPS）的提升和内存节省。

- 实验将60B参数模型从64个GPU扩展到400个GPU，当增加DP度时展示了**"超线性"**加速，这是由于每个GPU允许更大的批处理大小。

- 提供了GPU内存使用的详细测量，展示了分区优化器状态、梯度和参数如何显著减少每个GPU的需求。

- 作者训练了一个17B参数的语言模型（Turing-NLG），达到了新的最先进水平（Webtext-103困惑度为10.21），说明了实际应用性。

## 4. 这篇论文的缺点/局限性是什么？

- 虽然ZeRO在内存方面可以容纳万亿参数模型，但在当今（2020年）的硬件上端到端训练仍需不切实际的长时间（可能需要数月或更长）。

- 虽然它声称相对于标准DP，使用第3阶段只有1.5倍的通信开销，但在互连带宽有限的情况下，这种成本可能非常重要。

- 正确调整激活检查点分区（例如，决定何时卸载到CPU）可能需要特定领域的启发式方法，如果不仔细管理，可能会引入额外开销。

- 实现所宣传的效率有时需要许多具有特定高带宽互连的GPU（例如，节点内的NVSwitch，高速节点间链路）。

- 虽然transformer是主流架构，但论文并未深入探讨ZeRO如何处理具有不同内存模式的其他模型类型。

## 5. 基于该论文的合理下一步是什么？

- 研究更多动态卸载激活和模型状态的策略，考虑**异构内存层次**（例如，GPU HBM，CPU DRAM，NVMe）。

- 开发一个系统，根据实时内存使用、通信带宽和算术强度**自动**选择最佳分区计划或CPU卸载策略。

- 将ZeRO的分区方法和内存优化扩展到基于卷积的架构、图神经网络和新兴的大规模模型。

- 随着计算集群的增长，探索ZeRO的技术如何在具有数万GPU的百亿亿级HPC环境中扩展，可能完善通信集体。

- 为需要部分模型更新或参数高效方法（例如，LoRA，适配器）的**下游任务**调整ZeRO，确保在预训练和微调阶段都最小化内存使用。

## 附录

- **GPU HBM（高带宽内存）**：现代GPU使用的一种高速、封装内存，与传统GDDR内存相比提供非常高的带宽和低功耗。
- **NVMe（非易失性内存快车）**：固态硬盘的高性能存储接口协议，旨在减少延迟并改善输入/输出(I/O)操作。
- **NVSwitch**：NVIDIA的一种全连接、高速交换架构，允许同一服务器（例如DGX-2）中的GPU以非常高的带宽和低延迟进行通信。
- **Infiniband EDR（增强数据速率）**：一种提供高带宽和低延迟的网络互连技术，通常用于HPC集群和跨节点的GPU通信。
- **节点内**与**节点间**：节点内指同一物理机器内的设备（例如GPU）；节点间指集群中不同物理机器之间的设备。
- **DGX-2**：NVIDIA的系统，配备多达16个V100或A100 GPU，带有NVSwitch，用于高带宽、全对全GPU通信。
