---
title: "Summary: Efficient Memory Management for Large Language Model Serving with PagedAttention"
date: 2025-04-17T15:56:56+08:00
categories: ["paper_summary", "vllm"]
summary: "论文速览：'Efficient Memory Management for Large Language Model Serving with PagedAttention'"
---

> 本博客使用`o3`翻译，如有冲突请优先参考英文原文

## 论文下载

[原文 PDF](https://arxiv.org/pdf/2309.06180)

## 1. 这篇论文在讲什么？

- 提出 **PagedAttention**：把 Transformer 的 KV‑cache 按固定大小拆成“页”，取代传统的连续大张量存储。  
- 发布 **vLLM** 开源推理引擎，利用 **分页**、**块表** 与 **写时复制**（COW）把显存碎片压到近乎为零。  
- 分页机制让提示 / beam 等场景可以共享 KV，支持动态扩容、**抢占**（Swap 或 Recompute）及分布式模型并行。  
- 在相同延迟下，吞吐比 SOTA（FasterTransformer、Orca）高 **2–4 倍**。

## 2. 相比前人，这篇论文的新意在哪？

- 首次把 KV‑cache 当成“虚拟内存页”管理，而此前系统只关注调度或算子加速。  
- 设计 **块级映射表**：逻辑序列位置 ↔ 物理 GPU 块，一行代码动态分配/复用。  
- **页粒度 COW**：并行采样、Beam Search、共享前缀都能零拷贝共享 KV，以往系统只能整张量复制。  
- 针对自回归推理提出 **整序列驱逐 + 可选重算**，通用分页方案里没有。  
- 端到端实现 + CUDA 融合内核，分页额外开销仅 ~20 %。

## 3. 为了支撑观点，作者做了哪些实验？

- 在 ShareGPT、Alpaca 负载上，用 OPT‑13B/66B/175B 与 LLaMA‑13B 测 **吞吐/延迟**，对比 vLLM、Orca 三版本、FasterTransformer。  
- **批量大小分析**（Fig. 13）：同显存 vLLM 可并发请求是 Orca 的 2–4 倍。  
- **并行采样 & Beam Search**（Fig. 14）：共享越多收益越大，显存节省最高 55 %。  
- **共享前缀翻译**、**聊天机器人**（Fig. 16–17）：验证前缀缓存与长提示处理。  
- **内核微基准**（Fig. 18a）与 **块大小扫描**（Fig. 18b）：衡量分页开销与最佳页大小。  
- **Swap vs. Recompute**（Fig. 19）：展示两种抢占恢复策略的利弊。  
- **消融实验**：默认块大小 16 token 在利用率与碎片间最平衡。

## 4. 这篇论文还有哪些不足／局限？

- **块大小需手动调**，缺乏跨模型/GPU 的自适应策略。  
- 多卡并行仍需在各卡存各自 KV，**跨卡分页**尚未解决。  
- 仅验证 **解码式 LLM**；Encoder‑Decoder 或 MoE 需额外适配。  
- 抢占实验只在 PCIe A100 上做，缺少 NVLink、纯 GPU‑RAM 或 SSD 分层环境的数据。

## 5. 后续可以怎么拓展？

- 做 **自适应块大小**：按层或按负载动态选页尺寸。  
- 拓展到 **跨 GPU 分页**，让页可迁移或远程访问，实现 shard 之间 KV 去重。  
- **训练场景分页**：长上下文或持续学习时的激活/KV 也能页化管理。  
- 引入 **QoS 调度**：在混合解码模式下优先满足低延迟请求，其余流量用分页榨干显存。  
- 探索 **硬件级支持**，如显存内页表或 GPU MMU hint，把软件分页的 20 % 开销再降一截。

## Appendix

| 术语 | 释义 |
|------|------|
| **Copy‑on‑Write (COW)** | 多序列共享一页时，首次写入先复制出新页，再各自修改，避免冗余拷贝。 |
| **All‑or‑Nothing Eviction** | vLLM 的整序列驱逐策略：KV 要么全在显存，要么整体换出/重算。 |
| **Pre‑emption** | 显存耗尽时暂停序列：**Swap** 把块搬到 CPU，或 **Recompute** 之后再重算 KV。 |
| **Parallel Sampling** | 同一 prompt 随机生成多条结果，提示段 KV 完全共享。 |
| **Beam Search** | 逐步保留 top‑k 候选；前缀 KV 共享，分叉后块 COW，可省最多 55 % 显存。 |
| **Shared Prefix** | 多请求复用的系统提示或 few‑shot 样例，对应 KV 页可事先缓存重复映射。 |
| **OPT** | Meta 发布的开源解码式语言模型系列。 |
| **FasterTransformer (FT)** | NVIDIA 高性能 Transformer 内核库，但 KV 必须连续存储。 |
| **Orca** | 先前的迭代级调度系统，KV 连续分配导致碎片。 |
| **Iteration‑level Scheduling** | 每生成一步就插入／移除请求的细粒度批调度。 |
| **ShareGPT / Alpaca** | 长对话（ShareGPT）与短指令（Alpaca）负载，用于测试显存 & 计算两个极端场景。 |
