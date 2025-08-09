---
title: "Summary: SGLang: Efficient Execution of Structured Language Model Programs"
date: 2025-08-09T16:29:16+08:00
categories: ["paper_summary"]
summary: "论文速览：'SGLang: Efficient Execution of Structured Language Model Programs'"
---

> 本博客使用`GPT-5`翻译，如有冲突请优先参考英文原文

## Materials

- [Paper](https://arxiv.org/pdf/2312.07104)

- [Github](https://github.com/sgl-project/sglang)

## 1. 论文是关于什么的？

提出 **SGLang**，一种嵌入 Python 的 DSL，用于高效执行多次调用、结构化的 LLM 工作流。
关键运行时思想：用于 KV 缓存复用的 **RadixAttention**、用于快速受限（如 JSON/正则）解码的 **压缩有限状态机（cFSM）**，以及面向黑盒端点的 **API 预测执行**。

## 2. 与以往工作相比有哪些新意？

- 将 KV 缓存视为带 **缓存感知调度**的**基于树的 LRU 缓存（基数树）**，并进行前端/运行时协同设计；作者称这是首个同时支持**多级共享、LRU 淘汰、协同调度与分布式**场景的方案。
- 以往引擎（如 **vLLM/PagedAttention**）做内存分页与简单前缀复用，但不支持带 LRU 与调度的**树结构、多级复用**。
- 其他复用方向（如 **PromptCache**、**ChunkAttention**）探索模块化或前缀感知复用，但要么存在**准确性下降**风险，要么专注于**内核级改动**而非**缓存+调度+语言**的一体化设计。
- 引入 **压缩 FSM（cFSM）**，使确定路径上的多个 token 能在**一次**解码完成——相较以往逐 token 掩码，显著加速**受限解码**。

## 3. 为支撑论点做了哪些实验？

- 在 MMLU、HellaSwag、ReAct/生成式智能体、Tree-/Skeleton-of-Thought、JSON 解码、多轮对话及一个 DSPy RAG 流水线中，**吞吐最高提升至 6.4×、延迟最低降至 3.7×**。
- 在 **Mixtral-8×7B** 与 **Llama-70B** 上有类似增益。
- 在 **LLaVA-v1.5-7B（图像）** 与 **LLaVA-NeXT-34B（视频）** 上吞吐大幅提升；例如从 **0.18→1.15 图/秒** 与 **0.02→0.10 帧/秒**。
- **消融**显示各组件（树缓存、调度、前端提示/并行）均有贡献；在 JSON 解码中，**cFSM** 带来约 **1.6×** 吞吐提升。

## 4. 不足/局限

- 只与**较早版本的 vLLM**比较；随着基线演进，结果可能变化。
- 关于 **cFSM**，附录指出某些正则选择可能导致**概率分布扭曲**——这是一个需要进一步研究的准确性问题。

## 5. 合理的后续工作

- 将 **RadixAttention** 适配到**多级存储**（DRAM/磁盘），加入**模糊语义匹配**，缓解**缓存感知调度**中的潜在**饥饿**，并强化**编译器**以进行静态规划。
- 强化对**多模态工作流**的支持，并在**一致设置**下与更新的 KV 复用基线（如新版 vLLM、ChunkAttention）进行比较。

## 附录

- **基数树（radix tree）**：一种空间优化的前缀树，其**边**可存储**序列**（而非单一符号），从而支持高效前缀查找与插入。
- **结构化解码**：在**硬约束**（如正则/CFG）下，通过对非法 token **加掩码**来进行生成。
- **模糊语义匹配**：一个非正式术语，指按**语义**而非**表面文本**进行匹配（如向量/嵌入相似度），可容忍表层不一致，常用于 **RAG** 检索。
