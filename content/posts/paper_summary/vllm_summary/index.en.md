---
title: "Summary: Efficient Memory Management for Large Language Model Serving with PagedAttention"
date: 2025-04-17T15:56:56+08:00
categories: ["paper_summary", "vllm"]
summary: "Summary for paper 'Efficient Memory Management for Large Language Model Serving with PagedAttention'"
---

## Download the Paper

[Paper](https://arxiv.org/pdf/2309.06180)

## 1. What is the paper about?

- Proposes **PagedAttention** — an attention algorithm that stores Transformer KV‑cache in fixed‑size "pages" instead of one contiguous tensor.

- Introduces **vLLM**, an open‑source LLM serving system that uses **paging**, **block tables** and **copy‑on‑write** to manage KV‑cache with near‑zero fragmentation.

- Shows how paging enables prompt/beam sharing, dynamic growth, **preemption** (swap or recompute) and distributed model‑parallel execution.

- Demonstrates **2‑4 × higher throughput** than SOTA (FasterTransformer, Orca) at the same latency.

## 2. What is new about this specific paper, compared to prior work?

- Treats KV‑cache like virtual memory pages, whereas earlier systems focused on scheduling or kernel speed.

- **Block‑level mapping table** separating logical sequence positions from physical GPU blocks; supports on‑demand allocation and reuse.

- **Page‑granular copy‑on‑write** to share KV across parallel sampling, beam search and shared prefixes—previous systems duplicated entire tensors.

- **All‑or‑nothing eviction + optional recomputation** tailored to autoregressive workloads, not found in generic paging or model‑serving frameworks.

- Provides end‑to‑end implementation with fused CUDA kernels so paging adds only ~20 % kernel overhead.

## 3. What experiments were run to support the arguments in this paper?

- **Throughput/latency traces** on ShareGPT & Alpaca workloads with OPT‑13B/66B/175B and LLaMA‑13B, comparing vLLM vs. Orca variants & FasterTransformer.

- **Batch‑size analysis** (Fig. 13) showing vLLM batches 2–4 × more requests under same memory.

- **Parallel sampling & beam search** (Fig. 14) demonstrating bigger gains when KV sharing opportunities grow; memory‑saving quantified up to 55 %.

- **Shared‑prefix translation** and **chatbot** scenarios to highlight prefix caching and long‑prompt handling (Fig. 16–17).

- **Kernel micro‑benchmarks** (Fig. 18a) and **block‑size sweep** (Fig. 18b) to study paging overhead and optimal page size.

- **Swap vs. recompute microbenchmarks** (Fig. 19) to justify pre‑emption design.

- **Ablations** proving default block size = 16 tokens balances utilization and fragmentation.

## 4. What are the shortcomings/limitations of this paper?

- Block‑size **manually tuned**; no adaptive policy across models / GPUs.

- Multi‑GPU sharing still **duplicates KV across shards**; cross‑device paging left for future work.

- Evaluation limited to **decoder‑only LLMs**; encoder‑decoder or Mixture‑of‑Experts models may need further adaptation.

- Preemption tested only on PCIe‑based A100 clusters; swap performance CPU‑less accelerators or SSD tiers unknown.

## 5. What is a reasonable next step to build upon this paper?

- Dynamically choose block size per layer or workload to maximize utilization automatically.

- Extend block table so pages can migrate or be remotely accessed across GPUs, enabling KV deduplication in tensor parallel setups.

- **Apply paging to training** for long‑context or continual‑learning scenarios where activations/KV resemble dynamic, shareable states.

- **QoS‑aware scheduler** that mixes different decoding modes (greedy sampling, beam search) and **prioritizes latency‑critical requests** while using paging for background throughput.

- Explore hardware support (e.g., **on‑device SRAM page** tables or **GPU MMU hints**) to further cut the ~20 % kernel overhead introduced by software paging.

## Appendix

- **Copy‑on‑Write (COW)**: When a shared page needs modification, allocate a new page, copy data once, then let writers diverge.
- **All‑or‑Nothing Eviction**: vLLM's policy that swaps/recomputes all blocks of one sequence together, exploiting that KV pages are always accessed as a set.
- **Pre-emption**: Temporarily removing sequences when GPU memory is full; vLLM supports Swap (move pages to CPU) or Recompute (regenerate KV cache by rerunning the model on "prompt + generated tokens" once memory is available).
- **Parallel Sampling**: Generate several stochastic completions for one prompt simultaneously; prompts’ KV pages can be fully shared.
- **Beam Search**: Keeps top‑k hypotheses at each step; early‑prefix KV widely shared, divergent parts use COW; brings up to 55 % memory saving.
- **Shared Prefix**: Common system prompt or few‑shot examples reused across requests; cached pages mapped by many sequences.
- **OPT**: Open Pre-trained Transformer Language Models
- **FasterTransformer (FT)**: NVIDIA library with highly‑optimized Transformer kernels but contiguous KV tensors.
- **Orca**: Prior serving system using iteration‑level scheduling but still contiguous KV; suffers from fragmentation.
- **Iteration‑level Scheduling**: Insert or remove requests after every decoding step, avoiding long queueing and padding waste.
- **ShareGPT / Alpaca Workloads**: Real conversation (long) vs. instruction‑tuning (short) traces used to test memory‑ and compute‑bound extremes.
