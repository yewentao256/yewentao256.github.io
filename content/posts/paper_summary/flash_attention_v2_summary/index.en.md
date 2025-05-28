---
title: "Summary: FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning"
date: 2025-05-28T15:46:56+08:00
categories: ["paper_summary"]
summary: "Summary for paper 'FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning'"
---

## 0. Materials

- [Paper](https://arxiv.org/pdf/2307.08691)

- [Github](https://github.com/Dao-AILab/flash-attention)

## 1. What is the paper about?

- Introduces **FlashAttention-2**, a new GPU kernel for exact (non-approximate) Transformer attention.

- Targets long-context training/inference by **cutting memory traffic to O(N)** while pushing throughput close to matrix-multiply (GEMM) efficiency.

- Achieves up to **2–3 × speed-up over FlashAttention-v1** and ~10 × over naïve PyTorch, hitting ≈ **73 % of A100 peak FLOPs** and 72 % model-level FLOPs utilisation in GPT training.

## 2. What is new compared to prior work?

- Defers final soft-max rescaling and stores only `log-sum-exp` per row ⇒ far fewer scalar (non-matmul) FLOPs.

- In addition to *batch × heads*, thread-blocks now **split the sequence-length axis**, raising SM occupancy for long-sequence / small-batch regimes.

- Replaces **"split-K"** (each warp slices `K/V`) with **"split-Q"** (each warp slices `Q`) ⇒ almost zero inter-warp communication and shared-memory traffic.
d

## 3. What experiments were run to support the arguments in this paper?

- Measured forward, backward, and forward+backward TFLOPs / s across `L ∈ {512…16 k}`, head dims `{64, 128}`, with/without causal mask.

- Compared against PyTorch standard, xFormers-cutlass, FlashAttention-v1 CUDA, and FlashAttention-Triton.

- End-to-end training of GPT-style models (1.3 B & 2.7 B params, seq-len 2 k & 8 k) on 8× A100

- Same kernels **run unmodified on H100**, showing further raw-throughput gains (up to ≈ 335 TFLOPs/s).

## 4. What are the shortcomings/limitations of this paper?

- Current kernels target NVIDIA architectures; AMD/Intel GPUs and TPUs are not yet supported.

- Four hand-chosen tile shapes; no auto-tuner provided.

- No exploitation of H100-specific features (TMA, 4-gen Tensor-Cores, FP8) in the released code.

- Focuses on dense attention; does not address algorithmic sparsity / locality that could extend context lengths

## 5. What is a reasonable next step to build upon this paper?

- Port kernels to H100 with TMA/FP8, AMD ROCm, and other accelerators

- Embed FlashAttention-2 in **TVM/Triton autotuners** so optimal block sizes and warp layouts are discovered automatically.

- Fuse FlashAttention-2 kernels with block-sparse patterns to reach 100 k+ token context at high efficiency.

- Evaluate FP8/BF8 or hybrid **quantisation** to trim memory bandwidth further without losing accuracy.

- Integrate into vision, speech and multimodal Transformers to measure end-to-end gains beyond language modelling.

## Appendix

- **TMA (Tensor Memory Accelerator)**: A new Hopper-generation hardware path that streams tiles between HBM and registers/SRAM without explicit loads, freeing CUDA cores and reducing latency.
- **4-gen Tensor Cores**: The fourth-generation NVIDIA matrix-multiply units in H100 GPUs that add FP8/BF16 support and higher per-cycle throughput compared with A100’s third-generation units.
