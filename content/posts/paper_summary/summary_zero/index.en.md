---
title: "Summary: ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
date: 2025-04-02T15:06:56+08:00
categories: ["paper_summary"]
summary: "Summary for paper 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models'"
---

## Download the Paper

[Paper](https://arxiv.org/pdf/1910.02054)

> Note: Also read [distrubuted_training_strategy](../../distribution_training_strategy/) if you are interested in this topic.

## 1. What is the paper about?

- It introduces **ZeRO (Zero Redundancy Optimizer)**, a method to optimize memory usage for extremely large-scale NN training (potentially up to trillions of parameters).

- ZeRO-DP splits the **optimizer states**, **gradients**, and **parameters** across different devices, thereby reducing redundant memory usage and enabling larger batch sizes and faster training throughput.

- It also proposes ZeRO-R to handle **activation memory**, **temporary buffers**, and **memory fragmentation**, aiming to further reduce memory footprints while retaining efficient computation.

## 2. What is new about this specific paper, compared to prior work?

- Unlike **traditional DP** or **TP (tensor parallel)** (which either replicate the model on all devices or split layers vertically), ZeRO partitions the model’s states and only re-materializes them when needed.

- ZeRO’s approach retains the simplicity and low communication overhead of DP while drastically reducing memory overhead—something not achieved by prior TP or **pipeline-parallel (PP)** techniques.

- It demonstrates analytically and experimentally that ZeRO could feasibly train a trillion-parameter model by leveraging enough GPUs and employing its partitioning strategy.

- ZeRO does not require complicated model refactoring, whereas prior solutions, such as **Megatron-LM** or **G-Pipe**, often need significant changes to the model architecture or training loop.

## 3. What experiments were run to support the arguments in this paper?

- The authors trained GPT-2–style transformer models ranging from 1.5B parameters up to 170B parameters on hundreds of V100 GPUs.

- They compared ZeRO against baseline DP (**PyTorch DDP**) and a SOTA TP system (Megatron-LM) to show throughput (TFLOPS) improvements and memory savings.

- Experiments scaling a 60B model from 64 to 400 GPUs demonstrated **"super-linear"** speedups when increasing the DP degree, due to higher permissible batch sizes per GPU.

- Detailed measurements of GPU memory usage were provided, showing how partitioning optimizer states, gradients, and parameters significantly reduces per-GPU requirements.

- The authors trained a 17B-parameter language model (Turing-NLG) that achieved a new SOTA (Webtext-103 perplexity of 10.21), illustrating real-world applicability.

## 4. What are the shortcomings/limitations of this paper?

- Although ZeRO can fit a trillion-parameter model in terms of memory, training it end-to-end would still take an impractically long time on today’s (2020) hardware (potentially months or more).

- While it claims only a 1.5× communication overhead relative to standard DP using stage 3, that cost can be non-trivial in scenarios with limited interconnect bandwidth.

- Properly tuning activation checkpoint partitioning (e.g., deciding when to offload to CPU) may require domain-specific heuristics and can introduce additional overhead if not carefully managed.

- Achieving the advertised efficiency sometimes requires many GPUs with specific high-bandwidth interconnects (e.g., NVSwitch within a node, high-speed inter-node links).

- Although transformers are a dominant architecture, the paper does not deeply explore how ZeRO might handle other model types with different memory patterns.

## 5. What is a reasonable next step to build upon this paper?

- Investigate more strategies for dynamically offloading activations and model states, taking into account **heterogeneous memory tiers** (e.g., GPU HBM, CPU DRAM, NVMe).

- Develop a system that **automatically** selects the best partitioning schedule or CPU-offload policy based on real-time memory usage, communication bandwidth, and arithmetic intensity.

- Extend ZeRO’s partitioning approach and memory optimizations to convolution-based architectures, graph NNs, and emerging large-scale models.

- As compute clusters grow, explore how ZeRO's techniques scale in exascale HPC environments with tens of thousands of GPUs, potentially refining communication collectives.

- Adapt ZeRO for **downstream tasks** that require partial model updates or parameter-efficient methods (e.g., LoRA, adapters), ensuring that memory is minimized for both pre-training and fine-tuning stages.

## Appendix

- **GPU HBM (High-Bandwidth Memory)**: A type of high-speed, on-package memory used by modern GPUs, providing very high bandwidth and low power consumption compared to traditional GDDR memory.
- **NVMe (Non-Volatile Memory Express)**: A high-performance storage interface protocol for solid-state drives, designed to reduce latency and improve input/output (I/O) operations
- **NVSwitch**: A fully-connected, high-speed switch architecture by NVIDIA that allows GPUs in the same server (e.g., DGX-2) to communicate at very high bandwidth and low latency.
- **Infiniband EDR (Enhanced Data Rate)**: A network interconnect technology providing high bandwidth and low latency, often used for HPC clusters and GPU communication across nodes.
- **Intra-Node** vs. **Inter-Node**: Intra-node refers to devices (e.g., GPUs) within the same physical machine; inter-node refers to devices across different physical machines in a cluster.
- DGX-2: An NVIDIA system that packs up to 16 V100 or A100 GPUs with NVSwitch for high-bandwidth, all-to-all GPU communication.
