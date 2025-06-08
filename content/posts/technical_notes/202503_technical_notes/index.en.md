---
title: "2025 Technical Notes（3）"
date: 2025-05-23T13:44:56+08:00
categories: ["technical_notes"]
summary: "Technical notes during 2025 (3)."
---

>The content in this page here is translated by O3.

## Fundamentals

**Sequence Parallel** linearly reduces the memory footprint of the logits (the final full-vocabulary probability distribution), so Sequence Parallel is useful.

**NVLink** is a high-speed point-to-point interconnect that lets GPUs communicate directly (sharing a single memory pool) and offers far more bandwidth than PCIe. **NVSwitch** is a dedicated NVLink-switch ASIC that builds a fully connected, non-blocking topology for many GPUs. Wiring multiple NVLink GPUs into a fabric network solves communication-efficiency issues.

---

PPO (Proximal Policy Optimization) is essentially an RL algorithm that trains an agent to take the optimal action for each state. It mainly evaluates:

1. **State Value Function** — the expected cumulative reward (return value) of the current policy for a given state  
2. **Advantage Function** — how much better taking a certain action is compared with the **average** of all possible actions  
3. **Policy Distribution** — the probability of each action given the current state  

---

### PyTorch Reduce-Scatter operator

*1-D tensor:*

```py
dst   = torch.zeros(5, dtype=torch.float32)
index = torch.tensor([0, 1, 1, 2, 3])
src   = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
result_pytorch = dst.scatter_add(0, index, src)
print(result_pytorch)   # tensor([1., 5., 4., 5., 0.]) — easy to understand, just add to the positions
````

*2-D tensor (dim = 0):*

```py
# Elements are added to dst[target_row, same_col]
dst   = torch.zeros(3, 4, dtype=torch.float32)
index = torch.tensor([[0, 1, 2, 0],
                      [1, 2, 0, 1]])
src   = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8]], dtype=torch.float32)
result_pytorch = dst.scatter_add(0, index, src)
print(result_pytorch)
# tensor([[1., 0., 7., 4.],
#         [5., 2., 0., 8.],
#         [0., 6., 3., 0.]])
```

*2-D tensor (dim = 1):*

```py
# Elements are added to dst[same_row, target_col]
dst   = torch.zeros(3, 4, dtype=torch.float32)
index = torch.tensor([[0, 1, 2, 3],
                      [3, 2, 1, 0]])
src   = torch.tensor([[1, 2, 3, 4],
                      [5, 6, 7, 8]], dtype=torch.float32)
result_pytorch = dst.scatter_add(1, index, src)
print(result_pytorch)
# tensor([[1., 2., 3., 4.],
#         [8., 7., 6., 5.],
#         [0., 0., 0., 0.]])
```

A very useful application of `scatter_add` is **batched bin counting**, handy in NLP for counting token frequencies, for example:

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
# tensor([[1, 1, 1, 1, 0],
#         [1, 0, 2, 1, 0]])
```

## TVM

**Operator-level optimisation**: a key idea is to separate *compute* (e.g., multiplication) from the *schedule* (e.g., tiled looping). Even simple schedule tuning can reach 60 % of MKL’s performance; hitting 100 % takes more time, so TVM also supports ML-based schedule search (AutoTVM) that fills templates and auto-generates code.

There is also **graph-level optimisation** with two IR layers: **Relay** represents the NN computation graph, and **TIR** (Tensor IR) is the lower-level, hardware-oriented IR for each operator. A classic graph optimisation is operator fusion.

**Summary:** Convert the model to Relay → apply graph fusion → for every fused layer define a search space and run AutoTVM to generate hardware-specific binaries → test and deploy.

## NLP

### Basic concepts

* **Prefill**: process all input tokens, compute QKV and attention scores, and cache KV (compute-intensive).
* **Decode**: generate one token at a time until the stop condition (memory-intensive).

Modern practice separates prefill and decode or splits prefill into chunks to improve utilisation.

---

**Speculative Decoding** accelerates LLM inference while preserving quality.

Core idea: a small, fast **draft model** guesses a sequence of tokens; a larger, more accurate **target model** validates them. If the guess is correct, multiple tokens can be accepted in parallel instead of one by one, saving time.

Example: context “artificial intelligence”. The small model predicts “is changing our lives”. The large model’s correct answer is “is changing human life”. It validates “is changing human”, so four tokens “is changing” are accepted at once while the remainder “our” is discarded. From the first mismatch onward everything is dropped.

Speculative decoding can cut compute significantly and yield several-fold speed-ups (parallel acceptance).

---

**BERT**: `[CLS]` is placed at the very start to aggregate the whole sequence, especially for classification tasks. During fine-tuning for classification, `[CLS]` is usually taken as the sequence representation. `[SEP]` separates sentences or segments. Next-sentence prediction is a classification task mainly using `[CLS]`.

---

**NLL loss** vs. **cross-entropy**: With one-hot input labels, cross-entropy degenerates to NLL (only one term appears in the sum).

`NLL = −log p_θ(w* | context)`. We want the predicted `w*` to match the masked word. The higher `p(w*)` (closer to 1), the closer the loss gets to −0; with the minus sign it approaches +0.

---

**BLEU score**: widely used to evaluate machine translation and correlates well with human ratings.

1. Check n-gram overlap — matching longer phrases indicates better translation.
2. For each n-gram, compute
   `score = matched n-grams / total n-grams in candidate` (penalises overly long output).
3. **BP** (brevity penalty) punishes overly short output:
   `BP = 1` if the candidate is longer; `BP = e^(1 - ref_len / cand_len)` if the candidate is shorter.
4. `BLEU = BP × exp(weighted ln-average of four n-gram precisions)`

### Transformer time & space complexity

Matrix-multiply complexity:

For `A(M,N) @ B(N,P) → C(M,P)`, time complexity is `O(M·N·P)`. Space is `O(1)` for temporaries (or `O(MN + NP + MP)` counting all matrices).

**Attention (training):**

Input shape `(N, D)`:

1. QKV projection: `QKV = X @ W_qkv(D,D) → (N,D)` → `O(N·D²)`
2. Attention scores: `Q(N,D) @ Kᵀ(D,N) → (N,N)` → `O(N²·D)`
3. Weighted values: `(N,N) @ V(N,D) → (N,D)` → `O(N²·D)`

Overall: `O(N²·D)` (often simplified to `O(N²)`).

Memory: store Q, K, V, activations → `O(N² + 3·N·D)` → \~`O(N²)`.

**Attention (inference, token-by-token):**

* Token 1: `O(D²)` (QKV projection), cache KV.
* Token 2: `O(D² + 1·D)` (`(1,D) @ (D,1)` \~ `O(D)`).
* Token 3: `O(D² + 2·D)` …
* Token N: `O(D² + (N-1)·D)`.

Summing: `O(N·D² + D·N(N-1)/2) ≈ O(N²·D) ≈ O(N²)`.

Memory: KV cache grows linearly, `O(N)`.

### Loss computation example

Sentence: “I love deep learning”

Vocabulary `V = {"I": 0, "love": 1, "deep": 2, "learning": 3, "and": 4}`

Input sequence `input_t = [0, 1, 2]` (`"I love deep"`)
Target sequence `target_t = [1, 2, 3]` (`"love deep learning"`)

Model outputs logits, softmax to probabilities:

* After “I”: `[0.05, 0.55, 0.25, 0.10, 0.05]`
* After “I love”: `[0.08, 0.09, 0.55, 0.18, 0.10]`
* After “I love deep”: `[0.07, 0.08, 0.10, 0.60, 0.15]`

`loss = (−log 0.55 − log 0.55 − log 0.60) / 3 ≈ (0.60 + 0.60 + 0.51) / 3 ≈ 0.57`.

### Expert Parallel

Expert Parallel (MoE) places full expert networks on different devices. Unlike tensor parallelism, it does **not** split a single expert’s weights; it distributes whole experts across devices (basic EP).

If the GPU count exceeds the expert count, experts are sliced (**expert-slicing**) so that GPUs vertically/horizontally cut each expert. After processing their assigned tokens, experts exchange results via an all-to-all operation.

### Padding for parallel inference

1. **Pad to max**: pad every sequence in a batch to the longest length — wastes resources.
2. **Bucketing**: group requests with similar length — algorithmically complex and may incur waiting.
3. **Token-Level Scheduling**: refresh the batch at token granularity, leveraging paged-attention KV cache to locate context quickly. Instead of calling inference for each sequence separately, the final token of every sequence in the batch is processed together (`shape [B, 1]`).

Besides the last token, the KV-cache block location is also passed into the kernel.

Supplement: **Selective Batching** (ORCA)

Concatenate each sequence in a batch into one long sequence, then split into:

* **Attention path**: run attention per sequence.
* **Non-attention path**: run linear layers once for the entire concatenated sequence.

Merge the two paths and feed the result into the next layer.

## System-Design Example

### Global System Design

**Key question:** From a global perspective—across many instances of the inference engine—how should a unified orchestrator route and load-balance LoRA requests to minimise both TTFT and TPOT?

### 1 · Scheduling goals

1. **High hit-rate**: route a request to a node that already hosts the LoRA.
2. **Balanced load**: no hot LoRA overloads one machine; cold LoRA doesn’t wait forever.
3. **Minimal migration** during scale-up/down.
4. **Simple and practical** implementation.

### 2 · Core approach

Two-tier routing **+** two background threads.

#### 2.1 Consistent Hash with Bounded-Load (CH-BL)

*Consistent Hashing* places each server’s hash on a ring; a request hashes to a point, then moves clockwise to the first server. Adding/removing a node only affects its adjacent arc.

**CH-BL** adds a quota—each node may hold at most `(1 + ε) × average` keys. If the first node is “full”, the request moves to the next. Theory shows that with a small ε you typically find a host in one or two extra steps, so lookup is still `O(1)`.

**Implementation:** use `(adapter-id, base-model-id)` as the key and CH-BL for load-balancing. Including `base-model-id` matters because a LoRA only makes sense with its specific model.

#### 2.2 Power-of-Two Choices (P2C)

With CH-BL you naturally get replicas—keep moving until you meet `k` replicas. Among them choose two at random, compute

```bash
score = queue_len × α + gpu_util × β
```

Pick the lower score. P2C is `O(1)` fast and performs well in practice.

#### 2.3 Background thread #1: dynamic replicate/evict

Every 30 s check each LoRA’s hotness. If hotter, pick `k_new − k_old` idle nodes, prefetch the weights, and add them as replicas. If colder, mark as draining; after its queue empties, unload it.

#### 2.4 Background thread #2: version consistency

If a LoRA’s revision hash on the hub changes, push the new version to every node with the old hash.

The instance marks the old ΔW as **stale** and hot-swaps during idle GPU slices. The router keeps accepting requests for the old hash until every replica finishes swapping, ensuring **zero downtime**.
