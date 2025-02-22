---
title: "2025 01 Technical Notes"
date: 2025-01-31T14:02:12+08:00
categories: ["technical_notes"]
summary: "Technical notes during 2025 Jan."
---

>The content in this page here is translated by O3-mini-high.

## Robotic

### Reinforcement Learning (RL)

**Reinforcement Learning (RL)** is a branch of machine learning where agents learn to take actions in an environment to maximize long-term rewards. One popular algorithm is **Proximal Policy Optimization (PPO)**, a deep RL method based on policy optimization.

#### Value-Based Methods

These algorithms evaluate the quality of each action and select the best one to improve the policy. For example, in a maze scenario, if moving left has a high value while moving right leads into a trap, an algorithm like **Q-Learning**—which records the value for every action at each state—will choose the optimal move. Although simple and effective for small-scale discrete problems, value-based methods can suffer from greedy choices that may miss the global optimum, and they struggle with continuous action spaces (e.g., precise joint angles in robots).

*Note:* This approach differs from imitation learning; it does not require expert demonstrations but learns solely through interactions with the environment.

#### Policy-Based Methods

In contrast, policy-based methods directly parameterize the policy. The agent executes actions to receive rewards, aiming to maximize the long-term return. In robotics, pure RL can be both costly and risky—imagine programming a robot with the idea “I tend to avoid obstacles” or “I prefer to open treasure chests.” Policy-based methods are well-suited for continuous actions and offer greater flexibility, though they often suffer from lower sample efficiency and high variance during training.

A popular policy-based algorithm is **PPO**. In practice, the **Actor-Critic** framework is often used, where the critic evaluates the actions and the actor generates them. Furthermore, pretraining is crucial in RL—starting from scratch can be prohibitively time-consuming and even dangerous (especially for robots). Thus, using offline or unsupervised pretraining can significantly accelerate the learning process.

### Imitation Learning (IL)

**Imitation Learning (IL)**—or learning from demonstrations—involves mimicking expert data. This approach is particularly useful in domains such as robotic arm manipulation, service robotics, and autonomous driving. However, pure imitation learning often lacks generalization capabilities.

In practice, a common strategy is to first pretrain using expert demonstrations (e.g., via **Behavior Cloning**, where a network is trained in a supervised fashion to predict actions from states) and then fine-tune using reinforcement learning or even combine both strategies concurrently.

### Reinforcement Learning from Human Feedback (RLHF)

**RLHF** extends beyond robotics into broader deep learning applications. Its typical workflow is:

1. **Supervised Pretraining:** Initialize the model using supervised learning.
2. **Collect Human Feedback:** Gather feedback data where humans rate model outputs.
3. **Train a Reward Model:** Use the human scores as targets for a reward model that takes the LLM’s output and produces a reward.
4. **RL Fine-Tuning:** Utilize algorithms like PPO, with the reward model acting as the reward function, to fine-tune the LLM.
5. **Iterate:** Continuously collect data and refine the model.

---

## Linear Algebra

**Matrix Eigenvalues:** An eigenvalue is a scaling factor by which a matrix stretches or compresses a vector in the direction of its corresponding eigenvector. For instance, consider the matrix  
\( A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix} \).  
Its eigenvectors are \([1, 0]\) and \([0, 1]\), with eigenvalues 2 (scaling along the x-axis) and 3 (scaling along the y-axis), respectively.

**Quadratic Equation:**  
The discriminant is given by:  
`discriminant = b^2 - 4ac`  
and the solutions for a quadratic equation are:  
`x = (-b ± √discriminant) / (2a)`

For a 2×2 matrix  
\( A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \),  
its determinant is `det(A) = ad - bc` and the trace is `tr(A) = a + d`.  
The eigenvalues are found by solving:  
\[ \det(A - \lambda I) = 0 \]
which expands to:  
\[ \lambda^2 - (a+d)\lambda + (ad-bc) = 0 \]

---

## Distributed Training

### Strategy

#### Traditional Data Parallelism (DP)

- **Advantages:** Simple to implement and widely supported by community frameworks.
- **Disadvantages:** Each device maintains a full copy of the model weights and optimizer states.

This is where **Zero Redundancy Optimizer (ZeRO)** comes into play—by sharding only the optimizer state (Stage 1 of ZeRO), the upper layers still see a familiar DP setup, but the heavy optimizer state is partitioned, reducing memory overhead. The downside is that this requires framework support (e.g., DeepSpeed).

#### Tensor Parallelism

- **Advantages:** Increases throughput for individual operators.
- **Disadvantages:** Requires modifications to the model (although multi-head attention naturally lends itself to splitting) and incurs high communication costs.

#### Pipeline Parallelism (PP)

- **Advantages:** Saves memory.
- **Disadvantages:** May introduce pipeline bubbles and is often more challenging to debug.

For example, using **DeepSpeed**:

```python
model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                     model=model,
                                                     model_parameters=params)
# Behind the scenes, torch.distributed.init_process_group(...) is replaced by deepspeed.init_distributed()
```

DeepSpeed defaults to the NCCL backend and allows you to write training loops as usual:

```python
for step, batch in enumerate(data_loader):
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

When scaling to multiple nodes, DeepSpeed typically leverages Open MPI for high-performance message passing.

### RDMA

**Remote Direct Memory Access (RDMA)** enables direct memory access from the memory of one computer into that of another without involving the CPU, which significantly reduces latency. Common RDMA technologies include:

- **InfiniBand (IB)**
- **RoCE (RDMA over Converged Ethernet)**

Within a single server, high-speed interconnects like NVLink are used between GPUs, while RDMA (e.g., GPUdirect RDMA) is preferred for multi-node communication.

---

## Deep Learning

### Fundamentals

#### Pruning

Pruning reduces redundant weights (or even entire layers) in a network:

- **Weight Magnitude-Based:** Small weights are pruned.
- **Gradient Magnitude-Based:** Weights with minimal gradients (i.e., minimal impact on the loss) are pruned.
- **Activation-Based:** Low activations indicate little contribution and may be pruned.

After pruning (where pruned values are set to zero, making matrices sparse), specialized optimizations can reduce storage and computation. Since pruning can degrade performance, fine-tuning is often required in iterative cycles.

#### Distillation

Knowledge distillation involves transferring knowledge from a larger “teacher” model to a smaller “student” model using soft targets.

#### ONNX Runtime

ONNX Runtime supports multiple backends (CPU, GPU, TensorRT) and unifies different model formats for streamlined inference.

#### Object Detection Paradigms

- **RCNN:** Uses selective search to propose candidate regions, then classifies them.
- **Fast-RCNN:** Scans the entire image to generate feature maps and then extracts candidate boxes.
- **Faster-RCNN:** Uses neural networks to generate proposals directly.

For 3D detection, sensor fusion (e.g., combining camera data with LiDAR) is used. Although methods like NeRF focus on 3D reconstruction rather than bounding box detection, they serve as a foundation for later stages.

#### Finetuning

Finetuning involves freezing certain layers of a pretrained model and training the remaining layers on new data to adapt to specific tasks.

#### Loss Functions

- **Cross-Entropy Loss:**  
  \[
  \text{Loss} = -\sum_i y_i \cdot \log(y_{\text{softmax}, i})
  \]
  For example, if `y = [0, 1, 0]` and `y_softmax = [0.1, 0.8, 0.1]`, the loss is approximately 0.223.
  
- **Binary Cross-Entropy (BCE):**  
  \[
  \text{Loss} = -\left[y_i \cdot \log(y_p) + (1 - y_i) \cdot \log(1 - y_p)\right]
  \]
  where \(y_p\) is the probability output after a sigmoid activation.

#### Low-Rank Decomposition

Low-rank decomposition approximates weight matrices by factorizing them (e.g., via Singular Value Decomposition), reducing both storage and computational costs.

#### Out-Operator Optimization

The “out operator” can improve speed by:

- Reducing runtime malloc calls.
- Minimizing extra copies of data.

While PyTorch’s memory management minimizes malloc overhead, pre-allocating memory for out operators—especially during communication—can lead to significant performance gains. Note that CUDA malloc calls remain expensive.

### Quantization (TensorRT)

Quantization reduces model precision to speed up inference and reduce memory usage. There are two main approaches:

#### Post-Training Quantization (PTQ)

1. **Export the Model:** Convert the trained model to ONNX format. Ensure that all operators are supported (or implement custom plugins for unsupported ones, such as flash attention or RMSNorm in LLMs).
2. **Build the TensorRT Engine:** Serialize the engine into a `.plan` file or load it directly for inference.
3. **Calibration:** Because int8 values range from -128 to 127 while floating-point values can be much larger, calibration is needed. A scale factor is computed so that:
   \[
   \text{int8\_value} = \text{round}\left(\frac{\text{float\_value}}{\text{scale}}\right) + \text{zero\_point}
   \]
   Calibration methods include:
   - **MinMax Calibration:** Uses the minimum and maximum values, though it is sensitive to outliers.
   - **Entropy Calibration:** Minimizes KL divergence between the original and quantized distributions.
   - **Percentile Strategy:** Discards extreme values (e.g., the 99.9th percentile) to set calibration bounds.

#### Quantization-Aware Training (QAT)

QAT integrates quantization into the training process by simulating quantization error during forward passes while using techniques like fake quantization and the Straight-Through Estimator (STE) for backpropagation. The typical workflow is:

1. **Fake Quantization:** Simulate quantization by:
   - **Quantize:**  
     \[
     q = \text{clip}\left(\text{round}\left(\frac{x}{S}\right) + \text{zero\_point}, -128, 127\right)
     \]
     (The output remains in float format.)
   - **Dequantize:**  
     \[
     x_{\text{new}} = (q - \text{zero\_point}) \times S
     \]
2. **STE:** During backpropagation, non-differentiable operations like rounding are assumed to have a derivative of 1.

For instance, an original activation  
`x = [0.45, -1.23, 2.99, 0.02]`  
might be quantized (fakely) to `[5, -12, 30, 0]` and then dequantized to approximately `[0.5, -1.2, 3.0, 0.0]`, thereby simulating the error introduced by quantization.

### Common Issues

**Gradient Vanishing:**  
Common causes include:

- Excessive network depth, which can be mitigated by using proper network designs and residual connections.
- Activation function saturation (e.g., ReLU zeroing negatives or sigmoid saturating for extreme values).  
Mitigation strategies:
- Use alternative activations such as Leaky ReLU.
- Apply normalization techniques (e.g., BatchNorm).
- Employ appropriate weight initialization methods:
  - **Xavier Initialization:** Suitable for sigmoid/tanh, adjusts the initial range based on neuron count.
  - **He Initialization:** Better for ReLU networks due to its consideration of the non-linear activation’s behavior.

**Handling Unlabeled Data:**  
For scenarios like determining whether a video relates to restaurants:

- Use simple rule-based methods to extract weak labels (keywords like “restaurant”, “menu”, “food”).
- Extract features:
  - Use a pretrained classification model on video frames to detect food, menus, or dining scenes.
  - Use multimodal models (e.g., CLIP) to extract embeddings from video frames and evaluate restaurant-related scores.
- Alternatively, use unsupervised clustering to group data based on frame content and audio, then manually label clusters.  
Once initial pseudo-labels are generated, a preliminary model can be trained and iteratively improved via pseudo-labeling.

#### LogSoftmax Forward and Backward

Here is an example implementation using NumPy:

```python
import numpy as np

class LogSoftmax:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2, "x should have shape (batch_size, in_dim)"
        # log_softmax = x - log(sum(exp(x))) = x - (m + log(sum(exp(x-m))))
        # where m = max(x, axis=1, keepdims=True)
        max_val = np.max(x, axis=1, keepdims=True)
        self.exp = np.exp(x - max_val)
        self.sum_exp = np.sum(self.exp, axis=1, keepdims=True)
        log_value = np.log(self.sum_exp)
        return x - (max_val + log_value)

    def bprop(self, x: np.ndarray, dedy: np.ndarray) -> np.ndarray:
        softmax = self.exp / self.sum_exp
        sum_dedy = np.sum(dedy, axis=1, keepdims=True)
        dedx = dedy - softmax * sum_dedy
        return dedx
```

---

## Computer Vision (CV)

### Diffusion Models

#### Denoising Diffusion Probabilistic Models (DDPM)

Introduced in 2020, DDPMs gradually add noise in a forward process and train a neural network to reverse this process using an MSE loss.  

- **Forward Process:**  
  \[
  X_t = \sqrt{1-\beta_t}\, X_{t-1} + \sqrt{\beta_t}\, \epsilon
  \]
- **Reverse Process:**  
  \(X_{t-1}\) is approximated as \(X_t - \sqrt{\beta_t}\,\epsilon_0\) with additional random compensation.  
*Note:* The noise level is controlled by a schedule \(\beta_t\) (which increases linearly in DDPM). A downside is that reverse sampling can require hundreds of steps, making it slow.

#### Denoising Diffusion Implicit Models (DDIM)

Proposed in 2021, DDIM introduces deterministic reverse sampling:

- It removes the randomness inherent in DDPM.
- By skipping steps (e.g., reducing from 1000 to 50–100 steps), it greatly accelerates inference while maintaining image quality.
Deterministic reverse sampling is achieved by formulating the process as an implicit ordinary differential equation (ODE).

#### Latent Diffusion Models (LDM)

LDMs further optimize the diffusion process by shifting it from high-dimensional image space to a lower-dimensional latent space:

1. Use a pretrained Variational Autoencoder (VAE) to compress images.
2. Perform the diffusion process in latent space, significantly reducing computational costs.
3. Introduce conditioning (e.g., semantic maps, text, or images) during reverse denoising.
4. Finally, use a decoder to reconstruct the image.
While this method is faster, it is dependent on the quality of the VAE, which may result in some loss of fine details.

#### Stable Diffusion

Stable Diffusion is a specific implementation of latent diffusion models developed by Stability AI:

- It leverages **CLIP (Contrastive Language–Image Pre-training)** to extract text conditions that guide image generation, enabling strong multimodal capabilities.
- CLIP aligns images and captions into the same embedding space, allowing for controlled and semantically relevant generation.

---

## Natural Language Processing (NLP)

### Fundamentals2

1. **GPT (Generative Pre-trained Transformer):**  
   GPT models predict the next token in a sequence (left-to-right) using only the transformer’s decoder, making them well-suited for text generation. The pretraining objective is to model the probability of the next word given previous words.

2. **BERT (Bidirectional Encoder Representations from Transformers):**  
   BERT employs a bidirectional encoder, which allows it to consider both left and right context simultaneously. This makes it ideal for understanding tasks. Pretraining involves:
   - **Masked Language Modeling (MLM):** Randomly mask tokens and predict them.
   - **Next Sentence Prediction (NSP):** Determine if two sentences are contiguous.

3. **T5 (Text-to-Text Transfer Transformer):**  
   T5 uses both an encoder and a decoder, converting every task into a text-to-text format. Its pretraining task—**Span Corruption**—involves masking spans of text and training the model to recover them.

### LLM Operators

1. **Fused Multi-Head Attention:**  
   This operator fuses operations such as computing  
   \[
   \text{Softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \times V
   \]
   into a single kernel call. Here, Q, K, and V have shapes \((N, d)\) (with \(N\) being the sequence length). In multi-head attention, \(d\) is divided by the number of heads.

2. **Flash Attention:**  
   Traditional attention mechanisms require storing an \(N \times N\) activation matrix, leading to quadratic memory usage. Flash Attention optimizes this by:
   - Dividing Q, K, and V into smaller blocks (e.g., a block size of 128).
   - Computing local matrix multiplications and softmax operations within each block.
   - Fusing these operations into a single GPU kernel, reducing memory usage from \(O(N^2)\) to \(O(N \times d)\).  
   For example, if Q, K, and V are of shape (4, 2), the standard approach computes a (4, 4) matrix, whereas flash attention splits them into (2, 2) blocks, processing them in chunks and combining the results—using techniques like the max-shift trick to maintain numerical stability.

3. **RMSNorm:**  
   RMSNorm is a lightweight alternative to LayerNorm that omits mean subtraction. It is defined as:  
   \[
   \text{RMSNorm}(x) = \gamma \times \frac{x}{\text{RMS}(x)}
   \]
   where  
   \[
   \text{RMS}(x) = \sqrt{\frac{1}{d} \sum x^2}
   \]
   and \(\gamma\) is a learnable scaling parameter.

4. **Masked Softmax:**  
   This operator integrates a mask into the softmax computation by adding a large negative value to the positions to be masked. It is particularly useful in attention mechanisms to prevent attending to future tokens (e.g., in GPT models).

5. **Rotary Position Embedding (RoPE):**  
   Instead of fixed or learnable positional encodings, RoPE incorporates positional information using rotation matrices. In 2D, given a vector \(x = (x_1, x_2)\) and a rotation angle \(\theta\), the transformation is:
   \[
   \begin{pmatrix}
   \cos \theta & -\sin \theta \\
   \sin \theta & \cos \theta
   \end{pmatrix} x
   \]
   In high-dimensional space, the vector is split into \(d/2\) pairs (each forming a 2D plane), and each pair is rotated by an angle \(\theta\) (which may be a linear or logarithmic function of the position). Notably, the property \(\cos(a)\cos(b) + \sin(a)\sin(b) = \cos(a-b)\) helps the model learn relative positional information.

### Transformer Architecture

The transformer is typically structured as follows:

- **Encoder:**

```bash
(words → embedding) + positional encoding →
[Multihead self-attention → Norm → Feedforward Network (with ReLU) + Residual Connection → Norm] repeated N times
```

- **Decoder:**

```bash
(words → embedding) + positional encoding →
[Masked Multihead self-attention (masking future tokens) + Norm → Cross-attention + Norm → Feedforward Network → Norm] repeated N times → Linear → Softmax
```

**Why Self-Attention?**  

- RNNs process tokens sequentially and cannot be fully parallelized.
- CNNs can parallelize at the kernel level but require either many layers or large kernels to capture long-range dependencies, leading to high computational complexity.
- Self-attention enables direct interactions between all positions in a sequence and is highly parallelizable, especially when combined with multi-head attention.

#### LLama2 70B Training Example

For a 70-billion parameter model:

- **Dataset:** Approximately 2 trillion tokens (usually processed in one epoch).
- **Sequence Length:** Up to 4096 tokens.
- **Global Batch Size:**  
  \[
  \text{global\_batch\_size} = \text{micro\_batch\_size} \times \text{data parallel size} \times \text{pipeline parallel chunk size}
  \]
- One iteration processes:  
  \(\text{global\_batch\_size} \times \text{sequence length}\) tokens.
- **Training Example:**  
  With 210 TeraGS, 1024 GPUs, and 215,040 tokens/s, training takes roughly 108 days.  
  A sample configuration might be:  
  - DP (using ZeRO stage 1): 8  
  - TP: 16  
  - PP (chunk): 8  
  - Global batch size: 2048 → micro batch size: 16

---

## Recommendation Systems

### Job Recommendation in JobRight

For a job recommendation system (initially targeting the consumer side, with plans to expand to business clients), the primary metric is the application rate, as higher application counts correlate with better user retention.

1. **Recall Stage (10,000 items):**  
   - Highly dependent on the job title input by the user.
   - Use embedding-based search (e.g., “JavaScript” vs. “React Developer” should be recognized as similar).
   - Fine-tune data by categorizing jobs using a text taxonomy.
   - Leverage cosine similarity and additional filters.

2. **Coarse Ranking (Reduce from 10,000 to 500 items):**  
   - Apply hard rules (e.g., recency, title similarity).
   - Incorporate manually engineered scores (e.g., recent activity, popularity, publish time) to trim the list.
   - The recall scores can be used as an additional weighted factor.

3. **Fine Ranking:**  
   - Do not drop items; instead, use features from both the job (category, skills) and the user (past interactions, user attributes) to compute a ranking score.
   - The model is typically shallow and leverages cross-features (e.g., matching skills) using recommendation algorithms like GBDT or LightGBM.

4. **Re-Ranking:**  
   - Adjust rankings based on additional factors such as job freshness or company quality (e.g., ratings from Glassdoor).

This multi-stage system can respond within seconds.

### Open-Source Frameworks

#### Megatron

Developed by NVIDIA, Megatron is a framework for training large language models using PyTorch. It supports various transformer architectures (GPT, BERT, T5, etc.) and emphasizes:

- **Tensor Parallelism (TP)**
- **Pipeline Parallelism (PP)**
DeepSpeed can also be combined with Megatron (a common approach in the industry), and frameworks like ColossalAI offer alternatives with a more active community.

#### MLflow

An open-source lifecycle platform from Databricks that provides:

1. **ML Tracking:** Record and query experiments.
2. **MLflow Projects:** Standardized ML project descriptions for reproducibility.
3. **MLflow Models:** Universal model packaging.
4. **MLflow Model Registry:** Versioning and review management.

#### vLLM

vLLM is an inference system optimized for deploying large language models with high throughput and low latency, particularly in streaming scenarios. Innovations include:

- **PagedAttention:** Efficiently manages KV cache by partitioning keys/values into pages.
- Built-in batching to merge concurrent inference requests.

---

## System Design

### ML System Design Principles

When designing an ML system:

- **Define Requirements:** Clearly specify inputs, outputs, and non-functional requirements (latency, availability, scalability).
- **Document the Design:** Use text and diagrams to record the system’s architecture.
- **Three-Phase Approach:**
  1. **Data:** Data gathering, preprocessing, and (if needed) scaling.
  2. **Model:** Training (using techniques like k-fold validation), evaluation, and (if needed) scaling.
  3. **Deployment:** Managing different environments (dev, test, prod), online A/B testing, scaling, and optimization (e.g., quantization).
- **Feedback:** Continuously interact with stakeholders to refine the design.

### Inference Acceleration System

**Problem:** Design a machine learning inference acceleration system that guarantees low latency and high throughput, especially on resource-limited devices (e.g., mobile, edge devices, or overloaded servers).

**Inputs:**

- **Model:** Deep learning model (CNN, Transformer, recommendation model, etc.).
- **Input Data:** User behavior, image data, or text.
- **Compute Resources:** Available hardware (CPU, GPU, TPU, etc.).

**Outputs:**

- **Inference Results:** Predictions (e.g., recommended videos, image classifications, NLP responses).
- **Performance Metrics:** Latency, throughput (requests per second), model size, and accuracy.

**Design Considerations:**

- **Low Latency:**  
  - Compress the model using techniques like pruning (zeroing out unimportant weights based on magnitude, gradient, or activation) and post-training quantization (PTQ).
  - For PTQ, export the model to ONNX, ensure all operators are supported, and build a TensorRT engine with proper calibration (using min-max or entropy calibration).
- **High Throughput:**  
  - Scale horizontally using container orchestration tools like Kubernetes.
  - Implement caching (e.g., with Redis) for frequent requests.
  - Utilize message queues (like Kafka) and load balancers to distribute incoming requests efficiently.
  - For LLMs, consider frameworks like vLLM or DeepSpeed-Inference that optimize KV cache management through techniques such as PagedAttention.

**Hardware Acceleration:**

- **GPU Clusters:**  
  - For small, compressed models, a single GPU might suffice.
  - For larger models (e.g., LLMs), distribute inference using tensor and pipeline parallelism. NVLink can enhance GPU-to-GPU communication.
- **Alternative Accelerators:**  
  - TPUs or specialized inference accelerators may be used based on workload characteristics.
- **Supplementary Techniques:**  
  - Knowledge distillation can further reduce model size while maintaining performance.

**Scalability:**

- Use Kubernetes to containerize the service.
- Leverage Horizontal Pod Autoscaling (HPA) and Cluster Autoscaler to dynamically add nodes as demand increases.

**Handling Diverse Input Types:**

- **Time-Critical Requests:**  
  - Deploy dedicated servers for low-latency tasks.
- **Batch Processing:**  
  - For less urgent data, add a preprocessing layer (e.g., converting images or text to embeddings) and queue the data for inference.
- **Monitoring:**  
  - Use performance monitoring tools like Prometheus and conduct stress testing and A/B testing to validate system performance.

### Robot Gripper System

**Overall Design:**

- **Perception Layer:**  
  Incorporate visual sensors, torque sensors, and pose sensors.
- **Planning & Decision Layer:**  
  Process sensor data to perform object detection, path planning, and grasp strategy formulation.
- **Execution Layer:**  
  Control the robotic arm, joint motors, and gripper actuators. Note that since force is mutual, the gripper can read its own force without relying solely on external data.

**Key Data:**

- **Robotic Arm:** Angles, velocities, accelerations, and structural parameters (mass, DH parameters, etc.).
- **Environment and Object:** Data from visual and depth sensors regarding object classification, pose, state, and dimensions.
- **Gripper:** Opening range, torque feedback, and, if available, tactile or sliding sensor data.
- **Control & Safety:** Joint commands, execution results, joint limits, overload detection, and motor temperature monitoring.

*Summary:* The perception layer collects comprehensive data about the arm, environment, and gripper; the decision layer determines the optimal path and grasping strategy; and the execution layer issues commands to the actuators while continuously monitoring for safety and faults.

---

## CUDA

### CUTLASS

**CUTLASS** (CUDA Templates for Linear Algebra Subroutines) is NVIDIA’s C++ template library designed to simplify and accelerate general matrix multiplication (GEMM) operations. It provides reusable and composable components that are highly optimized for CUDA.

---

## Compiler Optimizations

**Loop Unrolling:**  
Loop unrolling can accelerate code execution by:

1. **Reducing Loop Overhead:** Minimizing jump instructions and loop control.
2. **Improving Branch Prediction:** Lessening the chance of branch mispredictions.
3. **Enhancing Instruction-Level Parallelism:** Allowing the compiler and CPU to better schedule and optimize instructions.

However, note that excessive unrolling can increase code size, potentially leading to instruction cache issues. Modern compilers often perform loop unrolling automatically, so manual intervention may yield only marginal improvements.
