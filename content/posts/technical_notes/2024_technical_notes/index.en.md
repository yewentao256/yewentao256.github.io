---
title: "2024 Technical Notes"
date: 2024-12-31T14:02:12+08:00
categories: ["technical_notes"]
summary: "Technical notes during 2024."
---

>The content in this page here is translated by O1.

## Fundamentals of Deep Learning

### Common Questions

**How to reduce overfitting?**

- Use dropout  
- Use normalization  
- Use regularization (add penalty term during training). Two approaches:  
  1. Add a penalty in the loss (e.g., \( \lambda \theta^2 \) for L2, or \(\lambda |\theta|\) for L1).  
  2. Directly apply weight decay (i.e., multiply weights by \(\lambda\)).  
- Increase the data size, use early stopping, etc.

### Zero

Why can’t Zero in half-precision (16-bit) train/ update parameters directly?  
Because gradient updates need high precision. Otherwise, small gradients may vanish.

### Activation

**Leaky ReLU**: Introduces a small negative slope so that it won’t produce zero derivatives.

\[
\text{LeakyReLU}(x) =
\begin{cases}
x, & x > 0 \\
\alpha x, & x \le 0
\end{cases}
\]
where \(\alpha\) is a small constant (e.g., 0.01).

### Backpropagation

#### Matrix Multiplication

Given \(\text{Out} = t_1 \times t_2\).

\[
\text{grad}_{t_1} = \text{grad\_output} \times t_2^T
\quad,\quad
\text{grad}_{t_2} = t_1^T \times \text{grad\_output}
\]

**Why?** One way to see it is to consider a small scalar case or do index-wise expansion.

#### Conv

By analogy, for convolution, the gradient w.r.t. the input is basically the “reverse convolution” of the gradients. You can derive it similarly by looking at small scalar examples.

### LoRA

LoRA updates the weight matrix without altering the original weights. The updated weight \(\mathbf{W}'\) can be expressed as:

\[
\mathbf{W}' = \mathbf{W} + \Delta \mathbf{W} = \mathbf{W} + \mathbf{A} \cdot \mathbf{B}
\]

where \(\mathbf{A}\) and \(\mathbf{B}\) are low-rank matrices. LoRA optimizes \(\mathbf{A}\) and \(\mathbf{B}\) while keeping the original \(\mathbf{W}\) fixed. It allows rapid fine-tuning on specific tasks without sacrificing the performance of the original model.

### Normalization

Why use normalization? It keeps each layer’s outputs in a relatively stable distribution (e.g., mean 0, variance 1), which helps the activation function. For example, if after BN we apply ReLU, normalizing helps avoid large numbers of negative values and speeds up convergence while mitigating overfitting.

Consider an input with shape **NCHW**:

- **Batch Norm (BN)**: normalizes across the batch dimension (N) plus \(\mathrm{HW}\). Subtract mean, divide by standard deviation.  
- **Layer Norm (LN)**: normalizes across a single sample’s feature dimension (C) along with \(\mathrm{H}\) and \(\mathrm{W}\).  
- **Group Norm (GN)**: splits channels into G groups, normalizes each group, then concatenates.  
- **Instance Norm (IN)**: normalizes only over **H** and **W** for each sample/channel.

### Optimizer

The fundamental update rule is:  
\[
\theta \mathrel{-}= \gamma \cdot G
\]

- **Momentum SGD**:  
  \[
  \theta \mathrel{-}= \gamma \cdot v,\quad
  v = \beta v_{t-1} + (1-\beta) G
  \]

- **Adam**:  
  \[
  \theta \mathrel{-}= \frac{\gamma \cdot m}{\sqrt{v} + \epsilon}
  \]  
  where  
  \[
  m = \beta_1 m_{t-1} + (1-\beta_1) G,\quad
  v = \beta_2 v_{t-1} + (1-\beta_2) G^2
  \]  
  There is also bias correction for \(m\) and \(v\) at the beginning:
  \[
  \hat{m} = \frac{m}{1-\beta_1^t},\quad
  \hat{v} = \frac{v}{1-\beta_2^t}
  \]

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

- **L2 Loss**: Mean Squared Error (MSE), \(\frac{1}{n}\sum (y_i - \hat{y})^2\)  
- **L1 Loss**: Mean Absolute Error (MAE), \(\frac{1}{n}\sum |y_i - \hat{y}|\)  
- **Huber Loss**: uses L2 loss for small errors, L1 loss for large errors.  
- **Binary Cross Entropy (BCE)**:  
  \[
  L = - \sum \Big[y_i \log (p_i) + (1-y_i) \log(1-p_i)\Big]
  \]

  Example:

  ```bash
  Sample    True Label(y)    Model Pred Prob(p)
  1             1                   0.9
  2             0                   0.2
  3             1                   0.6
  ```

  The loss would be \(-[\ln(0.9) + \ln(1 - 0.2) + \ln(0.6)] \approx 0.28\).

- **Multi-class Cross Entropy**:  
  \[
  L = - \sum y_i \log(p_i)
  \]
  Here \(p_i\) is the softmax probability of the \(i\)-th class.

In practice, for binary classification we usually combine a sigmoid with BCE, and for multi-class tasks we often combine softmax with cross entropy.

---

## NLP

### Perplexity

\[
\text{Perplexity} = \exp\Big(-\frac{1}{N}\sum \log P(w_i \mid w_1,\dots,w_{i-1})\Big)
\]

It measures the uncertainty in predicting the next token. A perfectly correct model would have a perplexity of 1. Higher perplexity indicates higher uncertainty.

### Transformer

**Essence of Attention**: “lookup” the context that best helps you predict the output.

**Transformer** is composed of an **Encoder** and a **Decoder**:

- The Encoder processes the input data and produces hidden states.
- The Decoder transforms the hidden states into the target sequence.

**Positional encoding**: uses sine or cosine functions (or an embedding) to encode position info, e.g., \(\sin(\frac{\text{pos}}{10000^{2i/d}})\). Another way is a learnable position embedding.

Within each Encoder/Decoder block, we have:

- Multi-head self-attention
- Feed Forward Network (two linear layers + activation like ReLU or GELU)
- Layer Normalization
- Residual connections

**Self-attention**: generate Query (Q), Key (K), and Value (V):
\[
Q = X W_q,\; K = X W_k,\; V = X W_v
\]
where \(X\) is input with shape \((n, d)\). Then:
\[
\text{Attention} = \mathrm{softmax}\Big(\frac{Q K^T}{\sqrt{d_h}}\Big) V
\]
(\(d_h\) is the hidden size per head, the factor \(\sqrt{d_h}\) is a scaling to avoid overly large dot products.)

**Cross-attention** (in the decoder) uses Q from the decoder’s hidden state and K, V from the encoder output.

There is also **masked self-attention** in the decoder to block future tokens during training for autoregressive tasks.

BERT typically uses two loss components:

1. **Masked Language Model (MLM)**  
2. **Next Sentence Prediction (NSP)**

### How to Embedding?

**How to get a pre-trained embedding?**

Previously, Word2Vec was popular (CBOW or Skip-gram). Now, BERT-style embeddings are more common:

- **WordPiece** tokenization (e.g., “un”, “##happi”, “##ness”) handles unknown words better than classic word-level embedding.  
- BERT obtains a context-dependent vector for each token.  
- BERT uses an embedding matrix (30,000 tokens), plus positional embeddings, plus segment embeddings, then feeds into the Transformer.  
- It is pre-trained on large-scale data using MLM + NSP.  

**GPT** differs from BERT in that it looks only at the left context, training to predict the next token. This makes GPT better for generative tasks.

### Transformer-Related Questions

**Why use Layer Norm instead of Batch Norm?**  
Batch sizes (N) can be variable in sequence tasks. Layer Norm normalizes along feature dimensions within one sample, which does not depend on batch size.

**Why is there a \(\sqrt{d}\) in the attention formula?**  
It acts as a scaling factor. If we didn’t scale, the softmax might become too “sharp.” The factor \(\sqrt{d}\) arises because, if Q and K have variance 1, their dot product’s variance grows with dimension \(d\).

**Hugging Face**: a widely-used unified API for tokenizers and model inference.

**T5**: Text-to-Text Transfer Transformer. It converts all NLP tasks into text-to-text format.

**RoBERTa**: removes Next Sentence Prediction and focuses more on MLM, among other training optimizations.

---

## CV

### Resnet50

Residual Network with 50 layers, used for classification.

Key points:

1. **Residual blocks**: skip connections mitigate gradient vanishing.  
2. Each block has CNN + BN + ReLU.  
3. Input size: \(224 \times 224\). Output: class probabilities.

Structure roughly:

```bash
Image (3*224*224)
-> Conv1 -> 64*112*112
-> Conv2 -> 256*56*56
-> Conv3 -> 512*28*28
-> Conv4 -> 1024*14*14
-> Conv5 -> 2048*7*7
-> Avg Pool -> 2048*1*1
-> FNN + Softmax -> 1000 classes
```

### Diffusion Model

**Forward process**:  
\[
X_t = \sqrt{1-\beta_t}\, X_{t-1} + \sqrt{\beta_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
\]

**Reverse process**:  
\[
X_{t-1} \approx X_t - \sqrt{\beta_t}\,\epsilon_0 + \sqrt{\beta_t}\,z,\quad z \sim \mathcal{N}(0,I)
\]
where \(\epsilon_0\) is predicted by a neural network.

**Loss**: MSE between the predicted noise \(\epsilon_0\) and the actual noise \(\epsilon\).

### GAN

#### Common Metrics

- **FID (Fréchet Inception Distance)**: measures distribution similarity between generated images and real images. Lower is better.  
- **PSNR (Peak Signal-to-Noise Ratio)**: focuses on pixel-level difference. Higher is better, but may not reflect perceptual quality well.  
- **SSIM (Structural Similarity Index Measure)**: focuses on structural similarity (luminance, contrast, structure). Higher is better.  
- **IS (Inception Score)**: uses a pretrained classifier (e.g., Inception) to evaluate realism (how confident the classifier is) and diversity (distribution spread). Higher is better.

#### Specific Models

- **UnetGenerator**: Convolution + BatchNorm + ReLU in an encoder-decoder structure that shrinks down to a bottleneck, then upsamples. Skip connections (concatenate encoder outputs) preserve spatial info.  
- **PatchDiscriminator**: outputs a \((1 \times 30 \times 30)\) probability map. It slides over the image in patches, focusing on local realism, and reduces parameter count.

### Swin Transformer

**Swin Transformer** introduces Transformers to CV:

- Split the image into non-overlapping windows to compute self-attention locally.  
- To enhance cross-window information, Swin uses **shifted windows** in consecutive layers so patches can interact with adjacent regions.

---

## Multimodal

### CLIP

**CLIP (Contrastive Language–Image Pre-training)**: a multimodal model that maps images and text into a shared semantic space. It is trained via contrastive learning so that matching image-text pairs have high similarity, while non-matching pairs have low similarity.

---

## Python-Related

### asyncio

`asyncio` — **coroutines** in Python.

Coroutines implement a single-threaded concurrency model. An event loop schedules multiple coroutines. During I/O wait, the loop switches to another coroutine. It’s ideal for I/O-bound tasks, especially asynchronous network I/O (non-blocking sockets, event notifications, etc.). File I/O, however, might still be blocking on many operating systems, though newer Linux kernels have `io_uring` and AIO to support asynchronous file operations.

Note that for CPU-bound tasks, asyncio may not help since it’s still single-threaded.

---

## RAG

For some tree-based structure, you may prefer it over llama-index. For example, `HierarchicalNodeParser` splits and returns a list of nodes. Internally, you might need `get_deeper_nodes` to retrieve the nodes at a particular level. Under the hood, it’s just iterating the list in a certain order. If you directly want to retrieve nodes from a certain tree level (like sentences), you can do it in \(O(1)\) if you maintain a dict of nodes by level.

**Common evaluation metrics** for RAG:

- **Accuracy**, **Precision**, **Recall**, **F1**  
- **Mean Reciprocal Rank (MRR)**: focuses on the position of the first relevant result; closer to 1 is better.  
- **Normalized Discounted Cumulative Gain (nDCG)**: considers ranked lists and relevance scores. Values closer to 1 are better.

---

## CUDA

**Shared memory** does not overflow silently; an overflow will raise errors.  

Modern hardware usually links memory usage to speed. Saving memory often saves time.  

In CUDA, the typical model is one **thread** handles a single computation. Many blocks are scheduled by the hardware. Instead of a `for` loop in one thread, you have thousands of threads, each doing a small piece of work.

---

## Machine Learning Basics

- **Linear Regression**: fit a line/plane for input-output relationships.  
- **Logistic Regression**: pass a linear model through a logistic (sigmoid) function for binary classification.  
- **Support Vector Machine (SVM)**: use different kernels (linear, polynomial, Gaussian) to map data into a high-dimensional space, then find a linear separating hyperplane.  

- **K-Nearest Neighbors (KNN)**: find the \(k\) nearest points, use majority vote or average to predict.  
- **Decision Tree**: iteratively split on feature conditions.  
- **Random Forest**: an ensemble of decision trees.  
- **Naive Bayes**: assumes independence among features (not always realistic).  
- **K-Means Clustering**: an unsupervised method that groups data into \(K\) clusters by minimizing within-cluster variance.  
- **Gradient Boosted Decision Trees (GBDT)**: builds decision trees in sequence, each new tree fitting the residual (negative gradient) from the previous model.
  - **XGBoost** is a common implementation.  
  - **LightGBM** uses histogram-based algorithms, leaf-wise growth, plus GOSS (focuses on large gradients) for efficiency.  
  - **CatBoost** has built-in support for categorical features.
