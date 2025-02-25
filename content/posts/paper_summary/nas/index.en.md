---
title: "Summary: NAS with RL"
date: 2025-02-25T10:15:13+08:00
categories: ["paper_summary"]
summary: "Summary for paper 'Neural Architecture Search with Reinforcement Learning'"
---

## Download the paper

[Paper_Link](https://arxiv.org/pdf/1611.01578)

## What is the paper about?

- The paper introduces **Neural Architecture Search (NAS)**, a method that uses a RNN as a “controller” to automatically generate neural network architectures.
- The controller is trained via **reinforcement learning**, with the validation accuracy of the “child network” serving as the reward signal.
- It demonstrates the ability to discover architectures for both **convolutional** (CIFAR-10) and **recurrent networks** (Penn Treebank) that rival or surpass human-designed models.

![image](nas_architecture.png)

## What is new about this specific paper, compared to prior work?

- It proposes a **flexible controller RNN** that can sample **variable-length** architecture descriptions, expanding beyond fixed hyperparameter settings seen in standard Bayesian or random search methods.
- It applies RL to directly optimize architectures for higher validation accuracy, rather than relying on supervised signals or manual heuristics.
- It shows **state-of-the-art performance** on CIFAR-10 and Penn Treebank, suggesting the automatically discovered architectures can compete with the best human-engineered models.

## What experiments were run to support the arguments in this paper?

**CIFAR-10** image classification:

- The authors let the controller search for convolutional networks with optional skip connections and pooling layers.
- The discovered architectures were trained and tested, achieving error rates better than leading manually designed networks.

**Penn Treebank** language modeling:

- The controller was used to search for novel **recurrent cell** architectures (beyond LSTM).
- The best discovered cell achieved test **perplexities better** than previous state-of-the-art models.
- The authors also tested the new cell on **character-level language modeling** (still on PTB) and briefly on a **machine translation** task (GNMT framework), showing further performance gains.

## What are the shortcomings/limitations of this paper?

- Training thousands of candidate architectures is **extremely expensive**, even with large-scale parallel resources.
- Although the search space is large, it is still **constrained by the design choices** (e.g., certain filter sizes, skip-connection types, or cell structures).
- While the discovered architectures worked well on CIFAR-10 and PTB, there is **no guarantee** they are optimal for other datasets or tasks.
- Setting up distributed reinforcement learning for large-scale architecture search requires **significant engineering effort**.

## What is a reasonable next step to build upon this paper?

- Investigate more efficient or sample-efficient search strategies, such as weight-sharing, early stopping, or more advanced reinforcement learning algorithms.
- Incorporate **more types of layers or operations** (e.g., attention mechanisms, dynamic routing) while keeping the search process tractable.
- Explore ways to transfer discovered architectures across tasks, possibly adapting or fine-tuning them with minimal retraining.
- Searching architectures under memory, latency, or power constraints to make the method more practical for real-world applications.

## Appendix

- CIFAR-10: A benchmark image classification dataset containing **60,000** `32×32` color images in **10 classes**.
- Penn Treebank (PTB): A benchmark dataset for language modeling used to evaluate the performance of RNN.
- Weight Sharing: Reusing the same weights across different parts of the network to reduce computational cost during the search process.
