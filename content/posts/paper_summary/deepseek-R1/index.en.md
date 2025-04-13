---
title: "Summary: DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
date: 2025-04-13T10:20:05+08:00
categories: ["paper_summary"]
summary: "Summary for paper 'Incentivizing Reasoning Capability in LLMs via Reinforcement Learning'"
---

## Download the Paper

[Paper](https://arxiv.org/pdf/2501.12948)

## 1. What is the paper about?

- It explores the development of reasoning models using **reinforcement learning (RL)**, specifically focusing on DeepSeek-R1 and DeepSeek-R1-Zero models.

- It investigates the potential of **large-scale RL** to enhance the reasoning capabilities of LLMs without relying on traditional **supervised fine-tuning (SFT)**.

- It explores **distillation** of reasoning models to smaller, more efficient models while maintaining high performance.

- It evaluates DeepSeek-R1's performance on various reasoning tasks and compares it to other leading models like OpenAI-o1 and GPT-4o.

## 2. What is new about this specific paper, compared to prior work?

- Unlike previous work that relied heavily on SFT, this paper introduces the use of pure **RL** to enhance reasoning capabilities without supervised data, especially in DeepSeek-R1-Zero.

- DeepSeek-R1 incorporates a small amount of **cold-start data** before applying RL, addressing issues like readability and language mixing, which were present in DeepSeek-R1-Zero.

- It demonstrates how reasoning capabilities can be **distilled** from larger models like DeepSeek-R1 into smaller models, achieving competitive performance even in compact models like DeepSeek-R1-Distill-Qwen-7B.

- It shows how techniques like **majority voting** can improve model performance significantly, such as increasing AIME 2024 performance from 71.0% to 86.7%.

## 3. What experiments were run to support the arguments in this paper?

- It evaluates DeepSeek-R1 and its variants (DeepSeek-R1-Zero, DeepSeek-R1-Distill) on multiple reasoning benchmarks, including **MMLU**, **AIME 2024**, **Codeforces**, **LiveCodeBench**, and others.

- It compares the performance of distilled models like DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B against larger models like OpenAI-o1 and GPT-4o.

- It tracks the performance of DeepSeek-R1-Zero during RL training, demonstrating its progression and improvements in various tasks over time.

- It compares the effect of majority voting (consensus) on performance, showing how this technique enhances results on benchmarks like AIME 2024.

## 4. What are the shortcomings/limitations of this paper?

- Despite improvements, DeepSeek-R1 still faces **language mixing issues**, especially when handling queries in languages other than English or Chinese.

- Large-scale RL training for reasoning tasks is computationally expensive and may not always be feasible, especially for smaller models.

- The model does not show significant improvement over DeepSeek-V3 on software engineering benchmarks due to the long evaluation times associated with RL processes.

- It acknowledges the issue of **reward hacking** when using reward models, which can lead to suboptimal training outcomes.

- The model's performance is sensitive to the format and type of prompts, and using few-shot prompting can degrade its results.

## 5. What is a reasonable next step to build upon this paper?

- Address language mixing by enhancing the model’s multilingual capabilities, particularly when handling queries in less commonly used languages.

- Investigate ways to make large-scale RL more computationally efficient, such as introducing asynchronous evaluations or alternative training strategies to speed up the process.

- Focus on improving performance on software engineering tasks, potentially through **rejection sampling** or more targeted RL data for engineering-specific domains.

- Combine RL with SFT in a more integrated manner, using RL to refine reasoning capabilities and SFT to maintain general-purpose task proficiency.

- Experiment with different types of prompting techniques and architectures to reduce sensitivity to prompt format and enhance the model's robustness in real-world applications.

## Appendix

- **Cold-Start Data**: Initial data used to stabilize the early phase of reinforcement learning (RL) training.

- **Majority Voting**: A method to improve performance by aggregating responses from multiple outputs and choosing the most frequent answer.

- **MMLU (Massive Multitask Language Understanding)**: A benchmark for testing general language understanding across multiple tasks.

- **AIME 2024 (American Invitational Mathematics Examination 2024)**: A math competition benchmark for testing mathematical reasoning abilities.

- **Codeforces**: A competitive programming platform where models are evaluated based on their ability to solve coding problems.

- **LiveCodeBench**: A benchmark for evaluating software engineering task performance.

- **Reward Hacking**: Exploiting the reward system in RL to achieve high scores without solving the task properly.

- **Supervised Fine-Tuning (SFT)**: Training a pre-trained model on a task-specific labeled dataset.

- **Reinforcement Learning (RL)**: A machine learning method where an agent learns by interacting with an environment and receiving rewards.
