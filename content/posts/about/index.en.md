---
title: "About Me"
date: 2023-02-03T16:49:14+08:00
hiddenFromHomePage: true
lastmod: 2025-08-17T09:44:56+08:00
summary: "Personal about page."
---

Hello Friend, welcome to my blog!

---

Contact:

- Email: `zhyanwentao@outlook.com`
- Github: [yewentao256](https://github.com/yewentao256)
- LinkedIn: [Wentao Ye](https://www.linkedin.com/in/yewentao/)

---

## 1. Patent

- One-iter Tool ([CN117312173A](https://patents.google.com/patent/CN117312173A/en?oq=CN117312173A)), patented in 2023, reducing model accuracy validation time from hours to minutes.

---

## 2. Awards

- **[100k Cornell Startup Award](https://tech.cornell.edu/news/cornell-tech-startup-awards-2025/)**
  - [PolyRook](https://polyrook.com/): Fast 3D environment generation.
- **National Third Prize**
  - China University Computer Contest, WeChat Big Data Challenge, 2021
  - **Rank 80 / 6,768 teams**
  - [Certificate](resources/2021_bdc.png)
- **National Third Prize**
  - China University Computer Contest, Huawei Big Data Challenge, 2020
  - **Rank 27 / 1,491 teams**
  - [Certificate](resources/2020_bdc.jpg)
- **National Second Prize**
  - China Service Outsourcing Innovation & Entrepreneurship Competition, 2020
  - **Top 1% / 6417 teams**
  - [Certificate](resources/2020_service_outsourcing.png)
- **National First Prize**
  - China University Computer Capability Challenge, 2019
  - **Runner-up / 414 teams**
  - [Certificate](resources/2019_runner_up.jpg)

---

## 3. Experience

- **Machine Learning Engineer**
  - **Red hat**
  - *Jun 2025 - Current*

- **Deep Learning Engineer**  
  - **SenseTime | SenseCore**  
  - *Jul 2022 - Aug 2024*

- **R&D Intern**  
  - **SenseTime | Research Institute (Deep Learning Frameworks)**  
  - *Jan 2021 - Jul 2022*

- **Co-founder & CTO**  
  - **Wuhan Hongyuan Investment & Technology Services Co., Ltd.**  
  - *Nov 2019 - Sep 2020*

- **Co-founder**  
  - **Yuye (Wuhan) Technology Development Co., Ltd.**  
  - *Jun 2019 - Nov 2019*

---

## 4. Education

- **Master of Computer Science**  
  - **Cornell University** | New York, USA  
  - *May 2025*
  - GPA: 4.21/4.0 (4.3 for A+)

- **Bachelor of Software Engineering**  
  - **Wuhan University** | Wuhan, China
  - *Jun 2022*
  - GPA: 3.91/4.0
  - National Scholarship (top 1%)

---

## 5. Selected OpenSource Projects

### Contributor

#### [PyTorch](https://github.com/pytorch/pytorch)

- <a href="https://github.com/pytorch/pytorch"><img src="https://img.shields.io/github/stars/pytorch/pytorch" alt="GitHub stars" height="20"></a>
- *May 2023 - Present*
- Optimized the [cuDNN Convolution](https://github.com/pytorch/pytorch/issues/115611) and [cuDNN BatchNorm operators](https://github.com/pytorch/pytorch/pull/123020), achieving a 15% performance boost in CNN training and inference for computer vision tasks
- 30+ [contributions](https://github.com/pytorch/pytorch/issues?q=author%3Ayewentao256) to Pytorch.
- Authored a [blog series](https://wentao.site/categories/pytorch/) with 15+ articles**, providing the developer community with insights into PyTorch’s core architecture and optimizations.
- Details at [My Contributions](https://github.com/pytorch/pytorch/issues?q=author%3Ayewentao256)

### Maintainer

#### [vLLM](https://github.com/vllm-project/vllm)

- <a href="https://github.com/vllm-project/vllm"><img src="https://img.shields.io/github/stars/vllm-project/vllm?style=social" alt="GitHub stars" height="20"></a>
- *Jun 2025 - Current*
- Code owner for quantization, batch-invariant execution, caching, weight loading and CUDA kernels
- Led design and implementation of batch-invariant, showcased in the [vLLM blog](https://blog.vllm.ai/2025/11/10/bitwise-consistent-train-inference.html) and mentioned at PyTorch Conference 2025.
- Optimized MoE shared-expert overlap scheduling, improving end-to-end throughput by ~6% and reducing time-to-first-token latency by 25%+.
- Integrated and tuned DeepGEMM on B200/H100 GPUs, delivering ~11% throughput gains on B200 and ~6% on H100 while preserving accuracy; Shipped [DeepSeek V3.2](https://github.com/vllm-project/vllm/pull/25896) support in one week.
- Developed and optimized low-precision quantization kernels (INT8/FP8) for LLM inference, speeding up models by ~13% on H100 and FP8 by ~7% on B200 without accuracy loss.
- Details at [My Contributions](https://github.com/vllm-project/vllm/issues?q=author%3Ayewentao256+) and [Bi-weekly Journal](https://wentao.site/vllm_contributions/)

#### [LazyLLM](https://github.com/LazyAGI/LazyLLM)

- <a href="https://github.com/LazyAGI/LazyLLM"><img src="https://img.shields.io/github/stars/LazyAGI/LazyLLM?style=social" alt="GitHub stars" height="20"></a>
- *May 2024 - Aug 2024*
- Built a Retrieval-Augmented Generation (RAG) system with a specialized tree architecture, which improved query performance by 50% over LlamaIndex by enhancing the efficiency of parent/child node retrieval.
- Details at [My Contributions](https://github.com/LazyAGI/LazyLLM/issues?q=author%3Ayewentao256+)

#### [MMCV](https://github.com/open-mmlab/mmcv) & PAVI Logger

- <a href="https://github.com/open-mmlab/mmcv"><img src="https://img.shields.io/github/stars/open-mmlab/mmcv?style=social" alt="GitHub stars" height="20"></a>
- *Jan 2021 - Dec 2022*
- **Rebuilt the PAVI data collection SDK**, achieving a 10× improvement in data upload efficiency through optimized parallel processing, significantly reducing ingestion time and enhancing performance for large-scale datasets.
- Integrated the proprietary PAVI Logger system into the MMCV library, enabling efficient and customizable logging for deep learning workflows, with the core system remaining private.

#### [DeepLink](https://github.com/DeepLink-org/deeplink.framework) & [DIOPI](https://github.com/DeepLink-org/DIOPI)

- *Apr 2023 - May 2024*
- Optimized Llama 2-70B training on 1024 NPUs by integrating distributed training strategies (ZeRO, Tensor Parallel, Pipeline Parallel) and operator-level optimizations. Achieved a 700% increase in TGS (Tokens/GPU/Second) and significantly boosted LLM performance.
- Details at [Deeplink](https://github.com/DeepLink-org/deeplink.framework/issues?q=author%3Ayewentao256+) and [DIOPI](https://github.com/DeepLink-org/DIOPI/issues?q=author%3Ayewentao256+)

### Owner

#### [GAN-Paint](https://github.com/yewentao256/GAN-Paint)

- *Nov 2024 - Jan 2025*
- Developed a lightweight GAN (generative adversarial network) for large-area image completion and cross-scene stitching, achieving realistic outputs on a single RTX 2070 GPU.
- Implemented an end-to-end training pipeline with efficient data preprocessing, masking strategies, and evaluation, completing model training within hours.

#### [MicroTorch](https://github.com/yewentao256/MicroTorch)

- *Jun 2023 - Aug 2024*
- Developed a **minimalistic deep learning framework** inspired by PyTorch, implementing core functionalities such as AutoGrad, dynamic computation graphs, and tensor operations.
- Designed to be lightweight and modular, making it ideal for educational purposes, with extensive examples to facilitate learning.

#### [CMU CSAPP](https://github.com/yewentao256/CSAPP_15213)

- *Dec 2022 - Feb 2024*
- Self-studied the **CMU CSAPP-15213** course and completed its associated labs, covering core concepts such as assembly optimization, multi-level cache, compiling and linking, exception control flow, virtual memory, and system-level I/O.
- [Blogs](https://wentao.site/categories/csapp/)

#### [TinyNN](https://github.com/yewentao256/TinyNN)

- *Nov 2022 - Dec 2022*
- Built **TinyNN**, a minimal implementation of **Fully Connected Neural Networks** and **Convolutional Neural Networks**, designed for educational and experimental purposes.

#### [You After Taking Drugs](https://github.com/yewentao256/You-after-taking-drugs)

- *Aug 2021*
- **Independently developed this system in 7 days** using computer vision algorithms; optimized for smooth performance on a single i3 CPU, ensuring a seamless user experience and earning client approval in the first review.
- Software Copyright: "After Taking Drugs (Facial Human Morphing Experience)" (2022SR0021854).

#### [Sicpy Compiler](https://github.com/yewentao256/sicpy)

- *Nov 2020 - Dec 2020*
- Designed and implemented an untyped programming language **Sicpy** and its corresponding compiler using **flex** and **bison**.
- Developed features including **lexical**, **syntax**, and **semantic analysis**, as well as **type inference** and **automatic garbage collection** via reference counting, providing a complete custom language framework for functional and imperative programming experimentation.

#### New Super Mario

- *Apr 2020*
- Group project with [Jifeng Wu](https://github.com/jifengwu2k), Jinran Tang and Taihe Li.

![gif](resources/mario.gif)

![gif](resources/mario2.gif)
