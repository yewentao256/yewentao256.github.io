---
title: "About Me"
date: 2023-02-03T16:49:14+08:00
hiddenFromHomePage: true
lastmod: 2025-05-31T10:35:26+08:00
summary: "Personal about page."
---

CS graduate from Cornell University (GPA: **4.21/4.0**, A+ = 4.3) with **3+ years** of experience in deep learning and LLM optimization, and 2 years in software development. Former deep learning engineer at [SenseTime](https://www.sensetime.com/en), recognized for achieving **up to 700% performance improvements on LLaMA2-70B across 1024 NPUs**. Dedicated **open-source contributor with 8k+ stars** across personal and collaborative projects. Award-winning developer and two-time startup co-founder, driving ML innovations from prototype to scalable production.

**Email**: `zhyanwentao@outlook.com` / `wy335@cornell.edu`

Github: [yewentao256](https://github.com/yewentao256)

LinkedIn: [Wentao Ye](https://www.linkedin.com/in/yewentao/)

My blog series:

- [Pytorch](https://wentao.site/categories/pytorch/)
- [Paper_Summary](https://wentao.site/categories/paper_summary/)
- [CSAPP](https://wentao.site/categories/csapp/)
- and [More Categories](https://wentao.site/categories/)

---

## 1. Patents

- One-iter Tool ([CN117312173A](https://patents.google.com/patent/CN117312173A/en?oq=CN117312173A)), patented in 2023, reducing model accuracy validation time **from hours to minutes**.
- Function-Level Task Scheduling Tool ([CN115033366A](https://patents.google.com/patent/CN115033366A/en)), patented in 2022, streamlining distributed training workflows.

---

## 2. Awards

- **[100k Cornell Startup Award Winner](https://tech.cornell.edu/news/cornell-tech-startup-awards-2025/)**
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

- **Bachelor of Software Engineering (Excellent Engineer Program)**  
  - **Wuhan UniversityChina** | Wuhan, China
  - *Jun 2022*
  - GPA: 3.91/4.0

---

## 5. OpenSource Projects

### Feature Contributor

#### PyTorch

- <a href="https://github.com/pytorch/pytorch/issues?q=author%3Ayewentao256"><img src="https://img.shields.io/github/stars/pytorch/pytorch" alt="GitHub stars" height="20"></a>
- *May 2023 - Present*
- **Optimized the CuDNN Convolution operator in PyTorch**, achieving a 15% performance boost in CNN training and inference for computer vision tasks; successfully merged into the PyTorch codebase.
- **Authored a [blog series](https://wentao.site/categories/pytorch/) with 10+ articles**, providing the developer community with insights into PyTorch’s core architecture and optimizations.

### Main Contributor

#### LazyLLM

- <a href="https://github.com/LazyAGI/LazyLLM/issues?q=author%3Ayewentao256+"><img src="https://img.shields.io/github/stars/LazyAGI/LazyLLM?style=social" alt="GitHub stars" height="20"></a>
- *May 2024 - Aug 2024*
- **Independently built a Retrieval-Augmented Generation (RAG) system** in LazyLLM with a specialized tree architecture, which improved query performance by 50% over LlamaIndex by enhancing the efficiency of parent/child node retrieval, optimizing response times for large language models.

#### DeepLink

- <a href="https://github.com/DeepLink-org/deeplink.framework/issues?q=author%3Ayewentao256+"><img src="https://img.shields.io/github/stars/DeepLink-org/deeplink.framework?style=social" alt="GitHub stars" height="20"></a>
- *May 2023 - May 2024*
- Designed and implemented the Op Inferrer, bypassing PyTorch's TensorIterator to increase the inference speed of binary, unary, and reduction operators by 5% across 40+ models, including large language models (LLMs).
- Identified and resolved CUDA performance bottlenecks by optimizing implementations within DeepLink and DIOPI, achieving an average 20% performance improvement across 30+ models; enhanced computational efficiency allowed ResNet50 to surpass PyTorch’s benchmark performance, providing significant speedups for high-demand tasks.

#### DIOPI

- <a href="https://github.com/DeepLink-org/DIOPI/issues?q=author%3Ayewentao256+"><img src="https://img.shields.io/github/stars/DeepLink-org/DIOPI?style=social" alt="GitHub stars" height="20"></a>
- *Apr 2023 - May 2024*
- **Developed 30+ machine learning operators** in DIOPI, enabling advanced functionalities across diverse hardware; implemented multi-chip adaptations to support CUDA, Cambricon, and Ascend architectures, enhancing cross-platform compatibility, reducing integration time and enhancing operational efficiency for large-scale systems.

#### MMCV & PAVI Logger

- <a href="https://github.com/open-mmlab/mmcv"><img src="https://img.shields.io/github/stars/open-mmlab/mmcv?style=social" alt="GitHub stars" height="20"></a>
- *Jan 2021 - Dec 2022*
- **Rebuilt the PAVI data collection SDK**, achieving a 10× improvement in data upload efficiency through optimized parallel processing, significantly reducing ingestion time and enhancing performance for large-scale datasets.
- Integrated the proprietary PAVI Logger system into the MMCV library, enabling efficient and customizable logging for deep learning workflows, with the core system remaining private.

### Owned Projects

#### TVM Tutorial

- <a href="https://github.com/yewentao256/TVM_tutorial"><img src="https://img.shields.io/github/stars/yewentao256/TVM_tutorial?style=social" alt="GitHub stars" height="20"></a>
- *Apr 2025*
- This tutorial series is designed for beginners to learn how to optimize deep learning operations with TVM. Through practical notebooks, we explore step-by-step performance tuning on both CPU and GPU.

#### GAN-Paint

- <a href="https://github.com/yewentao256/GAN-Paint"><img src="https://img.shields.io/github/stars/yewentao256/GAN-Paint?style=social" alt="GitHub stars" height="20"></a>
- *Nov 2024 - Jan 2025*
- Developed a lightweight GAN (generative adversarial network) for large-area image completion and cross-scene stitching, achieving realistic outputs on a single RTX 2070 GPU.
- Implemented an end-to-end training pipeline with efficient data preprocessing, masking strategies, and evaluation, completing model training within hours.

#### MicroTorch

- <a href="https://github.com/yewentao256/MicroTorch"><img src="https://img.shields.io/github/stars/yewentao256/MicroTorch?style=social" alt="GitHub stars" height="20"></a>
- *Jun 2023 - Aug 2024*
- Developed a **minimalistic deep learning framework** inspired by PyTorch, implementing core functionalities such as AutoGrad, dynamic computation graphs, and tensor operations.
- Designed to be lightweight and modular, making it ideal for educational purposes, with extensive examples to facilitate learning.

#### CMU CSAPP

- <a href="https://github.com/yewentao256/CSAPP_15213"><img src="https://img.shields.io/github/stars/yewentao256/CSAPP_15213?style=social" alt="GitHub stars" height="20"></a>
- *Dec 2022 - Feb 2024 (C, Assembly)*
- Self-studied the **CMU CSAPP-15213** course and completed its associated labs, covering core concepts such as assembly optimization, multi-level cache, compiling and linking, exception control flow, virtual memory, and system-level I/O.
- [Blogs](https://wentao.site/categories/csapp/)

#### TinyNN

- <a href="https://github.com/yewentao256/TinyNN"><img src="https://img.shields.io/github/stars/yewentao256/TinyNN?style=social" alt="GitHub stars" height="20"></a>
- *Nov 2022 - Dec 2022 (Python)*
- Built **TinyNN**, a minimal implementation of **Fully Connected Neural Networks** and **Convolutional Neural Networks**, designed for educational and experimental purposes.

#### You After Taking Drugs

- <a href="https://github.com/yewentao256/You-after-taking-drugs"><img src="https://img.shields.io/github/stars/yewentao256/You-after-taking-drugs?style=social" alt="GitHub stars" height="20"></a>
- *Aug 2021 (Python)*
- **Independently developed this system in 7 days** using computer vision algorithms; optimized for smooth performance on a single i3 CPU, ensuring a seamless user experience and earning client approval in the first review.
- Software Copyright: "After Taking Drugs (Facial Human Morphing Experience)" (2022SR0021854).

#### Sicpy Compiler

- <a href="https://github.com/yewentao256/sicpy"><img src="https://img.shields.io/github/stars/yewentao256/sicpy?style=social" alt="GitHub stars" height="20"></a>
- *Nov 2020 - Dec 2020 (C, Flex, Bison)*
- Designed and implemented an untyped programming language **Sicpy** and its corresponding compiler using **flex** and **bison**.
- Developed features including **lexical**, **syntax**, and **semantic analysis**, as well as **type inference** and **automatic garbage collection** via reference counting, providing a complete custom language framework for functional and imperative programming experimentation.

#### New Super Mario

- *Apr 2020 (C#, unity)*
- Group project with [Jifeng Wu](https://github.com/jifengwu2k), Jinran Tang and Taihe Li.

![gif](resources/mario.gif)

![gif](resources/mario2.gif)
