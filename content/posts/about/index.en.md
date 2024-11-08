---
title: "About"
date: 2023-02-03T16:49:14+08:00
hiddenFromHomePage: true
lastmod: 2024-11-07T15:58:26+08:00
summary: "Personal about page."
---

Hello Friend, Welcome to My Blog!

I am Wentao, currently a Master of Computer Science student at **Cornell University**, expected to graduate in May 2025.

**Email**: `zhyanwentao@outlook.com` / `wy335@cornell.edu`

Github: [yewentao256](https://github.com/yewentao256)

LinkedIn: [Wentao Ye](https://www.linkedin.com/in/yewentao/)

---

## 1. Patents

- Patent: One-iter Tool ([CN117312173A](https://patents.google.com/patent/CN117312173A/en?oq=CN117312173A)), patented in 2023, reducing model accuracy validation time from hours to minutes.
- Patent: Function-Level Task Scheduling Tool ([CN115033366A](https://patents.google.com/patent/CN115033366A/en)), patented in 2022, streamlining distributed training workflows.
- Software Copyright: "After Taking Drugs (Facial Human Morphing Experience)" (2022SR0021854), Published Aug 9, 2021.

---

## 2. Awards

- **National Third Prize**, China University Computer Contest - WeChat Big Data Challenge, 2021 (Top 2% / 6,000+ teams)
- **National Third Prize**, China University Computer Contest - Huawei Big Data Challenge, 2020 (Top 2% / 1,400+ teams)
- **National Second Prize**, China Service Outsourcing Innovation & Entrepreneurship Competition, 2020 (Top 2% / 700+ teams)
- **National First Prize (Runner-up)**, China University Computer Capability Challenge, 2019 (Top 1% / 400+ teams)

---

## 3. Experience

### **SenseTime**

**Deep Learning System Development Engineer**
*Jul 2022 - Aug 2024*

- Developed [LazyLLM](https://github.com/LazyAGI/LazyLLM/issues?q=author%3Ayewentao256+), independently building a custom **RAG** with a specialized tree architecture that **outperforms Llama Index by 50%** in fetching parent/child nodes, significantly improving query performance.
- Designed and implemented the Op Inferrer for the [DeepLink Framework](https://github.com/DeepLink-org/deeplink.framework/issues?q=author%3Ayewentao256+), bypassing PyTorch's TensorIterator to increase the inference speed of binary, unary, and reduction operators **by 5% across 40+ models**, including large language models (LLMs).
- Implemented multi-chip operator adaptations for [DIOPI](https://github.com/DeepLink-org/DIOPI/issues?q=author%3Ayewentao256+), supporting CUDA, Cambricon, and Ascend architectures.
- Independently identified performance bottlenecks and optimized CUDA model performance via DeepLink and DIOPI integration, achieving a **20% average improvement** across 30+ models, with ResNet50's performance **surpassing PyTorch**.
- Stabilized **Llama2 70B** training across 64 Ascend 910B chips using ZeRO + TP + PP and mixed-precision training, providing analysis reports and optimizing operator performance, **improving training TGS from 10% to 70% compared to A100 chips**.

### **SenseTime - R&D Intern**

**Research Institute (Deep Learning Frameworks)**
*Jan 2021 - Jul 2022*

- Independently designed and developed the backend of a training data visualization platform, using **FastAPI** and Ceph + MySQL + TiDB, deployed in Kubernetes, and integrated with CI/CD pipelines.
- Rebuilt a data collection SDK, increasing data upload efficiency **10×** through parallelism, accelerating model development pipelines.
- Led automatic model annotation project across multiple teams, reducing manual labeling costs by **60%**.

### **Wuhan Hongyuan Investment & Technology Services Co., Ltd.**

**Co-founder & CTO**
*Nov 2019 - Sep 2020*

- Led a team of 9 in software and hardware development for **10+** AI exhibition halls, including projects for the Henan Shangqiu City Procuratorate, increasing visitor engagement by **50%**. [Example](https://github.com/yewentao256/You-after-taking-drugs)

### **Yuye (Wuhan) Technology Development Co., Ltd.**

**Co-founder** | Wuhan, China  
*Jun 2019 - Nov 2019*

- **Led a team of 3** in developing a student competition teaming platform, attracting hundreds of users within the first month of launch.

---

## 4. Education

### Cornell University - Cornell Tech

**Master of Computer Science  | New York, USA**  
*Aug 2024 - May 2025*

### Wuhan University

**Bachelor of Software Engineering (Excellent Engineer Program) | Wuhan, China**  
*Sep 2018 - Jun 2022* | GPA: 3.91/4.0

- Led technical projects as Technical Director of Wuhan University Microsoft Student Club.
- Awarded National Scholarship (Top 1%) and twice awarded National Encouragement Scholarship (Top 2%)

---

## 5. Projects

### **[PyTorch Contribution](https://github.com/pytorch/pytorch/issues?q=author%3Ayewentao256)**

- *May 2023 - Present (C++, Python)*
- Authored **10+** [blog posts](https://wentao.site/categories/pytorch/) analyzing PyTorch internal mechanisms, covering Tensor Storage, CPU & Cuda Operators, Dispatcher, TensorIterator, AutoGrad and Distributed Training strategy.
- Identified and optimized the CuDNN Convolution operator, improving performance by **15%**, with the contribution merged.

### **[MicroTorch](https://github.com/yewentao256/MicroTorch)**

- *Jun 2023 - Present (C++, Python)*
- Developed a **minimalistic deep learning framework** inspired by PyTorch, implementing core functionalities such as AutoGrad, dynamic computation graphs, and tensor operations.
- Designed to be lightweight and modular, making it ideal for educational purposes, with extensive examples to facilitate learning.

### **[CMU CSAPP](https://github.com/yewentao256/CSAPP_15213)**

- *Dec 2022 - Feb 2024 (C, Assembly)*
- Self-studied the **CMU CSAPP-15213** course and completed its associated labs, covering core concepts such as assembly optimization, multi-level cache, compiling and linking, exception control flow, virtual memory, and system-level I/O.
- [Blogs](https://wentao.site/categories/csapp/)

### **[TinyNN](https://github.com/yewentao256/TinyNN)**

- *Nov 2022 - Dec 2022 (Python)*
- Built **TinyNN**, a minimal implementation of **Fully Connected Neural Networks (FCNN)** and **Convolutional Neural Networks (CNN)**, designed for educational and experimental purposes.

### **[You After Taking Drugs: Face Change Simulation System](https://github.com/yewentao256/You-after-taking-drugs)**

- *Aug 2021 (Python)*
- Developed a simulation system using **dlib**, **Image Warping**, and **Face Fusion** techniques, capable of generating face change predictions based on potential drug use effects.

### **[Sicpy Compiler](https://github.com/yewentao256/sicpy)**

- *Nov 2020 - Dec 2020 (C, Flex, Bison)*
- Designed and implemented an untyped programming language **Sicpy** and its corresponding compiler using **flex** and **bison**.
- Developed features including **lexical**, **syntax**, and **semantic analysis**, as well as **type inference** and **automatic garbage collection** via reference counting, providing a complete custom language framework for functional and imperative programming experimentation.

### **[CSE251](https://github.com/yewentao256/CSE251)**

- *Sep 2020 (C)*
- CSE 251 Programming in C @MSU

### **New Super Mario**

- *Apr 2020 (C#, unity)*
- Group project with [JiFeng Wu](https://github.com/jifengwu2k), Jinran Tang and Taihe Li.

![gif](resources/mario.gif)

![gif](resources/mario2.gif)
