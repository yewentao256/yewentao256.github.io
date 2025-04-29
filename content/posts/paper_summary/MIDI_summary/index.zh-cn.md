---
title: "Summary: MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation"
date: 2025-04-29T10:39:56+08:00
categories: ["paper_summary"]
summary: "论文速览：'MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation'"
---

> 本博客使用`o3`翻译，如有冲突请优先参考英文原文

## 0. Materials

- [Paper](https://arxiv.org/pdf/2412.03558)

- [Github](https://github.com/VAST-AI-Research/MIDI-3D)

## 1. 论文做了什么？  

![image](workflow.png)

工作流：Grounded-SAM 分割 → 输入：掩码 + 局部视图 + 全局视图（共 7 通道）→ 多实例扩散并行去噪（带跨实例注意力）。

- 提出了 **MIDI（Multi-Instance Diffusion）**，首个将预训练单物体扩散模型扩展为多物体生成器的框架，可凭单张图像生成完整室内 3D 场景  
- **N 个潜变量并行去噪**（每个物体一套潜变量，权重共享）并加入 **跨实例注意力**，使各物体 token 可互相关注，从而在生成阶段直接约束空间关系  
- 条件输入由裁剪物体图、其掩码及全局场景图组合；最终潜变量经解码得到网格并直接拼装，无需额外布局优化  

## 2. 与已有工作的对比创新  

- 取代 Gen3DSR、REPARO 等采用“分割 → 修补 → 逐物体生成 → 布局求解”的多阶段流程，MIDI 仅用 **一次扩散** 完成场景生成  
- 设计 **多实例注意力层**：将单物体自注意力推广为跨物体注意，令 token 可查询全部实例，单物体模型此前并不具备这一能力  
- 通过 **LoRA** 轻量微调 21 层 DiT 主干，仅更新少量参数而保持几何先验 ([Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748))  
- 训练集将经过清洗的 **3D-FRONT** 场景与单物体 **Objaverse** 混合，既学到空间关系又保留形状多样性  

## 3. 实验验证

- 在 **3D-FRONT** 与 **BlendSwap** 上定量评测：MIDI 在场景级与物体级 Chamfer-D、F-Score 与包围盒 IoU 均优于多阶段基线，且推理仅 40 s，对方需 4–9 min  
- 在 **Matterport3D**、**ScanNet** 上的定性对比显示，MIDI 相比 PanoRecon、Total3D、InstPIFu、SSR、DiffCAD、Gen3DSR、REPARO 具有更完整的几何与更准确的对齐  
- 向模型输入由 **SDXL** 生成的卡通/CG 场景，验证其出色的分布外泛化；同场景下 REPARO 易错放物体，而 MIDI 可保持布局一致  
- 消融实验：  
  - 调整多实例注意力层数（K = 0/5/21）→ K = 5 最优；K = 0 空间关系崩溃，K = 21 过拟合导致形变  
  - 去掉全局场景图或 Objaverse 混合 → IoU/F-Score 均下降，说明两者关键  
- 单张 A100 上完整场景生成仅 40 s  

## 4. 局限与不足

- 所有实例统一归一化至同一场景盒，小物体体素分辨率受限  
- **交互多样性** 受数据集限制；动态/关节关系（如人物握杯）尚未建模  
- 训练/推理时实例数受显存限制（N ≤ 5），杂物较多的大房间需分批处理  
- 背景布局（墙体、地面）不生成，需外部方法补全；不同于端到端整体重建  
- 依赖精确分割掩码，Grounded-SAM 误检将随之传播  

## 5. 下一步可行的研究方向

- 为每个物体在本地坐标系中高分辨生成，再回归 6-DoF 位姿拼装，兼顾解析度与联合优化  
- 引入 **交互丰富的数据集**（如 MOABA 等人-物交互合成数据）扩展多实例注意力到人-物、物-物动态关系  
- 探索显式注入 3D 位置编码的 **几何感知注意力**，提高空间推理效率并允许更深堆叠  
- 结合 **开放世界分割 + 文本提示**，扩展到户外街景或混合现实场景  
- 增加 **布局感知精化网络**，对小物体上采样并预测背景表面，输出完整可编辑 CAD 场景  

## 附录（术语与数据集）

- **DiT（Diffusion Transformer）**——以 Transformer 取代 U-Net 的扩散网络主干
- **VAE**——变分自编码器  
- **CFG**——Classifier-Free Guidance，推断时混合条件/无条件去噪以增强约束  
- **LoRA**——低秩适配，高效微调预训练权重  
- **Grounded-SAM**——结合 Grounding-DINO 与 SAM 的开放词汇实例分割  
- **Chamfer Distance (CD)**——点云间平均最近点距离，越低越近  
- **F-Score (3D)**——基于固定阈值的表面精确率/召回率调和均值，越高越好  
- **Volume IoU (IoU-B)**——预测与真值 3D 包围盒体积的交并比  
- **3D-FRONT**——大规模合成室内场景数据集  
- **Objaverse**——包含 80 万+ 3D 物体的大型开源集合  
- **Matterport3D / ScanNet**——常用真实室内 RGB-D 重建基准  
- **SDXL**——Stable Diffusion XL，高分辨率文本到图像模型  
- **SDF / Tri-plane / InstPIFu / Total3D / DiffCAD / Gen3DSR / REPARO**——与论文相关的 3D 表示与基线方法或管线  
