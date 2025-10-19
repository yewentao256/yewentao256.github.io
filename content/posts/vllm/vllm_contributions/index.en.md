---
title: "Bi-weekly Journal: Contributions to vLLM"
date: 2025-08-09T15:12:12+08:00
categories: ["vllm"]
summary: "My bi-weekly journal for contributions to vllm."
---

## Summary

My bi-weekly journal for contributions to vllm.

---

## Oct 1 - Oct 14

**Customer-related Bug Fix**:

- From Clayton (llm-d):  
  - Image issue with DeepGEMM: no kernel image is available for execution on the device: Gave technical support and fix in two days  
  - Log optimization [\#26322](https://github.com/vllm-project/vllm/pull/26322)  
- From Lu Fang (Meta)  
  - WIP: Improve vLLM CUDA Memory Utilization and Estimation [\#26300](https://github.com/vllm-project/vllm/issues/26300)

**Batch Invariant**:

- Closely collaborate with Bram Wasti, milestone doc: [vLLM Batch-Invariance Work List](https://docs.google.com/document/d/14msdTt0Y-NtANwLspIh3t_53ioCvfZ4ue9yN41Y_nHQ/edit?tab=t.0)  
- Landed Flashinfer support [\#26373](https://github.com/vllm-project/vllm/pull/26373)  
- WIP: Deepseek-v3 Batch Invariant on 8xH100 [https://github.com/vllm-project/vllm/pull/26609](https://github.com/vllm-project/vllm/pull/26609)  
- Several other small PRs

**Community Leadership**:

- Lead implementation  
  - Vectorize RMS norm variance using vectorize\_read\_with\_alignment [\#26234](https://github.com/vllm-project/vllm/pull/26234)  
  - Fix all of the mypy check, issue in [\#26533](https://github.com/vllm-project/vllm/issues/26533)  
    - \[CI\] Fix mypy for vllm/attention and vllm/compilation \#26482  
    - \[CI\] Fix mypy for vllm/distributed \#26593  
    - \[CI\] Fix mypy for vllm/engine and vllm/utils \#26540  
    - \[CI\] Fix mypy for vllm/executor \#26845  
  - Reduce Unit Test to Speed Up CI [\#22041](https://github.com/vllm-project/vllm/issues/22041)  
    - \[CI Perf\]Prune Tests in kernel/mamba \#26538  
    - Pruning kernel Core Tests \#26727  
- Mentioned and deep review  
  - \#26669: support flashinfer\_fp4 moe for 5090 gpu  
  - \#25619: \[UX\] Speedup DeepGEMM warmup with heuristics  
  - \#26438: \[Bug\]: TypeError: argument 'id': StreamInput must be either an integer or a list of integers  
  - \+ a lot more

**VLLM Contributions**:

- Huge Performance Improvement  
  - Enable E8M0 by Default on Hopper for DeepGEMM, 5% E2E throughput improvement: [\#26197](https://github.com/vllm-project/vllm/pull/26197)  
- Refactoring PRs merged  
  - [\#25293](https://github.com/vllm-project/vllm/pull/25293) [\#26743](https://github.com/vllm-project/vllm/pull/26743), [\#26601](https://github.com/vllm-project/vllm/pull/26601), [\#26044](https://github.com/vllm-project/vllm/pull/26044) \+ several more  
- Bug Fix PRs merged:  
  - [\#26532](https://github.com/vllm-project/vllm/pull/26532), [\#26528](https://github.com/vllm-project/vllm/pull/26528), [\#26448](https://github.com/vllm-project/vllm/pull/26448) \+ several more

## Sep 17 - Sep 30

DeepSeekV3.2 Support

- One week with a tight timeline, working through weekends, closely work with Chen Zhang, Yongye Zhu, Kaichao You, etc.  
- Main PR: [\#25896](https://github.com/vllm-project/vllm/pull/25896)  
- Release note: [https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html](https://blog.vllm.ai/2025/09/29/deepseek-v3-2.html)  
  - Wentao Ye in the Acknowledgements\!  
- My Work (All PRs combined)  
  - Everything with DeepGEMM  
  - Wheels, test script, B200 validation  
  - Weight loading issue etc like  [\#25909](https://github.com/vllm-project/vllm/pull/25909)

**Customer-related Bug Fix**:

- From Clayton(llm-d):  
  - Under review: Fixed Negative cuda memory usage: [\#25683](https://github.com/vllm-project/vllm/pull/25683)  
  - Fixed OOM issue: [\#25290](https://github.com/vllm-project/vllm/pull/25290)  
  - Fixed Cudagraph cache issue: [\#25093](https://github.com/vllm-project/vllm/pull/25093)  
- vLLM 0.11.0 release blocker
  - Issue related with B200 for Qwen3-VL  
  - Raised in [\#25582](https://github.com/vllm-project/vllm/issues/25582) and fixed by [\#25788](https://github.com/vllm-project/vllm/pull/25788), working closely with Roger Wang

**VLLM Contributions**:

- Several Refactoring/Fix PRs merged: [\#25958](https://github.com/vllm-project/vllm/pull/25958) [\#25710](https://github.com/vllm-project/vllm/pull/25710) [\#25519](https://github.com/vllm-project/vllm/pull/25519) [\#25518](https://github.com/vllm-project/vllm/pull/25518) [\#25517](https://github.com/vllm-project/vllm/pull/25517) \+ several more  
- Leadership:
  - Guide Community to produce better code  [\#22602](https://github.com/vllm-project/vllm/pull/22602)  
  - Feature Request to Optimize reshape\_and\_cache CUDA Kernel  [\#25705](https://github.com/vllm-project/vllm/issues/25705)  
  - Feature Request to Reduce unit test in CI [\#22041](https://github.com/vllm-project/vllm/issues/22041)  
- Mentioned by Community and Deep Review  
  - MOE flag related [\#23442](https://github.com/vllm-project/vllm/pull/23442)  
  - Cuda graph related [\#25829](https://github.com/vllm-project/vllm/pull/25829)  
  - Compiled issue [\#25843](https://github.com/vllm-project/vllm/pull/25843)  
  - A lot more …

## Sep 3 - Sep 16

**Performance Optimization**:

- Optimize DeepGEMM scale Contiguous Layout  
  - [https://github.com/vllm-project/vllm/pull/24783](https://github.com/vllm-project/vllm/pull/24783)
  - 5.5% Throughput Improvement  
- Ready for review: Triton Kernel for per\_block\_cast\_to\_fp8, 6x faster  
  - [https://github.com/vllm-project/vllm/pull/24611](https://github.com/vllm-project/vllm/pull/24611)  
  - 6x faster for the torch version

**Severe Bug Fix**:

- Clayton’s torch compile cache issue: [https://github.com/vllm-project/vllm/issues/24915](https://github.com/vllm-project/vllm/issues/24915)  
- Torch Inductor Graph issue:  
  - [https://github.com/vllm-project/vllm/pull/24772](https://github.com/vllm-project/vllm/pull/24772)

DBO support

- DBO PR get landed: [https://github.com/vllm-project/vllm/pull/23693](https://github.com/vllm-project/vllm/pull/23693) (Work together with Sage and Lucas)  
- HT support for DBO PR ready for review (combined with Lucas’ prefill support)  [https://github.com/vllm-project/vllm/pull/24845](https://github.com/vllm-project/vllm/pull/24845)

VLLM Contributions

- Several Refactoring/Fix PRs merged: [\#24902](https://github.com/vllm-project/vllm/pull/24902) [\#24887](https://github.com/vllm-project/vllm/pull/24887) [\#24774](https://github.com/vllm-project/vllm/pull/24774) [\#24696](https://github.com/vllm-project/vllm/pull/24696) [\#24674](https://github.com/vllm-project/vllm/pull/24674) \+ 4 other PRs  
- Several fix for CI: [\#24259](https://github.com/vllm-project/vllm/pull/24259) [\#24670](https://github.com/vllm-project/vllm/pull/24670)  
- Reviewed 40+ PRs

## Aug 20 - Sep 2

**Model Support for Deepseek V3.1**:

- Add Hopper DeepGEMM E8M0 for DeepSeekV3.1 scale\_fmt  
- [https://github.com/vllm-project/vllm/pull/23666](https://github.com/vllm-project/vllm/pull/23666)

**Performance Optimization**:

- Enable Piecewise CUDAGraph for DeepEP HT
  - [https://github.com/vllm-project/vllm/pull/24123](https://github.com/vllm-project/vllm/pull/24123)  
  - 33% E2E Throughput improvement for Decode  
- Enable DeepGEMM Linear on B200
  - [https://github.com/vllm-project/vllm/pull/23351](https://github.com/vllm-project/vllm/pull/23351)  
  - 1.5% E2E throughput improvement

Severe Bug Fix

- R1 Accuracy issue: routed\_scaling\_factor double mul  
  - [https://github.com/vllm-project/vllm/pull/24119](https://github.com/vllm-project/vllm/pull/24119)  
  - Meta is using vLLM main to deploy  
  - Meta reach out to express gratitude for the fast fix  
- Full Cuda graph Hang issue  
  - [https://github.com/vllm-project/vllm/pull/23595](https://github.com/vllm-project/vllm/pull/23595)  
  - Temporarily fix and will do more exploration later

DBO support

- [https://github.com/vllm-project/vllm/pull/23693](https://github.com/vllm-project/vllm/pull/23693) (Work together with Sage and Lucas)  
- HT single handle issue fixed

VLLM Contributions

- Several Refactoring/Fix PRs merged: [\#23287](https://github.com/vllm-project/vllm/pull/23287) [\#23858](https://github.com/vllm-project/vllm/pull/23858) [\#23689](https://github.com/vllm-project/vllm/pull/23689) [\#23660](https://github.com/vllm-project/vllm/pull/23660) [\#23591](https://github.com/vllm-project/vllm/pull/23591) [\#23370](https://github.com/vllm-project/vllm/pull/23370)  
- Reviewed 50+ PRs

## Aug 6 - Aug 19

I am nominated to be a vllm committer\! Thank so much to Kaichao [Michael Goin](mailto:mgoin@redhat.com) [Robert Shaw](mailto:robshaw@redhat.com),[Taneem Ibrahim](mailto:tibrahim@redhat.com), [Yuan Tang](mailto:yutang@redhat.com) and the vLLM community\!

[https://github.com/vllm-project/vllm/pull/22741](https://github.com/vllm-project/vllm/pull/22741)

**B200 Performance Optimization**:

- Cutlass MLA full cuda graph support  
  - [https://github.com/vllm-project/vllm/pull/22763](https://github.com/vllm-project/vllm/pull/22763)  
  - Also needed for DBO  
  - 6% E2E Throughput Improvement  
- Bug fix for FusedMoEModularKernel [\#22757](https://github.com/vllm-project/vllm/pull/22757)

**DBO support**:

- Several bugs fixed  
  - Fix set forward context error  
  - Fix assert error num\_tokens\_across\_dp is None  
  - Fix ubatch datatype issue  
  - Fix R1 accuracy issue  
- Build on B200 system, it is easy to benchmark now

**VLLM Contributions**:

- Several Refactoring PRs merged: [\#21968](https://github.com/vllm-project/vllm/pull/21968) [\#23137](https://github.com/vllm-project/vllm/pull/23137) [\#22860](https://github.com/vllm-project/vllm/pull/22860)  
- Reviewed 30+ PRs

## July 24 – Aug 5

**B200 Performance Optimization**:

- Per-token-group quant CUDA kernel  
  [#21476](https://github.com/vllm-project/vllm/pull/21476) — **15×** faster than the original Triton kernel (int8).  
  [#21867](https://github.com/vllm-project/vllm/pull/21867) — using `__nv_fp8_e4m3`, **10%** faster for FP8.  
  Works on all NVIDIA architectures, not only B200.
- NVFP4 optimization  
  Bug fix for Compressed Tensor NVFP4: [#21465](https://github.com/vllm-project/vllm/pull/21465)  
  Add FlashInfer MoE support for Compressed Tensor NVFP4: [#21639](https://github.com/vllm-project/vllm/pull/21639) — ~**15%** E2E throughput.
- Other perf wins  
  Non-contiguous support for FP8 quantization: [#21961](https://github.com/vllm-project/vllm/pull/21961) — ~**1%** E2E throughput.  
  Optimize `reshape_and_cache_flash` CUDA kernel: [#22036](https://github.com/vllm-project/vllm/pull/22036) — **20–40%** faster.

**B200 New DeepGemm Integration**:

- ✅ Done for this large scope! _Special thanks to the help from [Kaichao You](https://github.com/youkaichao) and [Chenggang Zhao](https://github.com/LyricZhao)_

- Unit test used to debug: [#21559](https://github.com/vllm-project/vllm/pull/21559) and log update: [#22208](https://github.com/vllm-project/vllm/pull/22208)  

**DBO Support**:

- WIP: Collaborated with Sage and Lucas — exciting new scope.

**Other Contributions**:

- Several code-refactoring PRs merged:  
  [#21631](https://github.com/vllm-project/vllm/pull/21631), [#21775](https://github.com/vllm-project/vllm/pull/21775), [#21787](https://github.com/vllm-project/vllm/pull/21787)
- Reviewed 10+ PRs.

---

## July 9 – July 23

**B200 Performance Optimization**:

- Per-token-group quant CUDA kernel for **FP8**:  
  [#21083](https://github.com/vllm-project/vllm/pull/21083) — ~**6%** E2E improvement; works on all NVIDIA architectures.  
- WIP at the time: per-token-group quant CUDA kernel for **int8** (later landed as [#21476](https://github.com/vllm-project/vllm/pull/21476)).
- NVFP4 optimization:  
  Bug fix for Compressed Tensor NVFP4 (ready to review then): [#21465](https://github.com/vllm-project/vllm/pull/21465)

**B200 New DeepGemm Integration**:

- Merged support for breaking DeepGEMM update on B200: [#20087](https://github.com/vllm-project/vllm/pull/20087)  
  Upstream DeepGEMM PR: deepseek-ai/DeepGEMM [#112](https://github.com/deepseek-ai/DeepGEMM/pull/112)
- Follow-up optimizations (all merged):  
  DeepEP low-latency bugfix: [#20833](https://github.com/vllm-project/vllm/pull/20833)  
  **~15%** E2E perf improvement: [#20841](https://github.com/vllm-project/vllm/pull/20841)  
  Breaking change fix: [#21187](https://github.com/vllm-project/vllm/pull/21187)  
  CUDA init error fix due to DeepGemm: [#21312](https://github.com/vllm-project/vllm/pull/21312)

**CI Bug Fixes**:

- Found and fixed quickly: [#20782](https://github.com/vllm-project/vllm/pull/20782), [#20845](https://github.com/vllm-project/vllm/pull/20845)

**Other Contributions**:

- Code-refactoring PRs merged: [#20770](https://github.com/vllm-project/vllm/pull/20770), [#20774](https://github.com/vllm-project/vllm/pull/20774), and others  
- Reviewed 10+ PRs.

---

## June 23 – July 8

**B200 Performance Optimization**:

- Quant vectorization utils optimization: [#20331](https://github.com/vllm-project/vllm/pull/20331)  
  +3% E2E for CUDA quant kernels; reusable for FP8 quant, `reshape_and_cache_flash`, etc.

**B200 New DeepGemm Integration**:

- WIP then: support new breaking DeepGEMM for B200: [#20087](https://github.com/vllm-project/vllm/pull/20087)  
  ~40% perf improvement for the GEMM kernel at specific batch sizes.  
  _Special thanks to [Michael Goin](https://github.com/mgoin) and [Varun Sundar Rabindranath](https://github.com/varun-sundar-rabindranath)._

**B200 DeepEP & PPLX Validation**:

- Bug fix: [#20094](https://github.com/vllm-project/vllm/pull/20094) — validation done.

**Severe CI Bug — Fixed**:

- Issue raised (blocker in main for ~1 month): [#20138](https://github.com/vllm-project/vllm/issues/20138)  
  Fix in two days: [#20204](https://github.com/vllm-project/vllm/pull/20204)

**Other Contributions**:

- Refactoring PRs merged: [#20187](https://github.com/vllm-project/vllm/pull/20187), [#20269](https://github.com/vllm-project/vllm/pull/20269), [#20334](https://github.com/vllm-project/vllm/pull/20334), plus more  
- Reviewed 10+ PRs.

---

## June 9 – June 20

**B200 Performance Optimization**:

- `align_moe_block_size` kernel optimization: [#19572](https://github.com/vllm-project/vllm/pull/19572) — ~**6%** E2E throughput.  
- Benchmark script refactor for GEMM: [#19627](https://github.com/vllm-project/vllm/pull/19627) — made future quant benchmarking easier.

**B200 DeepGemm Integration**:

- Initial integration: [#19820](https://github.com/vllm-project/vllm/pull/19820) — ~**40%** GEMM perf improvement.  
  _Thanks to [Robert Shaw](https://github.com/robertgshaw2-redhat)!_

**B200 DeepEP Integration**:

- Env setup & initial exploration.

**Other Contributions**:

- Helped review several PRs.

---

## June 2 – June 7

**B200 Performance Optimization**:

- Int8 quant kernel optimization: [#19233](https://github.com/vllm-project/vllm/pull/19233) — ~**10%** E2E throughput on B200.  
  _Thanks to [Michael Goin](https://github.com/mgoin)'s guidance!_  
  My first vLLM PR!

**Other Contributions**:

- Raised issues and reviewed several PRs.
