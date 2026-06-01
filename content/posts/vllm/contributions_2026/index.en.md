---
title: "Bi-weekly Journal: Contributions to vLLM (2026)"
date: 2026-05-30T14:12:12+08:00
categories: ["vllm"]
summary: "My bi-weekly journal for contributions to vllm. (2026)"
---

## Summary

My bi-weekly journal for contributions to vllm.

All contributions: [https://github.com/vllm-project/vllm/graphs/contributors](https://github.com/vllm-project/vllm/graphs/contributors)  
All PR reviews: [https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aopen+reviewed-by%3Ayewentao256+](https://github.com/vllm-project/vllm/pulls?q=is%3Apr+is%3Aopen+reviewed-by%3Ayewentao256+)

---

## May 13 \- May 26

**GPU Model Runner V2**:

- **Migration from v1 to v2**: [https://github.com/vllm-project/vllm/issues/41286](https://github.com/vllm-project/vllm/issues/41286)
- Landed: [https://github.com/vllm-project/vllm/pull/39337](https://github.com/vllm-project/vllm/pull/39337) Oracle for model runner v2 \- dense model by default \[1/N\]  
  - Landed [https://github.com/vllm-project/vllm/pull/41761](https://github.com/vllm-project/vllm/pull/41761) Bug fix: logprob dtype int64/int32 issue  
- Under review: [https://github.com/vllm-project/vllm/pull/42665](https://github.com/vllm-project/vllm/pull/42665) Migration from v1 to v2, with more Llama and Mistral dense models \[2/N\]  
  - Under review: [https://github.com/vllm-project/vllm/pull/42759](https://github.com/vllm-project/vllm/pull/42759) Migrate Reset cache for both v2 and v1 model runner  
  - Landed [https://github.com/vllm-project/vllm/pull/43233](https://github.com/vllm-project/vllm/pull/43233) Force v1 runner for tests  
  - Landed [https://github.com/vllm-project/vllm/pull/43139](https://github.com/vllm-project/vllm/pull/43139) Fix lora Triton Error \[CUDA\]: device-side assert triggered  
  - Landed [https://github.com/vllm-project/vllm/pull/42778](https://github.com/vllm-project/vllm/pull/42778) Fix prompt logprobs calculation Sizes of tensors must match error  
  - Landed [https://github.com/vllm-project/vllm/pull/42676](https://github.com/vllm-project/vllm/pull/42676) Fix kv\_connector pre\_forward order  
  - Landed [https://github.com/vllm-project/vllm/pull/42673](https://github.com/vllm-project/vllm/pull/42673) Support reload weights (sleep mode)  
- Under review: [https://github.com/vllm-project/vllm/pull/42667](https://github.com/vllm-project/vllm/pull/42667) Migration from v1 to v2, with Qwen and DSv2 MOE models \[3/N\]

Large Scaling Serving

- Landed: [https://github.com/vllm-project/vllm/pull/40841](https://github.com/vllm-project/vllm/pull/40841) **Support a node-local external DP mode with a single vllm serve and aggregated admin health endpoint**  
- Under review: [https://github.com/vllm-project/vllm/pull/43707](https://github.com/vllm-project/vllm/pull/43707)  Optimize shutdown logs, easier to follow and consistent  
- Under review: [https://github.com/vllm-project/vllm/pull/43688](https://github.com/vllm-project/vllm/pull/43688) \[Feature\] SSL support for dp supervisor

Kernel Optimization

- Under review: [https://github.com/vllm-project/vllm/pull/43706](https://github.com/vllm-project/vllm/pull/43706) Optimize cutlass fp8 scaled mm bypassing padding, 20% kernel performance improvement  
- Under review: [http://github.com/vllm-project/vllm/pull/43349](http://github.com/vllm-project/vllm/pull/43349) \[Perf\] Optimize remapped greedy draft token selection for Eagle3 and DFlash, 37\~81% kernel performance improvement  
- Under review: [https://github.com/vllm-project/vllm/pull/43137](https://github.com/vllm-project/vllm/pull/43137) **Optimize per\_token\_group\_quant using regsiter directly, 4.5% E2E Throughput improvement**  
- Under review: [https://github.com/vllm-project/vllm/pull/43014](https://github.com/vllm-project/vllm/pull/43014) Optimize moe permute by pre-allocate buffer, 9\~14% kernel performance improvement  
- Landed [https://github.com/vllm-project/vllm/pull/42988](https://github.com/vllm-project/vllm/pull/42988) \[Perf\] zeros \-\> empty to remove additional fill  
- Landed [https://github.com/vllm-project/vllm/pull/42774](https://github.com/vllm-project/vllm/pull/42774) **\[Perf\] Padded nvfp4 quant kernel to remove additional copy, 2.4%\~5.7% e2e performance improvement**  
- Landed [https://github.com/vllm-project/vllm/pull/42651](https://github.com/vllm-project/vllm/pull/42651) **\[Perf\] Optimize CutlassFP8ScaledMMLinearKernel when padding needed by pre-weight processing, 13.5% TTFT improvement**  
- Landed [https://github.com/vllm-project/vllm/pull/42561](https://github.com/vllm-project/vllm/pull/42561) \[Perf\] Optimize MLA attention \_v\_up\_proj bmm by removing additional copy

Batch Invariant

- Landed: [https://github.com/vllm-project/vllm/pull/42456](https://github.com/vllm-project/vllm/pull/42456) Support compile mode for batch invariance on SM80  
- Under review: [https://github.com/vllm-project/vllm/pull/42453](https://github.com/vllm-project/vllm/pull/42453) \[Feature\] Support batch invariant rms norm with residual

vLLM Contributions

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/43234](https://github.com/vllm-project/vllm/pull/43234)  
  - [https://github.com/vllm-project/vllm/pull/43358](https://github.com/vllm-project/vllm/pull/43358)  
  - [https://github.com/vllm-project/vllm/pull/42889](https://github.com/vllm-project/vllm/pull/42889)  
  - [https://github.com/vllm-project/vllm/pull/42767](https://github.com/vllm-project/vllm/pull/42767)  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/43261](https://github.com/vllm-project/vllm/pull/43261)  
  - [http://github.com/vllm-project/vllm/pull/42563](http://github.com/vllm-project/vllm/pull/42563)

## April 29 \- May 12

**GPU Model Runner V2**:

- **Migration from v1 to v2**: [https://github.com/vllm-project/vllm/issues/41286](https://github.com/vllm-project/vllm/issues/41286)
- Under review: [https://github.com/vllm-project/vllm/pull/39337](https://github.com/vllm-project/vllm/pull/39337) **Oracle for model runner v2 \- dense model by default \[1/N\]**  
  - Landed [https://github.com/vllm-project/vllm/pull/40559](https://github.com/vllm-project/vllm/pull/40559) Add logprob\_token\_ids support  
  - Landed [https://github.com/vllm-project/vllm/pull/41285](https://github.com/vllm-project/vllm/pull/41285) Fix v2 compile counter num\_gpu\_runner\_capture\_triggers and num\_cudagraph\_captured  
  - Landed [https://github.com/vllm-project/vllm/pull/41761](https://github.com/vllm-project/vllm/pull/41761) Bug fix: logprob dtype int64/int32 issue  
  - Under review: [https://github.com/vllm-project/vllm/pull/41667](https://github.com/vllm-project/vllm/pull/41667) Support stock torch compile for v2

Large Scaling Serving

- Under review: [https://github.com/vllm-project/vllm/pull/40841](https://github.com/vllm-project/vllm/pull/40841) **Support a node-local external DP mode with a single vllm serve and aggregated admin health endpoint**  
- Landed: [https://github.com/vllm-project/vllm/pull/40839](https://github.com/vllm-project/vllm/pull/40839) Fix status update address for non-MOE model within external dp mode  
- Landed: [https://github.com/vllm-project/vllm/pull/39832](https://github.com/vllm-project/vllm/pull/39832) \[KV Connector\] Remove compat support for pre-v0.12.0 constructor signatures without KVCacheConfig  
- Landed: [https://github.com/vllm-project/vllm/pull/42460](https://github.com/vllm-project/vllm/pull/42460) \[Perf\] Optimize MLA compute\_prefill\_context memory allocation

Batch Invariant

- Landed: [https://github.com/vllm-project/vllm/pull/40408](https://github.com/vllm-project/vllm/pull/40408) **Batch invariance with Cutlass fp8 support, 28.9% E2E latency improvement**  
- Under review: [https://github.com/vllm-project/vllm/pull/42456](https://github.com/vllm-project/vllm/pull/42456) Support compile mode for batch invariance on SM80  
- Under review: [https://github.com/vllm-project/vllm/pull/42453](https://github.com/vllm-project/vllm/pull/42453) \[Feature\] Support batch invariant rms norm with residual

Pooling Model (**Performance issue completed** [https://github.com/vllm-project/vllm/issues/35631](https://github.com/vllm-project/vllm/issues/35631))

- Landed: [https://github.com/vllm-project/vllm/pull/41163](https://github.com/vllm-project/vllm/pull/41163) Optimize AllPool.forward by slicing first, 51% faster in the method level benchmark

vLLM Contributions

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/41471](https://github.com/vllm-project/vllm/pull/41471)  
  - [https://github.com/vllm-project/vllm/pull/41417](https://github.com/vllm-project/vllm/pull/41417)  
  - [https://github.com/vllm-project/vllm/pull/42341](https://github.com/vllm-project/vllm/pull/42341)  
  - [https://github.com/vllm-project/vllm/pull/41993](https://github.com/vllm-project/vllm/pull/41993)  
  - etc..  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/42342](https://github.com/vllm-project/vllm/pull/42342)  
  - [https://github.com/vllm-project/vllm/pull/41261](https://github.com/vllm-project/vllm/pull/41261)  
  - [https://github.com/vllm-project/vllm/pull/42081](https://github.com/vllm-project/vllm/pull/42081)  
  - [https://github.com/vllm-project/vllm/pull/41288](https://github.com/vllm-project/vllm/pull/41288)  
  - etc…

## April 15 - April 28

**GPU Model Runner V2**:

- Under review: [https://github.com/vllm-project/vllm/pull/35214](https://github.com/vllm-project/vllm/pull/35214) Optimize Sampler Redundant Copy for Model Runner v2, 1.8% Throughput Improvement  
  - Under review: [https://github.com/vllm-project/vllm/pull/39337](https://github.com/vllm-project/vllm/pull/39337) **Oracle for model runner v2 \- dense model by default \[1/N\]**  
  - Landed [https://github.com/vllm-project/vllm/pull/40648](https://github.com/vllm-project/vllm/pull/40648) Fix block table IMA issue  
  - Under review [https://github.com/vllm-project/vllm/pull/40559](https://github.com/vllm-project/vllm/pull/40559) Add logprob\_token\_ids support  
  - Landed [https://github.com/vllm-project/vllm/pull/39937](https://github.com/vllm-project/vllm/pull/39937) Multiple prompt logprobs support

Large Scaling Serving

- Under review: [https://github.com/vllm-project/vllm/pull/38850](https://github.com/vllm-project/vllm/pull/38850) **Optimize DCP communication with reusable collective scratch buffers, 1.5%\~4% E2E throughput improvement**  
- Under review: [https://github.com/vllm-project/vllm/pull/40841](https://github.com/vllm-project/vllm/pull/40841) **Support a node-local external DP mode with a single vllm serve and aggregated admin health endpoint**  
- Under review: [https://github.com/vllm-project/vllm/pull/40839](https://github.com/vllm-project/vllm/pull/40839) Fix status update address for non-MOE model within external dp mode  
- Under review: [https://github.com/vllm-project/vllm/pull/40174](https://github.com/vllm-project/vllm/pull/40174) Add fast all2all kernel, tested for DCP, 1.1% Throughput improvement  
- Under review: [https://github.com/vllm-project/vllm/pull/39832](https://github.com/vllm-project/vllm/pull/39832) \[KV Connector\] Remove compat support for pre-v0.12.0 constructor signatures without KVCacheConfig

Batch Invariant

- Under review: [https://github.com/vllm-project/vllm/pull/40408](https://github.com/vllm-project/vllm/pull/40408) **Batch invariance with Cutlass fp8 support, 28.9% E2E latency improvement**  
- Landed: [https://github.com/vllm-project/vllm/pull/40413](https://github.com/vllm-project/vllm/pull/40413) Optimize batch invariant with fused rms norm, 2.1% E2E latency improvement  
- Landed: [https://github.com/vllm-project/vllm/pull/39820](https://github.com/vllm-project/vllm/pull/39820) Fix batch invariance nvfp4 support

Pooling Model (issue [https://github.com/vllm-project/vllm/issues/35631](https://github.com/vllm-project/vllm/issues/35631))

- Under review: [https://github.com/vllm-project/vllm/pull/41163](https://github.com/vllm-project/vllm/pull/41163) Optimize AllPool.forward by slicing first, 51% faster in the method level benchmark

vLLM Contributions

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/40640](https://github.com/vllm-project/vllm/pull/40640)  
  - [https://github.com/vllm-project/vllm/pull/40540](https://github.com/vllm-project/vllm/pull/40540)  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/40053](https://github.com/vllm-project/vllm/pull/40053)  
  - [https://github.com/vllm-project/vllm/pull/39938](https://github.com/vllm-project/vllm/pull/39938)

## April 1 \- April 14

**GPU Model Runner V2**:

- Under review: [https://github.com/vllm-project/vllm/pull/38390](https://github.com/vllm-project/vllm/pull/38390) E/P/D disaggregation support  
  - Under review: [https://github.com/vllm-project/vllm/pull/35214](https://github.com/vllm-project/vllm/pull/35214) Optimize Sampler Redundant Copy for Model Runner v2, 1.8% Throughput Improvement  
  - Under review: [https://github.com/vllm-project/vllm/pull/35206](https://github.com/vllm-project/vllm/pull/35206) Support Sequence Parallel for Model Runer v2 (Piecewise Cudagraph, PP=1)  
  - Under review: [https://github.com/vllm-project/vllm/pull/39337](https://github.com/vllm-project/vllm/pull/39337) **Oracle for model runner v2 \- dense model by default \[1/N\]**  
  - Landed [https://github.com/vllm-project/vllm/pull/39353](https://github.com/vllm-project/vllm/pull/39353) Fix flex attention kv blocks calculation issue

Large Scaling Serving

- Under review: [https://github.com/vllm-project/vllm/pull/38850](https://github.com/vllm-project/vllm/pull/38850) **Optimize DCP communication with reusable collective scratch buffers, 1.5%\~4% E2E throughput improvement**

Batch Invariant

- Landed: [https://github.com/vllm-project/vllm/pull/39320](https://github.com/vllm-project/vllm/pull/39320) Fix batch invariant test issue, bs=1 with max\_seq\_num \= 1  
- Landed [https://github.com/vllm-project/vllm/pull/39322](https://github.com/vllm-project/vllm/pull/39322)  batch invariance **nvfp4** support linear  
- Guide community to work on batch invariance, eg. [https://github.com/vllm-project/vllm/pull/39912](https://github.com/vllm-project/vllm/pull/39912)
- And more…

Pooling Model (issue [https://github.com/vllm-project/vllm/issues/35631](https://github.com/vllm-project/vllm/issues/35631))

- Under review: [https://github.com/vllm-project/vllm/pull/39533](https://github.com/vllm-project/vllm/pull/39533) **Batched projector for pooling model embed, 1.8% throughput improvement**  
- Landed: [https://github.com/vllm-project/vllm/pull/39113](https://github.com/vllm-project/vllm/pull/39113) **Optimize redundant sync for pooling model, 3.7% Throughput Improvement**

vLLM Contributions

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/38842](https://github.com/vllm-project/vllm/pull/38842)  
  - [https://github.com/vllm-project/vllm/pull/39750](https://github.com/vllm-project/vllm/pull/39750)  
  - [https://github.com/vllm-project/vllm/pull/39100](https://github.com/vllm-project/vllm/pull/39100)  
  - And more…  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/39347](https://github.com/vllm-project/vllm/pull/39347)  
  - [https://github.com/vllm-project/vllm/pull/39225](https://github.com/vllm-project/vllm/pull/39225)  
  - [https://github.com/vllm-project/vllm/pull/39219](https://github.com/vllm-project/vllm/pull/39219)  
  - [https://github.com/vllm-project/vllm/pull/39086](https://github.com/vllm-project/vllm/pull/39086)  
  - [https://github.com/vllm-project/vllm/pull/38915](https://github.com/vllm-project/vllm/pull/38915)  
  - And more …

## March 18 - March 31

**News: [https://vllm.ai/blog/mrv2](https://vllm.ai/blog/mrv2) first released\!**

**GPU Model Runner V2**:

- Under review: [https://github.com/vllm-project/vllm/pull/38390](https://github.com/vllm-project/vllm/pull/38390) E/P/D disaggregation support  
  - Under review: [https://github.com/vllm-project/vllm/pull/35214](https://github.com/vllm-project/vllm/pull/35214) Optimize Sampler Redundant Copy for Model Runner v2, 1.8% Throughput Improvement  
  - Under review: [https://github.com/vllm-project/vllm/pull/35206](https://github.com/vllm-project/vllm/pull/35206) Support Sequence Parallel for Model Runer v2 (Piecewise Cudagraph, PP=1)  
  - Landed: [https://github.com/vllm-project/vllm/pull/37488](https://github.com/vllm-project/vllm/pull/37488) **EPLB Support for GPU Model Runner v2**

Large Scaling Serving

- Under review: [https://github.com/vllm-project/vllm/pull/38287](https://github.com/vllm-project/vllm/pull/38287) Skip kv connector empty work, Around 1% Throughput Improvement  
- Landed: [https://github.com/vllm-project/vllm/pull/38383](https://github.com/vllm-project/vllm/pull/38383) Remove dead code in kv connector and model runner

Batch Invariant

- Under review: [https://github.com/vllm-project/vllm/pull/38039](https://github.com/vllm-project/vllm/pull/38039) Fix batch invariance for offline serving and aux stream  
- Landed: [https://github.com/vllm-project/vllm/pull/38014](https://github.com/vllm-project/vllm/pull/38014) Add batch invariant test for b200  
- Landed: [https://github.com/vllm-project/vllm/pull/37895](https://github.com/vllm-project/vllm/pull/37895) Add batch invariant test: Block FP8 \+ small MOE  
- Landed: [https://github.com/vllm-project/vllm/pull/37718](https://github.com/vllm-project/vllm/pull/37718) Fix fp8 deepgemm batch invariant  
- And more…

Kernel Optimization

- Landed: [https://github.com/vllm-project/vllm/pull/37340](https://github.com/vllm-project/vllm/pull/37340) Add tuned triton moe config for Qwen3.5 H200, 9.9% E2E throughput improvement

Pooling Model (issue [https://github.com/vllm-project/vllm/issues/35631](https://github.com/vllm-project/vllm/issues/35631))

- Landed: [https://github.com/vllm-project/vllm/pull/37347](https://github.com/vllm-project/vllm/pull/37347) Optimize token\_embed for pooling models, 1.0% token throughput improvement  
- Landed: [https://github.com/vllm-project/vllm/pull/38559](https://github.com/vllm-project/vllm/pull/38559) **Optimize mean pooling using chunks and index\_add, 5.9% E2E throughput improvement**  
- Landed: [https://github.com/vllm-project/vllm/pull/38139](https://github.com/vllm-project/vllm/pull/38139) **Remove redundant device copies for CPU-only pooling token IDs, 48.9% E2E throughput improvement**  
- And more…

vLLM Contributions

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/38153](https://github.com/vllm-project/vllm/pull/38153)  
  - [https://github.com/vllm-project/vllm/pull/38048](https://github.com/vllm-project/vllm/pull/38048)  
  - [https://github.com/vllm-project/vllm/pull/37808](https://github.com/vllm-project/vllm/pull/37808)  
  - [https://github.com/vllm-project/vllm/pull/37568](https://github.com/vllm-project/vllm/pull/37568)  
  - And more…  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/38573](https://github.com/vllm-project/vllm/pull/38573)  
  - [https://github.com/vllm-project/vllm/pull/37573](https://github.com/vllm-project/vllm/pull/37573)  
  - And more …

## March 3 - March 17

**GPU Model Runner V2**:

- Under review: [https://github.com/vllm-project/vllm/pull/35214](https://github.com/vllm-project/vllm/pull/35214) Optimize Sampler Redundant Copy for Model Runner v2, 1.8% Throughput Improvement  
  - Under review: [https://github.com/vllm-project/vllm/pull/35206](https://github.com/vllm-project/vllm/pull/35206) Support Sequence Parallel for Model Runer v2 (Piecewise Cudagraph, PP=1)  
  - Under review: [https://github.com/vllm-project/vllm/pull/37195](https://github.com/vllm-project/vllm/pull/37195) \[V0 Deprecation\] Deprecate virtual engine

**Large Scaling Serving**:

- Landed: [https://github.com/vllm-project/vllm/pull/35781](https://github.com/vllm-project/vllm/pull/35781) **Optimize scheduler overhead for PD disaggregation, around 5% E2E perf improvement**  
- Landed: [https://github.com/vllm-project/vllm/pull/36424](https://github.com/vllm-project/vllm/pull/36424) Remove dead code in KV connector  
- Landed: [https://github.com/vllm-project/vllm/pull/36170](https://github.com/vllm-project/vllm/pull/36170) Remove default ray dependency

**Kernel Optimization**:

- Under review: [https://github.com/vllm-project/vllm/pull/37340](https://github.com/vllm-project/vllm/pull/37340) **Add tuned triton moe config for Qwen3.5 H200, 9.9% E2E throughput improvement**

**Pooling Model**:

- Under review: [https://github.com/vllm-project/vllm/pull/37347](https://github.com/vllm-project/vllm/pull/37347) Optimize token\_embed for pooling models, 2.8% token throughput improvement  
- Landed: [https://github.com/vllm-project/vllm/pull/36710](https://github.com/vllm-project/vllm/pull/36710) **Optimize compute maxsim using batched version, 3.2% E2E throughput improvement**  
- Landed: [https://github.com/vllm-project/vllm/pull/36159](https://github.com/vllm-project/vllm/pull/36159) **Compute maxsim in worker side, reducing redundant copies, 2.7% E2E throughput improvement**  
- Landed: [https://github.com/vllm-project/vllm/pull/35427](https://github.com/vllm-project/vllm/pull/35427) Fix maxsim cuda platform and add cli to control it

**Other Contributions**:

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/37313](https://github.com/vllm-project/vllm/pull/37313)  
  - [https://github.com/vllm-project/vllm/pull/36171](https://github.com/vllm-project/vllm/pull/36171)  
  - [https://github.com/vllm-project/vllm/pull/36049](https://github.com/vllm-project/vllm/pull/36049)  
  - And more…  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/36693](https://github.com/vllm-project/vllm/pull/36693)  
  - [https://github.com/vllm-project/vllm/pull/36674](https://github.com/vllm-project/vllm/pull/36674)  
  - [https://github.com/vllm-project/vllm/pull/36529](https://github.com/vllm-project/vllm/pull/36529)  
  - And several more  

## Feb 18 - March 2

**GPU Model Runner V2**:

- [https://github.com/vllm-project/vllm/pull/35333](https://github.com/vllm-project/vllm/pull/35333) Optimize model runner v2 prepare\_inputs copy logic, 6.1% E2E throughput improvement. Nick has a PR after [https://github.com/vllm-project/vllm/pull/35561](https://github.com/vllm-project/vllm/pull/35561)  
- [https://github.com/vllm-project/vllm/pull/35214](https://github.com/vllm-project/vllm/pull/35214) **Optimize Sampler Redundant Copy for Model Runner v2, 1.8% Throughput Improvement**  
- [https://github.com/vllm-project/vllm/pull/35206](https://github.com/vllm-project/vllm/pull/35206) **Support Sequence Parallel for Model Runer v2 (Piecewise Cudagraph, PP=1)**  
- [https://github.com/vllm-project/vllm/pull/34903](https://github.com/vllm-project/vllm/pull/34903) Fix illegal memory access issue for model runner v2

**Async Scheduling**:

- Under review: [https://github.com/vllm-project/vllm/pull/34029](https://github.com/vllm-project/vllm/pull/34029) Optimize async scheduling redundant copy, 0.9% E2E throughput improvement  
- Under review: Optimize sampled\_token\_ids using numpy and remove tolist, 0.9% E2E throughput improvement [https://github.com/vllm-project/vllm/pull/35446](https://github.com/vllm-project/vllm/pull/35446)

**Large Scaling Serving**:

- Under review: [https://github.com/vllm-project/vllm/pull/35781](https://github.com/vllm-project/vllm/pull/35781) **Optimize scheduler overhead for PD disaggregation, around 5% E2E perf improvement**

**Pooling Model**:

- Landed: [https://github.com/vllm-project/vllm/pull/35427](https://github.com/vllm-project/vllm/pull/35427) \[Refactor\] Fix maxsim cuda platform and add cli to control it  
- Landed: [https://github.com/vllm-project/vllm/pull/35330](https://github.com/vllm-project/vllm/pull/35330) **Optimize maxsim scores computation for pooling models, 13.9% E2E throughput improvement**  
- Landed: [https://github.com/vllm-project/vllm/pull/35127](https://github.com/vllm-project/vllm/pull/35127) Optimize pooling model redundant copy, 1.8% throughput improvement

**Other Contributions**:

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/35634](https://github.com/vllm-project/vllm/pull/35634)  
  - [https://github.com/vllm-project/vllm/pull/35441](https://github.com/vllm-project/vllm/pull/35441)  
  - [https://github.com/vllm-project/vllm/pull/35418](https://github.com/vllm-project/vllm/pull/35418)  
  - And more…  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/35314](https://github.com/vllm-project/vllm/pull/35314)  
  - [https://github.com/vllm-project/vllm/pull/34961](https://github.com/vllm-project/vllm/pull/34961)  
- Lead Fix all of the mypy check, issue in [\#26533](https://github.com/vllm-project/vllm/issues/26533)  
- Batch invariant: Lead and track all progress in [https://github.com/vllm-project/vllm/issues/27433](https://github.com/vllm-project/vllm/issues/27433)

## Feb 4 - Feb 17

**GPU Model Runner V2**:

- Landed: [https://github.com/vllm-project/vllm/pull/34179](https://github.com/vllm-project/vllm/pull/34179) **\[Feature\] Decode Context Parallel support for GPU model runner v2**  
  - Co-authored with Summer and landed: [https://github.com/vllm-project/vllm/pull/33960](https://github.com/vllm-project/vllm/pull/33960) Pipeline Parallel support for Model Runner V2 (git diff shared)

**Async Scheduling**:

- Under review: [https://github.com/vllm-project/vllm/pull/34029](https://github.com/vllm-project/vllm/pull/34029) Optimize async scheduling redundant copy, 0.9% E2E throughput improvement  
- Landed: [https://github.com/vllm-project/vllm/pull/32975](https://github.com/vllm-project/vllm/pull/32975) Optimize detokenizer python logic  
- Landed: [https://github.com/vllm-project/vllm/pull/33612](https://github.com/vllm-project/vllm/pull/33612) **Optimize spec decoding \+ async scheduling, 1.5% Throughput improvement**

**Performance optimizations**:

- Landed: [https://github.com/vllm-project/vllm/pull/33449](https://github.com/vllm-project/vllm/pull/33449) Remove align block size logic in moe\_permute  
- Landed: [https://github.com/vllm-project/vllm/pull/33368](https://github.com/vllm-project/vllm/pull/33368) **Pipeline Parallel Async send/recv, 2.9% E2E throughput improvement**

**Batch Invariant**:

- Lead and track all progress in [https://github.com/vllm-project/vllm/issues/27433](https://github.com/vllm-project/vllm/issues/27433)  
- Leading for Community contributions
  - [\[Doc\] Add Mistral-7b-v0.3 model to the batch invariance validated model \#34584](https://github.com/vllm-project/vllm/pull/34584)  
  - [\[Feature\] Enable TRITON\_ATTN for Batch Invariance \#33688](https://github.com/vllm-project/vllm/pull/33688)  
  - [\[Core\] Add determinism warmup automation for batch invariant mode \#33537](https://github.com/vllm-project/vllm/pull/33537)  
  - [\[Doc\] Add Qwen2.5 models to batch invariance tested models \#33016](https://github.com/vllm-project/vllm/pull/33016)

**Other Contributions**:

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/34263](https://github.com/vllm-project/vllm/pull/34263)  
  - [https://github.com/vllm-project/vllm/pull/33593](https://github.com/vllm-project/vllm/pull/33593)  
  - [https://github.com/vllm-project/vllm/pull/33944](https://github.com/vllm-project/vllm/pull/33944)  
  - And more…  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/33998](https://github.com/vllm-project/vllm/pull/33998)  

## Jan 21 - Feb 3

**Async Scheduling**:

- Done: \[Feature\]: Async Scheduling \+ Pipeline Parallel Support (V1) [https://github.com/vllm-project/vllm/issues/32701](https://github.com/vllm-project/vllm/issues/32701)  
  - Landed [https://github.com/vllm-project/vllm/pull/32618](https://github.com/vllm-project/vllm/pull/32618) **Fully support for async scheduling \+ PP, 30.8% E2E throughput improvement, 31.8% TPOT improvement**  
- Other Optimizations  
  - Under review: [https://github.com/vllm-project/vllm/pull/32975](https://github.com/vllm-project/vllm/pull/32975) Optimize detokenizer python logic  
  - Under review: [https://github.com/vllm-project/vllm/pull/33612](https://github.com/vllm-project/vllm/pull/33612) Optimize spec decoding \+ async scheduling, 1.5% Throughput improvement

**Performance optimizations**:

- Landed [https://github.com/vllm-project/vllm/pull/32892](https://github.com/vllm-project/vllm/pull/32892) **Optimize moe\_permute kernel, 40%\~300% kernel performance improvement**  
- Under review: [https://github.com/vllm-project/vllm/pull/33593](https://github.com/vllm-project/vllm/pull/33593) Optimize Python Slice Operation using islice instead of \[:\]  
- Under review: [https://github.com/vllm-project/vllm/pull/33449](https://github.com/vllm-project/vllm/pull/33449) Remove align block size logic in moe\_permute  
- Under review: [https://github.com/vllm-project/vllm/pull/33368](https://github.com/vllm-project/vllm/pull/33368) **Pipeline Parallel Async send/recv, 2.9% E2E throughput improvement**  
- Landed Optimize dcp allocate tensor [https://github.com/vllm-project/vllm/pull/33102](https://github.com/vllm-project/vllm/pull/33102)

**Batch Invariant**:

- Lead and track all progress in [https://github.com/vllm-project/vllm/issues/27433](https://github.com/vllm-project/vllm/issues/27433)  
- Leading community to deliver bug fixes / features  

**Other Contributions**:

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/33108](https://github.com/vllm-project/vllm/pull/33108)  
  - [https://github.com/vllm-project/vllm/pull/32812](https://github.com/vllm-project/vllm/pull/32812)  
  - [https://github.com/vllm-project/vllm/pull/33722](https://github.com/vllm-project/vllm/pull/33722)  
  - And more…  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/32949](https://github.com/vllm-project/vllm/pull/32949)  

## Jan 5 - Jan 20

**Async Scheduling**:

- \[Feature\]: Async Scheduling \+ Pipeline Parallel Support  
  - Landed: [\[Feature\] Support async scheduling \+ PP with constraints \#32359](https://github.com/vllm-project/vllm/pull/32359)  
  - **Under review: [\[Feature\] Fully support for async scheduling \+ PP, 15% E2E throughput improvement, 16% TPOT improvement \#32618](https://github.com/vllm-project/vllm/pull/32618)**  
- Landed Optimizations  
  - [https://github.com/vllm-project/vllm/pull/32211](https://github.com/vllm-project/vllm/pull/32211) Optimize requests abort  
  - [https://github.com/vllm-project/vllm/pull/32056](https://github.com/vllm-project/vllm/pull/32056) Optimize async scheduling placeholder using empty  
  - [https://github.com/vllm-project/vllm/pull/32034](https://github.com/vllm-project/vllm/pull/32034) Remove numpy split in async scheduling

**Performance optimizations**:

- **(Done) Tracking issue: Optimizations for MOE cutlass models [https://github.com/vllm-project/vllm/issues/31755](https://github.com/vllm-project/vllm/issues/31755)**
- Tasks (All landed)  
  - Optimize grouped\_topk kernel [\[Perf\] Optimize group\_topk kernel, 1.9% Throughput improvement, 2.1% TPOT improvemnt \#30159](https://github.com/vllm-project/vllm/pull/30159)  
  - Optimize additional fill(0) in cutlass moe [\[Perf\] Optimize additional fill(0) in cutlass moe, 2.9% E2E throughput improvement, 10.8% TTFT improvement \#31754](https://github.com/vllm-project/vllm/pull/31754)  
  - Optimize cutlass moe problem size calculation [\[Perf\] Optimize cutlass moe problem size calculation, 5.3% E2E Throughput improvement, 2.2% TTFT improvement \#31830](https://github.com/vllm-project/vllm/pull/31830)  
  - Optimize group topk kernel further [\[Perf\] Optimize grouped topk kernel, 1.2%\~2% E2E Throughput improvement \#32058](https://github.com/vllm-project/vllm/pull/32058)

**Batch Invariant**:

- Lead and track all progress in [https://github.com/vllm-project/vllm/issues/27433](https://github.com/vllm-project/vllm/issues/27433)  
- **vLLM office hour speaker (Jan 08\)** [https://docs.google.com/presentation/d/1iaZkoyf2VDQFc3TB2DGld2MpZ-uLsVrXdtYWfT9hEBc/edit?slide=id.g39235cbcce8\_2\_318\#slide=id.g39235cbcce8\_2\_318](https://docs.google.com/presentation/d/1iaZkoyf2VDQFc3TB2DGld2MpZ-uLsVrXdtYWfT9hEBc/edit?slide=id.g39235cbcce8_2_318#slide=id.g39235cbcce8_2_318)  
  [https://www.youtube.com/watch?v=sDLR9DvEFq4](https://www.youtube.com/watch?v=sDLR9DvEFq4)  
  
**Other Contributions**:

- Refactoring PRs  
  - [https://github.com/vllm-project/vllm/pull/32692](https://github.com/vllm-project/vllm/pull/32692)
  - [https://github.com/vllm-project/vllm/pull/32610](https://github.com/vllm-project/vllm/pull/32610)  
  - [https://github.com/vllm-project/vllm/pull/32433](https://github.com/vllm-project/vllm/pull/32433)  
  - And several more…  
- Bug Fix PRs:  
  - [https://github.com/vllm-project/vllm/pull/32622](https://github.com/vllm-project/vllm/pull/32622)
- Lead Fix all of the mypy check, issue in [\#26533](https://github.com/vllm-project/vllm/issues/26533)  
  - [https://github.com/vllm-project/vllm/pull/32722](https://github.com/vllm-project/vllm/pull/32722)
