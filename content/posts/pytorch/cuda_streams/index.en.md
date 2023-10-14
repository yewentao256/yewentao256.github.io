---
title: "Pytorch Cuda Streams Introduction"
date: 2023-10-03T10:01:22+08:00
categories: ["pytorch"]
summary: "This article delves into the fundamental concepts of CUDA streams, parallel execution, and multi-GPU synchronization strategies. We analyze the advantages of using multiple CUDA streams and how to ensure proper task synchronization through CUDA events."
---

## Summary

This article delves into the fundamental concepts of CUDA streams, parallel execution, and multi-GPU synchronization strategies. We analyze the advantages of using multiple CUDA streams and how to ensure proper task synchronization through CUDA events.

## Basic Concepts

What is a Cuda stream? What are its main uses?

- Cuda streams can be understood as a queue for executing tasks sequentially. When tasks are submitted for execution on a CUDA stream, they are executed strictly in the order they were submitted.
- Tasks from different streams can be executed in parallel. This implies that we can use multiple streams to optimize tasks executed on the GPU.
- Note that task execution on a cuda stream and CPU is asynchronous by default.
- Main use: It allows developers to control parallelism, synchronization, and task execution order with finer granularity, leveraging GPU resources to optimize program execution efficiency.

By default, on which stream do PyTorch CUDA operations execute?

- Each GPU has a default Cuda stream.
- If the user doesn't specify any other stream explicitly, PyTorch's CUDA operations will execute on this stream.

## Parallel Execution

Why might we need to use multiple CUDA streams?

- We can use multiple CUDA streams to fully utilize the GPU, especially when there are multiple independent operations that can run in parallel.
- This method can be used not only for independent computational operations (such as multiplications and additions with no dependencies) but also for computation and data transfer (like `copy`) to further enhance performance.
- It's important to note that developers **must ensure that operations that depend on each other are placed on the same stream**. Otherwise, the computation results will be incorrect.

Why is the actual degree of parallelism dependent on the GPU's hardware resources?

There are many reasons, but we'll focus on the main ones:

- **Streaming Multiprocessors (SMs)** count: SMs are the basic units for parallel task processing. Each SM has a certain number of computational units (ALUs). For instance, V100 has 80 SMs, while GTX 1050 only has 6.
- Hardware capability level: NVIDIA GPUs have different **Compute Capability** levels. Devices with a level of 3.5 may support features like Hyper-Q and dynamic parallelism, while devices with a level of 2.0 do not.
- **Global memory bandwidth and caching**: Faster data transfer speeds (host-to-device, device-to-device) and memory access speeds can effectively increase parallelism.

Why do we need synchronization?

- **Data Accuracy**: If operations that depend on each other (computation and copy) are on different streams, we must synchronize to obtain the correct results.
- **Performance Measurement**: To accurately measure the execution time of GPU operations, we need to synchronize.
- **Error Checking**: Errors reported asynchronously might not be correct. Enabling synchronization helps in pinpointing issues.

## Data Transfer

How can we overlap GPU computations and data transfers using CUDA streams? A simple approach:

1. Create two Cuda streams: one for data transfer (Stream A) and one for computation (Stream B).
2. Initiate asynchronous transfer on Stream A.
3. Execute computations on Stream B.
4. Since operations are on different streams, they execute in parallel (overlap), improving performance.
5. Ensure all operations are complete through synchronization or events.

Some may ask, we just discussed that operations with dependencies must be synchronized, like first copying a tensor from host to GPU. We need to ensure that the data is fully transferred before starting the computation. So, what's the difference between synchronizing on different streams and executing on a single stream?

The key point is that we can start transferring the data for the next batch during computation:

While B is computing, if operations are executed on a single stream, we only compute without any data transfer. But if they're executed on two streams, we can use Stream A now for data transfer while B is computing, achieving basic overlapping parallelism, improving training efficiency.

Things to note:

- Not all GPUs support concurrent data transfer and computation. Older GPUs might not support it.
- Using too many streams or not managing them properly might lead to resource contention or scheduling issues, affecting efficiency.
- **Ensure data dependencies are handled correctly**.

## Events

How are CUDA events (`torch.cuda.Event`) related to CUDA streams?

`Events` are tools to mark specific points within a stream. We use events to monitor and synchronize the execution of streams. Their main uses are:

- **Synchronization**: As opposed to `cuda.synchronize` (which blocks the CPU and ensures that all operations on all streams on a device are complete), we can use events for finer-grained synchronization control. For instance, we can record an event in Stream A and wait for its completion in Stream B.
- **Performance Measurement**: Events can be used to measure the time of CUDA operations to further understand and optimize program performance.

How can we use CUDA events to precisely measure the time of a CUDA operation?

A simple approach using two CUDA events:

1. Before executing the operation, create and record an event (`start`).
2. Execute the operation.
3. After executing the operation, create and record another event (`end`).
4. Use `end.synchronize()` to ensure the operation is completed.
5. Use `start.elapsed_time(end)` to get the time.

## Multi-GPU Stream Synchronization

When using multiple GPUs, how can we ensure that stream operations on each GPU are synchronized correctly?

- Set the current device: Use `torch.cuda.set_device(device_id)` to ensure you're interacting with the right GPU.
- Use CUDA events: Events can synchronize across different GPUs, like recording an event in Stream A on GPU0 and then synchronizing the event in Stream B on GPU2.
- Explicitly synchronize a specific device using `torch.cuda.synchronize(device_id)`.
- Consider the dependencies between streams and ensure data dependencies are addressed.
- Avoid unnecessary synchronization and only synchronize when necessary to improve efficiency.

Moreover, we can also use **Direct Device-to-Device Communication (Peer-to-Peer, P2P)** to optimize synchronization efficiency: Through P2P, we can directly transfer data from one GPU to another without passing through the host, saving time and bandwidth (like NVIDIA's **NVLink** technology).

To achieve correctness and high performance in complex multi-GPU applications, besides considering stream synchronization, we also need to efficiently handle device-to-device communication, use primitives like `all-reduce`, `broadcast`, etc. For more details, refer to the author's previous article [distribution-training](../overview_of_distribution_training).

## Referrence

- [CUDA SEMANTICS](https://pytorch.org/docs/stable/notes/cuda.html)
- [TORCH CUDA](https://pytorch.org/docs/stable/cuda.html)
