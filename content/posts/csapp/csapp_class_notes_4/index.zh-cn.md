---
title: "CSAPP Class Notes(4)"
date: 2023-02-03T19:55:49+08:00
categories: ["csapp"]
summary: "My note while learning through CSAPP-15213 videos. Including Overview, Bits, Bytes, and Integers, Floating Point, Machine Level Programing, Program Optimization, Memory, Concurrency and Network."
---

## 等待被翻译

非常抱歉，看起来这篇博文还没有被翻译成中文，请等待一段时间

## 12. Network Programming

### 12.1 Basic Concepts

![image](/csapp/resources/network-hardware-organization.png)

In Linux, handling network is similar to handle a file (socket interface).

Lowest Level: **Ethernet** Segment. eg: MAC address(00:16:ea:e3:54:e6)

Next Level: Bridged Ethernet Segment

![image](/csapp/resources/network-briged-ethernet.png)

Next Level: **internets** - connecting multiple LANs (local area network) through routers

![image](/csapp/resources/network-internets.png)

### 12.2 Protocol

We use **Protocol** to send bits across incompatible LANs and WANs (wide area network)

![image](/csapp/resources/network-data-encapsulation.png)

**IP(Internet Protocol)**: basic naming scheme from host to host (doesn't know send for what process)

**UDP(Unreliable Datagram Protocol)**: Uses IP to deliver **unreliable** datagram from process to process

**TCP(Transmission Control Protocol)**: Uses IP to deliver **reliable** byte streams from process to process over connections.

usage: TCP/IP or UDP/IP

![image](/csapp/resources/network-organization-of-application.png)

### 12.3 Domain names

![image](/csapp/resources/network-domain-names.png)

DNS (Domain Name System): Used for mapping between IP addresses and domain names. Note: mapping is multiple to multiple.

### 12.4 Connections

Connections features:

- point to point: connects a pair of processes
- full-duplex: data can flow in both directions
- reliable

A **socket** is an endpoint of connection

A **port** identifies a process.

![image](/csapp/resources/network-socket-pair.png)

**socket pair**: to identify a connection.

A client-server model:

![image](/csapp/resources/network-service-client-model.png)

### 12.5 HTTP

HTTP: HyperText Transfer Protocol

Content: a sequence of bytes with an associated **MIME(Multipurpose Internet Mail Extensions)**

MIME eg: `text/html`, `image/png`

URL: Universal Resource Locator

### 12.6 CGI

CGI: Common Gateway Interface

Eg: `http://add.com/cgi-bin/adder?15213&18213` Here the `adder` is a CGI

### 12.7 Proxy

![image](/csapp/resources/network-proxy.png)

Develop your own proxy:

1. Sequential proxy: easy start
2. Concurrent proxy: multi-threading
3. Cache Web Objects: cache separate objects, using **LRU(Least Recently Used)** eviction.

## 13. Concurrent Programming

Classical problems: **Races**, **Deadlock**, **livelock/starvation/fairness**

### 13.1 Concurrent Servers

**Process-Based**: fork a new process to handle connections

![image](/csapp/resources/concurrent-process-based.png)

**Event-Based**: use `i/o multiplexing` technique, non-block in one process

**Thread-Based**: use multithreading in one process

![image](/csapp/resources/concurrent-threads.png)

Thread:

- has its own logical control flow
- shares the same code, data and kernel context
- has its own stack for local variables (not protected from other threads)
- State:
  - Joinable: can be reaped and killed by other threads(default)
  - Detached: Automatically be reaped on termination

### 13.2 Semaphores

Non-negative global integer synchronization variable. Maintained by `P` and `V`

```c++
mutex = 1
P(mutex)  // mutex -= 1
// do something
V(mutex)  // mutex += 1
```

Producer-Consumer Example

```c++
buffer[n]
mutex = 1   // semaphore
items = 0   // semaphore
slots = n   // semaphore
```

Reader-Writer Example--First Reader: as long as there is a reader, writer should wait

![image](/csapp/resources/synchronization-first-readers.png)

Note: `mutex` is only for `readcnt`

### 13.3 Prethreaded Concurrent server (thread pool)

![image](/csapp/resources/synchronization-pool-server.png)

Here, we do not need to create/destroy a thread each time a connection is built.

Main idea: put a task(descriptor) into a buffer, threads in pool try to fetch the task.

### 13.4 Thread Safe

Key: Fail to protect shared variables

Solution: solved by **lock/semaphores** or try to **make shared variables local**.
  
**Reentrant function**: access no shared variables when called by multiple threads. Safe and efficient

Do not race: i may = 50 when thread 0 try to fetch the value

![image](/csapp/resources/synchronization-race.png)

Race Result:

![image](/csapp/resources/synchronization-race-2.png)

Deadlock: t1 `P(s0)`, t2 `P(s1)`, t1 `P(s1)`, t2 `P(s0)` -- Deadlock!

Try to acquire resources in the same order to solve the deadlock.

Livelock: Just like deadlock, but this time it will release the lock and retry, so it looks like this:

```bash
start
t1 P(s0)
t2 P(s1)
t1 P(s1) fail
t2 P(s0) fail
t1 V(s0)
t2 V(s1)
goto start
```

### 13.5 Hardwares

Typical multi-core processor:

![image](/csapp/resources/hardware-multicore-processor.png)

Hyper-threading implementation:

![image](/csapp/resources/hyper-threading.png)

Snoopy Cache:

![image](/csapp/resources/snoopy-cache.png)

First, the program can not print `1, 100` or `100, 1`

However, if the cache is not protected, it can print `1, 100`

So, we have some mechanism to protect, like the writer-reader problem:

![image](/csapp/resources/snoopy-cache-2.png)

Then when read, instead of reading from main memory, it reads from another cache:

![image](/csapp/resources/snoopy-cache-3.png)

### 13.6 Sum Example

case1: sum to a global variable using multithreading

![image](/csapp/resources/sum-multithreading.png)

This can be very slow because the lock takes a lot of time.

case2: sum by each thread

![image](/csapp/resources/sum-multithreading-2.png)

This is much faster

### 13.7 Sort Example

Amdahl's law:

$T_{new}=(1-\alpha)T_{old} + (\alpha T_{old})/k$

Here we optimize the $\alpha$ part by $k$ times. When $k=\infty$, the time still be $(1-\alpha)T_{old}$, which means that we should pay attention to the bottleneck.

Quick Sort Algorithm:

![image](/csapp/resources/sort-multithreading.png)

Parallel Algorithm:

![image](/csapp/resources/sort-multithreading-2.png)

Performance:

![image](/csapp/resources/sort-multithreading-3.png)

fraction: How many threads will be spawned (this is a hyper-parameter can be set), with an input array of 134 217 728 values. If too much, thread overhead can be the main time-killer.
