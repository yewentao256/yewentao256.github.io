---
title: "Cachelab"
date: 2023-02-19T14:56:21+08:00
categories: ["csapp"]
summary: "In this lab, we will write a small C program that simulates the behavior of a cache memory and optimize a small matrix transpose function. Source: [https://github.com/yewentao256/CSAPP_15213/tree/main/cachelab]"
---

## Summary

In this lab, we will write a small C program that simulates the behavior of a cache memory and optimize a small matrix transpose function. Source: [https://github.com/yewentao256/CSAPP_15213/tree/main/cachelab]

## Introduction

In this lab, there are two parts:

1. Write a small C program (about 200-300 lines) that simulates the behavior of a cache memory.
2. Optimize a small matrix transpose function, with the goal of minimizing the number of cache misses.

## How to launch(Using docker)

Source from [Yansongsongsong](https://github.com/Yansongsongsong/CSAPP-Experiments)

Firstly using a docker:

`docker run -d -p 9912:22 --name datalab yansongsongsong/csapp:cachelab`

Then using vscode plugin **remote ssh**

`ssh root@127.0.0.1 -p 9912`

password: `THEPASSWORDYOUCREATED`

## Part A: Writing a Cache Simulator

In Part A we will write a cache simulator in `csim.c` that takes a `valgrind` memory trace as input, simulates the hit/miss behavior of a cache memory on this trace, and outputs the total number of hits, misses, and evictions.

### How to get command line options?

```c
int main(int argc, char **argv) {
  int opt, aflag = 0, nflag = 0;
  float xflag = 0.0;
  /* loop over arguments */
  while (-1 != (opt = getopt(argc, argv, "an:y:"))) {
    /* determine which argument was found */
    switch (opt) {
      case 'a':
        aflag = 1;
        break;
      case 'n':
        nflag = atoi(opt);
        break;
      case 'x':
        xflag = atof(opt);
        break;
      default:
        printf("unknown argument");
        break;
    }
  }
  return 0;
}

```

### How to get content from files?

```c
FILE * fp;

fp = fopen ("traces/yi.trace", "r");
if ( fp == NULL ) {
    // check here
}

char access_type;
unsigned long address;
int size;

while(fscanf(fp, " %c %lx,%d", &access_type, &address, &size) > 0){
    printf(" %c %lx,%d\n", access_type, address, size);
}

fclose(fp);   // always remember to free the memory you'v used
```

### How to realize LRU?

The basic idea is to realize by **queue**, but there is no std library for C.

So we use a `lru_counter`, when recently used, set counter to `0`, add one for other block. When eviction, remove the block that has the biggest counter.

```c
  // add 1 to lru_counter and find the biggest one
  // may do eviction(only happens when all of the lines are valid)
  for (int i = 0; i < E; i++) {
    if (max_counter < sets[s_index].lines[i].lru_counter) {
      eviction_index = i;
      max_counter = sets[s_index].lines[i].lru_counter;
    }
    sets[s_index].lines[i].lru_counter += 1;
  }
  if (need_evict) {
    // do your logic
  }
```

### The answer

Carefully read all the instructions above, we now know how to realize the simulator. Here are my codes which can get all of the scores in test.

```c
#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cachelab.h"

typedef struct Line {
  int valid;
  int tag;
  // for lru eviction, remove the biggest counter, set 0 when used
  int lru_counter;
} Line;

typedef struct Set {
  Line *lines;
} Set;

/* global variables */
unsigned int s = 0, E = 0, b = 0;
char access_log[20] = "";
Set *sets;
int hit_count = 0, miss_count = 0, eviction_count = 0;

void access_cache(unsigned long address) {
  unsigned int tag = address >> (s + b);
  int s_index = address >> b & ((1 << s) - 1);  // (1 << s)-1 for %
  int eviction_index = -1, max_counter = -1;
  int need_evict = 1;

  // first loop to see whether there is a match, or cold start
  for (size_t i = 0; i < E; i++) {
    if (sets[s_index].lines[i].valid) {
      // valid, compare tag
      if (sets[s_index].lines[i].tag == tag) {
        // hit
        hit_count += 1;
        sets[s_index].lines[i].lru_counter = 0;
        need_evict = 0;
        strcat(access_log, "hit ");
        break;
      } else {
        // tag mismatch, continue to next one
        continue;
      }
    } else {
      // cold start
      sets[s_index].lines[i].valid = 1;
      sets[s_index].lines[i].tag = tag;
      sets[s_index].lines[i].lru_counter = 0;
      strcat(access_log, "miss ");
      miss_count += 1;
      need_evict = 0;
      break;
    }
  }

  // second loop: add 1 to lru_counter and find the biggest one
  // may do eviction(only happens when all of the lines are valid)
  for (int i = 0; i < E; i++) {
    if (max_counter < sets[s_index].lines[i].lru_counter) {
      eviction_index = i;
      max_counter = sets[s_index].lines[i].lru_counter;
    }
    sets[s_index].lines[i].lru_counter += 1;
  }
  if (need_evict) {
    sets[s_index].lines[eviction_index].valid = 1;
    sets[s_index].lines[eviction_index].tag = tag;
    sets[s_index].lines[eviction_index].lru_counter = 0;
    strcat(access_log, "miss eviction ");
    miss_count += 1;
    eviction_count += 1;
  }
}

int main(int argc, char **argv) {
  /* P1: get user input*/
  int opt, hflag = 0, vflag = 0;
  char *tflag;
  // no ":" means an option, one ":" means there must have one param
  // two ":" means the option can have param
  const char *opt_string = "hvs:E:b:t:";
  /* loop over arguments */
  while ((opt = getopt(argc, argv, opt_string)) != -1) {
    /* determine which argument was found */
    switch (opt) {
      case 'h':
        hflag = 1;
        break;
      case 'v':
        vflag = 1;
        break;
      case 's':
        s = atoi(optarg);
        break;
      case 'E':
        E = atoi(optarg);
        break;
      case 'b':
        b = atoi(optarg);
        break;
      case 't':
        tflag = optarg;
        break;
      default:
        printf("unknown argument");
        break;
    }
  }

  /*P2: dealing with user input, initialize the Set*/
  if (hflag) {
    printf(
        "Welcome using my cache lab simulatorm, friend! I am yewentao or Peter "
        "Ye in English, here are some command options which may help you.\n "
        "-h: Optional help flag that prints usage info\n "
        "-v: Optional verbose flag that displays trace info\n "
        "-s <s>: Number of set index bits (S = 2^s is the number of sets)\n "
        "-E <E>: Associativity (number of lines per set)\n "
        "-b <b>: Number of block bits (B = 2^b is the block size)\n "
        "-t <tracefile>: Name of the valgrind trace to replay\n ");
    return 0;
  }
  unsigned int S = 1 << s;
  sets = (Set *)malloc(sizeof(Set) * S);
  for (int i = 0; i < S; i++) {
    Line *lines = (Line *)malloc(sizeof(Line) * E);
    sets[i].lines = lines;
  }

  /*P3: scan the file and process each line*/
  char access_type;
  unsigned long address;
  int size;

  FILE *fp;

  fp = fopen(tflag, "r");
  if (fp == NULL) {
    printf("Fail to open file %s! Please check the path you input", tflag);
    exit(0);
  }

  while (fscanf(fp, " %c %lx,%d", &access_type, &address, &size) > 0) {
    strcpy(access_log, "");  // clear the log string
    switch (access_type) {
      case 'L':
        access_cache(address);
        break;
      case 'M':
        access_cache(address);
        access_cache(address);
        break;
      case 'S':
        access_cache(address);
        break;
      default:
        break;
    }
    if (vflag && (access_type != 'I')) {
      printf("%c %lx,%d %s\n", access_type, address, size, access_log);
    }
  }

  /* P4: print the result*/
  printSummary(hit_count, miss_count, eviction_count);

  /*P5: free all of the memory we malloc*/
  for (int i = 0; i < S; i++) {
    free(sets[i].lines);
  }

  free(sets);
  fclose(fp);
  return 0;
}
```

## Part B：Optimizing Matrix Transpose

In Part B we will write a transpose function in `trans.c` that causes as few cache misses as possible.

**Note**: We don't recommend to spend too much time here, since it's **non-readable for your teammates** in real project, and it is only useful for specific CPU and Cache.

The param of cache is: `s=5, b=5, E=1`, so there are 32 sets, one line for each set and 32 bytes data in each line.

### 32 * 32

Our cache can save `8 int(32 bytes)` per line, so the common idea is to use `8 * 8` block to speed up.

Note: we can do this because there are **enough(32)** sets in cache, which satisfies our needs, if there are only 16 sets(eg: s = 4), we can't use this strategy. This is because if `s=4`, the first line in block A uses `set0`, the second line uses `sets4`... and the fourth line uses `set0` again, which will cause conflicts.

```c
  int i, j, m, n;
  for (i = 0; i < N; i += 8)
    for (j = 0; j < M; j += 8)
      for (m = i; m < i + 8; ++m)
        for (n = j; n < j + 8; ++n)
        {
          B[n][m] = A[m][n];
        }
```

ref code: **1183 misses**, our code: **343 misses**

Is there any ways to make more use of cache? Yes! We can use local variable to directly save all of the elements in one line of A, to reduce conflict with B. (A and B share the cache, furthermore, if you calculate the cahce set index, **the same array index in A and B share the same cache set line**)

```c
int i, j, k, v1, v2, v3, v4, v5, v6, v7, v8;
for (i = 0; i < 32; i += 8)
  for (j = 0; j < 32; j += 8)
    for (k = i; k < (i + 8); ++k) {
      v1 = A[k][j];
      v2 = A[k][j + 1];
      v3 = A[k][j + 2];
      v4 = A[k][j + 3];
      v5 = A[k][j + 4];
      v6 = A[k][j + 5];
      v7 = A[k][j + 6];
      v8 = A[k][j + 7];
      B[j][k] = v1;
      B[j + 1][k] = v2;
      B[j + 2][k] = v3;
      B[j + 3][k] = v4;
      B[j + 4][k] = v5;
      B[j + 5][k] = v6;
      B[j + 6][k] = v7;
      B[j + 7][k] = v8;
    }
```

ref code: **1183 misses**, our code: **287 misses**

### 64 * 64

Since there are more data, the cache can't hold block in `8*8`. Why? the first line in block uses `set0` for example, the second line uses `set8`... and the fourth line uses `set0` again which will cause conflicts.

So we make the block size smaller -- `4*4`.

```c
int i, j, k, v1, v2, v3, v4;
for (i = 0; i < M; i += 4)
  for(j = 0; j < M; j += 4)
    for(k = i; k < (i + 4); ++k)
    {
      v1 = A[k][j];
      v2 = A[k][j+1];
      v3 = A[k][j+2];
      v4 = A[k][j+3];
      B[j][k] = v1;
      B[j+1][k] = v2;
      B[j+2][k] = v3;
      B[j+3][k] = v4;
    }
```

ref code: **4723 misses**, our code: **1699 misses**

### 61 * 67

There are no specific rules for irregular matrix, simply test different block size and get the best result

- ref code: **4723 misses**
- `block_size = 4`: **2425 misses**
- `block_size = 8`: **2118 misses**
- `block_size = 16`: **1992 misses**
- `block_size = 17`: **1950 misses**,
- `block_size = 18`: **1961 misses**

we choose the best one:

```c
int i, j, k, l;
for (i = 0; i < N; i += 17) {
  for (j = 0; j < M; j += 17) {
    for (k = i; k < i + 17 && k < N; k++) {
      for (l = j; l < j + 17 && l < M; l++) {
        B[l][k] = A[k][l];
      }
    }
  }
}
```
