---
title: "MallocLab"
date: 2023-09-16T10:31:24+08:00
categories: ["csapp"]
summary: "在malloc lab中，我们将开发我们自己的`malloc`、`free`、`realloc`函数。通过这个实验，我们能更好理解内存布局与管理，并在空间利用率和时间效率上做权衡，实现一套高效内存管理机制。"
---

## Summary

在malloc lab中，我们将开发我们自己的`malloc`、`free`、`realloc`函数。通过这个实验，我们能更好理解内存布局与管理，并在空间利用率和时间效率上做权衡，实现一套高效内存管理机制。Source: [Github-Link-Here](https://github.com/yewentao256/CSAPP_15213/tree/main/malloclab)

## 等待被翻译

非常抱歉，看起来这篇博文还没有被翻译成中文，请等待一段时间

## How to launch(Using docker)

Source from [Yansongsongsong](https://github.com/Yansongsongsong/CSAPP-Experiments)

Firstly using a docker:

`docker run -d -p 9912:22 --name shelllab yansongsongsong/csapp:malloclab`

Then using vscode plugin **remote ssh**

`ssh root@127.0.0.1 -p 9912`

password: `THEPASSWORDYOUCREATED`

## How to validate

To compile the driver, type `make` to the shell.

To run the driver on a brief test trace, use the following command: `./mdriver -V -f traces/short1-bal.rep`

The result is like:

```bash
trace  valid  util     ops      secs    Kops
 0       yes   50%      12  0.000000  120000
Total          50%      12  0.000000  120000
```

Please note: It's crucial to aim for maximizing the utility rate while minimizing time consumption.

After making your changes, validate with all the traces using: `./mdriver -t ./traces -V`

## Solution: Explict free list (86/100)

This is a solution for **explicit free list**.

To further boost your scores, consider implementing a **segregated list** for optimization.

### Data Structure

```c
typedef struct Block {
  size_t size;  // includes sizeof(block) and footer
  struct Block *prev;
  struct Block *next;
  // size_t footer;  // footer is an extra space out of block
  // so the memory looks like:
  // block      payload = 0x200       footer
  //  0x0   0x12                0x212        0x216
} Block;

#define FOOTER(ptr) \
  ((size_t *)((char *)(ptr) + BLOCK_SIZE(ptr) - sizeof(size_t)))

#define SIZE_OF_BLOCK sizeof(Block) + sizeof(size_t)
```

- The **footer** (boundary tag) aids in coalescing.

- A single bit in the size field denotes if a block is allocated.

```c
// using lowest bit of size to show allocated or not
#define SET_ALLOCATED(b) ((b)->size |= 1)
#define SET_FREE(b) ((b)->size &= ~1)
#define IS_ALLOCATED(b) ((b)->size & 1)
#define BLOCK_SIZE(b) ((b)->size & ~1)
```

### `malloc`

- We employ the **First Fit** strategy.
- Upon finding a sufficiently large block, we split it. The desired block is allocated, while the remainder rejoins the explicit list.
- If no suitable block is available, we expand the heap.

### `free`

- The `free` operation attempts to **coalesce**: We inspect both the preceding and succeeding blocks. If available, we unify them to form a larger free block.
- The **Boundary Tag** plays a crucial role during this process.

### Realloc

There are four scenarios for `realloc`:

1. If the realloc size is smaller, the block is downsized (via splitting).
2. If the block being reallocated is the final block (excluding the epilogue), we can directly extend the heap.
3. We may enlarge the block by merging it with the subsequent free block.
4. If expansion is infeasible, a fresh block is allocated and the data copied over.

### Final Answer

```c
// head block, size = 0
Block head = {0, &head, &head};

size_t get_block_size(size_t size) {
  // get the aligned block size from initial size
  // eg: size = 10, then size = align(size + sizeof(block) + footer)
  // = align(10 + 16 + 4) = 32
  return (size + SIZE_OF_BLOCK + (ALIGNMENT - 1)) & ~(ALIGNMENT - 1);
}

void clear_block(Block *block) {
  *FOOTER(block) = 0;
  block->size = 0;
  block->prev = NULL;
  block->next = NULL;
}

void remove_from_list(Block *block) {
  // remove block from list
  block->prev->next = block->next;
  block->next->prev = block->prev;
}

// insert block to explicit list, prev_block shows the position to insert
void insert_to_list(Block *block, Block *prev_block) {
  block->prev = prev_block;
  block->next = prev_block->next;
  prev_block->next->prev = block;
  prev_block->next = block;
}

// Split block, this will generate a free block.
void split_block(Block *block, size_t size, int from_allocated) {
  Block *new_block = (Block *)((char *)block + size);
  new_block->size = BLOCK_SIZE(block) - size;
  *FOOTER(new_block) = BLOCK_SIZE(new_block);
  SET_FREE(new_block);

  block->size = size;
  *FOOTER(block) = size;

  if (from_allocated) {
    // Case when block is allocated (mm_realloc)
    // Insert new_block to the beginning of the list
    insert_to_list(new_block, &head);
  } else {
    // Case when block is free (find_fit)
    insert_to_list(new_block, block);
  }
}

// find a fit in explicit list
Block *find_fit(size_t size) {
  for (Block *block = head.next; block != &head; block = block->next) {
    // all the block in list should not be allocated
    assert(!IS_ALLOCATED(block));
    if (BLOCK_SIZE(block) >= size) {
      // Note that here block size and size all should be aligned
      assert(block->size % ALIGNMENT == 0);
      assert(size % ALIGNMENT == 0);
      // If a free block is larger than request, split it
      if (BLOCK_SIZE(block) > size + SIZE_OF_BLOCK) {
        split_block(block, size, 0);
      }
      remove_from_list(block);
      return block;
    }
  }
  return NULL;  // no suitable block
}

Block *coalesce(Block *block, int from_allocated) {
  // coalesce, return the pointer of final block
  Block *next_block = (Block *)((char *)block + BLOCK_SIZE(block));
  size_t prev_block_size = *((size_t *)((char *)block - sizeof(size_t)));
  Block *prev_block = (Block *)((char *)block - prev_block_size);

  int prev_free = prev_block_size > 0 &&
                  (char *)prev_block != (char *)(&head) &&
                  !IS_ALLOCATED(prev_block) && prev_block->size != 0;

  int next_free = (char *)next_block != (char *)(&head) &&
                  !IS_ALLOCATED(next_block) && next_block->size != 0;

  if (prev_free && next_free) {
    *FOOTER(prev_block) = 0;
    prev_block->size += BLOCK_SIZE(block) + BLOCK_SIZE(next_block);
    *FOOTER(prev_block) = BLOCK_SIZE(prev_block);

    remove_from_list(next_block);
    if (!from_allocated) {
      remove_from_list(block);
    }
    clear_block(next_block);
    clear_block(block);
    return prev_block;

  } else if (prev_free) {
    *FOOTER(prev_block) = 0;
    prev_block->size += BLOCK_SIZE(block);
    *FOOTER(prev_block) = BLOCK_SIZE(prev_block);
    if (!from_allocated) {
      remove_from_list(block);
    }
    clear_block(block);
    return prev_block;

  } else if (next_free) {
    *FOOTER(block) = 0;
    block->size += BLOCK_SIZE(next_block);
    *FOOTER(block) = BLOCK_SIZE(block);

    remove_from_list(next_block);
    clear_block(next_block);
    return block;

  } else {
    return block;
  }
}

void expand_heap(size_t size) {
  // each time expand heap, the size should be bigger than a single block
  assert(size > SIZE_OF_BLOCK);
  // try coalesce all of the blocks in explict list
  Block *coalesce_block = head.next;
  while (coalesce_block != &head) {
    assert(coalesce_block != NULL);
    coalesce_block = coalesce(coalesce_block, 0);
    coalesce_block = coalesce_block->next;  // Go to next block after
  }

  // Get the last block (current epilogue)
  // Note that only a size is needed, so we just sub sizeof(size_t)
  Block *last_block = (Block *)((char *)mem_sbrk(0) - sizeof(size_t));

  // Expand the heap
  Block *block = (Block *)mem_sbrk(size);
  if (block == (void *)-1) {
    perror("Error expanding heap");
    exit(1);
  }

  memset(block, 0, size);  // make sure block here is empty

  // If the last block was an epilogue, we overwrite it and adjust the size.
  if (IS_ALLOCATED(last_block) && BLOCK_SIZE(last_block) == 0) {
    block = last_block;
    size += sizeof(size_t);
  }

  // Set the size of the new block (excluding the epilogue at the end)
  block->size = size - sizeof(size_t);
  *FOOTER(block) = BLOCK_SIZE(block);

  insert_to_list(block, &head);

  // Add a new epilogue block at the end
  Block *epilogue = (Block *)((char *)block + BLOCK_SIZE(block));
  epilogue->size = 0;
  SET_ALLOCATED(epilogue);
}

/*
 * mm_init - initialize the malloc package.
 */
int mm_init(void) {
  // Clear all values in the blocks linked to the head.
  for (Block *block = head.next; block != &head;) {
    Block *temp = block;
    block = block->next;
    clear_block(temp);
  }

  // Reset the head block's pointers.
  head.size = 0;
  head.prev = &head;
  head.next = &head;

  // Add a prologue block, to avoid bug when coalescing the first block
  Block *prologue = (Block *)mem_sbrk(ALIGN(sizeof(Block)));
  prologue->size = ALIGN(sizeof(Block));
  SET_ALLOCATED(prologue);
  *FOOTER(prologue) = BLOCK_SIZE(prologue);

  // Expand the heap.
  expand_heap(EXPAND_HEAP_SIZE);
  return 0;
}

/*
 * mm_malloc - Allocate a block by incrementing the brk pointer.
 *     Always allocate a block whose size is a multiple of the alignment.
 */
void *mm_malloc(size_t size) {
  size = get_block_size(size);
  Block *block = find_fit(size);
  if (!block) {
    expand_heap(size > EXPAND_HEAP_SIZE ? size : EXPAND_HEAP_SIZE);
    block = find_fit(size);
    if (!block) {
      return NULL;
    }
  }
  SET_ALLOCATED(block);
  return (void *)((char *)block + sizeof(Block));  // return data to user
}

/*
 * mm_free - Freeing a block does nothing.
 */
void mm_free(void *ptr) {
  if (!ptr) return;  // null check
  Block *block = (Block *)((char *)ptr - sizeof(Block));
  SET_FREE(block);
  Block *coalescedBlock = coalesce(block, 1);

  if (coalescedBlock == block) {
    insert_to_list(coalescedBlock, &head);
  }
}

/*
 * mm_realloc - Resizes the memory block point to by ptr to size bytes.
 */
void *mm_realloc(void *ptr, size_t size) {
  if (!ptr) {
    // case when ptr == nullptr, malloc a new one
    return mm_malloc(size);
  }
  if (size == 0) {
    // case when ptr is used and size==0, equal to free
    mm_free(ptr);
    return NULL;
  }
  Block *block = (Block *)((char *)ptr - sizeof(Block));  // get block
  size_t realloc_block_size = get_block_size(size);

  // for a allocated block, next and prev is meaning less
  assert(IS_ALLOCATED(block));
  block->next = NULL;
  block->prev = NULL;
  // Case 1: If realloc size is smaller, shrink the block, and the new block
  // will be added to the explicit list. Note that at this time, block is
  // already in use, so we do not need to remove it from explicit list
  if (BLOCK_SIZE(block) > realloc_block_size + SIZE_OF_BLOCK) {
    split_block(block, realloc_block_size, 1);
    SET_ALLOCATED(block);
    return ptr;
  }

  // Case 2: The block to be reallocated is the last block(not epilogue)
  // At this time, we can directly expand the heap
  Block *next_block = (Block *)((char *)block + BLOCK_SIZE(block));
  Block *epilogue = (Block *)((char *)mem_sbrk(0) - sizeof(size_t));
  if (next_block == epilogue) {
    size_t extra_size = realloc_block_size - BLOCK_SIZE(block);
    if (extra_size > SIZE_OF_BLOCK) {
      // This is the last block. Simply expand heap.
      expand_heap(realloc_block_size - BLOCK_SIZE(block));
      // after we expand_heap, we add a block to the explicit list
      // so we should remove it now.
      remove_from_list(head.next);
      block->size = realloc_block_size;
      SET_ALLOCATED(block);
      return ptr;
    }
  }

  // Case 3: Attempt to extend the block by coalescing with the next free block
  // Note: there is no need to coalesce the previous block, if we do so
  // we need to memcpy, which is the same with mm_malloc (case 4)
  int next_free = (char *)next_block != (char *)(&head) &&
                  !IS_ALLOCATED(next_block) && next_block->size != 0;
  if (next_free) {
    *FOOTER(block) = 0;
    block->size += BLOCK_SIZE(next_block);
    *FOOTER(block) = BLOCK_SIZE(block);
    remove_from_list(next_block);
    clear_block(next_block);
    if (BLOCK_SIZE(block) > realloc_block_size + SIZE_OF_BLOCK) {
      split_block(block, realloc_block_size, 1);
      SET_ALLOCATED(block);
      return ptr;
    }
  }

  // Case 4: If the block can't be extended, allocate a new block
  void *new_ptr = mm_malloc(size);
  if (!new_ptr) {
    return NULL;
  }
  memcpy(new_ptr, ptr, size);
  mm_free(ptr);
  return new_ptr;
}
```

### Result

```bash
Results for mm malloc:
trace  valid  util     ops      secs    Kops
 0       yes   93%    5694  0.000656    8677
 1       yes   94%    5848  0.000405   14429
 2       yes   96%    6648  0.001274    5219
 3       yes   97%    5380  0.000680    7913
 4       yes   40%   14400  0.000499   28881
 5       yes   90%    4800  0.003578    1342
 6       yes   86%    4800  0.003834    1252
 7       yes   53%   12000  0.019015     631
 8       yes   47%   24000  0.019389    1238
 9       yes   98%   14401  0.000452   31839
10       yes   45%   14401  0.001873    7688
Total          76%  112372  0.051655    2175
```
