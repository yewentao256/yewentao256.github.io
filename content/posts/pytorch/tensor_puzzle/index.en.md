---
title: "Tensor Puzzles"
date: 2024-11-09T22:52:12+08:00
categories: ["Pytorch"]
summary: "This article provides solutions to the [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) by [Sasha Rush](https://github.com/srush)"
---

## Summary

This article provides solutions to the [Tensor Puzzles](https://github.com/srush/Tensor-Puzzles) by [Sasha Rush](https://github.com/srush).

For the more challenging puzzles (e.g., `compress`), we include detailed explanations as comments within the code.

## Puzzle 1 - ones

```py
def ones_spec(out):
    for i in range(len(out)):
        out[i] = 1

def ones(i: int) -> TT["i"]:
    return where(arange(i) > -1, 1, 0)
```

## Puzzle 2 - sum

```py
def sum_spec(a, out):
    out[0] = 0
    for i in range(len(a)):
        out[0] += a[i]

def sum(a: TT["i"]) -> TT[1]:
    return ones(a.shape[0]) @ a[:, None]
```

## Puzzle 3 - outer

```py
def outer_spec(a, b, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] = a[i] * b[j]

def outer(a: TT["i"], b: TT["j"]) -> TT["i", "j"]:
    return a[:, None] @ b[None, :]
```

## Puzzle 4 - diag

```py
def diag_spec(a, out):
    for i in range(len(a)):
        out[i] = a[i][i]

def diag(a: TT["i", "i"]) -> TT["i"]:
    return a[arange(a.shape[0]), arange(a.shape[0])]
```

## Puzzle 5 - eye

```py
def eye_spec(out):
    for i in range(len(out)):
        out[i][i] = 1

def eye(j: int) -> TT["j", "j"]:
    return where(arange(j)[:, None] == arange(j)[None, :], 1, 0)
```

## Puzzle 6 - triu

```py
def triu_spec(out):
    for i in range(len(out)):
        for j in range(len(out)):
            if i <= j:
                out[i][j] = 1
            else:
                out[i][j] = 0

def triu(j: int) -> TT["j", "j"]:
    return where(arange(j)[:, None] <= arange(j)[None, :], 1, 0)
```

## Puzzle 7 - cumsum

```py
def cumsum_spec(a, out):
    total = 0
    for i in range(len(out)):
        out[i] = total + a[i]
        total += a[i]

def cumsum(a: TT["i"]) -> TT["i"]:
    # Note: Triangle @ a, a will be implicited changed to (i, 1) to perform @
    return where(arange(a.shape[0])[:, None]>=arange(a.shape[0])[None, :],1,0)@a
```

## Puzzle 8 - diff

```py
def diff_spec(a, out):
    out[0] = a[0]
    for i in range(1, len(out)):
        out[i] = a[i] - a[i - 1]

def diff(a: TT["i"], i: int) -> TT["i"]:
    return a - where(arange(i) != 0, a[arange(i) - 1], 0)
```

## Puzzle 9 - vstack

```py
def vstack_spec(a, b, out):
    for i in range(len(out[0])):
        out[0][i] = a[i]
        out[1][i] = b[i]

def vstack(a: TT["i"], b: TT["i"]) -> TT[2, "i"]:
    return where(arange(2)[:, None] == ones(a.shape[0]), b, a)
```

## Puzzle 10 - roll

```py
def roll_spec(a, out):
    for i in range(len(out)):
        if i + 1 < len(out):
            out[i] = a[i + 1]
        else:
            out[i] = a[i + 1 - len(out)]

def roll(a: TT["i"], i: int) -> TT["i"]:
    return a[(arange(i) + 1) % i]
```

## Puzzle 11 - flip

```py
def flip_spec(a, out):
    for i in range(len(out)):
        out[i] = a[len(out) - i - 1]

def flip(a: TT["i"], i: int) -> TT["i"]:
    return a[i - arange(i) - 1]
```

## Puzzle 12 - compress (Hard)

```py
def compress_spec(g, v, out):
    j = 0
    for i in range(len(g)):
        if g[i]:
            out[j] = v[i]
            j += 1

def compress(g: TT["i", bool], v: TT["i"], i:int) -> TT["i"]:
    # Main idea: Using v @ m to map
    # Eg: v = [1, 2, 3], g = [0, 1, 1], result = [2, 3, 0]
    # [1, 2, 3] @ [[0, 0, 0],
    #              [1, 0, 0],
    #              [0, 1, 0]]  => [2, 3, 0]
    # How to get m?
    # `cumsum(1*g) - 1` to get the index of True
    # Eg: g = [1, 0, 1, 0, 1] => cumsum(1*g) - 1 = [0, 0, 1, 1, 2]
    # `arange(i) == (cumsum(1*g) - 1)[:, None]` to get the matrix
    # [[ True, False, False, False, False],
    #  [ True, False, False, False, False],
    #  [False,  True, False, False, False],
    #  [False,  True, False, False, False],
    #  [False, False,  True, False, False]]
    # Finally, we use where(g[:, None], matrix, 0) to get the m
    # [[ 1, 0, 0, 0, 0],
    #  [ 0, 0, 0, 0, 0],
    #  [ 0, 1, 0, 0, 0],
    #  [ 0, 0, 0, 0, 0],
    #  [ 0, 0, 1, 0, 0]]
    return v @ where(g[:, None], arange(i) == (cumsum(1*g) - 1)[:, None], 0)
```

## Puzzle 13 - pad_to

```py
def pad_to_spec(a, out):
    for i in range(min(len(out), len(a))):
        out[i] = a[i]


def pad_to(a: TT["i"], i: int, j: int) -> TT["j"]:
    # simalar to above, we use @ to fix this
    # eg: a = tensor([1, 0, 1, 0, 1]), i = 5, j = 4
    # 1 * (arange(i)[:, None] == arange(j) to get m
    # tensor([[1, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [0, 0, 1, 0],
    #         [0, 0, 0, 1],
    #         [0, 0, 0, 0]])
    # then a @ m to get [1, 0, 1, 0]
    return a @ (1 * (arange(i)[:, None] == arange(j)))
```

## Puzzle 14 - sequence_mask

```py
# eg:
# values = [[1, 2, 3],
#           [4, 5, 6],
#           [7, 8, 9]]
# length = [2, 1, 3]
# Output:
# [[1, 2, 0],
#  [4, 0, 0],
#  [7, 8, 9]]
def sequence_mask_spec(values, length, out):
    for i in range(len(out)):
        for j in range(len(out[0])):
            if j < length[i]:
                out[i][j] = values[i][j]
            else:
                out[i][j] = 0

def sequence_mask(values: TT["i", "j"], length: TT["i", int]) -> TT["i", "j"]:
    # arange(j)[None, :] < length[:, None]) eg:
    # [[0, 1, 2] < 2] -> [[True, True, False],
    #  [0, 1, 2] < 1] -> [[False, False, False],
    #  [0, 1, 2] < 3] -> [[True, True, True]]
    # then values * m = the sequence we need
    return values * (arange(values.shape[1])[None, :] < length[:, None])
```

## Puzzle 15 - bincount

```py
def bincount_spec(a, out):
    for i in range(len(a)):
        out[a[i]] += 1

def bincount(a: TT["i"], j: int) -> TT["j"]:
    # a = [1, 2, 2, 3, 4, 1, 0], j = 5
    # eye(j)[a] to get m
    # [[0, 1, 0, 0, 0],  # a[0] == 1
    #  [0, 0, 1, 0, 0],  # a[1] == 2
    #  [0, 0, 1, 0, 0],  # a[2] == 2
    #  [0, 0, 0, 1, 0],  # a[3] == 3
    #  [0, 0, 0, 0, 1],  # a[4] == 4
    #  [0, 1, 0, 0, 0],  # a[5] == 1
    #  [1, 0, 0, 0, 0]]  # a[6] == 0
    # then we use ones [1, 1, 1, 1, 1, 1, 1] @ m to get the result(like sum)
    return ones(a.shape[0]) @ eye(j)[a]
```

## Puzzle 16 - scatter_add

```py
# values = [3, 1, 4, 2], link = [0, 1, 0, 2]
# out = [7, 1, 2]: out[0] += 3 + 4 = 7, out[1] += 1, out[2] += 2
def scatter_add_spec(values, link, out):
    for j in range(len(values)):
        out[link[j]] += values[j]

def scatter_add(values: TT["i"], link: TT["i"], j: int) -> TT["j"]:
    # eye(j)[link] to get the m
    # then values @ m to get the result, like puzzle 15
    return values @ eye(j)[link]
```

## Puzzle 17 - flatten

```py
def flatten_spec(a, out):
    k = 0
    for i in range(len(a)):
        for j in range(len(a[0])):
            out[k] = a[i][j]
            k += 1

def flatten(a: TT["i", "j"], i:int, j:int) -> TT["i * j"]:
    # a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # arange(i*j) // j = [0, 0, 0, 1, 1, 1]
    # arange(i*j) % j = [0, 1, 2, 0, 1, 2]
    # so return a[0,0] a[0,1] a[0,2] a[1,0] a[1,1], a[1,2]
    return a[arange(i*j) // j, arange(i*j) % j]
```

## Puzzle 18 - linspace

```py
# i = 0.0, j = 1.0. n = 5
# out = [0.0, 0.25, 0.5, 0.75, 1.0]
def linspace_spec(i, j, out):
    for k in range(len(out)):
        out[k] = float(i.item() + (j.item() - i.item()) * k / max(1, len(out) - 1))

def linspace(i: TT[1], j: TT[1], n: int) -> TT["n", float]:
    # step array: (j - i) * arange(n) / max(1, n - 1)
    # using max(1, n-1) to avoid divide with 0
    return i + (j - i) * arange(n) / max(1, n - 1)
```

## Puzzle 19 - heaviside

```py
# a = torch.tensor([ -1.0, 0.0, 3.0, 0.0, 2.0 ])
# b = torch.tensor([ 10.0, 20.0, 30.0, 40.0, 50.0 ])
# out = [0, 20, 1, 40, 1]
def heaviside_spec(a, b, out):
    for k in range(len(out)):
        if a[k] == 0:
            out[k] = b[k]
        else:
            out[k] = int(a[k] > 0)

def heaviside(a: TT["i"], b: TT["i"]) -> TT["i"]:
    return (a > 0).int() + (a == 0).int() * b
```

## Puzzle 20 - repeat (1d)

```py
# a = torch.tensor([1, 2, 3])
# d = torch.tensor([2])
# out = tensor([[1, 2, 3], [1, 2, 3]])
def repeat_spec(a, d, out):
    for i in range(d[0]):
        for k in range(len(a)):
            out[i][k] = a[k]

def repeat(a: TT["i"], d: TT[1]) -> TT["d", "i"]:
    # broadcast a with d[0] times
    return ones(d[0])[:, None] * a[None, :]
```

## Puzzle 21 - bucketize

```py
# v = torch.tensor([-1.0, 0.0, 1.5, 3.0, 4.5, 6.0])
# boundaries = torch.tensor([1.0, 3.0, 5.0])
# out = [0, 0, 1, 2, 2, 3]
def bucketize_spec(v, boundaries, out):
    for i, val in enumerate(v):
        out[i] = 0
        for j in range(len(boundaries)-1):
            if val >= boundaries[j]:
                out[i] = j + 1
        if val >= boundaries[-1]:
            out[i] = len(boundaries)

def bucketize(v: TT["i"], boundaries: TT["j"]) -> TT["i"]:
    # tensor([5, 3, 3, 3, 3]) tensor([0, 4, 7])
    # 1 * (v[:, None] > boundaries[None, :] to get the m
    # tensor([[1, 1, 0],
    #         [1, 0, 0],
    #         [1, 0, 0],
    #         [1, 0, 0],
    #         [1, 0, 0]])
    # use m to @ ones(boundaries.shape[0]) to get the result tensor([2, 1, 1, 1, 1])
    return 1 * (v[:, None] >= boundaries[None, :]) @ ones(boundaries.shape[0])
```

## Speed Run Mode

```py
import inspect
fns = (ones, sum, outer, diag, eye, triu, cumsum, diff, vstack, roll, flip,
       compress, pad_to, sequence_mask, bincount, scatter_add)

for fn in fns:
    lines = [l for l in inspect.getsource(fn).split("\n") if not l.strip().startswith("#")]

    if len(lines) > 3:
        print(fn.__name__, len(lines[2]), "(more than 1 line)")
    else:
        print(fn.__name__, len(lines[1]))
```

```bash
ones 38
sum 40
outer 34
diag 52
eye 64
triu 64
cumsum 80
diff 57
vstack 62
roll 33
flip 31
compress 76
pad_to 54
sequence_mask 72
bincount 39
scatter_add 32
```
