---
title: "Understand Dynamic Programming"
date: 2022-03-11T11:13:50+08:00
categories: ["algorithm"]
summary: "基于两个实际的例子理解动态规划算法"
---

## Summary

基于两个实际的例子理解动态规划算法

动态规划其实某种意义上，就是高中数列题而已。

## 能用动态规划解决的问题

- 问题的答案构成了**数列**
- 大规模问题依赖小规模问题答案**递推**得到，例如：$f(n) = f(n-1) + 2$

## 应用动态规划

1. 建立状态转移方程，例如：$f(n) = f(n-1) + 2$
2. 确定初始值和边界值（结束条件）
3. 根据需要缓存结果。
4. 按顺序从小往大计算

## 例子——斐波那契数列

- 斐波那契数列：0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233……
- 目的：求第n个值
- 解法1：递归（反例）

```python
def fib(n):
    if n<2:
        return n
    else:
        return fib(n-1)+fib(n-2)

# 天荒地老亦不得出也！时间复杂度：O（2的n次方）——随着递归深入，计算任务倍增！
print(fib(100))
```

- 解法2：动态规划

```python
def fib(n):
    results = list(range(n+1))  # 缓存
    for i in range(n+1):
        if i<2:
            results[i] = i      # 初始值
        else:
            results[i] = results[i-1] + results[i-2]    # 状态转移方程，按顺序从小到大计算
    return results[-1]

# 秒算，时间复杂度：O（N），结果为354224848179261915075
print(fib(100))
```

## 例子——不同路径选择

- 题目来源：[leetcode](https://leetcode-cn.com/problems/unique-paths/%20/)
- 题目说明：
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ），机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。问总共有多少条不同的路径？
- 样例：
输入：m = 3, n = 7(3行7列)
输出：28
- 思路
  - 建立状态转移方程（i，j格子的值为左边格子的值+上边格子的值）：$f(i,j) = f(i-1,j)+f(i,j-1)$
  - 初始值与结束条件：初始值$f(0,0) = 0, f(m,n)结束$
  - 缓存并复用结果：需要用二维数组存储中间结果
  - 按顺序从小到大计算：两个循环逐行逐列分析

- 代码：

```python
def count_paths(m,n):
    results = [[1 for _ in range(n)] for _ in range(m)]

    # 第0行第0列都是1，剪枝跳过
    for i in range(1, m):           # 行计算
        for j in range(1, n):       # 列计算
            results[i][j] = results[i-1][j] + results[i][j-1]     # 应用状态转移方程，且复用中间结果
    
    return results[-1][-1]

# 结果为28
print(count_paths(3,7))
```

## Reference

[@zhen tan](https://www.zhihu.com/question/39948290/answer/883302989)

---
*Confused about some of the content? Feel free to report an issue [here](https://github.com/yewentao256/yewentao256.github.io/issues/new).*