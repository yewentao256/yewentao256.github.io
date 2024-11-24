---
title: "Understand Lightgbm"
date: 2021-12-17T11:01:36+08:00
categories: ["algorithm"]
summary: "Understanding Lightgbm: Gradient Boosting Decision Tree, including decision tree and gradient boosting."
---

## Summary

Understanding Lightgbm: Gradient Boosting Decision Tree, including decision tree and gradient boosting.

## To be translated

Oh Sorry!

This blog has't been translated to English, please wait for a little while...

GBDT包含两个重要的内容：**Decision Tree**（实际中采用CART回归树）和**Gradient Boosting**（梯度提升）

## CART回归树

- 为什么不用CART分类树？因为GBDT每次迭代拟合的是**梯度值**，梯度值是连续值所以用回归树

## Gradient Boosting

梯度提升树（Gradient Boosting Tree）是提升树（Boosting Tree）的一种改进，这里先介绍一下提升树

### 提升树

- 通俗理解：假如有个人30岁，我们首先用20岁去拟合，发现损失有10岁，这时我们用6岁去拟合剩下的损失，发现差距还有4岁，第三轮我们用3岁拟合剩下的差距，差距就只有一岁了。如果我们的迭代轮数还没有完，可以继续迭代。最后将每次拟合的岁数加起来便是模型输出的结果。

- 算法：
    1. 初始化$f_0(x) = 0$
    2. 对$m = 1,2,...,M$

        (a) 计算残差$r_{mi} = y_i - f_{m-1}(x)$

        (b) 拟合残差$r_{mi}$ 学习CART回归树得到$h_m(x)$

        (c) 更新$f_m(x) = f_{m-1} + h_m(x)$

    3. 得到提升树$$f_M(x) = \sum_{m=1}^M h_m(x)$$

---

- 当损失函数L为平方损失/指数损失函数时，提升树每一步优化很简单。这里以平方损失函数为例：

$$L(y, f_{t-1}(x)+h_t(x)) = (y-f_{t-1}(x)-h_t(x))^2 = (r-h_t(x))^2$$

这里的$r = y - f_{t-1}(x)$即为我们的残差$h_t(x)$为本轮迭代得到的弱学习器

### 梯度提升树

- 但对于一般的损失函数而言，每一步优化起来没有那么容易。所以Friedman提出了梯度提升算法，利用最速下降的近似方式，关键是利用损失函数的负梯度作为提升树算法中的残差的近似值。
- 如果选择损失函数为平方损失，那么负梯度为

$$-[{∂L(y,f(x_i)) \over ∂f(x_i)}]_{_{f(x) = f_{t-1}(x)}} = y-f(x_i)$$

- 我们发现GBDT基于平方损失的回归问题其负梯度就是残差。（备注：如果是分类问题那么损失函数是`logloss`）

## Lightgbm

基于传统GBDT，lightgbm做了以下优化与改进

### 1. 基于直方图的算法提升效率

- 决策树中最耗时的部分为寻找最佳分割点，通常寻找方法为预排序算法，将特征取值预排序并枚举可能分割点。但此方法不够高效
- 训练中连续特征分箱构建直方图，这样虽然精度略微降低，但寻找分割点的内存消耗和训练速度都更为高效。
- 复杂度：

$$O(data * features) → O(bins * features)$$

而我们知道bins是远小于data数量的，所以更加高效。

### 2. 带深度限制的leaf-wise叶子生长策略

- 大部分决策树算法采用逐层加深的方法生长树

- lightgbm采用leaf-wise（最佳分裂节点优先）的生长策略
每次选择损失减小得最多的节点方向生长。

- 此举容易造成过拟合，因此我们有max_depth限制树的最大深度。

### 3. Goss算法

- 直观理解：不损害数据分布的前提下，丢弃小梯度的数据样本（小梯度表示训练误差较小，大多数情况下已经被良好训练），中心放在梯度大的难以学习的数据上。

- 算法：先将梯度绝对值由大到小排序，排序后选择a%的样本（大梯度样本，全部保留），剩下数据中抽取b%样本（小梯度）。之后再计算信息增益时通过常数$(1-a) \over b$ 增大小梯度样本权重，如此可以尽量不改变数据分布（减少对模型准确性影响）。

### 4. EFB算法

- 直观理解：对于高维稀疏数据中的互斥的特征（不同时取0），捆绑为一个特征，大大提高GBDT训练速度。

- 复杂度：

$$O(bins * features) → O(bins * bundle)$$

## 总结

- 决策树 decision tree

    常用如ID3、C4.5、CART（可以分类也可以回归）等

- 梯度提升 gradient boosting

    数学推导发现负梯度即残差，所以，梯度提升意味着残差更小。

- 梯度提升决策树

    decision tree + gradient boosting

- lightgbm——微软提出的梯度提升决策树模型

    相较于xgb，它在性能和效果上都有着更好的表现

light的优点：

1. 基于直方图的决策树算法

    连续特征分箱为离散值，分箱构建直方图，使得复杂度由`O（data * feature）` ->  `O（bins * feature）`

2. 带有深度限制的leaf wise叶子生长策略

    大部分决策树是逐层加深（level-wise）的策略，而lightgbm则是以损失减小最多的方向生长结点，同样叶子数量情况下损失更少。但有过拟合的风险，因此需要通过`max_depth`限制树深

3. GOSS 算法(Gradient-based One-Side Sampling 基于梯度的单侧采样)

    保留较大梯度的样本，小梯度样本仅采样部分。把更多精力放在学习相对难的部分上。
    为了补偿对数据分布的影响，会相应调大小梯度样本的权重
    （选a%的大梯度，选b%的小梯度，信息增益计算时`(1-a)/b`这个常数调大权重）。

4. EFB算法(Exclusive Feature Bundling 互斥特征捆绑)

    高维数据稀疏时使用，在稀疏特征空间中，许多特征是互斥的，即它们从不同时取非零值。
    我们就可以在此时把互斥特征捆绑为一个特征，使得复杂度从`O（bins * feature）`变为`O（bins * bundle）`
    这样就可以在不损失精度的情况下大大提升训练速度

## Reference

- [LightGBM](https://github.com/microsoft/LightGBM)
- [GBDT算法原理以及实例理解](https://blog.csdn.net/zpalyq110/article/details/79527653)
