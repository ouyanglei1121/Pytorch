#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/8/3
# 创建tensor
# 从numpy导入的float其实是double类型


# 第一种方法
import torch
import numpy as np
a = np.array([2, 3.3])
print(torch.from_numpy(a))

#第二种方法
b = torch.tensor([2, 3.3])   # torch.tensor(a)   a为真实的数据  Tensor（a）  a为数据的维度shape
print(b)


# 申请未初始化tensor的数据 未初始化的数据也是有数据的  可能会出现一些不规则的数据   未初始化的tensor一定要跟写入数据的后续步骤
# 1.
a = torch.empty(2, 3)
print('a', a)
# 2.
b = torch.FloatTensor(2, 3, 5)
print('b', b)
# 3.
c = torch.IntTensor(2, 4, 5)
print('c', c)

# 随机取值 torch.rand(a)  a 为维度 torch.rand_like(a)  a为指定的tensor
a = torch.rand(2, 5)
print('a', a)
b = torch.rand_like(a)
print('b', b)
a = torch.randint(1, 10, [2, 3])    #  1为最小值， 10为最大值，但不包含10， [2, 3]为维度shape
print('a', a)

# 正态分布  randn()  均值为0 方差为1
a = torch.randn(3, 3)
print('a', a)
# 自定义正太分布的均值和方差
b = torch.normal(mean=torch.full([10], 0), std=torch.arange(1, 0, -0.1))
print('b', b.reshape(2, 5))
# 将生成的tensor指定为某个值
c = torch.full([2, 3], 7)  #  [2, 3]代表维度
print(c)

# 生成固定个数
a = torch.linspace(0, 10, steps=5)
print('a', a)
b = torch.logspace(1, 9, steps=4)  # base默认值为10
print('b', b)

# 随机打散
a = torch.randperm(10)  # [0, 10)
print('a', a)