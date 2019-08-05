#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: ouyang lei time:2019/8/4
import torch
""""
基本的运算：
    矩阵相乘
    torch.mm(a, b)  仅限2维相乘
    torch.matmul(a, b)  可多维相乘
    a@b    可多维相乘
"""
#  @
a = torch.rand(4, 784)
x = torch.rand(4, 784)
w = torch.rand(512, 784)
print((x@w.t()).shape)
# matmul
a = torch.rand(4, 3, 28, 64)
b = torch.rand(4, 3, 64, 32)
c = torch.matmul(a, b)  # torch.matmul(a, b)  只将a和b的最后两个维度进行相乘即（28， 64）@（64， 32）= （28， 32）
print(c.shape)
t = torch.rand(4, 1, 64, 32)
s = torch.matmul(a, t)  # 启动了broadcast自动扩展机制（广播）
print(s.shape)

# pow()
a = torch.full([2, 2], 3)
print(a.pow(2))
# sqrt()
print(a.pow(2).sqrt())
# rsqrt()  平方根的倒数
print(a.pow(2).rsqrt())

a = torch.exp(torch.ones(2, 2))
print(a)
print(torch.log(a))

"""
  近似值
    .floor()  往下取整
    .ceil()   往上取整
    .round()  四舍五入
    .trunc()  取整数部分
    .frac()   取小数部分
"""
a = torch.tensor(3.14)
print(a.floor())
print(a.ceil())
print(a.round())
print(a.trunc())
print(a.frac())

'''
    clamp()裁剪  一般用于梯度裁剪
'''
a = torch.rand(2, 3)*15
print(a)
print(a.clamp(10))   # 一个值时为最小值
print(a.clamp(1, 10))  # 两个值时为最小值和最大值





