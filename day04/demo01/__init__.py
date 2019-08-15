"""统计属性-2"""
import torch
a = torch.randn(4, 10)
print(a.max(dim=1))
print("-===-=-============------max()=---------==========--------")
print(a.max(dim=1, keepdim=True))  # keepdim=True保持返回的维度跟a的维度是一样的a.dim = 2,
print("-===-=-============--------topk()=============---------==========--------")
print(a.topk(3, dim=1, largest=False))  # 返回每行最小的3个数字， largest默认为True，表示最大的，为False表示为最小的
print(a.topk(3, dim=0, largest=False))  # 返回每列最小的3个数字
print("-===-=-============--------kthavalue()=============---------==========--------")
print(a.kthvalue(8, dim=1, keepdim=True))  # 返回每行第8小的数字
print("-===-=-============--------compare(比较）--------==========--------")
b = torch.randn(4, 10)
c = torch.eq(a, b)  # a=b 判断两个tensor相应的元素是否相等
d = torch.gt(a, b)  # a>b
x = torch.equal(a, b)  # 判断两个tensor是否相等，返回True或False
print(c)
print(d)
print(x)









