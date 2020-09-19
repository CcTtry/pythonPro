# @Project -> File   ：flyTest -> pro1
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/16 11:13
# @Desc   ：

import openpyxl
import sys
import random
from gurobipy import *
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import  pandas as pd
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, savefig

shape = 613
random.seed(1)
sys.setrecursionlimit(100000)
alpha1 = 25
alpha2 = 15
beta1 = 20
beta2 = 25
theta = 30
sigma = 0.001
delta = 0.001
import os

excel1 = pd.read_excel("数据集1终稿.xlsx")
coordinates_2 = excel1.values[1:, 1:4]  # 全部行 2-5列

correction_vector = list(excel1.values[1:, 4])  # 取所有行的第5列
correction_vector[0] = 2
correction_vector[612] = 3
fig = plt.figure()
ax = Axes3D(fig)
A_B = [coordinates_2[shap]]

# 计算欧式距离
A = np.zeros((shape, shape))
for i in range(shape):
    for j in range(shape):
        A[i, j] = np.sqrt((coordinates_2[i, 0] - coordinates_2[j, 0])**2 +
                          (coordinates_2[i, 1] - coordinates_2[j, 1])**2 +
                          (coordinates_2[i, 2] - coordinates_2[j, 2])**2)

# 将A转换成tupledict类型
dict_A = {}
for i in range(shape):
    for j in range(shape):
        dict_A[i, j] = A[i, j]
dict_A = tupledict(dict_A)
# 将dict_A用于存储剪枝后的邻接矩阵


# 问题123 程序3
# 剪枝过程， V 为垂直校正点集合，H为水平校正点集合，共shape-2个点；起点1个，终点1个
V = []
H = []
for i in range(shape):
    if correction_vector[i] == 1:
        V.append(i)
    elif correction_vector[i] == 0:
        H.append(i)
    else:
        pass

# 问题123 程序4
C = np.ones(shape, shape)
for i in range(1, shape - 1):  # 不包含起点和终点
    for j in V:
        if A[i, j] > min(alpha1, alpha2)/delta:
            dict_A[i, j] = 0
            c[i, j] = 0

for i in range(1, shape - 1):  # 不包含起点和终点
    for j in H:
        if A[i, j] >= min(beta1, beta2)/delta:
            dict_A[i, j] = 0
            C[i, j] = 0

for i in range(shape - 1):
    if dict_A[i, shape - 1] > theta/delta:
        dict_A[i, shape - 1] = 0
        C[i, shape - 1] = 0

for i in range(shape):  # 没有自环
    C[i, i] = 0

edge = []
for i in range(shape):
    for j in range(shape):
        if dict_A[i, j] != 0:
            edge.append((i, j))
        else:
            pass

# 以上为预处理的程序
# 将 dict_A用于存储剪枝后的邻接矩阵
# gurobipy 中的tuplelist类是Python中list的子类， tupledict是dict的子类
# tupledict是python类dict的子类，由键值两部分组成。
# key 为 上文提到的tuplelist，value为gurobi的变量var类 tupledict可以方便地索引下标以及创建表达式

#  剪枝过程 V为垂直校正点几何， H为水平校正点集合，共shape-2个点，起点1个，终点1个。

# 问题1 建模 程序1
model1 = Model()
# 添加新变量x[i,j], i = 0 to shape-1, j=0 to shape-1
x = model1.addVar(shape, shape, vtype=GRB.BINARY, name='x')  # 二进制的变量
# 添加新决策 变量h[i], v[i] i = 0 to shape-1
h = model1.addVar(shape, vtype=GRB.CONTINUOUS, name='h')  # 连续的变量
v = model1.addVar(shape, vtype=GRB.CONTINUOUS, name='v')
# 添加限制条件 x[i, j] ==0 根据C矩阵的信息进行剪枝
for i in range(shape):
    for j in range(shape):
        if C[i, j] == 0:
            model1.addConstr(x[i, j] == 0)
        # 添加限制

# 添加限制条件起始点 中间点 终点的初度入度条件  中间节点的限制条件需要 节点的入度等于出度
test1 = [0]*shape  # 出度表达式
test2 = [0]*shape  # 入度表达式
# test1[i] 表示i节点的出度
for(i, j) in edge:
    test1[i] = test1[i] + x[i, j]

# test2[i] 表示i节点的入度
for(i, j) in edge:
    test2[i] = test2[i] + x[j, i]

for i in range(shape):
    if i == 0:
        model1.addConstr(test1[i] == 1)
        model1.addConstr(test2[i] == 0)
    elif 0 < i < shape - 1:
        model1.addConstr(test1[i] == test2[i])
    else:
        model1.addConstr(test1[i] == 0)
        model1.addConstr(test2[i] == 1)

# 添加限制条件对h，v进行约束，M=10000
for (i, j) in edge:
    # 计算欧式距离 A[i,j]*delta表示 便宜的差距
    model1.addConstr(correction_vector[i] * h[i] + delta*A[i, j] - h[j] <=
                     10000 - 10000*x[i, j])



