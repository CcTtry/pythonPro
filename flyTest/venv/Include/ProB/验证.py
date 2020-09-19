# @Project -> File   ：flyTest -> 验证
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/19 17:15
# @Desc   ：

import openpyxl
import sys
import random
from gurobipy import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, savefig
import os
import re

N = 354
class Solution:
    def __init__(self, coordinates, ids = None, excel3_285=None, excel3_313=None):
        self.coordinates = coordinates                # 表4的数据
        self.starts = {}   # np.zeros((len(coordinates_4)))  # 根据ID存储范围起点
        self.ends = {}   # np.zeros((len(coordinates_4)))    # 根据ID存储范围终点
        self.excelDf = excel3_285         # 表3样本285的数据
        self.excel3_313 = excel3_313        # 表3样本313的数据
        self.ids = ids                                 # 354列的ID属性名
        self.theta = 10  # 空值超过了这个阈值，就删除该列属性的值
        self.delLines = []   # 用于存储需要删除的行
        self.delCols = []
        self.delColsIndex = []
        self.averages = {}    # ID->averaeg
        self.sigma = {}    # ID->sigma
        self.vi = {}    # ID->剩余误差
        # self.validCount = np.zeros(len(coordinates_4))
        # self.starts = starts
        # self.ends = ends

    # 分割字符串  获得数值的范围
    def mySplit(self, mystr):
        endstr = mystr[-1]
        beginstr = mystr[0]
        low = 0
        high = 0
        if endstr == ')' or endstr == '）':  # 第二个数带括号  # 西文字符 中文字符
            for i in range(len(mystr) - 1, -1, -1):
                if mystr[i] != '(' and mystr[i] != '（':  # 西文字符 中文字符
                    continue
                else:
                    high = (mystr[i + 1:len(mystr) - 1])
                    break
        else:  # 第二个数不带括号
            for i in range(len(mystr) - 1, -1, -1):
                if mystr[i] != '-':
                    continue
                else:
                    high = (mystr[i + 1:len(mystr)])
                    break
        # 第一个数的处理
        if beginstr == '-':
            for i in range(1, len(mystr)):
                if mystr[i] != '-':
                    continue
                else:
                    low = (mystr[0:i])
                    break
        elif beginstr != '-':
            for i in range(len(mystr)):
                if mystr[i] != '-':
                    continue
                else:
                    low = (mystr[0:i])
                    break
        return float(low), float(high)

    # 获取每一个属性的取值范围
    def getStartEnd(self, coordinates_4):
        for i in range(len(coordinates_4)):
            temp = re.split('-', coordinates_4[i][3])
            begin = float(-99999)
            end = float(999999)
            if len(temp) > 2:
                begin, end = self.mySplit(coordinates_4[i][3])
            else:
                begin, end = float(temp[0]), float(temp[1])
            self.starts[coordinates_4[i][1]] = begin
            self.ends[coordinates_4[i][1]] = end
            # print("index:{:<8}name={:<30}  idName={}  range ={} begin={:<10}  end={:<10}  delta={:<10} {} {}".format(coordinates_4[i][0], coordinates_4[i][1],
            #       coordinates_4[i][1], coordinates_4[i][3], self.starts[coordinates_4[i][1]], self.ends[coordinates_4[i][1]], coordinates_4[i][5], type(begin), type(end)))

    # 获取当前所有列的平均值
    def getAvg(self):
        print("len of ids is {}".format(len(self.ids)))
        # print("value is {}".format(self.excelDf["S-ZORB.AT_5201.PV"]))

        for i in range(len(self.ids)):
            id = self.ids[i]
            self.averages[ids[i]] = self.excelDf[id].mean()

    # 获取当前所有列的sigma的值
    def getSigma(self):
        length = len(self.excelDf)
        for i in range(len(self.ids)):
            temp_x2 = 0.0
            for j in range(length):
                temp_x2 += self.excelDf[self.ids[i]][j] ** 2
            xi2 = (self.averages[self.ids[i]] * length) ** 2  # 累加的平方和
            self.sigma[self.ids[i]] = (temp_x2 - xi2 / length) / math.sqrt(length - 1)

    # （1）对于只含有部分时间点的位点，如果其残缺数据较多，无法补充，将此类位点删除；
    def pro1_1and2(self):
        N = len(self.ids)
        i = 0
        while i < N:  # 根据ids的列数来确定列数  使用for i in range 无法动态更新
            # 设置阈值N  空值数据大于等于N的时候，删除这一列的内容 #
            # 对于只含有部分时间点的位点，如果其残缺数据较多，无法补充，将此类位点删除
            # 删除325个样本中数据全部为空值的位点
            temp_id = self.ids[i]
            if self.excelDf[temp_id].isna().sum() > self.theta: # or self.excelDf[self.ids[i]].isna().sum() == len(self.excelDf)
                del self.excelDf[self.ids[i]]
                temp = self.ids[i]
                # print(len(self.ids))
                self.ids.remove(temp)  # 删除对应列
                # print(temp in ids)
                # print(" # 删除对应列 ")
                # print(len(self.ids))
                # i = i + 1
                N = N - 1
            i = i + 1
        print(self.excelDf[self.ids[0]].mean())

    # （3）对于部分数据为空值的位点，空值处用其前后两个小时数据的平均值代替；
    def pro1_3(self):
        N = len(self.ids)
        i = 0
        while i < N:  # 剩余有多少列
            # 对于部分数据为空值的位点，空值处用其前后两个小时数据(总体的)的平均值代替
            for j in range(len(self.excelDf)):
                prob = self.excelDf[self.ids[i]][j]
                if pd.isna(prob):
                    temp = self.excelDf[self.ids[i]].mean()
                    self.excelDf[self.ids[i]][j] = temp
            i = i + 1

    # （4）根据工艺要求与操作经验，总结出原始数据变量的操作范围，然后采用最大最小的限幅方法剔除一部分不在此范围的样本；
    # 删除包含不合要求的行， 删除不符合要求的列
    def pro1_4(self):
        N = len(self.ids)  # 数据的条目
        i = 0
        while i < N:
            count = 0
            temp_id = self.ids[i]
            temp_start = self.starts[temp_id]
            temp_end = self.ends[temp_id]
            for j in range(len(self.excelDf)):

                temp_value = self.excelDf[temp_id][j]


                if self.ids[i] == "S-ZORB.SIS_LT_1001.PV":  # 特定一列的取值处理
                    self.excelDf[self.ids[i]][j] /= 10000.00
                if temp_value >= temp_start and temp_value <= temp_end:  # 是否在取值范围内
                    continue
                else:
                    count += 1
                    self.delLines.append(j)
                    # print(self.ids[i], end="")
                    # print("  {}  {}  ".format(i, j))
                if j == len(self.excelDf)-1:
                    if count >= self.theta:
                        print(count, end="  ")
                        self.delCols.append(self.ids[i])
                        self.delColsIndex.append(i)
                        # print("被删除的列： ", end="")
                        # print(self.ids[i])
                        N = N - 1

                        print("{}   [{:.8f}, {:.8f}]     ".format(temp_id, self.excelDf[temp_id].min(), self.excelDf[temp_id].max()), end="")
                        # print("[{:.3f}, {:.3f}]     ".format(self.excelDf[temp_id].min(), self.excelDf[temp_id].max()))
                        # print("{}  ".format(temp_id))  # , self.excelDf[temp_id].min(), self.excelDf[temp_id].max())
                        print("[{:.8f}, {:.8f}] ".format(temp_start, temp_end))
                        del self.excelDf[self.ids[i]]
                        self.ids.remove(temp_id)
                        self.delLines = []
                        self.delCols = []
                        self.delColsIndex = []
                        count = 0
                        break
                    else:
                        # print(count)
                        self.delCols.append(self.ids[i])
                        self.delColsIndex.append(i)
                        # print("被删除的列： ", end="")
                        # print(self.ids[i])
                        N = N - 1

                        # print("{}   [{:.6f}, {:.6f}]     ".format(temp_id, self.excelDf[temp_id].min(), self.excelDf[temp_id].max()))
                        # print("[{:.3f}, {:.3f}]     ".format(self.excelDf[temp_id].min(), self.excelDf[temp_id].max()))
                        # print("{}  ".format(temp_id))  # , self.excelDf[temp_id].min(), self.excelDf[temp_id].max())
                        # print("[{:.3f}, {:.3f}] ".format(temp_start, temp_end))
                        # del self.excelDf[self.ids[i]]
                        # self.ids.remove(temp_id)
                        self.delLines = []
                        self.delCols = []
                        self.delColsIndex = []
                        count = 0
                        break
            i = i+1
        N = len(self.ids)  # 数据的条目l
        for i in range(N):
            for j in range(len(self.excelDf)):
                # print("  {}  {}  ".format(i, j))
                temp_id = self.ids[i]
                temp_value = self.excelDf[temp_id][j]
                temp_start = self.starts[temp_id]
                temp_end = self.ends[temp_id]

                if self.ids[i] == "S-ZORB.SIS_LT_1001.PV":   #  某一列的取值处理
                    self.excelDf[self.ids[i]][j] /= 10000.00
                if temp_value >= temp_start and temp_value <= temp_end:  # 是否在取值范围内
                    continue
                else:
                    self.excelDf.drop(labels=j, axis=0, inplace=True)
                    # 当我们在清洗数据时往往会将带有空值的行删除，不论是DataFrame
                    # 还是Series的index都将不再是连续的索引了，那么这个时候我们可以
                    # 使用reset_index()方法来重置它们的索引，以便后续的操作。
                    self.excelDf = self.excelDf.reset_index(drop=True)
                    i = i - 1
                    break

    # （5）根据拉依达准则（3σ准则）去除异常值。
    def pro1_5(self):
        self.getAvg()
        self.getSigma()
        length = len(self.excelDf)
        for i in range(len(self.ids)):   # 每一列计算一次
            for j in range(length):
                tem_vi = abs(self.averages[self.ids[i]] - self.excelDf[ids[i]][j])
                if tem_vi > 3 * self.sigma[self.ids[i]]:
                    # self.excelDf[self.ids[i]][j] = float(np.nan)
                    self.excelDf.drop(labels=j, axis=0, inplace=True)
                    self.excelDf = self.excelDf.reset_index(drop=True)
                    length = length - 1
                    i = i - 1
                    break

# 开始执行程序
# 第一行会默认为列名，不会读取第一行作为内容
excel4 = pd.read_excel("附件四：354个操作变量信息.xlsx", sheet_name='Sheet1')
coordinates_4 = excel4.values[0:, 0:6]  # 全部行 1-6列  不包含第7列
ids1 = list(excel4.values[:, 1])   # 获取列标签

temp = list([i for i in range(16)])  # 补充开头没有列标签的16列
ids1 = temp + ids1    # 16 到 16+354范围为有效
featureMap = {

}

for i in range(N):
    featureMap[coordinates_4[i][1]] = coordinates_4[i][2]

excel_00 = pd.read_excel('特征选择结果-30维.xlsx')
coordinates_00 = excel_00.values[0:, :]
colsMap = list(excel_00.columns.values)
for i in colsMap:
    print(featureMap[i])
# for i in range(N):
# print(excel_00.columns.values)
