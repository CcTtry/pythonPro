# @Project -> File   ：flyTest -> getRange
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/17 16:04
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

    def mySplit(self, mystr):  # 分割字符串  获得数值的范围
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

    def getStartEnd(self, coordinates_4):

        for i in range(len(coordinates_4)):  # 获取每一个属性的取值范围
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

    def getAvg(self):   # 获取当前所有列的平均值\sigma的值
        print("len of ids is {}".format(len(self.ids)))
        # print("value is {}".format(self.excelDf["S-ZORB.AT_5201.PV"]))

        for i in range(len(self.ids)):
            id = self.ids[i]
            self.averages[ids[i]] = self.excelDf[id].mean()

    def getSigma(self):  # 获取当前所有列的sigma的值
        length = len(self.excelDf)
        for i in range(len(self.ids)):
            temp_x2 = 0.0
            for j in range(length):
                temp_x2 += self.excelDf[self.ids[i]][j] ** 2
            xi2 = (self.averages[self.ids[i]] * length) ** 2  # 累加的平方和
            self.sigma[self.ids[i]] = (temp_x2 - xi2 / length) / math.sqrt(length - 1)


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

    def pro1_3(self):
        N = len(self.ids)
        i = 0
        while i < N: # 剩余有多少列
            # 对于部分数据为空值的位点，空值处用其前后两个小时数据(总体的)的平均值代替
            for j in range(len(self.excelDf)):
                prob = self.excelDf[self.ids[i]][j]
                if pd.isna(prob):
                    #  对于部分数据为空值的位点，空值处用其前后两个小时数据的平均值代替
                    # print("hello", self.ids[i])
                    print(i, j)
                    # print(self.excelDf[self.ids[i]][j])
                    # print(self.excelDf[self.ids[i]].mean())
                    temp = self.excelDf[self.ids[i]].mean()
                    self.excelDf[self.ids[i]][j] = temp
                    # print(self.excelDf[self.ids[i]][j])
            i = i + 1
            # print("i={}".format(i))

    def pro1_4(self):
        # （4）根据工艺要求与操作经验，总结出原始数据变量的操作范围，然后采用最大最小的限幅方法剔除一部分不在此范围的样本；
        # 删除包含不合要求的行， 删除不符合要求的列
        # self.coordinates = excel4.values[:, :]
        N = len(self.ids)  # 数据的条目
        print("长度为{}".format(N))
        i = 0
        while i < N:
            count = 0
            for j in range(len(self.excelDf)):
                # print("  {}  {}  ".format(i, j))
                temp_id = self.ids[i]
                if self.ids[i] == "S-ZORB.SIS_LT_1001.PV":
                    self.excelDf[self.ids[i]][j] /= 10000.00
                if self.excelDf[temp_id][j] >= self.starts[temp_id] and self.excelDf[temp_id][j] <= self.ends[temp_id]:
                    continue
                else:
                    count += 1
                    self.delLines.append(j)
                    # print(self.ids[i], end="")
                    # print("  {}  {}  ".format(i, j))
                if count >= self.theta:
                    self.delCols.append(self.ids[i])
                    self.delColsIndex.append(i)
                    print("这列被删除了    " * 10, end="")
                    print(self.ids[i])
                    N = N - 1
                    del self.excelDf[self.ids[i]]
                    self.ids.remove(temp_id)
                    self.delLines = []
                    self.delCols = []
                    self.delColsIndex = []
                    count = 0
                    break
            i = i+1
        N = len(self.ids)  # 数据的条目
        for i in range(N):
            for j in range(len(self.excelDf)):
                # print("  {}  {}  ".format(i, j))
                temp_id = self.ids[i]
                temp_value = self.excelDf[temp_id][j]
                temp_start = self.starts[temp_id]
                temp_end = self.ends[temp_id]

                if self.ids[i] == "S-ZORB.SIS_LT_1001.PV":
                    self.excelDf[self.ids[i]][j] /= 10000.00
                #
                # try :
                #     print("{}  {}  {}  {}".format(self.excelDf[temp_id][j], self.starts[temp_id], self.excelDf[temp_id][j], self.ends[temp_id]))
                # except:
                #     print(temp_id)
                #     print("{} {} {}".format(i, j, self.excelDf[temp_id][j-1]))
                #     print("{}".format(len(self.excelDf)))
                if  temp_value >= temp_start and temp_value <= temp_end:
                    continue
                else:
                    self.excelDf.drop(labels=j, axis=0, inplace=True)
                    # 当我们在清洗数据时往往会将带有空值的行删除，不论是DataFrame
                    # 还是Series的index都将不再是连续的索引了，那么这个时候我们可以
                    # 使用reset_index()方法来重置它们的索引，以便后续的操作。
                    self.excelDf = self.excelDf.reset_index(drop=True)
                    # print("长度为 {} {}  {}   {} \n".format(i, j, len(self.excelDf), len(self.ids)))
                    i = i - 1
                    break




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
                    # 当我们在清洗数据时往往会将带有空值的行删除，不论是DataFrame
                    # 还是Series的index都将不再是连续的索引了，那么这个时候我们可以
                    # 使用reset_index()方法来重置它们的索引，以便后续的操作。
        # 删除缺失行

        # self.excelDf.dropna(axis=0, how='any', inplace=True)


# 第一行会默认为列名，不会读取第一行
excel4 = pd.read_excel("附件四：354个操作变量信息.xlsx", sheet_name='Sheet1')
coordinates_4 = excel4.values[0:, 0:6]  # 全部行 1-6列  不包含第7列
ids = list(excel4.values[:, 1])

index_col = [int(i) for i in range(40)]
excel3_285 = pd.read_excel('附件三：285号和313号样本原始数据11.xlsx',
                       sheet_name='操作变量', header=None,
                       skiprows=3, skipfooter=41, names=ids)
# skipfooter = 41 跳过末尾的41行的数据

excel3_313 = pd.read_excel('附件三：285号和313号样本原始数据11.xlsx',
                       sheet_name='操作变量', header=None,
                       skiprows=44, names=ids)
# 取消列标签的初始化，略过前44行的数据， 指定列标签


print(len(excel3_285))
excel3_285 = excel3_285.reset_index()  # dataframe = dataframe.reset_index()  # THE FIX
excel3_313 = excel3_313.reset_index()  # dataframe = dataframe.reset_index()  # THE FIX
# excel3_285.drop(labels=1,  axis=0, inplace=True)
print(len(excel3_285))
# exit()
# 全部设置为浮点类型
for i in range(N):
    excel3_285[ids[i]] = excel3_285[ids[i]].values.astype(float)
    excel3_313[ids[i]] = excel3_313[ids[i]].values.astype(float)


#  我们使用drop()函数，此函数有一个列表形参labels，写的时候可以加上labels=[xxx]，
#  也可以不加，列表内罗列要删除行或者列的名称，默认是行名称，如果要删除列，则要增加参数axis=1

sl285 = Solution(coordinates_4, ids, excel3_285)
sl313 = Solution(coordinates_4, ids, excel3_313)
sl285.getStartEnd(sl285.coordinates)
sl313.getStartEnd(sl313.coordinates)
# exit()

# sl285.theta = XX  设置阈值，超过了阈值就删除相应的列
# sl313.theta = XX
# （1）对于只含有部分时间点的位点，如果其残缺数据较多，无法补充，将此类位点删除；
# （2）删除325个样本中数据全部为空值的位点；
sl285.pro1_1and2()
sl313.pro1_1and2()

# 对于部分数据为空值的位点，空值处用其前后两个小时数据(总体的)的平均值代替
# （3）对于部分数据为空值的位点，空值处用其前后两个小时数据的平均值代替；??? 只有几分钟？(使用有效值的平均值来替换)
sl285.pro1_3()
sl313.pro1_3()
print(len(sl285.excelDf))
print(len(sl285.ids))
print("pro1_3 end\n\n")
# exit()
# （4）根据工艺要求与操作经验，总结出原始数据变量的操作范围，然后采用最大最小的限幅方法剔除一部分不在此范围的样本；
sl285.pro1_4()
print("pro1_4-285 end、n\n\n\n")
sl313.pro1_4()
print("pro1_4 end、n\n\n\n")
# exit()
# for i in range(len(sl285.excelDf)):
#     print(sl285.excelDf['S-ZORB.CAL_H2.PV'][i])
#     print(sl285.excelDf['S-ZORB.AT_5201.PV'][i])
# print(len(sl285.excelDf))
# print(sl285.excelDf['S-ZORB.CAL_H2.PV'])
# exit()
# （5）根据拉依达准则（3σ准则）去除异常值。
sl285.pro1_5()
print("pro1_5-285 end、n\n\n\n")
sl313.pro1_5()
print("pro1_5 end、n\n\n\n")

exit()



exit()
print("\n\n")



# 自定义dataframe时如何为个别元素赋空值


# 字符串类型的，使用None赋值为空值
# 数值类型的，使用numpy.NaN赋值为空值
# 时间类型的，使用pandas.NaT赋值为空值

# 样本编号285
#  根据工艺要求与操作经验，总结出原始数据变量的操作范围，然后采用最大最小的限幅方法剔除一部分不在此范围的样本
for i in range(N):
    count = 0
    for j in range(len(excel3_285)):
        if ids[i] == "S-ZORB.SIS_LT_1001.PV":
            excel3_285[ids[i]][j] /= 10000.00
            # print(excel3_285[ids[i]][j])
        if excel3_285[ids[i]][j] >= sl.starts[ids[i]] and excel3_285[ids[i]][j] <= sl.ends[ids[i]]:
            continue
        else:
            count += 1
            # print("具体={:<10.4f}   范围为{:>10}-{:>10}  ".format(excel3_285[ids[i]][j], sl.starts[ids[i]], sl.ends[ids[i]]))
            excel3_285[ids[i]][j] = float(np.nan)
            # if pd.isna(excel3_285[ids[i]][j]):
            #     print("成功了")
        if count == len(excel3_285):
            print("没想到啊范围不对应={:<10.4f}   范围为{:>10}-{:>10}  id={}".format(excel3_285[ids[i]][j], sl.starts[ids[i]], sl.ends[ids[i]], ids[i]))
            sl.delCols.append(ids[i])
            sl.delColsIndex.append(i)

        # 以上为不符合范围的列表

# 删除包含不合要求的行， 删除不符合要求的列
excel3_285.dropna(axis=0, how='any', inplace=True)
for i in sl.delCols:
    del excel3_285[i]

# for i in range(N):
#     tem_avg = sl.

exit()


print(len(coordinates_4))
# excel3.colums = list(ids)
i = 1

print("\n\n\n\n\n")
# del excel3[ids[0]]  # 删除指定列名称 的列
# coordinates_285 = excel3.values[0:40, :]  # 获取了样本号为285的数据  0-39 行
# coordinates_313 = excel3.values[41:82, :]  # 获取了样本号为313的数据 41-81行
# print(type(coordinates_285))


# 设置阈值N  空值数据大于等于N的时候，删除这一列的内容 #

    # # 对于只含有部分时间点的位点，如果其残缺数据较多，无法补充，将此类位点删除
    # # 删除325个样本中数据全部为空值的位点
    # if excel4[ids[i]].isna().sum() > sl.theta or excel4[ids[i]].isna().sum()==len(coordinates_285):
    #     del excel3[ids[i]]
    #     N = N - 1
coordinates_285 = excel3.values[0:40, :]  # 获取了样本号为285的数据  0-39 行
coordinates_313 = excel3.values[41:82, :]  # 获取了样本号为313的数据 41-81行


# for i in range(len(coordinates_313)):
#     print(i, coordinates_313[i][0])
# 判断某一列为空的数据个数  其中单位指的是列标签
#
print(ids[0])
print(excel3[ids[0]].mean())  # 22.0  指定列的平均值
print(excel4["单位"].isna().sum())
count = 0
for i in range(len(coordinates_4)):
    if pd.isna(coordinates_4[i][4]):
        count += 1
    # print("")
print(count)
print(excel3.values[0:40, 0].mean())  # 22.0  指定列的平均值
# for index, row in excel3.iteritems():
#     if i == 0:
#         print(index)
#         # break
#     print(i, index)
#     i +=1
    # print(i, index)
    # i = i+1

# print(coordinates_285.columns[0])
# exit()
# del coordinates_4[ids[1][0]]












# for i in range(len(coordinates_4)):
#     df['坐落地址'].isnull.sum()
# for i in range(len(coordinates_285)):
#     print(coordinates_285[i])
# print("\n\n")
# for i in range(len(coordinates_313)):
#     print(coordinates_313[i])
#     # if pd.isna(coordinates_4[i][4]):  # 判断读取的是否为空值
#     #     coordinates_4[i][4] = "   test NAN"
#     #     valid_count
#     # print(
#     #     "index:{:<8}name={:<30} 单位{:<20} ".format(coordinates_4[i][0], coordinates_4[i][1],
#     #                                                                  coordinates_4[i][4]))
#

#
# for i in range(len(coordinates_485)):
#     print("index:{:<30}    {:<30} range={:<15}   delta={:<10} ".format(coordinates_485[i][0],
#                                                                        coordinates_485[i][1],
#                                                                        coordinates_485[i][2],
#                                                                        coordinates_485[i][3])
#           )





