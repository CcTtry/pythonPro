# @Project -> File   ：flyTest -> 展示图2
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/19 11:21
# @Desc   ：
# @Project -> File   ：flyTest -> draw
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/15 21:31
# @Desc   ：

from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts import options as opts
import matplotlib.pyplot as plt
import matplotlib
import xlrd
#解决中文显示问题
import matplotlib.pyplot as plt
import  pandas as pd
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, savefig

import numpy as np
# ids = list(["0", "S-ZORB.TC_2801.PV", "S-ZORB.TE_2103.PV", "S-ZORB.TE_2005.PV", "S-ZORB.TE_9301.PV"])
ids = list(["0", "还原器温度",	"反应器上部温度",	"反应器底部温度", "1.0MPa蒸汽进装置温度"])
for i in range(len(ids)):
    print(ids[i])
excel3_285 = pd.read_excel('画图.xlsx',
                           skipfooter=41,
                           names=ids)
# skipfooter = 41 跳过末尾的41行的数据

excel3_313 = pd.read_excel('画图.xlsx', skiprows=43, header=None, names=ids)
print(excel3_285.columns.values)

# for i, j in excel3_285.iteritems():
#     print(i, j)
# exit()
for i in range(len(ids)):
    excel3_285[ids[i]] = excel3_285[ids[i]].values.astype(float)
    excel3_313[ids[i]] = excel3_313[ids[i]].values.astype(float)

print(len(excel3_285))
# for i in range(len(excel3_285)):
#     print(excel3_285[ids[1]][i])
# print("\n\n")
# for i in range(len(excel3_313)):
#     print(excel3_313[ids[1]][i])
# exit()
x1 = excel3_285.values[:, 0]
print(x1)
# exit()
y1 = excel3_285.values[:, 1]
y2 = excel3_285.values[:, 2]
y3 = excel3_285.values[:, 3]
y4 = excel3_285.values[:, 4]


# plt.rcParams['font.sans-serif'] = ['KaiTi'] # 指定默认字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体支持
plt.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
fig = plt.figure(figsize=(12, 5))

# p1 = plt.step(x2, AR, label="Arrival rate")  # 锯齿图
plt.ylabel("温度℃", { 'family': 'sans-serif', 'weight': 'bold', 'size': 15})
plt.xticks(np.arange(0, 123, 6))
plt.xlabel("分钟", {'family': 'sans-serif', 'weight': 'bold', 'size': 15})
# 添加一条坐标轴，y轴的
# plt.twinx()

plt.yticks(np.arange(201.42290, 201.42970, (201.42970 - 201.42290)/12))

p1 = plt.plot(x1, y4, color="blue", marker='*', label=ids[4])

# plt.ylabel("Average stay time(ms)", {'family': 'Times New Roman', 'weight': 'bold', 'size': 25})

# p = p2 + p3
p = p1
labs = [l.get_label() for l in p]
# plt.yticks([x for x in range(max(xs) + 1) if x % 2 == 0])  # x标记step设置为2
plt.xticks(fontproperties='Times New Roman', size=15)
plt.legend(p, labs, loc='upper right', prop={'family': 'sans-serif', 'size': 16})
plt.savefig("23pic.png", bbox_inches='tight')
plt.show()


'''
0: ‘best'
1: ‘upper right'
2: ‘upper left'
3: ‘lower left'
4: ‘lower right'
5: ‘right'
6: ‘center left'
7: ‘center right'
8: ‘lower center'
9: ‘upper center'
10: ‘center'
'''