# @Project -> File   ：flyTest -> pro2
# @IDE    ：PyCharm
# @Author ：Ctry
# @Date   ：2020/9/16 10:41
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

data_path = os.path.join('数据集1终稿.xlsx')
# 打开 excel 文件,获取工作簿对象
wb = openpyxl.load_workbook(data_path)

# 获取指定的表单
sheet = wb['data1']
point_list = []
for row in sheet[3:sheet.max_row]:
    point_list.append([row[1].value, row[2].value, row[3].value, row[4].value, row[5].value])
point_num = len(point_list)
print("\npintnum =  {} ".format(point_num))


def get_distance(start_index, end_index):
    x1, y1, z1 = point_list[start_index][0:3]
    x2, y2, z2 = point_list[end_index][0:3]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5


def judge_z(start_index, end_index=point_num - 1):
    x1, y1, z1 = point_list[start_index][0:3]
    x2, y2, z2 = point_list[end_index][0:3]
    if abs(z1 - z2) > 5000:
        return False
    else:
        return True


def judge(start_index, end_index, horizontal_error, vertical_error):
    end_point_type = point_list[end_index][3]
    distance = get_distance(start_index, end_index)
    delta_error = distance * sigma
    end_point_horizontal_error = horizontal_error + delta_error
    end_point_vertical_error = vertical_error + delta_error

    if judge_z(start_index):
        if end_index != point_num - 1:  # 下一个点不是终点
            if end_point_type == 0: # 水平
                if end_point_horizontal_error <= beta2 and end_point_vertical_error <= beta1:
                    is_pass = True
                else:
                    is_pass = False
            elif end_point_type == 1:  # 垂直
                if end_point_horizontal_error <= alpha2 and end_point_vertical_error <= alpha1:
                    is_pass = True
                else:
                    is_pass = False
            else:
                is_pass = False
        else:  # 下一个点是终点
            if end_point_horizontal_error <= theta and end_point_vertical_error <= theta:
                is_pass = True

            else:
                is_pass = False
    else:
         is_pass = False

    after_end_point_horizontal_error = end_point_horizontal_error
    after_end_point_vertical_error = end_point_vertical_error
    if is_pass:
        if end_point_type == 0:
            after_end_point_horizontal_error = 0
            after_end_point_vertical_error = end_point_vertical_error
        elif end_point_type == 1:
            after_end_point_horizontal_error = end_point_horizontal_error
            after_end_point_vertical_error = 0
    return is_pass, end_point_horizontal_error, end_point_vertical_error, after_end_point_horizontal_error, after_end_point_vertical_error


def rank_distance(point_index_list):
    ranked_list = sorted(point_index_list, key=lambda index_list: get_distance(index_list[0], point_num - 1))
    return ranked_list


def get_all_distance(index_list):
    distance = 0
    for i in range(len(index_list) - 1):
        start_index = index_list[i]
        end_index = index_list[i + 1]
        distance = distance + get_distance(start_index, end_index)
    return distance


vis = point_num * [0]
order = []
temp_all_distance = 1000000
temp_order = []

def find_path(start_index=0, horizontal_error=0, vertical_error=0):
    global temp_all_distance
    global order, temp_order
    order.append(start_index)
    candidate_list = []
    for index in range(point_num):
        if index != start_index:
            is_pass, end_point_horizontal_error, end_point_vertical_error, after_end_point_horizontal_error, after_end_point_vertical_error \
                = judge(start_index, index, horizontal_error, vertical_error)
            if is_pass and get_distance(start_index, point_num - 1) > get_distance(index, point_num - 1):
                candidate_list.append(index)
    if len(candidate_list) == 0:
        order.pop()
        return

    candidate_list.sort(key=lambda index_list: get_distance(index_list, point_num - 1))
    # random.shuffle(candidate_list)
    if len(candidate_list) >= 5:
        candidate_list = candidate_list[0:5]
    for candidate in candidate_list:
        if candidate == point_num - 1:
            order.append(candidate)
            all_distance = get_all_distance(order)
            if all_distance < temp_all_distance:
                temp_all_distance = all_distance
                temp_order = order.copy()
                print('\n' + 'small', order, all_distance)
            order.pop()
            break
        if len(order) > 13:
            break
        is_pass, end_point_horizontal_error, end_point_vertical_error, after_end_point_horizontal_error, after_end_point_vertical_error = judge(
        start_index, candidate, horizontal_error, vertical_error)
        find_path(candidate, after_end_point_horizontal_error, after_end_point_vertical_error)

    order.pop()
    return
# 定义新的变量w为字典，简直为(k, i, j)，表示由k点至i点，再到j点(i, j)之间的圆弧+切线段的长度
# 首先生成列表triple_edge，元素为元组(k,i,j), 表示(k,i)与  (i,j)都在可行边集edge里面
node = [0, 503, 294, 91, 607, 540, 250, 340, 277, 612]

import matplotlib.pyplot as plt
import  pandas as pd
from  mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import plot, savefig

excel1 = pd.read_excel("数据集1终稿.xlsx")
coordinates_2 = excel1.values[1:, 1:4]  # 全部行 2-5列

correction_vector = list(excel1.values[1:, 4])  # 取所有行的第5列
correction_vector[0] = 2
correction_vector[612] = 3
fig = plt.figure()
ax = Axes3D(fig)
A_B = [coordinates_2[shap]]