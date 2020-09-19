from torch import nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch


class lstm(nn.Module):
    def __init__(self, input_size=24, hidden_size=4, output_size=1, num_layer=2):
        super(lstm, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# 数据预处理
labels_path = './附件一：325个样本数据.xlsx'
labels_read = pd.read_excel(labels_path)

RONLoss = labels_read.loc[0:325, 'RON-Loss']
SS = labels_read.loc[0:325, '硫含量-2,μg/g']
RON = labels_read.loc[0:325, '辛烷值RON-2']
# data_read = labels_read.drop('RON-Loss', axis=1)
# data_read = labels_read.drop('硫含量-2,μg/g', axis=1)
# data_read = labels_read.drop('辛烷值RON-2', axis=1)

feature_path = './特征选择结果-逐步回归.xlsx'
feature_read = pd.read_excel(feature_path)
dataset = feature_read.values   # 获得csv的值
# dataset = dataset.astype('double')

'''归一化'''
# max_value = np.max(dataset)  # 获得最大值
# min_value = np.min(dataset)  # 获得最小值
# scalar = max_value - min_value  # 获得间隔数量
# dataset = list(map(lambda x: x / scalar, dataset))  # 归一化


# 创建好输入输出
# data_X, data_Y = create_dataset(dataset)
data_X = np.array(feature_read)
data_Y = np.array(RONLoss)
print(data_X.shape, data_Y.shape)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

# 设置LSTM模型数据类型形状
train_X = train_X.reshape(-1, 1, 24)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 24)

train_x = torch.from_numpy(train_X).float().cuda()
train_y = torch.from_numpy(train_Y).float().cuda()
test_x = torch.from_numpy(test_X).float().cuda()

model = lstm(24, 4, 1, 2).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# 开始训练
for e in range(1000000):
    # train_x = torch.tensor(train_x, dtype=torch.double)
    # train_y = torch.tensor(train_y, dtype=torch.double)

    # 前向传播
    model = model.train()
    out = model.forward(train_x)
    loss = criterion(out, train_y)
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (e + 1) % 200 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

        model = model.eval()  # 转换成测试模式
        # data_X = data_X.reshape(-1, 1, 24)
        # data_X = torch.from_numpy(data_X).float()
        pred_test = model(test_x).cpu()  # 测试集的预测结果

        pred_test = pred_test.view(-1).data.numpy()
        plt.plot(pred_test, 'r', label='prediction')
        plt.plot(test_Y, 'b', label='real')
        plt.legend(loc='best')
    if (e + 1) % 20000 == 0:
        state = {'net': model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':(e+1)}
        dir = 'parameter' + (e+1).__str__() + '.pkl'
        torch.save(state, dir)
        print("succ \n\n\n")
        # plt.show()




