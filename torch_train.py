import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch_data import Torch_data_set
import visualization as v
from dataPreprocessor import *

NUM_EPOCHS = 2
BATCH_SIZE = 128

print(torch.cuda.is_available())
device = "cuda:0" if torch.cuda.is_available()  else "cpu"


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Conv1 = nn.Conv2d(1, 20, 5, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.Conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.Conv1(x))
        x = self.pool(x)
        x = F.relu(self.Conv2(x))
        x = self.pool(x)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    train_data = raw_file_idx3_process('./data/train/train-images-idx3-ubyte')
    train_label = raw_file_idx1_process('./data/train/train-labels-idx1-ubyte')
    train_set = Torch_data_set(train_data, train_label)
    train_loader = tud.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

    # net 对象
    net = Net()
    # 损失函数 交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    # 优化器 SGD 随机梯度下降 学习率0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

    # 使用gpu
    net = net.to(device)

    # 循环 训练
    for epoch in range(NUM_EPOCHS):
        print("\nepoch: ", epoch + 1)
        b = v.Progress_bar()
        for index, data in enumerate(train_loader):
            b.bar(index, train_set.__len__(), "Training: ", BATCH_SIZE)
            image, label = data
            image, label = image.to(device), label.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 获取output layer的值
            predict = net(image)
            # 获取误差
            loss = loss_fn(predict, label)
            print("loss: ", torch.sum(loss).cpu().item(), end='')
            # bp
            loss.backward()
            # 正向传播
            optimizer.step()

    torch.save(net, "torch_model.pth")


if __name__ == '__main__':
    train()
