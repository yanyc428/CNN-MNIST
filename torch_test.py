import torch
from dataPreprocessor import *
import torch.utils.data as tud
from torch_data import Torch_data_set
from torch_train import Net


net = Net()
net = torch.load("torch_model.pth")

BATCH_SIZE = 10

test_data = raw_file_idx3_process( "data/test/t10k-images-idx3-ubyte")

test_label = raw_file_idx1_process("data/test/t10k-labels-idx1-ubyte")

test_set = Torch_data_set(test_data, test_label)
test_loader = tud.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

total = 0
bingo = 0

print("\n")
with torch.no_grad():
    for index, data in enumerate(test_loader):
        inputs, label = data
        inputs, label = inputs.to(device), label.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data,1)
        # 对total和bingo进行修改
        total += label.size(0)
        bingo += (predicted == label).sum().item()
        # 显示第128号的图像
        if index == 117:
            print(predicted.cpu().numpy())
            inputs = [inp.reshape(28, 28) for inp in inputs.cpu()]
            show_image(inputs, 2, 5)
        # 打印正确率
        print("\rAccuracy: {:.2f}".format(bingo/total), end='')

