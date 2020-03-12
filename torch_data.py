import torch
import torch.utils.data as tud
import torchvision.transforms as T
from PIL import Image


class Torch_data_set(tud.Dataset):
    def __init__(self, data, label):
        super(Torch_data_set, self).__init__()
        self.data = data
        self.label = label

    def __len__(self):
        if len(self.data) == len(self.label):
            return len(self.data)
        else:
            raise ValueError("wrong input size")

    def __getitem__(self, index):
        # print(self.data[index].shape)
        # img = Image.fromarray(self.data[index], "L")
        # tensor_data = T.ToTensor()(img)
        # tensor_label = torch.from_numpy(self.label[index])
        tensor_data = torch.from_numpy(self.data)[index]
        tensor_label = torch.from_numpy(self.label)[index]

        return tensor_data.type(torch.float32), tensor_label.type(torch.int64)





