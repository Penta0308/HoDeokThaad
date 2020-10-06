from PIL import Image, ImageFont, ImageDraw
import torch
import numpy
from sklearn import model_selection
import sqlite3
from pprint import pprint

from torch.utils.data import TensorDataset, DataLoader

import modules

fnt = ImageFont.truetype("D2Coding-Ver1.3.2-20180524.ttf", 16, encoding="UTF-8")


def normalize(t: str):
    im = Image.new("1", (384, 16), (0,))  # White
    dr = ImageDraw.Draw(im)
    dr.text((0, 0), t, font=fnt, fill=(1,))
    return numpy.asarray(numpy.float32(im.split()[0]))


def loadtraindata():
    _data_x = []
    _data_y = []

    for fd in modules.config_get("aidb"):
        with open(fd["file"], 'r', encoding="UTF-8") as f:
            line = f.readline()
            while line:
                _data_x.append(normalize(line))
                _data_y.append(fd["value"])
                line = f.readline()
    return _data_x, _data_y


data, label = loadtraindata()
data = numpy.array(data, dtype='float32')
label = numpy.array(label, dtype='int8')

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.1)

train_X = torch.from_numpy(train_X).float()
train_Y = torch.from_numpy(train_Y).byte()

test_X = torch.from_numpy(test_X).float()
test_Y = torch.from_numpy(test_Y).byte()

train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=32, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)

        self.fc1 = torch.nn.Linear(20 * 2 * 93, 50)  # 2=(((((16-5)+1)/2)-5)+1)/2, 93=(((((384-5)+1)/2)-5)+1)/2
        self.fc2 = torch.nn.Linear(50, 2)

    def forward(self, x):
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv1(x)), 2)
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = x.view(-1, 20 * 2 * 93)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.nn.functional.log_softmax(x, dim=1)


model = Net()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = torch.autograd.Variable(train_x), torch.autograd.Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        print(output)
        print(train_x)
        print(train_y)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
    if (epoch + 1) % 10 == 0:
        print(epoch + 1, total_loss)

test_x, test_y = torch.autograd.Variable(test_X), torch.autograd.Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
accuracy = sum(test_y.data.numpy() == result.numpy()) / len(test_y.data.numpy())
print(accuracy)
