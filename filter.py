import pandas as pandas
from PIL import Image, ImageFont, ImageDraw
import torch
import numpy
from sklearn import model_selection
import hgtk
import flask
from urllib import parse

from torch.utils.data import TensorDataset, DataLoader

import modules


iscuda = torch.cuda.is_available()

fnt = ImageFont.truetype("D2Coding-Ver1.3.2-20180524.ttf", 32, encoding="UTF-8")

webserver = flask.Flask("HoDeokThaad")

def normalize(t: str):
    u = []
    for i in t:
        if hgtk.checker.is_hangul(i):
            i = hgtk.letter.decompose(i)
        u += i
    t = ''.join(u)
    print(t)
    im = Image.new("L", (768, 32), (0,))  # White
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
                _data_x.append(numpy.asarray([normalize(line),]) / 255)
                _data_y.append(fd["value"])
                line = f.readline()
    return _data_x, _data_y

data, label = loadtraindata()

print(pandas.DataFrame(data[0][0]).shape)

data = numpy.array(data, dtype='float32')
label = numpy.array(label, dtype='int8')

train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data, label, test_size=0.1)

train_X = torch.from_numpy(train_X)
train_Y = torch.from_numpy(train_Y)
test_X = torch.from_numpy(test_X)
test_Y = torch.from_numpy(test_Y)

if iscuda:
    train_X = train_X.cuda()
    train_Y = train_Y.cuda()
    test_X = test_X.cuda()
    test_Y = test_Y.cuda()

train_X = train_X.float()
train_Y = train_Y.long()
test_X = test_X.float()
test_Y = test_Y.long()

train = TensorDataset(train_X, train_Y)
train_loader = DataLoader(train, batch_size=16, shuffle=True)


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        conv1 = torch.nn.Conv2d(1, 1024, (32, 64), (1, 2))  # 4@25*2
        # activation ReLU
        conv2 = torch.nn.Conv2d(1024, 8192, (2, 6), (1, 1))  # 16@20*1
        # activation ReLU
        conv3 = torch.nn.Conv2d(8192, 16384, (1, 6), (1, 1))  # 16@15*1

        self.conv_module = torch.nn.Sequential(
            conv1,
            torch.nn.ReLU(),
            conv2,
            torch.nn.ReLU(),
            conv3
        )

        fc1 = torch.nn.Linear(16384 * 15 * 1, 1200)
        # activation ReLU
        fc2 = torch.nn.Linear(1200, 400)
        # activation ReLU
        fc3 = torch.nn.Linear(400, 80)
        # activation ReLU
        fc4 = torch.nn.Linear(10, 2)

        self.fc_module = torch.nn.Sequential(
            fc1,
            torch.nn.ReLU(),
            fc2,
            torch.nn.ReLU(),
            fc3,
            torch.nn.ReLU(),
            fc4
        )

        if iscuda:
            self.conv_module = self.conv_module.cuda()
            self.fc_module = self.fc_module.cuda()

    def forward(self, x):
        out = self.conv_module(x)  # 16@4*4
        # make linear
        dim = 1
        for d in out.size()[1:]:
            dim = dim * d
        out = out.view(-1, dim)
        out = self.fc_module(out)
        return torch.nn.functional.softmax(out, dim=1)


model = CNNClassifier()

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(modules.config_get("aicount")):
    total_loss = 0
    for train_x, train_y in train_loader:
        train_x, train_y = torch.autograd.Variable(train_x), torch.autograd.Variable(train_y)
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.data.item()
    print(epoch + 1, total_loss)

test_x, test_y = torch.autograd.Variable(test_X), torch.autograd.Variable(test_Y)
result = torch.max(model(test_x).data, 1)[1]
tyd = test_y.data
txd = test_x.data
if iscuda:
    tyd = tyd.cpu()
    txd = txd.cpu()
    result = result.cpu()
accuracy = sum(tyd.numpy() == result.numpy()) / len(txd.numpy())
print(accuracy)

@webserver.route("/evaluate/<t>")
def flask_eval(t):
    t = parse.unquote(t)
    r = normalize(t)
    r = torch.from_numpy(r).float()
    r = r.unsqueeze(0)
    r = r.unsqueeze(0)
    if iscuda:
        r = r.cuda()
        return str(model(r).data.cpu())
    else:
        return str(model(r).data)

webserver.run(host="0.0.0.0", port=modules.config_get("port"))
