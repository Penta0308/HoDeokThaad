from urllib import parse

import flask
from flask import request

import modules

import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np
from konlpy.tag import Kkma

kkma = Kkma()

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FILTER_SIZES = [2, 3, 4, 5, 6, 7]


def tokenizer(text):
    token = [t for t in kkma.morphs(text)]
    if len(token) < max(FILTER_SIZES):
        for i in range(0, max(FILTER_SIZES) - len(token)):
            token.append('<PAD>')
    return token


TEXT = data.Field(tokenize=tokenizer, batch_first=True)
LABEL = data.LabelField(dtype=torch.float)

raw_data = data.TabularDataset(path='aidb.csv', format='csv',
                               fields={'text': ('text', TEXT), 'label': ('label', LABEL)})

train_data, test_data, valid_data = raw_data.split(split_ratio=[0.8, 0.1, 0.1], stratified=False, strata_field='label',
                                                   random_state=random.seed(SEED))

MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors='fasttext.simple.300d',
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)

BATCH_SIZE = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=device)

import torch.nn as nn
import torch.nn.functional as F


def print_shape(name, data):
    print(f'{name} has shape {data.shape}')


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([nn.Conv2d(in_channels=1,
                                              out_channels=n_filters,
                                              kernel_size=(fs, embedding_dim))
                                    for fs in filter_sizes])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # print_shape('text', text)
        # text = [batch_size, sent_len]

        embedded = self.embedding(text)
        # print_shape('embedded', embedded)
        # embedded = [batch_size, sent_len, emb_dim]

        embedded = embedded.unsqueeze(1)
        # print_shape('embedded', embedded)
        # embedded = [batch_size, 1, sent_len, emb_dim]

        # print_shape('self.convs[0](embedded)', self.convs[0](embedded))
        # self.convs[0](embedded) = [batch_size, n_filters, sent_len-filter_sizes[n]+1, 1 ]
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]

        # print_shape('F.max_pool1d(conved[0], conved[0].shape[2])', F.max_pool1d(conved[0], conved[0].shape[2]))
        # F.max_pool1d(conved[0], conved[0].shape[2]) = [batch_size, n_filters, 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # print_shape('cat', cat)
        # cat = [batch_size, n_filters * len(filter_size)]

        res = self.fc(cat)
        # print_shape('res', res)
        # res = [batch_size, output_dim]

        return self.fc(cat)


INPUT_DIM = len(TEXT.vocab)
EMBEDDING_DIM = 300
N_FILTERS = 100
OUTPUT_DIM = 1
DROPOUT = 0.5
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]

model = CNN(INPUT_DIM, EMBEDDING_DIM, N_FILTERS, FILTER_SIZES, OUTPUT_DIM, DROPOUT, PAD_IDX)

model = model.to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'모델의 파라미터 수는 {count_parameters(model):,} 개 입니다.')

pretrained_weight = TEXT.vocab.vectors
print(pretrained_weight.shape, model.embedding.weight.data.shape)

model.embedding.weight.data.copy_(pretrained_weight)

UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

import torch.optim as optim

optimizer = optim.Adam(model.parameters())

criterion = nn.BCEWithLogitsLoss()

model = model.to(device)
criterion = criterion.to(device)


def binary_accuracy(preds, y):
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        acc = binary_accuracy(predictions, batch.label)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


import time


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = modules.config_get("aicount")
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    start_time = time.time()

    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'hodeokthaad-model.pt')

    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

webserver = flask.Flask("HoDeokThaad")
webserver.config['JSON_AS_ASCII'] = False

@webserver.route("/")
def flask_eval():
    resp = {}
    try:
        t = tokenizer(parse.unquote(request.args['t'], encoding='UTF-8'))
    except KeyError:
        resp['code'] = 400
        return flask.jsonify(resp)

    if len(t) == 0:
        resp['code'] = 500
        return flask.jsonify(resp)

    eval_data = data.Dataset(
        [data.Example.fromlist([t, "0"], fields=[('text', TEXT), ('label', LABEL)]), ],
        fields={'text': TEXT, 'label': LABEL})

    eval_iterator = data.BucketIterator.splits(
        (eval_data,),
        batch_size=1,
        sort_key=lambda x: len(x.text),
        sort_within_batch=True,
        device=device)[0]

    resp['code'] = 200
    resp['data'] = {'token': t}

    with torch.no_grad():
        for batch in eval_iterator:
            resp['data']['rate'] = float(torch.sigmoid(model(batch.text).squeeze(1)[0])) * -1.0
            return flask.jsonify(resp)


webserver.run(host="0.0.0.0", port=modules.config_get("port"))
