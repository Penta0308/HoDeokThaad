import modules

import os
import shutil
import torch
from pytorch_transformers import BertForSequenceClassification, BertConfig
from torch.optim import *
import torch.nn.functional as F
from torchtext import data
import random
import pickle


# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'

TEXT = data.Field(tokenize=modules.tokenizer, batch_first=True)
LABEL = data.LabelField(dtype=torch.long)

raw_data = data.TabularDataset(path='aidb.csv', format='csv',
                               fields={'text': ('text', TEXT), 'label': ('label', LABEL)})

train_data, test_data = raw_data.split(split_ratio=[0.9, 0.1], stratified=False, strata_field='label',
                                       random_state=random.seed(modules.SEED))

MAX_VOCAB_SIZE = 16384
BATCH_SIZE = 64

TEXT.build_vocab(train_data,
                 max_size=MAX_VOCAB_SIZE,
                 vectors='fasttext.simple.300d',
                 unk_init=torch.Tensor.normal_)

LABEL.build_vocab(train_data)

train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=BATCH_SIZE,
    sort_key=lambda x: len(x.text),
    sort_within_batch=True,
    device=DEVICE)

model = BertForSequenceClassification(
    BertConfig(
        vocab_size=MAX_VOCAB_SIZE,
        max_position_embeddings=512,
        intermediate_size=1024,
        hidden_size=512,
        num_attention_heads=8,
        num_hidden_layers=6,
        type_vocab_size=5,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_labels=2
    )
)
model.to(DEVICE)

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

model.bert.embeddings.word_embeddings.weight.data[PAD_IDX] = torch.zeros(512)
model.bert.embeddings.word_embeddings.weight.data[UNK_IDX] = torch.zeros(512)


print(f'Parameter: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

optimizer = RMSprop(model.parameters(), lr=0.000005)

itr = 1
epochs = 20
total_loss = 0
total_len = 0
total_correct = 0

for epoch in range(epochs):
    model.train()
    print(f"Epoch {epoch + 1}/{epochs}")
    for batch in train_iterator:
        optimizer.zero_grad()

        outputs = model(batch.text, labels=batch.label)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        correct = pred.eq(batch.label)
        total_correct += correct.sum().item()
        total_len += len(batch.label)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"  Train Accuracy: {total_correct / total_len:.3f}, Loss: {total_loss / total_len:.4f}")
    total_loss = 0
    total_len = 0
    total_correct = 0

    model.eval()
    for batch in test_iterator:
        outputs = model(batch.text, labels=batch.label)
        loss, logits = outputs

        pred = torch.argmax(F.softmax(logits, dim=1), dim=1)
        correct = pred.eq(batch.label)
        total_correct += correct.sum().item()
        total_len += len(batch.label)

    print(f"  Test Accuracy: {total_correct / total_len:.3f}")
    total_len = 0
    total_correct = 0

    itr += 1

shutil.rmtree("data/model/", ignore_errors=True)
os.mkdir("data/model/")
model.save_pretrained("data/model/")

output = open("data/TEXT.vocab", 'wb')
pickle.dump(TEXT.vocab, output)
output.close()
output = open("data/LABEL.vocab", 'wb')
pickle.dump(LABEL.vocab, output)
output.close()
