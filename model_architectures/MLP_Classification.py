import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import re
import time
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torchtext
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from collections import Counter

class MLP_Classification(nn.Module):
  def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
    super(MLP_Classification, self).__init__()
    self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
    self.fc1 = nn.Linear(embed_dim, hidden_dim)
    self.fc2 = nn.Linear(hidden_dim, int(hidden_dim / 2))
    self.fc3 = nn.Linear(int(hidden_dim / 2), num_class)
    self.relu = nn.ReLU()
    self.sigmoid = nn.Sigmoid()
    self.init_weights()

  def init_weights(self):
    initrange = 0.5
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.fc1.weight.data.uniform_(-initrange, initrange)
    self.fc1.bias.data.zero_()
    self.fc2.weight.data.uniform_(-initrange, initrange)
    self.fc2.bias.data.zero_()
    self.fc3.weight.data.uniform_(-initrange, initrange)
    self.fc3.bias.data.zero_()

  def forward(self, text, offsets):
    embedded = self.embedding(text, offsets)
    fc = self.relu(self.fc1(embedded))
    fc = self.relu(self.fc2(fc))
    fc = self.relu(self.fc3(fc))
    return self.sigmoid(fc)

def readData(filename):
  df = pd.read_csv(filename)
  df['list'] = df[df.columns[2:]].values.tolist()
  new_df = df[['comment_text', 'list']].copy()
  return new_df

def data_preprocess(text):
  # clean text
  text = re.sub("\'", "", text)
  text = re.sub("[^a-zA-Z]", " ", text)
  text = ' '.join(text.split())
  text = text.lower()
  # remove stopwords
  stop_words = set(stopwords.words('english'))
  no_stopword_text = [w for w in text.split() if not w in stop_words]
  return ' '.join(no_stopword_text)

def build_vocab(data, tokenizer):
  counter = Counter()
  for comment in data['comment_text']:
    counter.update(tokenizer(comment))
  return Vocab(counter, min_freq=1)
 
def collate_batch(batch):
  label_list, text_list, offsets = [], [], [0]
  for (text, label) in batch:
    label_list.append(label)
    text_list.append(torch.tensor(text, dtype=torch.int64))
    offsets.append(torch.tensor(text).size(0))
  label_list = torch.tensor(label_list, dtype=torch.int64)
  text_list = torch.cat(text_list)
  offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
  return text_list.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), \
         label_list.to(torch.device("cuda" if torch.cuda.is_available() else "cpu")), \
         offsets.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def train(model, dataloader, optimizer, criterion, epoch):
  model.train()
  total_acc, total_count = 0, 0
  log_interval = 500
  start_time = time.time()

  for idx, (text, label, offsets) in enumerate(dataloader):
    optimizer.zero_grad()
    predited_label = model(text, offsets)
    label = label.type_as(predited_label)
    loss = criterion(predited_label, label)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
    optimizer.step()
    for i in range(label.size(0)):
      if (predited_label.round()[i] == label[i]).sum() == 6:
        total_acc += 1.0
    total_count += label.size(0)
    if idx % log_interval == 0 and idx > 0:
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches '
            '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader), total_acc/total_count))
      total_acc, total_count = 0, 0
      start_time = time.time()

def evaluate(model, dataloader, criterion, test):
  model.eval()
  each_acc = [0] * 6
  total_acc, total_count = 0, 0
  total_predited, total_label = [], []
  with torch.no_grad():
    for idx, (text, label, offsets) in enumerate(dataloader):
      predited_label = model(text, offsets)
      label = label.type_as(predited_label)
      loss = criterion(predited_label, label)
      for i in range(label.size(0)):
        if (predited_label.round()[i] == label[i]).sum() == 6:
          total_acc += 1.0
        if (predited_label.round()[i][0] == label[i][0]):
          each_acc[0] = each_acc[0] + 1.0
        if (predited_label.round()[i][1] == label[i][1]):
          each_acc[1] = each_acc[1] + 1.0
        if (predited_label.round()[i][2] == label[i][2]):
          each_acc[2] = each_acc[2] + 1.0
        if (predited_label.round()[i][3] == label[i][3]):
          each_acc[3] = each_acc[3] + 1.0
        if (predited_label.round()[i][4] == label[i][4]):
          each_acc[4] = each_acc[4] + 1.0
        if (predited_label.round()[i][5] == label[i][5]):
          each_acc[5] = each_acc[5] + 1.0
      total_count += label.size(0)
      if test:
        if idx == 0:
          total_predited = predited_label.round()
          total_label = label
        else:
          total_predited = torch.cat((total_predited, predited_label.round()), 0)
          total_label = torch.cat((total_label, label), 0)
  total_acc = total_acc / total_count
  each_acc = [x / total_count for x in each_acc]
  return total_acc, each_acc, total_predited, total_label

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print('Reading Data...')
  path = os.getcwd()
  data = readData(os.path.join(path, 'data', 'train.csv'))
  print('Complete reading data')
  print()

  print('Data preprocessing...')
  #data = data[:10000]
  data['comment_text'] = data['comment_text'].apply(lambda x: data_preprocess(x))
  
  #tokenizer = get_tokenizer('basic_english')
  #vocab = build_vocab(data, tokenizer)

  print('Creating Dataset...')
  x_train, x_test, y_train, y_test = train_test_split(data['comment_text'], data['list'], test_size=0.2, random_state=9)

  x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, random_state=9)

  train_data = {'comment_text': x_train,
                'label': y_train}
  train_data = pd.DataFrame(train_data)
  valid_data = {'comment_text': x_valid,
                'label': y_valid}
  valid_data = pd.DataFrame(valid_data)
  test_data = {'comment_text': x_test,
               'label': y_test}
  test_data = pd.DataFrame(test_data)
  print('Complete preparing Dataset')
  print('Full Dataset: {}'.format(data.shape))
  print('Train Dataset: {}'.format(train_data.shape))
  print('Valid Dataset: {}'.format(valid_data.shape))
  print('Test Dataset: {}'.format(test_data.shape))
  
  tokenizer = get_tokenizer('basic_english')
  vocab = build_vocab(train_data, tokenizer)

  train_data["comment_text"] = train_data["comment_text"].apply(lambda x: [vocab[token] for token in tokenizer(x)])
  valid_data["comment_text"] = valid_data["comment_text"].apply(lambda x: [vocab[token] for token in tokenizer(x)])
  test_data["comment_text"] = test_data["comment_text"].apply(lambda x: [vocab[token] for token in tokenizer(x)])
  train_iter = iter(zip(train_data["comment_text"], train_data["label"]))
  valid_iter = iter(zip(valid_data["comment_text"], valid_data["label"]))
  test_iter = iter(zip(test_data["comment_text"], test_data["label"]))

  print('Set up Model...')
  EPOCHS = 100
  LR = 0.001
  BATCH_SIZE = 64
  num_class = 6
  vocab_size = len(vocab)
  embed_size = 128
  hidden_size = 256
  model = MLP_Classification(vocab_size, embed_size, hidden_size, num_class).to(device)

  criterion = torch.nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=LR)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
  total_accu = None
  

  train_dataloader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
  valid_dataloader = DataLoader(list(valid_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
  test_dataloader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, shuffle = True, collate_fn=collate_batch)
  print('Complete setup')
  print()

  print('Start training...\n')
  last = False
  for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(model, train_dataloader, optimizer, criterion, epoch)
    if epoch == EPOCHS:
      last = True
    accu_val, each_val, total_predited, total_label = evaluate(model, valid_dataloader, criterion, test=False)
    target_names = ['toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate']
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
      total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f}'.format(epoch, time.time()-epoch_start_time, accu_val))
    print('-' * 59)
  
  print('Checking the results of test dataset.')
  accu_test, each_val, total_predited, total_label = evaluate(model, test_dataloader, criterion, test=True)
  print('test accuracy {:8.3f}'.format(accu_test))
  print('| toxic: {:8.3f} | severe_toxic: {:8.3f} | obscene: {:8.3f}'.format(each_val[0], each_val[1], each_val[2]))
  print('| threat: {:8.3f} | insult: {:8.3f} | identity_hate: {:8.3f}'.format(each_val[3], each_val[4], each_val[5]))
  confusion_matrix = multilabel_confusion_matrix(total_label.cpu().numpy(), total_predited.cpu().numpy())
  print(confusion_matrix)
  print(classification_report(total_label.cpu().numpy(), total_predited.cpu().numpy(), target_names=target_names))

if __name__ == '__main__':  
  main()
