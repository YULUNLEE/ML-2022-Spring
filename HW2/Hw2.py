import os
import random
import pandas as pd
import torch
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import gc
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]

    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n)
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, train_val_seed=1337):
    class_num = 41 # NOTE: pre-computed, should not need change
    mode = 'train' if (split == 'train' or split == 'val') else 'test'

    label_dict = {}
    if mode != 'test':
      phone_file = open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines()

      for line in phone_file:
          line = line.strip('\n').split(' ')
          label_dict[line[0]] = [int(p) for p in line[1:]]

    if split == 'train' or split == 'val':
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.seed(train_val_seed)
        random.shuffle(usage_list)
        percent = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:percent] if split == 'train' else usage_list[percent:]
    elif split == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    X = torch.empty(max_len, 39 * concat_nframes)
    if mode != 'test':
      y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes)
        if mode != 'test':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode != 'test':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode != 'test':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode != 'test':
      print(y.shape)
      return X, y
    else:
      return X

class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        # self.fc = nn.Sequential(
        #     BasicBlock(input_dim, hidden_dim),
        #     *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
        #     nn.Linear(hidden_dim, output_dim)
        # )
        self.fc = nn.Sequential(nn.Linear(input_dim, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                nn.Linear(512,2048),
                                nn.ReLU(),
                                nn.BatchNorm1d(2048),
                                nn.Dropout(0.2),
                                nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.BatchNorm1d(512),
                                nn.Dropout(0.1),
                                nn.Linear(512,256),
                                nn.ReLU(),
                                nn.BatchNorm1d(256),
                                nn.Linear(256, 64),
                                nn.ReLU(),
                                nn.BatchNorm1d(64),
                                nn.Linear(64,output_dim))

    def forward(self, x):
        x = self.fc(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=39,
            hidden_size=512,         # rnn hidden unit
            num_layers=5,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=False,
            dropout = 0.1
        )

        self.classifier = nn.Sequential(
                                        nn.Linear(512, 256),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(256),
                                        nn.Linear(256,64),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(64),
                                        nn.Linear(64, 41),
                                        # nn.ReLU(),
                                        # nn.BatchNorm1d(64),
                                        # nn.Linear(64, 41),
                                        )
        # 初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.w_omega = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_dim, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        self.dropout =nn.Dropout(0.5)
    def attention_net(self, x, query, mask=None):  # 軟性注意力機制（key=value=x）
        d_k = query.size(-1)  # d_k為query的維度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分機制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  # 對最後一個維度歸一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 對權重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # h0 = torch.zeros(2*2, x.size(0), 1024)  # 同样考虑向前层和向后层
        # c0 = torch.zeros(2*2, x.size(0), 1024)
        # Attention_out, _ = self.attention(x, x, x) #multi-head attention
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        query = self.dropout(r_out)
        attn_output, attention = self.attention_net(r_out, query)  # 和LSTM的不同就在於這一句
        out=self.classifier(attn_output)
        return out
'''
class GRUNet(nn.Module):
    def __init__(self, input_dim=39, hidden_dim=512, output_dim=41, n_layers=3, drop_prob=0.1):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Sequential(nn.Linear(hidden_dim, 128),
                                nn.ReLU(),
                                nn.Linear(128, output_dim))
        self.dropout = nn.Dropout(0.2)

    # x,query：[batch, seq_len, hidden_dim*2]
    def attention_net(self, x, query, mask=None):  # 軟性注意力機制（key=value=x）
        d_k = query.size(-1)  # d_k為query的維度
        scores = torch.matmul(query, x.transpose(1, 2)) / math.sqrt(d_k)  # 打分機制  scores:[batch, seq_len, seq_len]
        p_attn = F.softmax(scores, dim=-1)  # 對最後一個維度歸一化得分
        context = torch.matmul(p_attn, x).sum(1)  # 對權重化的x求和，[batch, seq_len, hidden_dim*2]->[batch, hidden_dim*2]
        return context, p_attn


    def forward(self, x):
        out, h = self.gru(x, None)
        query = self.dropout(out)
        attn_output, attention = self.attention_net(out, query)  # 和LSTM的不同就在於這一句
        out = self.fc(attn_output)
        return out
'''



# data prarameters
concat_nframes = 41              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.9              # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 101                        # random seed
batch_size = 256               # batch size
num_epoch = 30                   # the number of training epoch
learning_rate = 0.0001          # learning rate
# model_path = './model.ckpt'     # the path where the checkpoint will be saved
model_path = 'D:/Hw2_model/'     # the path where the checkpoint will be saved
# load_model_path = "D:/Hw2_model/model_0.ckpt"
# model parameters
input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
hidden_layers = 3               # the number of hidden layers
hidden_dim = 512                # the hidden dim
print("input dim : ",input_dim)

# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio)

print("------------------- data Fc ------------------------")
print(train_X.shape)
print(train_y.shape)
print(val_X.shape)
print(val_y.shape)
#LSTM data preprocessinig
train_X_LSTM = train_X.view(train_X.size(0), concat_nframes, 39)
val_X_LSTM = val_X.view(val_X.size(0), concat_nframes, 39)

print("------------------- data LSTM ------------------------")
print(train_X_LSTM.shape)
print(train_y.shape)
print(val_X_LSTM.shape)
print(val_y.shape)


# get dataset
train_set = LibriDataset(train_X_LSTM, train_y)
val_set = LibriDataset(val_X_LSTM, val_y)

# remove raw feature to save memory
del train_X, train_y#, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {device}')

#fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# fix random seed
same_seeds(seed)

# create model, define a loss function, and optimizer
# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model = LSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.005)


# model.load_state_dict(torch.load(load_model_path))
# print(f"load model :{load_model_path}")
print(model)

best_acc = 0.0
for epoch in range(1, num_epoch):

    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train()  # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        for parameter in model.parameters():
            penalty = torch.sum(parameter**2)
        loss = criterion(outputs, labels)+0.001*penalty
        loss.backward()
        optimizer.step()

        _, train_pred = torch.max(outputs, 1)  # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
            ))
    torch.save(model.state_dict(), model_path+f"model_{epoch}.ckpt")
    # validation
    if len(val_set) > 0:
        model.eval()  # set the model to evaluation mode
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader)):
                features, labels = batch
                features = features.to(device)
                labels = labels.to(device)
                outputs = model(features)

                loss = criterion(outputs, labels)

                _, val_pred = torch.max(outputs, 1)
                val_acc += (
                            val_pred.cpu() == labels.cpu()).sum().item()  # get the index of the class with the highest probability
                val_loss += loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader),
                val_acc / len(val_set), val_loss / len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(),  model_path+f"model_{epoch}.ckpt")
                print('saving model with acc {:.3f}'.format(best_acc / len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc / len(train_set), train_loss / len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path+"model_last.ckpt")
    print('saving model at last epoch')

'''
# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes)
test_X_LSTM = test_X.view(test_X.size(0), concat_nframes, 39)
test_set = LibriDataset(test_X_LSTM, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
# model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
model = LSTM().to(device)
model.load_state_dict(torch.load(model_path))

test_acc = 0.0
test_lengths = 0
pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)

with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))
'''