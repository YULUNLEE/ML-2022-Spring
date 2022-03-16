# Numerical Operations
import math
import numpy as np

# Reading/Writing Data
import pandas as pd
import os
import csv
import random
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression, f_regression
# For Progress Bar
from tqdm import tqdm

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# For plotting learning curve
# from torch.utils.tensorboard import SummaryWriter



def same_seed(seed):
    '''Fixes random number generator seeds for reproducibility.'''
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

def predict(test_loader, model, device):
    model.eval() # Set your model to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds

class COVID19Dataset(Dataset):
    '''
    x: Features.
    y: Targets, if none, do prediction.
    '''
    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.FloatTensor(y)
        self.x = torch.FloatTensor(x)

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        else:
            return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class My_Model(nn.Module):
    def __init__(self, input_dim):
        super(My_Model, self).__init__()
        # TODO: modify model's structure, be aware of dimensions.
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
        )

        #LSTM
        # self.layers = nn.Sequential(
        #     nn.LSTM(
        #     input_size=81,
        #     hidden_size=64,
        #     num_layers=1,
        #     bidirectional=True,
        #     batch_first=True,
        # ),
        # nn.Linear(64, 1)
        #
        # )

    def forward(self, x):
        x = self.layers(x)
        x = x.squeeze(1) # (B, 1) -> (B)
        return x


def select_feat(train_data, valid_data, test_data, select_all=True):
    '''Selects useful features to perform regression'''
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(1, raw_x_train.shape[1]))
    else:
        # feat_idx =  list(range(37, 117))  # TODO: Select suitable feature columns.
        # # feat_idx.remove(50)
        # # feat_idx.remove(51)
        # feat_idx.remove(52)
        # # feat_idx.remove(66)
        # # feat_idx.remove(67)
        # feat_idx.remove(68)
        # # feat_idx.remove(82)
        # # feat_idx.remove(83)
        # feat_idx.remove(84)
        # # feat_idx.remove(98)
        # # feat_idx.remove(99)
        # feat_idx.remove(100)
        # # feat_idx.remove(114)
        # # feat_idx.remove(115)
        # feat_idx.remove(116)
        # # for i in range(58,70):
        # #     feat_idx.append(i)
        # # for i in range(74,86):
        # #     feat_idx.append(i)
        # # for i in range(90,102):
        # #     feat_idx.append(i)
        # # for i in range(102,117):
        # #     feat_idx.append(i)
        # feat_idx = [38,39,40,41,42,43,44,53,54,55,56,57,58,59,60,69,70,71,72,73,74,75,76,85,86,87,88,89,90,91,92,101,102,103,104,105,106,107,108]
        # feat_idx = [38, 39, 40, 41, 53, 54, 55, 56, 57, 69, 70, 71, 72, 73, 85, 86,
        #             87, 88, 89, 101, 102, 103, 104, 105]
        feat_idx = [38,39,40,41,42,46,47,48,53,54,55,56,57,58,62,63,64,69,70,71,72,73,74,78,79,80,85,86,87,88,89,90,94,95,96,101,102,103,104,105,106,110,111,112]
        # feat_idx = [38,39,40,41,44,53,54,55,56,57,60,69,70,71,72,73,76,81,85,86,87,88,89,92,101,102,103,104,105,108,113,114]
        for i in range(1,38):
            feat_idx.append(i)

        print(feat_idx)

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


def trainer(train_loader, valid_loader, model, config, device):
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.
    # criterion = nn.SmoothL1Loss()
    # Define your optimization algorithm.
    # TODO: Please check https://pytorch.org/docs/stable/optim.html to get more available algorithms.
    # TODO: L2 regularization (optimizer(weight decay...) or implement by your self).
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], amsgrad=True, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

    # writer = SummaryWriter()  # Writer of tensoboard.

    if not os.path.isdir('./models'):
        os.mkdir('./models')  # Create directory of saving models.

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []

        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            for param in model.parameters():
                regulation_loss = torch.sum(torch.abs(param)**2)
            loss = criterion(pred, y)+ 0.001*regulation_loss
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        # writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        # writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            try:
                os.remove(f"./models/model_224_{best_loss}.ckpt")
            except:
                pass
            best_loss = mean_valid_loss
            # torch.save(model.state_dict(), config['save_path'])  # Save your best model
            torch.save(model.state_dict(), f"./models/model_224_{best_loss}.ckpt")  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
        scheduler.step()
    torch.save(model.state_dict(), f"./models/model_last.ckpt")  # Save your best model
if __name__=='__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'seed': 101,      # Your seed number, you can pick your lucky number. :)
        'select_all': False,   # Whether to use all features.
        'valid_ratio': 0.1,   # validation_size = train_size * valid_ratio
        'n_epochs': 5000,     # Number of epochs.
        'batch_size': 256,
        'learning_rate': 1e-4,
        'early_stop': 600,    # If model has not improved for this many consecutive epochs, stop training.
        'save_path': f'./models/model_222_81_0.586010780185461.ckpt'  # Your model will be saved here.
    }

    # Set seed for reproducibility
    # for i in range(100000):
    #     config['seed'] = i
    same_seed(config['seed'])


    # train_data size: 2699 x 118 (id + 37 states + 16 features x 5 days)
    # test_data size: 1078 x 117 (without last day's positive rate)
    train_data, test_data = pd.read_csv('./covid.train.csv').values, pd.read_csv('./covid.test.csv').values
    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    # Print out the data size.
    print(f"""train_data size: {train_data.shape} 
    valid_data size: {valid_data.shape} 
    test_data size: {test_data.shape}""")

    # Select features
    x_train_list_mod = []
    x_valid_list_mod = []
    x_test_list_mod =[]
    np.set_printoptions(threshold = np.inf)
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])
    print(x_train.shape)
    # # 只選得分最好的k＝2個features
    # x_new = SelectKBest(score_func=mutual_info_regression, k = 50).fit_transform(x_train, y_train)
    # print(x_new[0])

    x_train_list = x_train.tolist()
    x_valid_list = x_valid.tolist()
    x_test_list = x_test.tolist()
    print(type(x_train_list))
    # Select features
    #train
    # for i in range(len(x_train_list)):
    #     for j in range(37):
    #         if x_train_list[i][j] == 1.0:
    #             x_train_list[i].append(j)
    # for i in range(len(x_train_list)):
    #     x_train_list_mod.append(x_train_list[i][37:])
    # x_train_list_mod_array = np.array(x_train_list_mod)
    # print(x_train_list_mod_array)
    #
    # # val
    # for i in range(len(x_valid_list)):
    #     for j in range(37):
    #         if x_valid_list[i][j] == 1.0:
    #             x_valid_list[i].append(j)
    # for i in range(len(x_valid_list)):
    #     x_valid_list_mod.append(x_valid_list[i][37:])
    # x_valid_list_mod_array = np.array(x_valid_list_mod)
    # print(x_valid_list_mod_array.shape)
    #
    # # test
    # for i in range(len(x_test_list)):
    #     for j in range(37):
    #         if x_test_list[i][j] == 1.0:
    #             x_test_list[i].append(j)
    # for i in range(len(x_test_list)):
    #     x_test_list_mod.append(x_test_list[i][37:])
    # x_test_list_mod_array = np.array(x_test_list_mod)
    # print(x_test_list_mod_array.shape)

    #std
    # means = np.mean(x_train, axis=0)
    # stds = np.std(x_train, axis=0)
    # normalized_x_train = (x_train - means) / stds
    #
    # means1 = np.mean(x_valid, axis=0)
    # stds1 = np.std(x_valid, axis=0)
    # normalized_x_val = (x_valid - means1) / stds1
    #
    # means2 = np.mean(x_test, axis=0)
    # stds2 = np.std(x_test, axis=0)
    # normalized_x_test = (x_test - means2) / stds2


    # Print out the number of features.
    print(f'number of features: {x_train.shape[1]}')

    train_dataset, valid_dataset, test_dataset = COVID19Dataset(x_train, y_train), \
                                                COVID19Dataset(x_valid, y_valid), \
                                                COVID19Dataset(x_test)

    # Pytorch data loader loads pytorch dataset into batches.
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    #train

    # model = My_Model(input_dim=x_train.shape[1]).to(device)  # put your model and data on the same computation device.
    # model.load_state_dict(torch.load(config['save_path']))
    # trainer(train_loader, valid_loader, model, config, device)


def save_pred(preds, file):
    ''' Save predictions to specified file '''
    with open(file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])
#
model = My_Model(input_dim=x_train.shape[1]).to(device)
model.load_state_dict(torch.load(config['save_path']))
preds = predict(test_loader, model, device)
save_pred(preds, 'pred.csv')