import matplotlib.pyplot as plt

def no_axis_show(img, title='', cmap=None):
  # imshow, and set the interpolation mode to be "nearest"。
  fig = plt.imshow(img, interpolation='nearest', cmap=cmap)
  # do not show the axes in the images.
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.title(title)

titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
plt.figure(figsize=(18, 18))
for i in range(10):
  plt.subplot(1, 10, i+1)
  fig = no_axis_show(plt.imread(f'real_or_drawing/train_data/{i}/{500*i}.bmp'), title=titles[i])
plt.show()

plt.figure(figsize=(18, 18))
for i in range(10):
  plt.subplot(1, 10, i+1)
  fig = no_axis_show(plt.imread(f'real_or_drawing/test_data/0/' + str(i).rjust(5, '0') + '.bmp'))
plt.show()

import cv2
import matplotlib.pyplot as plt
titles = ['horse', 'bed', 'clock', 'apple', 'cat', 'plane', 'television', 'dog', 'dolphin', 'spider']
plt.figure(figsize=(18, 18))

original_img = plt.imread(f'real_or_drawing/train_data/0/0.bmp')
plt.subplot(1, 5, 1)
no_axis_show(original_img, title='original')

gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title='gray scale', cmap='gray')

gray_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)
plt.subplot(1, 5, 2)
no_axis_show(gray_img, title='gray scale', cmap='gray')

canny_50100 = cv2.Canny(gray_img, 50, 100)
plt.subplot(1, 5, 3)
no_axis_show(canny_50100, title='Canny(50, 100)', cmap='gray')

canny_150200 = cv2.Canny(gray_img, 150, 200)
plt.subplot(1, 5, 4)
no_axis_show(canny_150200, title='Canny(150, 200)', cmap='gray')

canny_250300 = cv2.Canny(gray_img, 250, 300)
plt.subplot(1, 5, 5)
no_axis_show(canny_250300, title='Canny(250, 300)', cmap='gray')

plt.show()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

source_transform = transforms.Compose([
  # Turn RGB to grayscale. (Bacause Canny do not support RGB images.)
  transforms.Grayscale(),
  # cv2 do not support skimage.Image, so we transform it to np.array,
  # and then adopt cv2.Canny algorithm.
  transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
  # Transform np.array back to the skimage.Image.
  transforms.ToPILImage(),
  # 50% Horizontal Flip. (For Augmentation)
  transforms.RandomHorizontalFlip(),
  # Rotate +- 15 degrees. (For Augmentation), and filled with zero
  # if there's empty pixel after rotation.
  transforms.RandomRotation(15, fill=(0,)),
  # Transform to tensor for model inputs.
  transforms.ToTensor(),
])
target_transform = transforms.Compose([
  # Turn RGB to grayscale.
  transforms.Grayscale(),
  # Resize: size of source data is 32x32, thus we need to
  #  enlarge the size of target data from 28x28 to 32x32。
  transforms.Resize((32, 32)),
  # 50% Horizontal Flip. (For Augmentation)
  transforms.RandomHorizontalFlip(),
  # Rotate +- 15 degrees. (For Augmentation), and filled with zero
  # if there's empty pixel after rotation.
  transforms.RandomRotation(15, fill=(0,)),
  # Transform to tensor for model inputs.
  transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=64, shuffle=False)


# class ResBlock(nn.Module):
#   def __init__(self, inchannel, outchannel, stride=1):
#     super(ResBlock, self).__init__()
#     # 这里定义了残差块内连续的2个卷积层
#     self.left = nn.Sequential(
#       nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
#       nn.BatchNorm2d(outchannel),
#       nn.ReLU(inplace=True),
#       nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
#       nn.BatchNorm2d(outchannel)
#     )
#     self.shortcut = nn.Sequential()
#     if stride != 1 or inchannel != outchannel:
#       # shortcut，这里为了跟2个卷积层的结果结构一致，要做处理
#       self.shortcut = nn.Sequential(
#         nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
#         nn.BatchNorm2d(outchannel)
#       )
#
#   def forward(self, x):
#     out = self.left(x)
#     # 将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
#     out = out + self.shortcut(x)
#     out = F.relu(out)
#
#     return out
#
#
# class ResNet(nn.Module):
#   def __init__(self, ResBlock):
#     super(ResNet, self).__init__()
#     self.inchannel = 64
#     self.conv1 = nn.Sequential(
#       nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
#       nn.BatchNorm2d(64),
#       nn.ReLU()
#     )
#     self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
#     self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
#     self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
#     self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
#
#   # 这个函数主要是用来，重复同一个残差块
#   def make_layer(self, block, channels, num_blocks, stride):
#     strides = [stride] + [1] * (num_blocks - 1)
#     layers = []
#     for stride in strides:
#       layers.append(block(self.inchannel, channels, stride))
#       self.inchannel = channels
#     return nn.Sequential(*layers)
#
#   def forward(self, x):
#     # 在这里，整个ResNet18的结构就很清晰了
#     out = self.conv1(x)
#     out = self.layer1(out)
#     out = self.layer2(out)
#     out = self.layer3(out)
#     out = self.layer4(out)
#     out = F.avg_pool2d(out, 4)
#     out = out.squeeze()
#     return out
#
#
# def ResNet18():
#   return ResNet(ResBlock)


class Residual_Network(nn.Module):
  def __init__(self):
    super(Residual_Network, self).__init__()

    self.cnn_layer1 = nn.Sequential(
      nn.Conv2d(1, 64, 3, 1, 1),
      nn.BatchNorm2d(64),
    )

    self.cnn_layer2 = nn.Sequential(
      nn.Conv2d(64, 64, 3, 1, 1),
      nn.BatchNorm2d(64),
    )

    self.cnn_layer3 = nn.Sequential(
      nn.Conv2d(64, 128, 3, 2, 1),
      nn.BatchNorm2d(128),
    )

    self.cnn_layer4 = nn.Sequential(
      nn.Conv2d(128, 128, 3, 1, 1),
      nn.BatchNorm2d(128),
    )
    self.cnn_layer5 = nn.Sequential(
      nn.Conv2d(128, 256, 3, 2, 1),
      nn.BatchNorm2d(256),
    )
    self.cnn_layer6 = nn.Sequential(
      nn.Conv2d(256, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
    )
    self.relu = nn.ReLU()
    self.pooling = nn.MaxPool2d(2)

  def forward(self, x):
    # input (x): [batch_size, 3, 128, 128]
    # output: [batch_size, 11]

    # Extract features by convolutional layers.
    x1 = self.cnn_layer1(x)
    x1 = self.relu(x1)


    x2 = self.cnn_layer2(x1)
    x2 = torch.add(x1, x2)
    x2 = self.relu(x2)


    x3 = self.cnn_layer3(x2)
    x3 = self.relu(x3)
    x3 = self.pooling(x3)

    x4 = self.cnn_layer4(x3)
    x4 = torch.add(x3, x4)
    x4 = self.relu(x4)
    x4 = self.pooling(x4)

    x5 = self.cnn_layer5(x4)
    x5 = self.relu(x5)
    x5 = self.pooling(x5)

    x6 = self.cnn_layer6(x5)
    x6 = torch.add(x5, x6)
    x6 = self.relu(x6)

    # print(x6.shape)
    # The extracted feature map must be flatten before going to fully-connected layers.
    xout = x6.flatten(1)

    return xout



class FeatureExtractor(nn.Module):

  def __init__(self):
    super(FeatureExtractor, self).__init__()

    self.conv = nn.Sequential(
      nn.Conv2d(1, 64, 3, 1, 1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(64, 128, 3, 1, 1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(128, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(256, 256, 3, 1, 1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(2),

      nn.Conv2d(256, 512, 3, 1, 1),
      nn.BatchNorm2d(512),
      nn.ReLU(),
      nn.MaxPool2d(2)
    )

  def forward(self, x):
    x = self.conv(x).squeeze()
    return x


class LabelPredictor(nn.Module):

  def __init__(self):
    super(LabelPredictor, self).__init__()

    self.layer = nn.Sequential(
      nn.Linear(512, 512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.ReLU(),

      nn.Linear(512, 10),
    )

  def forward(self, h):
    c = self.layer(h)
    return c


class DomainClassifier(nn.Module):

  def __init__(self):
    super(DomainClassifier, self).__init__()

    self.layer = nn.Sequential(
      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),

      nn.Linear(512, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(),

      nn.Linear(512, 1),
    )

  def forward(self, h):
    y = self.layer(h)
    return y


feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

import math

num_epoch = 2000
train_step = 0
total_steps = len(source_dataloader) * num_epoch
print("total steps : ", total_steps)


def get_Lambda(train_step):
  global total_steps
  Lambda = (2 / (1 + math.exp(-10 * (train_step / total_steps)))) - 1
  return Lambda


def adjust_learning_rate(optimizer, train_step, total_steps):
  lr = 0.01 / (1 + 10 * (train_step / total_steps)) ** 0.75
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
  return lr


def train_epoch(source_dataloader, target_dataloader):
  '''
    Args:
      source_dataloader: source data的dataloader
      target_dataloader: target data的dataloader
      lamb: control the balance of domain adaptatoin and classification.
  '''

  # D loss: Domain Classifier的loss
  # F loss: Feature Extrator & Label Predictor的loss
  running_D_loss, running_F_loss = 0.0, 0.0
  total_hit, total_num = 0.0, 0.0
  global train_step
  for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):
    lr_F = adjust_learning_rate(optimizer_F, train_step, total_steps)
    lr_C = adjust_learning_rate(optimizer_C, train_step, total_steps)
    lr_D = adjust_learning_rate(optimizer_D, train_step, total_steps)

    source_data = source_data.cuda()
    source_label = source_label.cuda()
    target_data = target_data.cuda()

    # Mixed the source data and target data, or it'll mislead the running params
    #   of batch_norm. (runnning mean/var of soucre and target data are different.)
    mixed_data = torch.cat([source_data, target_data], dim=0)
    domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
    # set domain label of source data to be 1.
    domain_label[:source_data.shape[0]] = 1

    # Step 1 : train domain classifier
    feature = feature_extractor(mixed_data)
    # We don't need to train feature extractor in step 1.
    # Thus we detach the feature neuron to avoid backpropgation.
    domain_logits = domain_classifier(feature.detach())
    loss = domain_criterion(domain_logits, domain_label)
    running_D_loss += loss.item()
    loss.backward()
    optimizer_D.step()

    # Step 2 : train feature extractor and label classifier
    class_logits = label_predictor(feature[:source_data.shape[0]])
    domain_logits = domain_classifier(feature)
    # loss = cross entropy of classification - lamb * domain binary cross entropy.
    #  The reason why using subtraction is similar to generator loss in disciminator of GAN
    loss = class_criterion(class_logits, source_label) - get_Lambda(train_step) * domain_criterion(domain_logits,
                                                                                                   domain_label)
    running_F_loss += loss.item()
    loss.backward()
    optimizer_F.step()
    optimizer_C.step()

    optimizer_D.zero_grad()
    optimizer_F.zero_grad()
    optimizer_C.zero_grad()

    total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
    total_num += source_data.shape[0]
    print(i, end='\r')
    train_step += 1

  return running_D_loss / (i + 1), running_F_loss / (i + 1), total_hit / total_num, get_Lambda(train_step), lr_F





# feature_extractor.load_state_dict(torch.load(f'extractor_model_79.4.bin'))
# label_predictor.load_state_dict(torch.load(f'predictor_model_79.4.bin'))

# # train 200 epochs
# for epoch in range(num_epoch):
#   train_D_loss, train_F_loss, train_acc, Lambda, lr = train_epoch(source_dataloader, target_dataloader)
#
#   # if epoch == 200:
#   #   torch.save(feature_extractor.state_dict(), f'extractor_model79.4_200.bin')
#   #   torch.save(label_predictor.state_dict(), f'predictor_model79.4_200.bin')
#   # elif epoch == 500:
#   #   torch.save(feature_extractor.state_dict(), f'extractor_model79.4_500.bin')
#   #   torch.save(label_predictor.state_dict(), f'predictor_model79.4_500.bin')
#
#
#   print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}, Lambda {:6.4f}, LR {:6.4f}'.format(epoch,
#                                                                                                                  train_D_loss,
#                                                                                                                  train_F_loss,
#                                                                                                                  train_acc,
#                                                                                                                  Lambda,
#                                                                                                                lr))

torch.save(feature_extractor.state_dict(), f'extractor_model.bin')
torch.save(label_predictor.state_dict(), f'predictor_model.bin')

# feature_extractor.load_state_dict(torch.load(f'extractor_model.bin'))
# label_predictor.load_state_dict(torch.load(f'predictor_model.bin'))

result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()

    class_logits = label_predictor(feature_extractor(test_data))
    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)
# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('submission_.csv',index=False)
