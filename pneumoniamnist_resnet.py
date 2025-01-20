# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 21:15:34 2024

@author: Allen
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator


data_flag = 'pneumoniamnist'
download = True

EPOCH_RANGE = [3]
#EPOCH_RANGE = [3,3,3,3,3,3,3,3,3,3] #Multiple runs of 3 epochs to get average of auc and acc
#EPOCH_RANGE = [1,2,3,4,5,6,7,8,9,10] #Testing out diffeent epoch values
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

pil_dataset = DataClass(split='train', download=download)

train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_loader_at_eval = data.DataLoader(dataset=train_dataset, batch_size=2*BATCH_SIZE, shuffle=False)
test_loader = data.DataLoader(dataset=test_dataset, batch_size=2*BATCH_SIZE, shuffle=False)

print(train_dataset)
print("===================")
print(test_dataset)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(in_channels, num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

    
model = ResNet18(in_channels=n_channels, num_classes=n_classes)

if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train

if len(EPOCH_RANGE) > 1:
    print('==> Training multiple epochs ...')

#train_f1 = []
#test_f1 = []
eval_metrics_list = []

for NUM_EPOCHS in EPOCH_RANGE:    

    print('==> Training ...')
    print('Epoch: %d' % (NUM_EPOCHS))
    
    for epoch in range(NUM_EPOCHS):
        train_correct = 0
        train_total = 0
        test_correct = 0
        test_total = 0
        
        model.train()
        for inputs, targets in tqdm(train_loader):
            # forward + backward + optimize
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if task == 'multi-label, binary-class':
                targets = targets.to(torch.float32)
                loss = criterion(outputs, targets)
            else:
                targets = targets.squeeze().long()
                loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
    # evaluation
    
    def test(split):
        model.eval()
        y_true = torch.tensor([])
        y_score = torch.tensor([])
        
        data_loader = train_loader_at_eval if split == 'train' else test_loader
    
        with torch.no_grad():
            for inputs, targets in data_loader:
                outputs = model(inputs)
    
                if task == 'multi-label, binary-class':
                    targets = targets.to(torch.float32)
                    outputs = outputs.softmax(dim=-1)
                else:
                    targets = targets.squeeze().long()
                    outputs = outputs.softmax(dim=-1)
                    targets = targets.float().resize_(len(targets), 1)
    
                y_true = torch.cat((y_true, targets), 0)
                y_score = torch.cat((y_score, outputs), 0)
            

            #y_true_f1 = torch.squeeze(y_true)
            y_true_eval = y_true.numpy()
            y_score_eval = y_score.detach().numpy()
            
            
            
            #f1_score = multiclass_f1_score(y_score, y_true_f1)
            evaluator = Evaluator(data_flag, split)
            eval_metrics = evaluator.evaluate(y_score_eval)
            #if split == 'train':
                #train_f1.append(f1_score)
            if split == 'test':
                #test_f1.append(f1_score)
                eval_metrics_list.append(eval_metrics)
    
            
        
            print('%s  auc: %.3f  acc:%.3f' % (split, *eval_metrics))
            #print("F1 score: %.3f" % (f1_score))
       
    print('==> Evaluating ...')
    test('train')
    test('test')
    
#train_f1_avg = sum(train_f1)/len(train_f1)
#test_f1_avg = sum(test_f1)/len(test_f1)

#print("Avg F1 score for training: %f \nAvg F1 score for testing: %f" %(train_f1_avg, test_f1_avg))


auc_list = []
acc_list = []

for i in range(len(eval_metrics_list)):
    auc_list.append(eval_metrics_list[i][0])
    
for i in range(len(eval_metrics_list)):
    acc_list.append(eval_metrics_list[i][1])
    
auc_avg = np.sum(auc_list)/len(auc_list)
acc_avg = np.sum(acc_list)/len(acc_list)
auc_max = max(auc_list)
acc_max = max(acc_list)
print("avg auc: %f \navg acc: %f \nmax auc: %f \nmax acc: %f" %(auc_avg, acc_avg, auc_max, acc_max))
    
plt.plot(range(len(EPOCH_RANGE)),auc_list, label='auc')
plt.plot(range(len(EPOCH_RANGE)),acc_list, label='acc')
plt.plot(range(len(EPOCH_RANGE)),[acc_avg]*len(EPOCH_RANGE), label='acc avg')
plt.plot(range(len(EPOCH_RANGE)),[auc_avg]*len(EPOCH_RANGE), label='auc avg')
#legend = plt.legend(loc='centre right', shadow=False, fontsize='small')
plt.xlabel('Epochs')
plt.show
