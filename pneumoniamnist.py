# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 12:35:48 2024

@author: Allen
"""

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torcheval.metrics.functional import multiclass_f1_score


import medmnist
from medmnist import INFO, Evaluator

data_flag = 'chestmnist'
download = True

EPOCH_RANGE = [3,3,3,3,3,3,3,3,3,3] #Multiple runs of 3 epochs to get average of auc and acc
#EPOCH_RANGE = [1,2,3,4,5,6,7,8,9,10] #Testing out diffeent epoch values
BATCH_SIZE = 128
lr = 0.001
eval_metrics_list = []

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


class Net(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = Net(in_channels=n_channels, num_classes=n_classes)

# define loss function and optimizer

##criterion = multiclass_f1_score()


if task == "multi-label, binary-class":
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

    
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

# train


print('==> Training multiple epochs ...')

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
    
            y_true = y_true.numpy()
            y_score = y_score.detach().numpy()
            
            evaluator = Evaluator(data_flag, split)
            metrics = evaluator.evaluate(y_score)
            if split == 'test':
                eval_metrics_list.append(metrics)
            
        
            print('%s  auc: %.3f  acc:%.3f' % (split, *metrics))
       
    print('==> Evaluating ...')
    test('train')
    test('test')
    
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
#legend = plt.legend(loc='middle right', shadow=False, fontsize='small')
plt.xlabel('Epochs')
plt.show

