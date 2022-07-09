import torch
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np
import os

import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
from torch.utils import data
import torch.optim as optim

from torchvision.io import read_image
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Lambda

import matplotlib.pyplot as plt
import wandb


class MODEL(nn.Module):
    def __init__(self, out_label):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, out_label)

    def forward(self, X):
        return self.resnet(X)


class MyDataset(Dataset):
    def __init__(self, annotations, img_dir, mode=None, target_transform=None):
        super().__init__()
        self.img_labels = annotations
        self.img_dir = img_dir
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
        self.transform = preprocess
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir[idx])
        with Image.open(img_path) as im:
            image = im
            label = self.img_labels.iloc[idx]
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
test_data = pd.read_csv('data/test.csv')
train_data = pd.read_csv('data/train.csv')
classes = train_data['label'].unique().tolist()
print("train_data:", train_data.shape, "test_data shape:", test_data.shape, "\nlabel size:", len(classes))

num_epochs, lr, bs = 30, 0.001, 128
NUM_SAVE = 5

target_transform = Lambda(lambda y: torch.tensor(classes.index(y)))

training_data = MyDataset(train_data['label'], train_data['image'], 'train', target_transform)
train_dataloader = DataLoader(training_data, batch_size=bs, shuffle=True)

testing_data = MyDataset(test_data['image'], test_data['image'], 'test', None)
test_dataloader = DataLoader(testing_data, batch_size=bs, shuffle=False)
print("train_data length:", len(training_data), "test_data length:", len(test_data))

net = MODEL(out_label=len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

wandb.init(project="kaggle_predict_leaves",
           config={"learning_rate": lr,
                   "batch_size": bs,
                   "total_run": num_epochs,
                   "optim": optimizer
                   }
           )

# 4. 训练过程
wandb.watch(net)
step = 0
for epoch in tqdm(range(num_epochs)):  # loop over the dataset multiple times

    # 训练集
    train_accs = []
    for i, data in enumerate(train_dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # record
        step += 1
        acc = (outputs.argmax(dim=-1) == labels).float().mean()
        train_accs.append(acc)
        wandb.log({'loss': loss, 'step': step})
        del inputs, labels

    train_accuracy = sum(train_accs) / len(train_accs)
    wandb.log({'accuracy': train_accuracy, 'epoch': epoch})
    print("No.", epoch, "  Accuracy:" + "{:.2f}%".format(train_accuracy.item() * 100))
    if (epoch % (num_epochs // NUM_SAVE) == 0) and epoch != 0:
        torch.save(net.state_dict(), 'checkpoint_' + str(epoch))
        print("Model Saved")

wandb.finish()
print("Finished Training, accuracy:" + "{:.2f}%".format(train_accuracy.item() * 100))