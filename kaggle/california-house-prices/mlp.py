import torch
import pandas as pd
import torch.nn as nn
from torch.nn import functional as F
from d2l import torch as d2l
from tqdm import tqdm
import numpy as np
from torch.utils import data
import wandb

NUM_SAVE = 50


class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, 896)
        self.layer2 = nn.Linear(896, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.out(x)


def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def log_rmse(net, loss, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def train(net, loss, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size, device):
    wandb.watch(net)
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in tqdm(range(num_epochs)):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(X)
            l = loss(outputs, y)
            l.backward()
            optimizer.step()
        record_loss = log_rmse(net.to(device), loss, train_features, train_labels)
        wandb.log({'loss': record_loss, 'epoch': epoch})
        train_ls.append(record_loss)
        if (epoch % NUM_SAVE == 0 and epoch != 0) or (epoch == num_epochs - 1):
            torch.save(net.state_dict(), 'checkpoint_' + str(epoch))
            print('save checkpoints on:', epoch, 'rmse loss value is:', record_loss)
        del X, y
        net.to(device)
    wandb.finish()
    return train_ls, test_ls
