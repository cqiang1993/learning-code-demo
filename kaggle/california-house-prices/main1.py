import pandas as pd
import torch
from d2l import torch as d2l


# 获取数据
# kaggle competitions download -c california-house-prices -p data
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
# 长度
print(len(train))
print(len(test))

# 维度
print(train.shape)
print(test.shape)

