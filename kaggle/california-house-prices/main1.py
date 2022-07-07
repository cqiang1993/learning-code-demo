import pandas as pd
import torch
import torch.nn as nn
import filter
import mlp as mlp
import wandb

# 确认跑道
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
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

# 预处理数据
# 删除无用数据 序列、地址、描述、省份、邮编 等等等
redundant_cols = ['Id', 'Address', 'Summary', 'State', 'Zip', 'Last Sold On', 'Last Sold Price', 'Listed On',
                  'Listed Price',
                  'Elementary School', 'Middle School', 'High School', 'Region', 'Lot']
describe_cols = ['Flooring', 'Heating features', 'Cooling features', 'Appliances included', 'Laundry features',
                 'Parking features', 'Heating', 'Cooling', 'Parking', 'Bedrooms']

# 补充缺失数据 以及格式化数据
train, test = filter.handle_data([train, test], redundant_cols, describe_cols)

# 把train和test去除id后放一起，train也要去掉label
all_features = pd.concat((train.iloc[:, 1:], test.iloc[:, 1:]))
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train.shape[0]
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float).cuda()
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float).cuda()
train_labels = torch.tensor(train['Sold Price'].values.reshape(-1, 1), dtype=torch.float).cuda()

criterion = nn.MSELoss()
in_features = train_features.shape[1]
net = mlp.MLP(in_features)

k, num_epochs, lr, weight_decay, batch_size = 5, 2000, 0.005, 0.05, 256
wandb.init(project="kaggle_1",
           config={"learning_rate": lr,
                   "weight_decay": weight_decay,
                   "batch_size": batch_size,
                   "total_run": num_epochs,
                   "network": "in->256->64"}
           )
print("network:", net.to(device))

train_ls, valid_ls = mlp.train(net, criterion, train_features, train_labels, None, None, num_epochs, lr, weight_decay,
                               batch_size, device)

# 使用现有训练好的net
net.to(device)
# 将网络应用于测试集。
preds = net(test_features).detach().numpy()

# 将其重新格式化以导出到Kaggle
test['Sold Price'] = pd.Series(preds.reshape(1, -1)[0])
submission = pd.concat([test['Id'], test['Sold Price']], axis=1)
submission.to_csv('submission.csv', index=False)
