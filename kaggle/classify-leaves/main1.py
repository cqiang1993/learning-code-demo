import pandas as pd
import torch as torch
import wandb
from torchvision.transforms import ToTensor, Lambda
import myDataset
from torch.utils.data import Dataset,DataLoader
from tqdm.notebook import tqdm
import Model as model
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

print(len(train))
print(len(test))

print(train.shape)
print(test.shape)

train_data = train
val_data = train
# train_data = train.sample(n=int(len(train) * 0.9))
# val_data = train.sample(n=int(len(train) * 0.1))
classes = train_data['label'].unique().tolist()
print(len(train_data))

num_epochs, lr, bs, weight_decay = 50, 0.001, 64, 2e-4
NUM_SAVE = 5

wandb.init(project="kaggle_predict_leaves",
           config={ "learning_rate": lr,
                    "batch_size": bs,
                    "total_run": num_epochs,
                    "weight decay":weight_decay,
                    "optim": "AdamW"
                  }
          )

target_transform = Lambda(lambda y: torch.tensor(classes.index(y)))


training_data = myDataset.MyDataset(train_data['label'], train_data['image'], 'train', target_transform)
train_dataloader = DataLoader(training_data, batch_size=bs, shuffle=True)

# val_data = myDataset.MyDataset(val_data['label'], val_data['image'], 'val', target_transform)
# val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=False)

testing_data = myDataset.MyDataset(test['image'], test['image'], 'test', None)
test_dataloader = DataLoader(testing_data, batch_size=bs, shuffle=False)
print("train_data length:",len(training_data),"test_data length:",len(test))

net = model.MODEL(out_label=len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
params_1x = [param for name, param in net.resnet.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
optimizer = optim.AdamW([{'params': params_1x},
                         {'params': net.resnet.fc.parameters(),
                          'lr': lr * 10}],
                        lr=lr, weight_decay=weight_decay)

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
    wandb.log({'train accuracy': train_accuracy, 'epoch': epoch})

    # # 验证集
    # val_accs = []
    # for i, data in enumerate(val_dataloader, 0):
    #     inputs, labels = data
    #     inputs, labels = inputs.to(device), labels.to(device)
    #     outputs = net(inputs)
    #     acc = (outputs.argmax(dim=-1) == labels).float().mean()
    #     val_accs.append(acc)
    #     del inputs, labels
    # val_accuracy = sum(val_accs) / len(val_accs)
    # wandb.log({'accuracy': val_accuracy, 'epoch': epoch})
    # print("No.", epoch, "accuracy:" + "{:.2f}%".format(train_accuracy.item() * 100))
    # if (epoch % (num_epochs // NUM_SAVE) == 0) and epoch != 0:
    #     torch.save(net.state_dict(), 'checkpoint_' + str(epoch))
    #     print("Model Saved")

wandb.finish()
print('Finished Training, the last loss is:', loss.item())

torch.cuda.empty_cache()
# 将网络应用于测试集。
pre_label=[]
# net.to('cpu')
for i, data in enumerate(tqdm(test_dataloader,0)):
    inputs, labels = data
    inputs = inputs.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    for j in range(len(predicted)):
        pre_label.append(classes[predicted[j]])

# 提交 submission
submission = pd.concat([test['image'], pd.DataFrame(pre_label,columns =['label'])], axis=1)
submission.to_csv('submission.csv', index=False)