import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# 1. 基础模块 ###
class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        cnt = torch.tensor(0)
        self.register_buffer('cnt', cnt)

    def forward(self, x):
        self.cnt += 1
        # print("In forward: ", self.cnt, "Rank: ", self.fc.weight.device)
        return torch.sigmoid(self.fc(x))


class SimpleDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]


# 2. 初始化我们的模型、数据、各种配置  ####
## DDP：从外部得到local_rank参数。从外面得到local_rank参数，在调用DDP的时候，其会自动给出这个参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

## DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')

## 假设我们有一些数据
n_sample = 100
n_dim = 10
batch_size = 25
X = torch.randn(n_sample, n_dim)  # 100个样本，每个样本有10个特征
Y = torch.randint(0, 2, (n_sample,)).float()

dataset = SimpleDataset(X, Y)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

## 构造模型
model = SimpleModel(n_dim).to(local_rank)
## DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))

## DDP: 构造DDP model —————— 必须在 init_process_group 之后才可以调用 DDP
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

## DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
loss_func = nn.BCELoss().to(local_rank)

# 3. 网络训练  ###
model.train()
num_epoch = 100
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    data_loader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in data_loader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label.unsqueeze(1))
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()

    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0 and epoch == num_epoch - 1:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)
