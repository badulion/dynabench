from dynabench.dataset import DynabenchIterator, download_equation
from torch.utils.data import DataLoader
from dynabench.dataset._transforms import MakeKNNGraph, Grid2Cloud, Compose

import torch.optim as optim
import torch.nn as nn

transform = Compose([Grid2Cloud(), MakeKNNGraph(k=8)])

#download_equation('advection', structure='cloud', resolution='low')

advection_train_iterator = DynabenchIterator(split="train",
                                           equation='advection',
                                           structure='grid',
                                           resolution='low',
                                           lookback=1,
                                           squeeze_lookback_dim=True,
                                           rollout=1,
                                           transforms=transform,
)

train_loader = DataLoader(advection_train_iterator, batch_size=16, shuffle=True)

for epoch in range(10):
    for i, data_item in enumerate(train_loader):
        if len(data_item) == 3:
            print(data_item[0].shape, data_item[1].shape, data_item[2].shape)
        elif len(data_item) == 4:
            print(data_item[0].shape, data_item[1].shape, data_item[2].shape, data_item[3].shape)
        break
    break