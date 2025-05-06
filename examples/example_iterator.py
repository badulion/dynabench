from dynabench.dataset import DynabenchIterator, download_equation
from torch.utils.data import DataLoader
from dynabench.dataset.transforms import EdgeList, Grid2Cloud, Compose, ToDict

import torch.optim as optim
import torch.nn as nn

transform = Compose([Grid2Cloud(), EdgeList(k=8), ToDict()])

#download_equation('advection', structure='cloud', resolution='low')

advection_train_iterator = DynabenchIterator(split="train",
                                           equation='advection',
                                           structure='grid',
                                           resolution='full',
                                           lookback=1,
                                           squeeze_lookback_dim=True,
                                           rollout=1,
                                           base_path='/home/andi/coding/data/dynabench',
                                           transforms=transform,
)

train_loader = DataLoader(advection_train_iterator, batch_size=16, shuffle=True)

for epoch in range(10):
    for i, data_item in enumerate(train_loader):
        for key, value in data_item.items():
            print(f"{key}: {value.shape}")
        break
    break