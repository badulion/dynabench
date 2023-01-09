from torch.utils.data import DataLoader as DataLoader_pytorch
from torch_geometric.loader import DataLoader as DataLoader_graph
from tqdm import tqdm
import pytorch_lightning as pl
from typing import Any
from torch.nn.functional import mse_loss

import torch

from src.model.mlp import MLP
from src.dataset.dataset_graph import DynaBench
from src.model.gat import GATNet


class Model(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = GATNet(8, 1, 128, 5)
        self.loss = mse_loss

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.x)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.x)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)

if __name__ == '__main__':
    ds = DynaBench(equation="wave", lookback=8, rollout=16,task="forecast", mode="train")
    data = DataLoader_graph(ds, batch_size=64, num_workers=4, shuffle=True)
    print(ds[0][0])

    #trainer = pl.Trainer(accelerator='cpu', devices=1, default_root_dir="results")
    #model = Model()

    #trainer.fit(model, data)
    #trainer.test(model, data)

    
