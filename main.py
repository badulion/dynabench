from torch.utils.data import DataLoader as DataLoader_pytorch
from torch_geometric.loader import DataLoader as DataLoader_graph
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Any
from torch.nn.functional import mse_loss

import torch

from src.model.mlp import MLP
from src.dataset.dataset_graph import DynaBench
from src.model.gat import GATNet


class Model(pl.LightningModule):
    def __init__(self, batch_size=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = GATNet(34, 4, 128, 4)
        self.loss = mse_loss
        self.batch_size = batch_size

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.x)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.x)
        self.log('val_loss', loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y.x)
        self.log('test_loss', loss, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)

if __name__ == '__main__':
    ds_train = DynaBench(equation="gas_dynamics", lookback=8, rollout=1, support="full", task="evolution", mode="train")
    dataloader_train = DataLoader_graph(ds_train, batch_size=32, num_workers=4, shuffle=True)


    ds_val = DynaBench(equation="gas_dynamics", lookback=8, rollout=1, support="full", task="evolution", mode="val")
    dataloader_val = DataLoader_graph(ds_val, batch_size=32, num_workers=4, shuffle=False)


    logger = TensorBoardLogger("results/tb_logs", name="my_model")
    trainer = pl.Trainer(accelerator='gpu', 
                         devices=1, 
                         default_root_dir="results", 
                         max_epochs=300,
                         logger=logger)

    model = Model(batch_size=64)
    #model = Model.load_from_checkpoint("results/tb_logs/my_model/version_0/checkpoints/epoch=99-step=31100.ckpt", batch_size=64)

    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)

    #trainer.test(model, data)

    
