from pytorch_lightning import LightningModule
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch import stack, randn_like

from typing import Any, Optional

class Model(LightningModule):
    def __init__(self, net: Module, lr: float = 1e-3, training_noise=0.0, batch_size: Optional[int]=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = net
        self.loss = mse_loss
        self.batch_size = batch_size
        self.lr = lr
        self.training_noise = training_noise

    def forward(self, x, rollout: Optional[int] = 1):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, points = batch

        x.x += self.training_noise * randn_like(x.x) # add gaussian noise during training

        y_hat = self(x)
        loss = self.loss(y_hat.x, y.x)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat.x, y.x)
        self.log('val_loss', loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, points = batch
        y_hat = self(x)
        loss = self.loss(y_hat.x, y.x)
        self.log('test_loss', loss, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=self.lr)