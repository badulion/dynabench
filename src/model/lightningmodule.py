from pytorch_lightning import LightningModule
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch import stack, concat, randn_like, sum, mean

from typing import Any, Optional
from copy import copy
import gin

@gin.configurable
class Model(LightningModule):
    def __init__(self, net: Module, lr: float = 1e-3, training_noise=0.0, batch_size: Optional[int]=None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = net
        self.loss = mse_loss
        self.batch_size = batch_size
        self.lr = lr
        self.training_noise = training_noise

    def forward(self, x, rollout: Optional[int] = None):
        if rollout:
            x_graph = copy(x)
            predictions=[]
            for _ in range(rollout):
                x_old = copy(x)
                pred = self.net(x)
                predictions.append(pred.x)
                x.x = concat([x_old.x[:,pred.x.size(1):], pred.x], dim=1)
            x_graph.x = stack(predictions, dim=-1)
            return x_graph
        else:
            return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, points = batch

        x.x += self.training_noise * randn_like(x.x) # add gaussian noise during training
        
        rollout = None if y.x.dim() < 3 else 1
        y_hat = self(x, rollout=rollout)
        loss = self.loss(y_hat.x, y.x)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, points = batch
        rollout = None if y.x.dim() < 3 else 1
        y_hat = self(x, rollout=rollout)
        loss = self.loss(y_hat.x, y.x)
        self.log('val_loss', loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, points = batch
        rollout = None if y.x.dim() < 3 else y.x.size(2)
        y_hat = self(x, rollout=rollout)
        loss = mean((y_hat.x-y.x)**2, dim=(0,1))
        if rollout:
            for i in range(rollout):
                self.log(f"test_rollout_{i+1}", loss[i], batch_size=self.batch_size)
            self.log('test_rollout_last', loss[-1], batch_size=self.batch_size)
            self.log('test_rollout_mean', mean(loss), batch_size=self.batch_size)
            loss = loss[0]

        self.log('test_loss', loss, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=self.lr)