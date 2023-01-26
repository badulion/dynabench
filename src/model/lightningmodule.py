from pytorch_lightning import LightningModule
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch import stack, concat, randn_like, sum, mean

from typing import Any, Optional
from copy import copy

class ModelPyG(LightningModule):
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
            x_graph.x = stack(predictions, dim=0)
            return x_graph.x
        else:
            return self.net(x).x

    def training_step(self, batch, batch_idx):
        x, y, points = batch

        # add noise during training
        x.x += self.training_noise * randn_like(x.x) # add gaussian noise during training
        
        # check for rollout
        rollout = 1 if isinstance(y, list) else None
        y = y[0].x.unsqueeze(0) if isinstance(y, list) else y.x

        # forward pass
        y_hat = self(x, rollout=rollout)
        loss = self.loss(y_hat, y)

        # logging
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, points = batch

        # check for rollout
        rollout = 1 if isinstance(y, list) else None
        y = y[0].x.unsqueeze(0) if isinstance(y, list) else y.x

        # forward pass
        y_hat = self(x, rollout=rollout)
        loss = self.loss(y_hat, y)

        # logging
        self.log('val_loss', loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, points = batch

        # check for rollout
        rollout = len(y) if isinstance(y, list) else None
        y = stack([roll.x for roll in y]) if isinstance(y, list) else y.x

        y_hat = self(x, rollout=rollout)
        loss = mean((y_hat-y)**2, dim=(1, 2))


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