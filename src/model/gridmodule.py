from pytorch_lightning import LightningModule
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch import stack, concat, randn_like, sum, mean

from typing import Any, Optional
from copy import copy

class GridModule(LightningModule):
    def __init__(self, net: Module, 
                 lr: float = 1e-3, 
                 training_noise: float=0.0, 
                 batch_size: Optional[int]=None, 
                 *args: Any, 
                 **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.net = net
        self.loss = mse_loss
        self.batch_size = batch_size
        self.lr = lr
        self.training_noise = training_noise

    def forward(self, x, rollout):
        predictions=[]
        for _ in range(rollout):
            x_old = copy(x)
            pred = self.net(x)
            predictions.append(pred)
            x = concat([x_old[:,pred.size(1):], pred], dim=1)
        x = stack(predictions, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y, points = batch['.x'], batch['.y'], batch['.points']

        x += self.training_noise * randn_like(x) # add gaussian noise during training
        
        y_hat = self(x, rollout=1)
        
        loss = self.loss(y_hat, y)
        metrics = {
            'train_loss': loss
        }
        for logger in self.loggers:
            logger.log_metrics(metrics=metrics, step=self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, points = batch['.x'], batch['.y'], batch['.points']
        
        y_hat = self(x, rollout=1)

        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, points = batch['.x'], batch['.y'], batch['.points']
        
        rollout = y.size(1)
        y_hat = self(x, rollout=rollout)
        
        loss = mean((y_hat-y)**2, dim=(0,2,3,4))
        
        for i in range(rollout):
            self.log(f"test_rollout_{i+1}", loss[i], batch_size=self.batch_size)
        self.log('test_rollout_last', loss[-1], batch_size=self.batch_size)
        self.log('test_rollout_mean', mean(loss), batch_size=self.batch_size)
        loss = loss[0]

        self.log('test_loss', loss, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=self.lr)