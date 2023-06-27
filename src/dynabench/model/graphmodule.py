from pytorch_lightning import LightningModule
from torch.nn import Module
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch import stack, concat, randn_like, sum, mean

from typing import Any, Optional
from copy import copy

class GraphModule(LightningModule):
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
        x_graph = copy(x)
        predictions=[]
        for _ in range(rollout):
            x_old = copy(x)
            pred = self.net(x)
            predictions.append(pred.x)
            x.x = concat([x_old.x[:,pred.x.size(1):], pred.x], dim=1)
        x_graph.x = stack(predictions, dim=0)
        return x_graph.x

    def training_step(self, batch, batch_idx):
        x, y, points = batch['.x'], batch['.y'], batch['.points']

        x.x += self.training_noise * randn_like(x.x) # add gaussian noise during training
        
        y_hat = self(x, rollout=1)
        y = y[0].x.unsqueeze(0)
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
        y = y[0].x.unsqueeze(0)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss, batch_size=self.batch_size, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, points = batch['.x'], batch['.y'], batch['.points']
        
        rollout = len(y)
        y_hat = self(x, rollout=rollout)
        y = stack([roll.x for roll in y])
        loss = mean((y_hat-y)**2, dim=(-1,-2))
        
        for i in range(rollout):
            self.log(f"test_rollout_{i+1}", loss[i], batch_size=self.batch_size)
        self.log('test_rollout_last', loss[-1], batch_size=self.batch_size)
        self.log('test_rollout_mean', mean(loss), batch_size=self.batch_size)
        loss = loss[0]

        self.log('test_loss', loss, batch_size=self.batch_size)

        return loss

    def configure_optimizers(self):
        return Adam(self.net.parameters(), lr=self.lr)