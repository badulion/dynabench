import sys
import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# import model and lightning moule
from src.model.continuous_conv.cconv_lightning_graph import ConCov
from src.model.continuous_conv.cconv_lightning_graph import LitModel
# import dataloader
from src.dataset import dataloader as DB

parser = argparse.ArgumentParser(
                    prog = 'Test Continuous Convolution',
                    description = 'Neural Network')
parser.add_argument('--equation', dest='eq', action='store', default="wave",
                    help='specify equation for dynabench')
parser.add_argument('--epochs', dest='epochs', action='store', default=10,
                    help='specify epochs to be trained')
args = parser.parse_args()


###### CONFIGURATION ######
# TRAIN
eq = args.eq                # equation for data
epochs      = args.epochs   # epochs
batch_size  = 32            # batches
print(f"NOW using EQUATION {eq} and TRAINING with {epochs} epochs")
# MODEL
lb          = 8             # lookback
f_in        = 1             # features_in
hl          = 8             # hidden layers for cconv
coord_dim   = 2             # dimensionality of points
hs          = 16             # hidden size: features_out for hidden cconv layers
# CCONV LAYER
knn         = 10            # default 10 neighbors
hidden_mlp  = 32            # hidden mlp size for each layer
hl_mlp      = 1            # hidden layer for MLP in cconv layer

if eq == "gas_dynamics":
    f_in = 4
if eq == "advection":
    coord_dim = 1
if eq == "kuramato_sivashinsky":
    coord_dim = 4
###########################

dynabench = DB.DynaBenchDataModule(batch_size=batch_size, equation=eq, base_path="data", structure="graph",lookback=lb, num_workers=8)
# logger tensorboard --logdir=logs/lightning_logs/
tb_logger = pl_loggers.TensorBoardLogger(save_dir="outputs/tb_logs")
csv_logger = pl_loggers.CSVLogger(save_dir="outputs/tb_logs")
# Trainer
trainer = pl.Trainer(logger=[tb_logger, csv_logger], limit_val_batches=10, max_epochs=epochs, accelerator="gpu", devices=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min")], val_check_interval=0.8) #, , profiler="simple"
# net
net = ConCov(k=knn, hidden_mlp=hidden_mlp, points_dim=coord_dim, input_size=lb*f_in, output_size=f_in, hidden_layers=hl, hidden_size=hs, hl_mlp=hl_mlp)
# Lightning model
model = LitModel(net)
# train model
trainer.fit(model, dynabench)
# resume from checkpoint
# trainer.fit(model, ckpt_path="logs/lightning_logs/...")
# test model
trainer.test(model, dynabench)