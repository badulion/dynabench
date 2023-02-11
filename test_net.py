import sys
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# import model and lightning moule
from src.model.continuous_conv.cconv_lightning_graph import ConCov
from src.model.continuous_conv.cconv_lightning_graph import LitModel
# import dataloader
from src.dataset import dataloader as DB

eq = sys.argv[1]            # equation for data
#eq = "wave"                # debug
print(f"NOW using EQUATION {eq}")
###### CONFIGURATION ######
# TRAIN
epochs      = 10            # epochs
batch_size  = 32            # batches
# MODEL
lb          = 8             # lookback
f_in        = 1             # features_in
hl          = 6             # hidden layers for cconv
coord_dim   = 2             # dimensionality of points
hs          = 16            # hidden size: features_out for hidden cconv layers
# CCONV LAYER
knn         = 10            # default 10 neighbors
hidden_mlp  = 32            # hidden mlp size for each layer
hl_mlp      = 2             # hidden layer for MLP in cconv layer

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
trainer = pl.Trainer(logger=[tb_logger, csv_logger], limit_val_batches=100, limit_train_batches=100, max_epochs=epochs, accelerator="gpu", devices=1, callbacks=[EarlyStopping(monitor="val_loss", mode="min")], val_check_interval=0.5) #, , profiler="simple"
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