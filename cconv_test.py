import argparse
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
# import model and lightning moule
from src.model.continuous_conv.cconv_lightning_graph import ConCov
from src.model.continuous_conv.cconv_lightning_graph import LitModel
# import dataloader
from src.dataset import dataloader as DB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                        prog = 'Test Continuous Convolution',
                        description = 'Neural Network')
    parser.add_argument('--equation', dest='eq', action='store', default="wave",
                        help='specify equation for dynabench')
    parser.add_argument('--epochs', dest='epochs', action='store', default=10,
                        help='specify epochs to be trained', type=int)
    parser.add_argument('--batches', dest='batches', action='store', default=32,
                        help='specify batch size', type=int)
    args = parser.parse_args()


###### CONFIGURATION ######
    # TRAIN
    EQUATION = args.eq       # equation for data
    EPOCHS = args.epochs   # epochs
    BATCH_SIZE = args.batches  # batches
    print("----------------------------------------------------------")
    print(f"USING: EQUATION {EQUATION}, EPOCHS {EPOCHS}, BATCHSIZE {BATCH_SIZE}")
    print("----------------------------------------------------------")
else:
    EQUATION = "wave"
    EPOCHS = 1
    BATCH_SIZE = 32
### STANDARD PARAMS ########
# MODEL
LOOKBACK = 8             # lookback
FEATURES_IN = 1             # features_in
COORDINATE_DIM = 2             # dimensionality of points
HIDDEN_LAYER = 2             # hidden layers for cconv   x-1-10
# CCONV LAYER
KNN_NUM = 10            # default 10 neighbors
HIDDEN_SIZE_MLP = 128           # hidden mlp size for each layer   x-32-256
HIDDEN_LAYER_MLP = 1             # hidden layer for MLP in cconv layer   x-1-2

if EQUATION == "gas_dynamics":
    # specific changes for gas_dynamics
    FEATURES_IN = 4
    LOOKBACK = 4
    print(f"SET FEATURS_IN = {FEATURES_IN} AND LOOKBACK= {LOOKBACK}")
    print("----------------------------------------------------------")
if EQUATION == "brusselator":
    # specific changes for brusselator
    FEATURES_IN = 2
    print(f"SET FEATURS_IN = {FEATURES_IN}")
    print("----------------------------------------------------------")

HIDDEN_SIZE          = FEATURES_IN*LOOKBACK       # hidden size: features_out for hidden cconv layers
###########################

dynabench = DB.DynaBenchDataModule(batch_size=BATCH_SIZE,
                                   equation=EQUATION, base_path="data",
                                   structure="graph",
                                   lookback=LOOKBACK,
                                   num_workers=8)
# logger tensorboard --logdir=logs/lightning_logs/
tb_logger = pl_loggers.TensorBoardLogger(save_dir="outputs/tb_logs_"+EQUATION)
csv_logger = pl_loggers.CSVLogger(save_dir="outputs/csv_logs_"+EQUATION)
# Trainer
trainer = pl.Trainer(logger=[tb_logger, csv_logger],
                     max_epochs=EPOCHS,
                     accelerator="gpu",
                     devices=1,
                     val_check_interval=0.5)#,
                     #callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
                     #profiler="simple",
                     #auto_lr_find=True)
                     
# net
net = ConCov(k=KNN_NUM,
             hidden_mlp=HIDDEN_SIZE_MLP,
             points_dim=COORDINATE_DIM,
             input_size=LOOKBACK*FEATURES_IN,
             output_size=FEATURES_IN,
             hidden_layers=HIDDEN_LAYER,
             hidden_size=HIDDEN_SIZE,
             hl_mlp=HIDDEN_LAYER_MLP)
# Lightning model
model = LitModel(net)
# train model
#trainer.tune(model, dynabench)
trainer.fit(model, dynabench)
# resume from checkpoint
# trainer.fit(model, ckpt_path="logs/lightning_logs/...")
# test model
trainer.test(model, dynabench)