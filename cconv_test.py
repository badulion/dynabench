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
    parser.add_argument('--epochs', dest='epochs', action='store', default=1,
                        help='specify epochs to be trained', type=int)
    parser.add_argument('--batches', dest='batches', action='store', default=16,
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
    BATCH_SIZE = 16
### STANDARD PARAMS ########
# MODEL
LOOKBACK = 8                # lookback
FEATURES_IN = 1             # features_in
COORDINATE_DIM = 2          # dimensionality of points
HIDDEN_LAYER = 1            # hidden layers for cconv
# CCONV LAYER
KNN_NUM = 10                # default 10 neighbors
HIDDEN_SIZE_MLP = 129       # hidden mlp size for each layer
HIDDEN_LAYER_MLP = 2        # hidden layer for MLP in cconv layer
HIDDEN_SIZE_DIVIDENT = 1    # Division of number of hidden size
