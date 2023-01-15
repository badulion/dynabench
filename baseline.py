
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.dataset.dataloader import DynaBenchDataModule

# models
from src.model.gat import GATNet
from src.model.gcn import GCN
from src.model.feast import FeaStNet
from src.model.point_net import PointNet
from src.model.point_gnn import PointGNN
from src.model.point_transformer import PointTransformer

from src.model.baseline_persistence import BaselinePersistence
from src.model.baseline_zero import BaselineZero

from src.model.lightningmodule import Model
import gin
import argparse

@gin.configurable
def logging_dir(equation, support, task, model):
    return f"{equation}/{task}/{support}/{model}"


@gin.configurable
def calc_input_features(equation):
    if equation == "gas_dynamics":
        return 4
    else:
        return 1

@gin.configurable
def calc_input_size(equation, lookback):
    return calc_input_features(equation) * lookback


def parse_args():
    global args, parser
    parser = argparse.ArgumentParser(
                prog = 'Baseline Evaluation Script',
                description = 'This program runs the experiments for the DynaBench Dataset.',
                epilog = 'Thank you!')

    parser.add_argument('-e', '--equation', type=str, default='wave', help="Equation to use.", 
                        choices=['wave', 'gas_dynamics', 'brusselator', 'kuramoto_sivashinsky'])
    parser.add_argument('-s', '--support', type=str, default='high', help="Support (number of points) to use.", 
                        choices=['low', 'mid', 'high', 'full'])
    parser.add_argument('-t', '--task', type=str, default='forecast', help="Task to solve.", 
                        choices=['forecast', 'evolution'])
    parser.add_argument('-b', '--baseline', type=str, default='persistence', help="Which baseline to use.", 
                        choices=['zero', 'persistence'])
    
    args = parser.parse_args()



def make_gin_config():
    global Model, TensorBoardLogger, CSVLogger, Trainer, DynaBenchDataModule
    global ModelCheckpoint, EarlyStopping
    global GATNet, GCN, FeaStNet, PointNet, PointGNN, PointTransformer
    global BaselinePersistence, BaselineZero

    # register external objects and parse gin config tree
    TensorBoardLogger = gin.external_configurable(TensorBoardLogger)
    CSVLogger = gin.external_configurable(CSVLogger)
    ModelCheckpoint = gin.external_configurable(ModelCheckpoint)
    EarlyStopping = gin.external_configurable(EarlyStopping)
    Trainer = gin.external_configurable(Trainer)


    Model = gin.external_configurable(Model)
    DynaBenchDataModule = gin.external_configurable(DynaBenchDataModule)

    BaselinePersistence = gin.external_configurable(BaselinePersistence)
    BaselineZero = gin.external_configurable(BaselineZero)


    gin.constant("equation", args.equation)
    gin.constant("support", args.support)
    gin.constant("task", args.task)
    gin.constant("model", args.baseline)

    gin.parse_config_file('config/config.gin')
    gin.parse_config_file(f'config/baseline/{args.baseline}.gin')
    gin.finalize()

if __name__ == '__main__':
    parse_args()
    make_gin_config()

    # train the model
    datamodule = DynaBenchDataModule()
    trainer = Trainer()
    model = Model()

    trainer.test(model, datamodule=datamodule)