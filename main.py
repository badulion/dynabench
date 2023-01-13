
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

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
def logging_dir(equation, support, task):
    return f"{equation}/{task}/{support}"

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
                prog = 'Training and Evaluation Script',
                description = 'This program runs the experiments for the DynaBench Dataset.',
                epilog = 'Thank you!')

    parser.add_argument('-e', '--equation', type=str, default='wave', help="Equation to use.", 
                        choices=['wave', 'gas_dynamics', 'brusselator', 'kuramoto_sivashinsky'])
    parser.add_argument('-s', '--support', type=str, default='high', help="Support (number of points) to use.", 
                        choices=['low', 'mid', 'high', 'full'])
    parser.add_argument('-t', '--task', type=str, default='forecast', help="Task to solve.", 
                        choices=['forecast', 'evolution'])
    parser.add_argument('-m', '--model', type=str, default='point_gnn', help="Model to train.", 
                        choices=['feast', 'gat', 'gcn', 'point_gnn', 'point_net', 'point_transformer'])
    
    args = parser.parse_args()



def make_gin_config():
    global Model, TensorBoardLogger, Trainer, DynaBenchDataModule
    global GATNet, GCN, FeaStNet, PointNet, PointGNN, PointTransformer
    global BaselinePersistence, BaselineZero

    # register external objects and parse gin config tree
    Model = gin.external_configurable(Model)
    TensorBoardLogger = gin.external_configurable(TensorBoardLogger)
    Trainer = gin.external_configurable(Trainer)
    DynaBenchDataModule = gin.external_configurable(DynaBenchDataModule)

    GATNet = gin.external_configurable(GATNet)
    GCN = gin.external_configurable(GCN)
    FeaStNet = gin.external_configurable(FeaStNet)
    PointNet = gin.external_configurable(PointNet)
    PointGNN = gin.external_configurable(PointGNN)
    PointTransformer = gin.external_configurable(PointTransformer)

    BaselinePersistence = gin.external_configurable(BaselinePersistence)


    gin.constant("equation", args.equation)
    gin.constant("support", args.support)
    gin.constant("task", args.task)

    gin.parse_config_file('config/config.gin')
    gin.parse_config_file(f'config/model/{args.model}.gin')
    gin.finalize()

if __name__ == '__main__':
    parse_args()
    make_gin_config()

    # train the model
    datamodule = DynaBenchDataModule()
    trainer = Trainer()
    model = Model()

    trainer.fit(model, datamodule)
    #trainer.test(datamodule=datamodule)