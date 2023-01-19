
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.dataset.dataloader import DynaBenchDataModule

# models
from src.model.gnn import GATNet, GCN
from src.model.point import FeaStNet, PointNet, PointGNN, PointTransformer
from src.model.baseline import BaselinePersistence, BaselineZero

from src.model.lightningmodule import Model
from src.utils import ExperimentConfig, Experiment
import gin
import argparse


def parse_args():
    global args, parser
    parser = argparse.ArgumentParser(
                prog = 'Training and Evaluation Script',
                description = 'This program runs the experiments for the DynaBench Dataset.',
                epilog = 'Thank you!')

    parser.add_argument('-e', '--equation', type=str, default='wave', help="Equation to use.", 
                        choices=['wave', 'gas_dynamics', 'brusselator', 'kuramoto_sivashinsky'])
    parser.add_argument('-s', '--support', type=str, default='cloud', help="Support type (grid or cloud) to use.", 
                        choices=['grid', 'cloud'])
    parser.add_argument('-n', '--num-points', type=str, default='high', help="Number of points to use.", 
                        choices=['low', 'high'])
    parser.add_argument('-t', '--task', type=str, default='forecast', help="Task to solve.", 
                        choices=['forecast', 'evolution'])
    parser.add_argument('-m', '--model', type=str, default='point_gnn', help="Model to train.", 
                        choices=['feast', 'gat', 'gcn', 'point_gnn', 'point_net', 'point_transformer', 'zero', 'persistence'])
    
    args = parser.parse_args()



def make_gin_config():
    global TensorBoardLogger, CSVLogger, Trainer
    global ModelCheckpoint, EarlyStopping

    # register external objects and parse gin config tree
    TensorBoardLogger = gin.external_configurable(TensorBoardLogger)
    CSVLogger = gin.external_configurable(CSVLogger)
    ModelCheckpoint = gin.external_configurable(ModelCheckpoint)
    EarlyStopping = gin.external_configurable(EarlyStopping)
    Trainer = gin.external_configurable(Trainer)


    gin.constant("equation", args.equation)
    gin.constant("support", args.support)
    gin.constant("num_points", args.num_points)
    gin.constant("task", args.task)
    gin.constant("model", args.model)

    gin.parse_config_file('config/config.gin')
    gin.parse_config_file(f'config/model/{args.model}.gin')
    gin.finalize()

if __name__ == '__main__':
    parse_args()
    make_gin_config()

    # train the model
    experiment = Experiment()
    experiment.run() 