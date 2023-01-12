
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger



from src.dataset.dataloader import DynaBenchDataModule

# models
from src.model.gat import GATNet
from src.model.point_net import PointNet
from src.model.point_gnn import PointGNN
from src.model.point_transformer import PointTransformer

from src.model.lightningmodule import Model
import gin



if __name__ == '__main__':
    # register external objects and parse gin config tree
    Model = gin.external_configurable(Model)
    TensorBoardLogger = gin.external_configurable(TensorBoardLogger)
    Trainer = gin.external_configurable(Trainer)
    DynaBenchDataModule = gin.external_configurable(DynaBenchDataModule)

    GATNet = gin.external_configurable(GATNet)
    PointNet = gin.external_configurable(PointNet)
    PointGNN = gin.external_configurable(PointGNN)
    PointTransformer = gin.external_configurable(PointTransformer)
    gin.parse_config_file('config/config.gin')


    # train the model
    datamodule = DynaBenchDataModule()
    trainer = Trainer()
    model = Model()

    trainer.fit(model, datamodule)


    
