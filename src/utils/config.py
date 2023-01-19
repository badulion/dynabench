import gin
import pytorch_lightning as pl
from ..model.baseline import is_baseline

@gin.configurable
class ExperimentConfig:

    @staticmethod
    @gin.configurable
    def logging_dir(task: str, 
                    support: str, 
                    num_points: str, 
                    equation: str, 
                    model: str) -> str:
        return f"{task}/{support}/{num_points}/{equation}/{model}"

    @staticmethod
    @gin.configurable
    def calc_input_features(equation: str) -> int:
        if equation == "gas_dynamics":
            return 4
        elif equation == "brusselator":
            return 2
        elif equation == "wave":
            return 1
        elif equation == "kuramoto_sivashinsky":
            return 1
        else:
            raise RuntimeError("Equation name not recognized. Check your config files.")
        
    @staticmethod
    @gin.configurable
    def calc_input_size(equation: str, lookback: int) -> int:
        return ExperimentConfig.calc_input_features(equation) * lookback

@gin.configurable
class Experiment:
    def __init__(self,
                 model: pl.LightningModule,
                 datamodule: pl.LightningDataModule,
                 trainer: pl.Trainer) -> None:
        self.model = model
        self.datamodule = datamodule
        self.trainer = trainer
    
    def run(self) -> None:
        if is_baseline(self.model.net):
            self.trainer.test(model=self.model, datamodule=self.datamodule)
        else:
            self.trainer.fit(model=self.model, datamodule=self.datamodule)
            self.trainer.test(datamodule=self.datamodule, ckpt_path='best')