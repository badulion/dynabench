import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg : DictConfig) -> None:
    
    model = instantiate(cfg.model)
    datamodule = instantiate(cfg.datamodule)
    trainer = instantiate(cfg.trainer)
    
    if cfg.experiment.model in ['zero', 'persistence', 'difference']:
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(datamodule=datamodule, ckpt_path='best')

if __name__ == "__main__":
    main()