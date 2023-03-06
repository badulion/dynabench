import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


from src.dataset.datapipe import create_datapipes
#from torch.utils.data import DataLoader
from torch_geometric.data import DataLoader

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg : DictConfig) -> None:
    
    model = instantiate(cfg.task)
    datamodule = DataLoader(create_datapipes(equation="advection", as_graph=True, lookback=8, rollout=16), batch_size=16, num_workers=8, shuffle=True)
    trainer = instantiate(cfg.trainer)
    
    if cfg.experiment.model in ['zero', 'persistence', 'difference']:
        trainer.test(model=model, datamodule=datamodule)
    else:
        trainer.fit(model=model, train_dataloaders = datamodule, val_dataloaders = datamodule)
        trainer.test(datamodule=datamodule, ckpt_path='best')

if __name__ == "__main__":
    main()