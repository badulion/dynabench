import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    model = instantiate(cfg.model)
    dm = instantiate(cfg.datamodule)
    t = instantiate(cfg.trainer)

    t.fit(model, dm)

if __name__ == "__main__":
    main()