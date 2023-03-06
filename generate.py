import hydra
import os
import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from src.utils.logging import create_logger

logger = create_logger(__name__)



@hydra.main(version_base=None, config_path="config", config_name="generate")
def main(cfg : DictConfig) -> None:

    logger.info(f"Starting to generate data.")
    logger.info(f"Generating data for the {cfg.equation_name} equation.")
    logger.info(f"The selected equation will be solved {cfg.num_simulations} times.")

    eq_path = f"data/{cfg.equation_name}"
    seed_list = np.loadtxt("config/seeds.txt", dtype=int)

    for i in range(cfg.num_simulations):
        # make sure everything is seeded
        seed = seed_list[i]
        np.random.seed(seed)

        # solve instance
        equation = instantiate(cfg.equation)
        solver = instantiate(cfg.generator, equation=equation, save_dir=eq_path, save_name=seed)
        solver.solve()
    logger.info(f"Finished generating data.")
    

if __name__ == "__main__":
    main()