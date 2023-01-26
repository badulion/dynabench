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
    num_existing_equations = len(os.listdir(eq_path)) if os.path.exists(eq_path) else 0

    for i in range(cfg.num_simulations):
        # make sure everything is seeded
        seed = num_existing_equations + i + cfg.seed
        np.random.seed(seed)

        # solve instance
        equation = instantiate(cfg.equation)
        solver = instantiate(cfg.generator, equation=equation, save_dir=eq_path)
        solver.solve()
    logger.info(f"Finished generating data.")
    

if __name__ == "__main__":
    main()