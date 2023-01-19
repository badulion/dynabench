from src.generator.equations import GasDynamicsPDE, BrusselatorPDE, WavePDE, KuramotoSivashinskyPDE
from src.generator import PDESolver
import numpy as np
import os
import argparse
from src.utils.logging import create_logger
import gin


logger = create_logger(__name__)
parser = argparse.ArgumentParser(
            prog = 'Data Generator',
            description = 'This program generates data for the DynaBench Dataset by solving each of the equations \
                            Gas-Dynamics, Brusselator, Wave, Kuramoto-Sivashinksy several times and postprocessing the generated solutions.',
            epilog = 'Thank you!')

parser.add_argument('-n', '--num', type=int, default=1, help="Number of times each equation is simulated.")
parser.add_argument('-s', '--seed', type=int, default=42, help="The seed to use for generating.")
parser.add_argument('-e', '--equation', type=str, default='gas_dynamics', choices=['gas_dynamics', 'brusselator', 'wave', 'kuramoto_sivashinsky'], help="The equation to be solved.")

args = parser.parse_args()


logger.info(f"Starting to generate data.")
logger.info(f"The selected equation will be solved {args.num} times.")

gin.parse_config_file("config/generator/default.gin") # can be changed 
gin.finalize()


logger.info(f"Generating data for the {args.equation} equation.")
eq_path = f"data/{args.equation}"
num_existing_equations = len(os.listdir(eq_path)) if os.path.exists(eq_path) else 0
for i in range(args.num):
    seed = num_existing_equations + i + args.seed
    np.random.seed(seed)
    solver = PDESolver(args.equation, f"data/{args.equation}")
    solver.solve()
logger.info(f"Finished generating data.")