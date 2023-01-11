from src.generator.equations import GasDynamicsPDE, BrusselatorPDE, WavePDE, KuramotoSivashinskyPDE
from src.generator.solver import PDESolver
import numpy as np
import sys
import argparse
from src.utils.logging import create_logger

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
np.random.seed(args.seed)


logger.info(f"Starting to generate data.")
logger.info(f"The selected equation will be solved {args.num} times.")

if args.equation == 'gas_dynamics':
    equationModule = GasDynamicsPDE
elif args.equation == 'brusselator':
    equationModule = BrusselatorPDE
elif args.equation == 'wave':
    equationModule = WavePDE
else:
    equationModule = KuramotoSivashinskyPDE



logger.info(f"Generating data for the {args.equation} equation.")
for i in range(args.num):
    eq = equationModule()
    solver = PDESolver(eq, f"data/{args.equation}", grid_size=64, t_range=100)
    solver.solve()
logger.info(f"Finished generating data.")