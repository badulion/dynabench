from src.generator.equations import GasDynamicsPDE, BrusselatorPDE, WavePDE, KuramotoSivashinskyPDE
from src.generator.solver import PDESolver
import numpy as np
import sys
import argparse
from src.utils.logging import create_logger

np.random.seed(42)
logger = create_logger(__name__)
parser = argparse.ArgumentParser(
            prog = 'Data Generator',
            description = 'This program generates data for the DynaBench Dataset by solving each of the equations \
                            Gas-Dynamics, Brusselator, Wave, Kuramoto-Sivashinksy several times and postprocessing the generated solutions.',
            epilog = 'Thank you!')

parser.add_argument('-n', '--num', type=int, default=1, help="Number of times each equation is simulated.")
args = parser.parse_args()


logger.info(f"Starting to generate equations.")
logger.info(f"Each equation will be solved {args.num} times.")


logger.info("Generating data for the Gas Dynamics equation.")
for i in range(args.num):
    eq = GasDynamicsPDE()
    solver = PDESolver(eq, "data/gas_dynamics", grid_size=64, t_range=100)
    solver.solve()


logger.info("Generating data for the Brusslator equation.")
for i in range(args.num):
    eq = BrusselatorPDE()
    solver = PDESolver(eq, "data/brusselator", grid_size=64, t_range=100)
    solver.solve()


logger.info("Generating data for the Wave equation.")
for i in range(args.num):
    eq = WavePDE()
    solver = PDESolver(eq, "data/wave", grid_size=64, t_range=100)
    solver.solve()


logger.info("Generating data for the Kuramoto-Sivashinsky equation.")
for i in range(args.num):
    eq = KuramotoSivashinskyPDE()
    solver = PDESolver(eq, "data/kuramoto_sivashinsky", grid_size=64, t_range=100)
    solver.solve()

logger.info(f"Finished generating equations.")