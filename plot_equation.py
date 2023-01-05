from src.generator.equations import GasDynamicsPDE, BrusselatorPDE, WavePDE, KuramotoSivashinskyPDE
from src.generator.solver import PDESolver
import numpy as np
import sys
import argparse

np.random.seed(42)
parser = argparse.ArgumentParser(
            prog = 'Data Generator',
            description = 'This program generates data for the DynaBench Dataset by solving each of the equations \
                            Gas-Dynamics, Brusselator, Wave, Kuramoto-Sivashinksy several times and postprocessing the generated solutions.',
            epilog = 'Thank you!')

parser.add_argument('-e', '--equation', type=str, default='wave', help="Equation to plot.", 
                    choices=['wave', 'gas_dynamics', 'brusselator', 'kuramoto_sivashinsky'])
args = parser.parse_args()


if args.equation == "wave":
    eq = WavePDE()
elif args.equation == "gas_dynamics":
    eq = GasDynamicsPDE()
elif args.equation == "brusselator":
    eq = BrusselatorPDE()
else:
    eq = KuramotoSivashinskyPDE()

solver = PDESolver(eq, f"data/plots/args.equation", grid_size=64, t_range=100, save_interval=0.5)
solver.solve()
solver.make_movie('equation_example_wave.gif')
