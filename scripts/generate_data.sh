#!/bin/sh

python generate.py -n 30 -e gas_dynamics 
python generate.py -n 30 -e wave 
python generate.py -n 30 -e brusselator
python generate.py -n 30 -e kuramoto_sivashinsky