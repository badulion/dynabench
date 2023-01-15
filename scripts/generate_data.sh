#!/bin/sh

python gen.py -n 30 -e gas_dynamics 
python gen.py -n 30 -e wave 
python gen.py -n 30 -e brusselator
python gen.py -n 30 -e kuramoto_sivashinsky