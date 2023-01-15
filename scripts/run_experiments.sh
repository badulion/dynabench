#!/bin/sh

# run baselines
python baseline.py -e brusselator -b zero -t forecast -s high
python baseline.py -e kuramoto_sivashinsky -b zero -t forecast -s high
python baseline.py -e gas_dynamics -b zero -t forecast -s high
python baseline.py -e wave -b zero -t forecast -s high

python baseline.py -e brusselator -b persistence -t forecast -s high
python baseline.py -e kuramoto_sivashinsky -b persistence -t forecast -s high
python baseline.py -e gas_dynamics -b persistence -t forecast -s high
python baseline.py -e wave -b persistence -t forecast -s high

# run models
python main.py -e gas_dynamics -m point_net -t forecast -s high
python main.py -e gas_dynamics -m point_gnn -t forecast -s high
python main.py -e gas_dynamics -m point_transformer -t forecast -s high
python main.py -e gas_dynamics -m gcn -t forecast -s high
python main.py -e gas_dynamics -m gat -t forecast -s high
python main.py -e gas_dynamics -m feast -t forecast -s high

python main.py -e wave -m point_net -t forecast -s high
python main.py -e wave -m point_gnn -t forecast -s high
python main.py -e wave -m point_transformer -t forecast -s high
python main.py -e wave -m gcn -t forecast -s high
python main.py -e wave -m gat -t forecast -s high
python main.py -e wave -m feast -t forecast -s high

python main.py -e kuramoto_sivashinsky -m point_net -t forecast -s high
python main.py -e kuramoto_sivashinsky -m point_gnn -t forecast -s high
python main.py -e kuramoto_sivashinsky -m point_transformer -t forecast -s high
python main.py -e kuramoto_sivashinsky -m gcn -t forecast -s high
python main.py -e kuramoto_sivashinsky -m gat -t forecast -s high
python main.py -e kuramoto_sivashinsky -m feast -t forecast -s high

python main.py -e brusselator -m point_net -t forecast -s high
python main.py -e brusselator -m point_gnn -t forecast -s high
python main.py -e brusselator -m point_transformer -t forecast -s high
python main.py -e brusselator -m gcn -t forecast -s high
python main.py -e brusselator -m gat -t forecast -s high
python main.py -e brusselator -m feast -t forecast -s high
