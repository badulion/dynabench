#!/bin/sh

# run baselines
python baseline.py -e brusselator -b zero -t evolution -s high
python baseline.py -e kuramoto_sivashinsky -b zero -t evolution -s high
python baseline.py -e gas_dynamics -b zero -t evolution -s high
python baseline.py -e wave -b zero -t evolution -s high

python baseline.py -e brusselator -b persistence -t evolution -s high
python baseline.py -e kuramoto_sivashinsky -b persistence -t evolution -s high
python baseline.py -e gas_dynamics -b persistence -t evolution -s high
python baseline.py -e wave -b persistence -t evolution -s high

# run models
python main.py -e gas_dynamics -m point_net -t evolution -s high
python main.py -e gas_dynamics -m point_gnn -t evolution -s high
python main.py -e gas_dynamics -m point_transformer -t evolution -s high
python main.py -e gas_dynamics -m gcn -t evolution -s high
python main.py -e gas_dynamics -m gat -t evolution -s high
python main.py -e gas_dynamics -m feast -t evolution -s high

python main.py -e wave -m point_net -t evolution -s high
python main.py -e wave -m point_gnn -t evolution -s high
python main.py -e wave -m point_transformer -t evolution -s high
python main.py -e wave -m gcn -t evolution -s high
python main.py -e wave -m gat -t evolution -s high
python main.py -e wave -m feast -t evolution -s high

python main.py -e kuramoto_sivashinsky -m point_net -t evolution -s high
python main.py -e kuramoto_sivashinsky -m point_gnn -t evolution -s high
python main.py -e kuramoto_sivashinsky -m point_transformer -t evolution -s high
python main.py -e kuramoto_sivashinsky -m gcn -t evolution -s high
python main.py -e kuramoto_sivashinsky -m gat -t evolution -s high
python main.py -e kuramoto_sivashinsky -m feast -t evolution -s high

python main.py -e brusselator -m point_net -t evolution -s high
python main.py -e brusselator -m point_gnn -t evolution -s high
python main.py -e brusselator -m point_transformer -t evolution -s high
python main.py -e brusselator -m gcn -t evolution -s high
python main.py -e brusselator -m gat -t evolution -s high
python main.py -e brusselator -m feast -t evolution -s high
