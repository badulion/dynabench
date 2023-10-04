#python generate.py -m equation=advection,burgers,gas_dynamics,kuramoto_sivashinsky,reaction_diffusion,wave num_simulations=10 split=val,test
#python generate.py -m equation=advection,burgers,gas_dynamics,kuramoto_sivashinsky,reaction_diffusion,wave num_simulations=50 split=train
python generate.py -m equation=burgers,gas_dynamics,kuramoto_sivashinsky,reaction_diffusion,wave num_simulations=10 split=val,test start_from=10
python generate.py -m equation=advection,burgers,gas_dynamics,kuramoto_sivashinsky,reaction_diffusion,wave num_simulations=50 split=train start_from=50