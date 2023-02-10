#!/bin/sh
#!/bin/sh

#MODEL_LIST=(persistence point_gnn point_net point_transformer gat gcn feast)
MODEL_LIST=(difference point_gnn point_net point_transformer gat gcn feast)
EQUATIONS=(wave gas_dynamics brusselator kuramoto_sivashinsky)

for MODEL in ${MODEL_LIST[*]}
do
for EQ in ${EQUATIONS[*]}
do 
python main.py equation=$EQ model=$MODEL support=cloud num_points=low task=evolution training_noise=0.1
done
done