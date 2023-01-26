#!/bin/sh
python -m venv venv2
source venv/bin/activate
pip install -r requirements.txt
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu116.html