from src.dataset import DynaBenchBase, DynaBenchGraph

ds = DynaBenchGraph(equation="wave", lookback=8, rollout=16, support="cloud", task="forecast")

x, y, p = ds[0]

print(x)