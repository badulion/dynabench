from src.dataset import DynaBenchGraph

ds = DynaBenchGraph(equation="wave")

x, y, p = ds[0]
print(x)