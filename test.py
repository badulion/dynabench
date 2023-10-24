from dynabench.dataset import DynabenchIterator



ds = DynabenchIterator(equation="advection")
x, y, p = ds[0]

print(p)