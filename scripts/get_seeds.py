import numpy as np

initial_seed = 42

gen = np.random.default_rng(42)

seeds = gen.choice(range(1000000, 10000000), 7000, replace=False)

np.savetxt("config/seeds/train.txt", seeds[:5000], fmt="%d")
np.savetxt("config/seeds/val.txt", seeds[5000:6000], fmt="%d")
np.savetxt("config/seeds/test.txt", seeds[6000:7000], fmt="%d")