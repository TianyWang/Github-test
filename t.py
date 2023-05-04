import numpy as np

for i in range(10): 
    fix_seed = 3407
    # np.random.seed(fix_seed)
    print(np.random.randn(1, 3))
    print(np.random.randn(1, 2))