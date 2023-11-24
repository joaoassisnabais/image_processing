import numpy as np 

a = [0, 1, 3]
b = np.random.randint(0, 20, size=(2, 5)).T
print(b)
b = b[a]
print(b)