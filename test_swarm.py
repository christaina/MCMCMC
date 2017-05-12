from swarm_optimisation import swarm_opt
import math

def x_squared(x):
    return x[0] ** 2

def sin(x):
    return math.sin(x[0])

print(swarm_opt([5], x_squared, n_iter=20, n_cand=20))
print(swarm_opt([5], sin, n_iter=20, n_cand=20))
