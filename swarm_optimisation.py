import numpy as np

def swarm_opt(x, func, scale=0.1, n_cand=10, n_iter=50, random_state=0):
    rng = np.random.RandomState(0)
    cov = scale * np.eye(len(x))

    for i in range(n_iter):

        x_cands = rng.multivariate_normal(x, cov, n_cand)
        func_values = [func(x_cand) for x_cand in x_cands]
        min_arg = np.argmin(func_values)
        x_new = x_cands[min_arg]

        old_best = func(x)
        new_best = func(x_new)
        if new_best > old_best:
            break
        x = x_new[:]
        print("Iter %d" % i)
        print(func(x))
    return x, func(x)
