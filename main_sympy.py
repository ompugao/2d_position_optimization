import nlopt
import sympy
import numpy as np
import matplotlib.pyplot as plt
import time

num_beacons = 6
x_op = np.array(sympy.symbols('x:2:%d'%num_beacons)).reshape(2, num_beacons)
distances = []
distances.append((0, 1, 3))
distances.append((0, 2, 1))
distances.append((1, 2, 3))

distances.append((0, 3, 4))
distances.append((2, 3, 4))
distances.append((0, 4, 3))
distances.append((1, 4, 3.2))
distances.append((1, 4, 3.2))
distances.append((1, 5, 10))
distances.append((4, 5, 10))

cost_op = 0
for m in distances:
    id1, id2, dist = m
    cost_op += (np.sum((x_op[:,id1] - x_op[:,id2])**2) - dist**2)**2

def cost(x, grad):
    mapping = {}
    for i in range(num_beacons):
        mapping[x_op[0, i]] = x[i]
        mapping[x_op[1, i]] = x[i+num_beacons]

    if grad.size > 0:
        for i in range(num_beacons):
            # dx
            grad[i] = sympy.diff(cost_op, x_op[0, i]).subs(mapping)
            # dy 
            grad[i+num_beacons] = sympy.diff(cost_op, x_op[1, i]).subs(mapping)

    cost = cost_op.subs(mapping)
    return float(cost)

#opt = nlopt.opt(nlopt.LD_MMA, num_beacons*2)
opt = nlopt.opt(nlopt.LN_NELDERMEAD, num_beacons*2)
# opt.set_lower_bounds(np.ones(num_beacons*2)*(-10.0))
# opt.set_upper_bounds(np.ones(num_beacons*2)*10.0)
opt.set_min_objective(cost)
opt.set_xtol_rel(0.001)
starttime = time.time()
print('optimization start!')
optimum = opt.optimize(np.random.random(num_beacons*2).tolist())
print('time elapsed: %s'%(time.time()-starttime))
minf = opt.last_optimum_value()
print('optimum: ', optimum)
print('minf: ', minf)

listpos = optimum.reshape(2, num_beacons)

plt.plot(listpos[0, :], listpos[1, :], 'x')
for m in distances:
    id1, id2, dist = m
    print('%s should be equal to %s'%(np.hypot(*(listpos[:, id1] - listpos[:, id2])), dist))
    plt.plot([listpos[0, id1], listpos[0, id2]], [listpos[1, id1], listpos[1, id2]], '-')
plt.show()

from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

