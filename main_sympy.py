import nlopt
import sympy
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import time

class OptimizationException(Exception):
    pass

class OptimizationMode(object):
    TWO_DIMENSION = 0
    THREE_DIMENSION = 1

class Constraint(object):
    def __init__(self, num_beacons, *args, **kwargs):
        self.num_beacons = num_beacons

    def __call__(self, opt, op):
        pass

class ConstraintX(Constraint):
    def __init__(self, num_beacons, id, x):
        super().__init__(num_beacons)
        self.id = id
        self.x = x

    def __call__(self, opt, x_op):
        def cost(x, grad):
            if grad.size > 0:
                for g in grad:
                    g = 0
                grad[i] = 2 * (x[i] - self.x)

            return (float(x[i]) - self.x)**2
        opt.add_equality_constraint(cost)

class ConstraintY(Constraint):
    def __init__(self, num_beacons, id, y):
        super().__init__(num_beacons)
        self.id = id
        self.y = y

    def __call__(self, opt, x_op):
        def cost(x, grad):
            if grad.size > 0:
                for g in grad:
                    g = 0
                grad[i+self.num_beacons] = 2 * (x[i+self.num_beacons] - self.y)

            return (float(x[i+self.num_beacons]) - self.y)**2
        opt.add_equality_constraint(cost)

class PositionsOptimizer(object):
    def __init__(self, num_beacons, mode):
        self.num_beacons = num_beacons
        self.mode = OptimizationMode.TWO_DIMENSION
        self.distances = []
        self.mode = mode
        self.initial_guess = np.zeros((2, num_beacons))
        self.constraints = []
        if self.mode != OptimizationMode.TWO_DIMENSION and self.mode != OptimizationMode.THREE_DIMENSION:
            raise OptimizationException("invalid mode: %s"%(mode))
        if self.mode == OptimizationMode.THREE_DIMENSION:
            self.heights = np.zeros(self.num_beacons)

    def set_measured_distance(self, id1, id2, distance):
        self.distances.append((id1, id2, distance))

    def set_height(self, id, height):
        if self.mode != OptimizationMode.THREE_DIMENSION:
            raise OptimizationException('cannot set height in two dimensional mode')
        if not (0 <= id < self.num_beacons):
            raise OptimizationException('invalid id')
        self.heights[id] = height

    def set_initial_guess(self, id, x, y):
        self.initial_guess[0, id] = x
        self.initial_guess[1, id] = y

    def set_x_constraint(self, id, x):
        self.constraints.append(ConstraintX(self.num_beacons, id, x))

    def set_y_constraint(self, id, y):
        self.constraints.append(ConstraintY(self.num_beacons, id, y))

    def set_position_constraint(self, id, x, y):
        self.constraints.append(ConstraintX(self.num_beacons, id, x))
        self.constraints.append(ConstraintY(self.num_beacons, id, y))

    def optimize(self,):
        x_op = np.array(sympy.symbols('x:2:%d'%self.num_beacons)).reshape(2, self.num_beacons)
        cost_op = 0
        for m in self.distances:
            id1, id2, dist = m
            cost_op += (np.sum((np.hstack([x_op[:,id1], self.heights[id1]]) - np.hstack([x_op[:,id2], self.heights[id2]]))**2) - dist**2)**2

        def cost(x, grad):
            mapping = {}
            for i in range(self.num_beacons):
                mapping[x_op[0, i]] = x[i]
                mapping[x_op[1, i]] = x[i+self.num_beacons]

            if grad.size > 0:
                for i in range(self.num_beacons):
                    # dx
                    grad[i] = sympy.diff(cost_op, x_op[0, i]).subs(mapping)
                    # dy 
                    grad[i+self.num_beacons] = sympy.diff(cost_op, x_op[1, i]).subs(mapping)

            cost = cost_op.subs(mapping)
            return float(cost)

        opt = nlopt.opt(nlopt.LD_MMA, self.num_beacons*2)
        #opt = nlopt.opt(nlopt.LN_NELDERMEAD, self.num_beacons*2)
        # opt.set_lower_bounds(np.ones(num_beacons*2)*(-10.0))
        # opt.set_upper_bounds(np.ones(num_beacons*2)*10.0)
        for c in self.constraints:
            c(opt, x_op)
        opt.set_min_objective(cost)
        opt.set_xtol_rel(0.001)
        starttime = time.time()
        print('optimization start!')
        #optimum = opt.optimize(np.random.random(self.num_beacons*2).tolist())
        optimum = opt.optimize(self.initial_guess.flatten())
        print('time elapsed: %s'%(time.time()-starttime))
        minf = opt.last_optimum_value()
        print('optimum: ', optimum)
        print('minf: ', minf)

        twod_positions = optimum.reshape(2, self.num_beacons)
        threed_positions = np.vstack([twod_positions, self.heights])
        return threed_positions, self.initial_guess, self.heights

    def plot(self, threed_positions, initial_guess, heights):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_aspect('equal')

        xs = threed_positions[0, :]
        ys = threed_positions[1, :]
        zs = threed_positions[2, :]

        initial_xs = initial_guess[0, :]
        initial_ys = initial_guess[1, :]
        initial_zs = self.heights


        max_range = np.array([xs.max()-xs.min(), ys.max()-ys.min(), zs.max()-zs.min()]).max() * 0.5
        max_range *= 1.05

        mid_x = (xs.max()+xs.min()) * 0.5
        mid_y = (ys.max()+ys.min()) * 0.5
        mid_z = (zs.max()+zs.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.scatter(xs, ys, zs)
        ax.scatter(initial_xs, initial_ys, initial_zs)

        for m in self.distances:
            id1, id2, dist = m
            print('%s should be equal to %s'%(np.linalg.norm(threed_positions[:, id1] - threed_positions[:, id2]), dist))
            ax.plot([threed_positions[0, id1], threed_positions[0, id2]], [threed_positions[1, id1], threed_positions[1, id2]], [threed_positions[2, id1], threed_positions[2, id2]], '-')

        plt.show()


if __name__ == '__main__':
    optimizer = PositionsOptimizer(5, OptimizationMode.THREE_DIMENSION)
    optimizer.set_measured_distance(0, 1, np.linalg.norm([5, 1]))
    optimizer.set_measured_distance(0, 2, np.linalg.norm([5, 1]))
    optimizer.set_measured_distance(1, 2, np.linalg.norm([5, 5, 2]))
    optimizer.set_measured_distance(0, 3, np.linalg.norm([5, 6, 2]))
    optimizer.set_measured_distance(2, 3, np.linalg.norm([6, 1]))
    optimizer.set_measured_distance(0, 4, np.linalg.norm([5, 12, 4]))
    optimizer.set_measured_distance(1, 4, np.linalg.norm([5, 7, 5]))
    optimizer.set_measured_distance(3, 4, np.linalg.norm([6, 2]))
    # optimizer.set_position_constraint(0, 0, 0)
    # optimizer.set_x_constraint(1, 0)
    # optimizer.set_y_constraint(2, 0)
    optimizer.set_height(0, 2)
    optimizer.set_height(1, 1)
    optimizer.set_height(2, 3)
    optimizer.set_height(3, 4)
    optimizer.set_height(4, 6)
    optimizer.set_initial_guess(0, 0, 0)
    optimizer.set_initial_guess(1, 3.6, 0.1)
    optimizer.set_initial_guess(2, 0.1, 3.3)
    optimizer.set_initial_guess(3, 3.8, 5.2)
    optimizer.set_initial_guess(4, 13, 2.9)
    positions, initial_guess, heights = optimizer.optimize()
    optimizer.plot(positions, initial_guess, heights)

    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())

"""
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
"""

