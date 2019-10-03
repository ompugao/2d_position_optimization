import sys
sys.path.append('..')
from main_sympy import *
import data_reader

if __name__ == '__main__':
    d = data_reader.read('./unfreezed_rawdistances.log')
    mapping = {}
    mapping[1] = 0
    mapping[2] = 1
    mapping[3] = 2
    mapping[4] = 3
    mapping[5] = 4
    mapping[6] = 5
    mapping[7] = 6
    mapping[8] = 7

    mapping_inv = {v:k for k, v in mapping.items()}

    optimizer = PositionsOptimizer(8, OptimizationMode.THREE_DIMENSION)
    for i in range(1, 9):
        for j in range(i+1, 9):
            if i in d and j in d[i]:
                optimizer.set_measured_distance(mapping[i], mapping[j], np.median(d[i][j]))

    for id in mapping:
        optimizer.set_height(mapping[id], 0.8)
    optimizer.set_initial_guess(mapping[1], 0.0, 0.0)
    optimizer.set_initial_guess(mapping[2], 20.0, 0.0)
    optimizer.set_initial_guess(mapping[3], 0.0, 10.0)
    optimizer.set_initial_guess(mapping[4], 20.0, 10.0)
    optimizer.set_initial_guess(mapping[5], 40.0, 0.0)
    optimizer.set_initial_guess(mapping[6], 60.0, 0.0)
    optimizer.set_initial_guess(mapping[7], 40.0, 10.0)
    optimizer.set_initial_guess(mapping[8], 60.0, 10.0)
    positions, initial_guess, heights = optimizer.optimize()
    optimizer.plot(positions, initial_guess, heights)


    lp = LocalizationProcessor(positions, mapping)

    pos = np.zeros(3)
    for d2 in data_reader.iterative_read('./freezed_rawdistances.log'):
        print("=============")
        print(d2)
        print("------")
        pos = lp.process(d2, pos)
        lp.plot(d2, pos, fig=lp.fig)
    from IPython.terminal import embed; ipshell=embed.InteractiveShellEmbed(config=embed.load_default_config())(local_ns=locals())


