import matplotlib.pyplot as plt
import numpy as np


def value_function_policy_plot(V, policy, env_map):
    plt.figure(figsize=(7, 7))
    plt.imshow(V, cmap='viridis', interpolation='none')  # , clim=(0, 1)
    ax = plt.gca()
    ax.set_xticks(np.arange(V.shape[0]) - .5)
    ax.set_yticks(np.arange(V.shape[1]) - .5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for y in range(V.shape[0]):
        for x in range(V.shape[1]):
            plt.text(x-0.25, y+0.35, format(V[y, x], '.2f'),
                     color='white', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            plt.text(x - 0.25, y - 0.25, str(env_map[y][x]),
                     color='white', size=12, verticalalignment='center',
                     horizontalalignment='center', fontweight='bold')
            for i, prob in enumerate(policy[y, x]):
                dx = 0.0
                dy = 0.0
                if i == 0:  # Up
                    dy = -prob / 3.0
                elif i == 1:  # Right
                    dx = prob / 3.0
                elif i == 2:  # Down
                    dy = prob / 3.0
                elif i == 3:  # Left
                    dx = -prob / 3.0
                if dx == 0.0 and dy == 0.0:
                    pass
                plt.arrow(x, y, dx, dy, width=0.01, color='black')

    plt.grid(color='black', lw=1, ls='-')
    plt.colorbar()
    plt.show()


def estimate_policy_array_from_samples(model):
    policy = np.zeros((4, 4, 4))
    for x in range(4):
        for y in range(4):
            agent_actions = [model.action([x, y]) for _ in range(50)]
            uniques, counts = np.unique(agent_actions, return_counts=True)

            for (u, c) in zip(uniques, counts):
                policy[x, y, u] = c / 50

    return policy
