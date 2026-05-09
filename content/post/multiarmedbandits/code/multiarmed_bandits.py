import matplotlib.pyplot as plt
import numpy as np

"""
This module contains the implementation of the epsilon-greedy algorithm for the multi-armed bandit problem.
"""


class Bandit:
    def __init__(self, m):
        self.m = m
        self.mean = 0
        self.N = 0
        self.means = []

    def pull(self):
        """Pull the bandit arm and return a random sample from a normal distribution with mean m."""
        return np.random.randn() + self.m

    def update(self, x):
        """Update the mean of the bandit with the new value x."""
        self.N += 1
        self.means.append(self.mean)
        self.mean = (1 - 1.0 / self.N) * self.mean + 1.0 / self.N * x

    def __str__(self):
        return (
            "Bandit with true mean: "
            + str(self.m)
            + " and estimated mean: "
            + str(self.mean)
        )


def run_experiment(bandit_means, eps, N, plot=False, rule="epsilon"):
    """Run an experiment with the epsilon-greedy algorithm.

    Args:
    m1, m2, m3: float - true means of the bandits
    eps: float - probability of exploration
    N: int - number of iterations
    """

    bandits = [Bandit(m) for m in bandit_means]

    data = np.empty(N)
    choices = np.empty(N)

    for i in range(N):
        if rule == "epsilon":
            eps = 1.0 / (i + 1)
        elif rule == "rate":
            eps = 2.0 * np.log(i + 1) / (i + 1)
            # print(eps)

        # epsilon greedy
        p = np.random.random()
        if p < eps or i == 0:
            j = np.random.choice(len(bandits))
        else:
            j = np.argmax([b.mean for b in bandits])
        choices[i] = j
        x = bandits[j].pull()
        bandits[j].update(x)
        # for the plot
        data[i] = x

    if plot:
        cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

        ax = plt.subplot(211)

        # plot moving average ctr
        ax.plot(cumulative_average, linestyle="-", linewidth=5, alpha=0.5)
        for m in bandit_means:
            ax.plot(np.ones(N) * m, alpha=0.15)

        ax.set_xscale("log")
        ax.set_title(f"sum: {np.sum(data)}")

        # add a subplot below with the choices
        ax = plt.subplot(212)
        ax.plot(choices, alpha=0.5, color="k")
        c = [bandits[int(i)].m for i in choices]
        s = ax.scatter(np.arange(N), choices, c=c, alpha=0.5)
        ax.set_xlim(0, N)
        ax.set_ylim(0, len(bandits) - 1)
        plt.colorbar(s)
        plt.show()

    return bandits, data
