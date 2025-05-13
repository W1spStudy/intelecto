import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Benchmark functions


def himmelblau(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))


def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2


def eggholder(x, y):
    return -(y + 47) * np.sin(np.sqrt(abs(x/2 + (y + 47)))) - x * np.sin(np.sqrt(abs(x - (y + 47))))

# Bat Algorithm implementation


class BatAlgorithm:
    def __init__(self, func, bounds, n_bats=30, n_iter=100, A=0.5, r=0.5, Qmin=0, Qmax=2):
        self.func = func
        self.bounds = bounds  # [(xmin, xmax), (ymin, ymax)]
        self.n_bats = n_bats
        self.n_iter = n_iter
        self.A = A  # loudness
        self.r = r  # pulse rate
        self.Qmin = Qmin
        self.Qmax = Qmax

        self.dim = 2
        self.Q = np.zeros(n_bats)  # frequency
        self.v = np.zeros((n_bats, self.dim))  # velocity
        self.S = np.zeros((n_bats, self.dim))  # positions

        # Initialize positions
        for i in range(self.dim):
            self.S[:, i] = np.random.uniform(
                bounds[i][0], bounds[i][1], n_bats)

        self.fitness = np.array(
            [self.func(self.S[i, 0], self.S[i, 1]) for i in range(n_bats)])
        self.best = self.S[self.fitness.argmin()]
        self.best_fitness = self.fitness.min()

    def simple_bounds(self, s):
        s_new = np.copy(s)
        for i in range(self.dim):
            s_new[:, i] = np.clip(
                s_new[:, i], self.bounds[i][0], self.bounds[i][1])
        return s_new

    def run(self):
        for t in range(self.n_iter):
            self.Q = self.Qmin + (self.Qmax - self.Qmin) * \
                np.random.rand(self.n_bats)
            self.v += (self.S - self.best) * self.Q[:, None]
            S_new = self.S + self.v

            # Apply bounds
            S_new = self.simple_bounds(S_new)

            # Pulse rate condition
            rand = np.random.rand(self.n_bats)
            mask = rand > self.r
            eps = np.random.uniform(-1, 1, (self.n_bats, self.dim))
            S_new[mask] = self.best + eps[mask] * self.A

            # Evaluate new solutions
            Fnew = np.array([self.func(S_new[i, 0], S_new[i, 1])
                            for i in range(self.n_bats)])

            # Update if better and if loudness condition met
            improved = (Fnew <= self.fitness) & (
                np.random.rand(self.n_bats) < self.A)
            self.S[improved] = S_new[improved]
            self.fitness[improved] = Fnew[improved]

            # Update the current best
            if self.fitness.min() < self.best_fitness:
                self.best = self.S[self.fitness.argmin()]
                self.best_fitness = self.fitness.min()

            yield self.S, self.best, self.best_fitness

# Visualization function


def visualize_bat_algorithm(func, bounds, n_iter=100, title="Bat Algorithm Optimization"):
    bat_algo = BatAlgorithm(func, bounds, n_iter=n_iter)

    # Create meshgrid for contour plot
    x = np.linspace(bounds[0][0], bounds[0][1], 200)
    y = np.linspace(bounds[1][0], bounds[1][1], 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    fig.colorbar(contour)
    scat = ax.scatter([], [], c='red', s=30, label='Bats')
    best_point = ax.scatter([], [], c='yellow', s=100,
                            marker='*', label='Best')

    ax.set_title(title)
    ax.set_xlim(bounds[0])
    ax.set_ylim(bounds[1])
    ax.legend()

    def update(frame):
        positions, best, best_fitness = frame
        scat.set_offsets(positions)
        best_point.set_offsets(best)
        ax.set_xlabel(f"Best fitness: {best_fitness:.4f}")
        return scat, best_point

    anim = FuncAnimation(fig, update, frames=bat_algo.run(),
                         interval=100, repeat=False)
    plt.show()


# Run visualizations for the required functions
if __name__ == "__main__":
    functions = [
        (himmelblau, [(-5, 5), (-5, 5)], "Himmelblau Function"),
        (rastrigin, [(-5.12, 5.12), (-5.12, 5.12)], "Rastrigin Function"),
        (rosenbrock, [(-2, 2), (-1, 3)], "Rosenbrock Function"),
        (eggholder, [(-512, 512), (-512, 512)], "Eggholder Function"),
    ]

    for func, bounds, name in functions:
        print(f"Running Bat Algorithm on {name}")
        visualize_bat_algorithm(func, bounds, n_iter=100,
                                title=f"Bat Algorithm Optimization - {name}")
