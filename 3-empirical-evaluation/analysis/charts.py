# ~\~ language=Python filename=analysis/charts.py
# ~\~ begin <<lit/main.md|analysis/charts.py>>[0]
# ~\~ begin <<lit/main.md|preprocessing>>[0]
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
import time
from pandas.core.tools.numeric import to_numeric
from subprocess import run, PIPE
from itertools import product

# pipe the instance generator into the solver

algs = ["bf", "bb", "dpc", "dpw", "redux"]

def generate(n, w, c):
    params = {
        "seed": 42,
        "n_items": n,
        "n_instances": 1,
        "max_weight": w,
        "max_cost": c,
    }
    # run the instance generator
    instance = dict({"contents": os.popen(
        "gen/kg2 \
        -r {seed} \
        -n {n_items} \
        -N {n_instances} \
        -W {max_weight} \
        -C {max_cost}".format(**params)
    ).read()}, **params)
    # TODO: permute the order of items in the instance
    return instance

def solve(alg, instance):
    solver = run(
        ["target/release/main", alg],
        stdout = PIPE,
        stderr = PIPE,
        input = instance["contents"],
        encoding = "ascii",
        cwd = "solver/"
    )
    if solver.returncode != 0:
        print(solver)
        raise Exception("solver failed")

    # return only the first number, i.e. the cost of the solution
    # convert to an integer
    return int(solver.stdout.split()[0])

data = []

# enumerate the parameter values of a dataset for instance generation and
# algorithm benchmarking.
def dataset(id, **kwargs):
    params = dict({
        # defaults
        "algs": algs,
        "n_items": [28],
        "max_weight": [5000],
        "max_cost": [5000],
    }, **kwargs)

    return list(product(
        [id],
        params["algs"],
        params["n_items"],
        params["max_weight"],
        params["max_cost"],
    ))

# benchmark configurations
# we don't want a full cartesian product (too slow to fully explore), so we'll
# use a union of subsets, each tailored to the particular algorithm
configs = dataset(
    "weight range",
    algs = ["bf", "dpw"],
    max_weight = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
) + dataset(
    "cost range",
    algs = ["bf", "dpc"],
    max_cost = [500, 1000, 5000, 10000, 50000, 100000, 500000],
) + dataset(
    "n_items range",
    n_items = [4, 10, 15, 20, 25, 28],
)

for (id, alg, n, w, c) in configs:
    print("config in set", id, "\tparams", alg, n, w, c)

    inst = generate(n, w, c)
    # measure the time taken by the call to the solver
    start = time.time()
    cost = solve(alg, inst)
    end = time.time()
    data.append(dict(inst,
        id = id,
        cost = cost,
        alg = alg,
        t = end - start,
        contents = None
    ))


# ~\~ end

# ~\~ begin <<lit/main.md|performance-chart>>[0]

# plot the mean runtimes and max errors

figsize = (14, 8)

plot_labels = dict(
    seed = "Seed",
    t = "Doba běhu (sec)",
    n_items = "Velikost instance",
    max_cost = "Maximální cena",
    max_weight = "Maximální váha",
    n_instances = "Počet instancí v zadání",
)

alg_labels = dict(
    bf = "Brute force",
    bb = "Branch & bound",
    dpc = "Dynamic programming (cost)",
    dpw = "Dynamic programming (weight)",
    redux = "Greedy redux",
)

def plot(x_axis, y_axis, id, title):
    fig, ax = plt.subplots(figsize = figsize)
    plt.title(title)
    plt.xlabel(plot_labels[x_axis])
    plt.ylabel(plot_labels[y_axis])
    for alg in algs:
        ds = [d for d in data if d["id"] == id and d["alg"] == alg]
        if len(ds) != 0:
            plt.plot(
                [d[x_axis] for d in ds],
                [d[y_axis] for d in ds],
                "--o",
                label = alg_labels[alg]
            )

    plt.legend()
    plt.savefig("docs/assets/{}.svg".format(id))

print("rendering plots")
plot("n_items",    "t", "n_items range", "Průměrná doba běhu vzhledem k velikosti instance")
plot("max_weight", "t", "weight range",  "Průměrná doba běhu vzhledem k maximální váze")
plot("max_cost",   "t", "cost range",    "Průměrná doba běhu vzhledem k maximální ceně")

    # fig, ax = plt.subplots(figsize = figsize)
    # plt.title("Závislost maximální chyby na velikosti instance")
    # plt.xlabel("Velikost instance")
    # plt.ylabel("Maximální chyba")
    # plt.xticks(n_values)
    # yticks = np.append(ax.get_yticks(), [0.1, 0.01])
    # ax.set_yticks(yticks)
    # ax.grid(linestyle = "dotted")
    # for alg in algs:
    #     plt.plot([n for n in data[s][alg]], [data[s][alg][n]["error"]["max"] for n in data[s][alg]], label=alg)
    # plt.legend()
    # plt.savefig("docs/assets/{}_max_errors.svg".format(s))
# ~\~ end
# ~\~ end
