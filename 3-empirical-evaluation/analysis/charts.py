# ~\~ language=Python filename=analysis/charts.py
# ~\~ begin <<lit/main.md|analysis/charts.py>>[0]
# ~\~ begin <<lit/main.md|preprocessing>>[0]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import json
import os
import time
from pandas.core.tools.numeric import to_numeric
from subprocess import run, PIPE
from itertools import product, chain

# pipe the instance generator into the solver

algs = ["bf", "bb", "dpc", "dpw", "redux"]
data = []

def generate(**kwargs):
    res = []
    kwargs["granularity"] = kwargs["granularity_and_light_heavy_balance"][0]
    kwargs["light_heavy_balance"] = kwargs["granularity_and_light_heavy_balance"][1]
    for seed in range(42, 42 + kwargs["n_runs"]):
        params = dict({
            "seed": seed,
            "n_instances": 1,
        }, **kwargs)
        # run the instance generator
        instance = dict({"contents": os.popen(
            "gen/kg2 \
            -r {seed} \
            -n {n_items} \
            -N {n_instances} \
            -W {max_weight} \
            -C {max_cost} \
            -k {granularity} \
            -w {light_heavy_balance} \
            -c {cost_weight_correlation} \
            -m {capacity_weight_sum_ratio} \
            ".format(**params)
        ).read()}, **params)
        # TODO: permute the order of items in the instance
        res.append(instance)
    return res

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

    # return only the cost of the solution
    return int(solver.stdout.split()[0])

# enumerate the parameter values of a dataset for instance generation and
# algorithm benchmarking.
def dataset(id, **kwargs):
    params = dict({
        # defaults
        "id": [id],
        "alg": algs,
        "n_runs": [1],
        "n_items": [27],
        "max_weight": [5000],
        "max_cost": [5000],
        "granularity_and_light_heavy_balance": [(1, "bal")],
        "capacity_weight_sum_ratio": [0.8],
        "cost_weight_correlation": ["uni"],
    }, **kwargs)

    key_order = [k for k in params]
    cartesian = list(product(
        *[params[key] for key in key_order]
    ))

    return {
        key: [row[key_order.index(key)] for row in cartesian] for key in params
    }

def merge_datasets(*dss):
    return {
        k: chain(*(ds[k] for ds in dss))
        for k in dss[0]
    }


n_samples = 2 # FIXME

# benchmark configurations
# we don't want a full cartesian product (too slow to fully explore), so we'll
# use a union of subsets, each tailored to the particular algorithm
configs = merge_datasets(dataset(
#     "weight range",
#     alg = ["bf", "dpw"],
#     max_weight = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
# ), dataset(
#     "cost range",
#     alg = ["bf", "dpc"],
#     max_cost = [500, 1000, 5000, 10000, 50000, 100000, 500000],
# ), dataset(
#     "n_items range",
#     n_items = [4, 10, 15, 20, 25, 28],
# ), dataset(
    "granularity exploration",
    alg = ["bb", "dpc", "dpw", "redux"],
    n_runs = [n_samples],
    granularity_and_light_heavy_balance = [
        (1, "light"), (2, "light"), (3, "light"), (1, "heavy"), (2, "heavy"), (3, "heavy")
    ],
), dataset(
    "capacity weight sum ratio exploration",
    alg = ["bb", "dpc", "dpw", "redux"],
    n_runs = [n_samples],
    capacity_weight_sum_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
))

for config in [dict(zip(configs, v)) for v in zip(*configs.values())]:
    param_iter = iter(config.values())
    next(param_iter) # skip id
    print("config in set", config["id"], "\tparams", *param_iter)

    for inst in generate(**config):
        # measure the time taken by the call to the solver
        start = time.time()
        cost = solve(config["alg"], inst)
        end = time.time()
        data.append(dict(inst,
            cost = cost,
            alg = config["alg"],
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
    granularity = "Granularita",
    light_heavy_balance = "Rozložení váhy předmětů",
    cost = "Cena řešení",
    capacity_weight_sum_ratio = "Poměr kapacity a součtu vah",
)

alg_labels = dict(
    bf = "Brute force",
    bb = "Branch & bound",
    dpc = "Dynamic programming (cost)",
    dpw = "Dynamic programming (weight)",
    redux = "Greedy redux",
)

def plot(x_axis, y_axis, id, title, data = data, filename = None):
    if filename is None:
        filename = id
    print("\t{}".format(title))
    fig, ax = plt.subplots(figsize = figsize)
    # for alg in algs:
    ds = [d for d in data if d["id"] == id] #and d["alg"] == alg]
    # create a frame from the list
    df = pd.DataFrame(ds)

    # do a violin plot grouped by the algorithm name
    sns.violinplot(x = x_axis, y = y_axis, data = df, hue = "alg", ax = ax)
    plt.title(title)
    plt.xlabel(plot_labels[x_axis])
    plt.ylabel(plot_labels[y_axis])

            # plt.boxplot(
            #     [(d[x_axis], d[y_axis]) for d in ds],
            #     "o",
            #     label = alg_labels[alg]
            # )

    plt.legend()
    plt.savefig("docs/assets/{}.svg".format(filename))

print("rendering plots")
# plot("n_items",     "t", "n_items range",           "Průměrná doba běhu vzhledem k velikosti instance")
# plot("max_weight",  "t", "weight range",            "Průměrná doba běhu vzhledem k maximální váze")
# plot("max_cost",    "t", "cost range",              "Průměrná doba běhu vzhledem k maximální ceně")

# TODO: do this properly, which means
# - do like a violin plot or at least a box plot
# - take the light/heavy balance into account

for balance in ["light", "heavy"]:
    plot(
        "granularity",
        "t",
        "granularity exploration",
        "Doba běhu vzhledem ke granularitě (preference {})".format(balance),
        data = [d for d in data if d["light_heavy_balance"] == balance],
        filename = "granularity exploration {}".format(balance),
    )

plot(
    "capacity_weight_sum_ratio",
    "t",
    "capacity weight sum ratio exploration",
    "Doba běhu vzhledem k poměru kapacity a součtu vah",
)

# ~\~ end
# ~\~ end
