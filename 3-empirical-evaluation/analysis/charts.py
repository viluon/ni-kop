# ~\~ language=Python filename=analysis/charts.py
# ~\~ begin <<lit/main.md|analysis/charts.py>>[0]
# ~\~ begin <<lit/main.md|preprocessing>>[0]
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as st
import json
import os
import time
from pandas.core.tools.numeric import to_numeric
from subprocess import run, PIPE
from itertools import product, chain
import textwrap as tr

# pipe the instance generator into the solver

algs = ["bf", "bb", "dpc", "dpw", "redux"]
data = []

# adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress_bar(iteration, total, length = 60):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '=' * filledLength + ' ' * (length - filledLength)
    print(f'\r[{bar}] {percent}%', end = "\r")
    if iteration == total:
        print()

def generate(**kwargs):
    res = []
    kwargs["granularity"] = kwargs["granularity_and_light_heavy_balance"][0]
    kwargs["light_heavy_balance"] = kwargs["granularity_and_light_heavy_balance"][1]
    for seed in range(kwargs["seed"], kwargs["seed"] + kwargs["n_runs"]):
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

        for p in range(0, instance["n_permutations"]):
            kg_perm = run(
                "gen/kg_perm \
                -d 0 \
                -N 1 \
                -r {} \
                ".format(p).split(),
                stdout = PIPE,
                stderr = PIPE,
                input = instance["contents"],
                encoding = "ascii",
            )

            res.append(dict({
                "contents": kg_perm.stdout,
                "perm_id": p,
            }, **instance))

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
        "seed": [42],
        "n_runs": [1],
        "n_permutations": [1],
        "n_repetitions": [1],
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
        k: list(chain(*(ds[k] for ds in dss)))
        for k in dss[0]
    }


n_samples = 2 # FIXME

# benchmark configurations
# we don't want a full cartesian product (too slow to fully explore), so we'll
# use a union of subsets, each tailored to the particular algorithm
configs = merge_datasets(dataset(
    "weight range",
    alg = ["bf", "dpw"],
    max_weight = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
), dataset(
    "cost range",
    alg = ["bf", "dpc"],
    max_cost = [500, 1000, 5000, 10000, 50000, 100000, 500000],
), dataset(
    "n_items range",
    n_items = [4, 10, 15, 20, 25, 28],
), dataset(
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
), dataset(
    "branch and bound robustness",
    seed = [420],
    n_items = [18], # FIXME
    alg = ["bf", "bb", "dpw", "redux"],
    n_permutations = [20],
    n_repetitions = [10],
))

iteration = 0
total = sum([r * p * rep for (r, p, rep) in zip(configs["n_runs"], configs["n_permutations"], configs["n_repetitions"])])
for config in [dict(zip(configs, v)) for v in zip(*configs.values())]:
    param_iter = iter(config.values())
    next(param_iter) # skip id
    print(end = "\033[2K") # clear the current line (to get rid of the progress bar)
    print(config["id"], "\tparams", *param_iter)
    progress_bar(iteration, total)

    for inst in generate(**config):
        for rep in range(0, config["n_repetitions"]):
            # measure the time taken by the call to the solver
            start = time.time()
            cost = solve(config["alg"], inst)
            end = time.time()
            data.append(dict(inst,
                cost = cost,
                alg = config["alg"],
                t = end - start,
                repetition = rep,
                contents = None
            ))
            iteration = iteration + 1
            progress_bar(iteration, total)

print()

# ~\~ end

# ~\~ begin <<lit/main.md|performance-chart>>[0]

# plot the measurements

figsize = (14, 8)

plot_labels = dict(
    seed = "Seed",
    t = "Doba běhu [s]",
    cost = "Cena řešení",
    perm_id = "ID permutace",
    n_items = "Velikost instance",
    max_cost = "Maximální cena",
    max_weight = "Maximální váha",
    n_instances = "Počet instancí v zadání",
    granularity = "Granularita",
    light_heavy_balance = "Rozložení váhy předmětů",
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
    ds = [d for d in data if d["id"] == id]
    # create a frame from the list
    df = pd.DataFrame(ds)

    # do a boxplot grouped by the algorithm name
    sns.boxplot(
        x = x_axis,
        y = y_axis,
        data = df,
        hue = "alg",
        ax = ax,
        linewidth = 0.8,
    )

    # render the datapoints as dots with horizontal jitter
    sns.stripplot(
        x = x_axis,
        y = y_axis,
        data = df,
        hue = "alg",
        ax = ax,
        jitter = True,
        size = 4,
        dodge = True,
        linewidth = 0.2,
        alpha = 0.4,
        edgecolor = "white",
    )

    plt.title(title)
    plt.xlabel(plot_labels[x_axis])
    plt.ylabel(plot_labels[y_axis])

    constant_columns = [
        col for col in df.columns[df.nunique() <= 1]
            if (col not in ["id", "n_instances", "contents"])
    ]

    caption = "\n".join(tr.wrap("Konfigurace: {}".format({
        k: df[k][0] for k in constant_columns
    }), width = 170))

    fig.text(
        0.09,
        0.05,
        caption,
        fontsize = "small",
        fontfamily = "monospace",
        verticalalignment = "top",
    )

    handles, labels = ax.get_legend_handles_labels()
    labels = [alg_labels[l] for l in labels]

    plt.legend(handles[0 : int(len(handles) / 2)], labels[0 : int(len(labels) / 2)])
    plt.savefig("docs/assets/{}.svg".format(filename))

print("rendering plots")
plot("n_items",     "t", "n_items range",           "Průměrná doba běhu vzhledem k velikosti instance")
plot("max_weight",  "t", "weight range",            "Průměrná doba běhu vzhledem k maximální váze")
plot("max_cost",    "t", "cost range",              "Průměrná doba běhu vzhledem k maximální ceně")


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

plot(
    "perm_id",
    "t",
    "branch and bound robustness",
    "Doba běhu přes několik permutací jedné instance",
)

plot(
    "perm_id",
    "cost",
    "branch and bound robustness",
    "Cena řešení přes několik permutací jedné instance",
    filename = "branch and bound robustness - cost"
)

# ~\~ end
# ~\~ end
