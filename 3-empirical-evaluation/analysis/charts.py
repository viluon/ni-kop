# ~\~ language=Python filename=analysis/charts.py
# ~\~ begin <<lit/main.md|analysis/charts.py>>[0]
# ~\~ begin <<lit/main.md|python-imports>>[0]
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
# ~\~ end


show_progress = os.environ.get("JUPYTER") == None
algs = ["bf", "bb", "dpc", "dpw", "redux"]
data = []

# adapted from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def progress_bar(iteration, total, length = 60):
    if not show_progress:
        return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '=' * filledLength + ' ' * (length - filledLength)
    print(f'\r[{bar}] {percent}%', end = "\r")
    if iteration == total:
        print()

def generate(**kwargs):
    # ~\~ begin <<lit/main.md|generate-instance>>[0]
    res = []
    kwargs["granularity"] = kwargs["granularity_and_light_heavy_balance"][0]
    kwargs["light_heavy_balance"] = kwargs["granularity_and_light_heavy_balance"][1]
    del kwargs["granularity_and_light_heavy_balance"]
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
    # ~\~ end

def solve(alg, instance):
    # ~\~ begin <<lit/main.md|invoke-solver>>[0]
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
    # ~\~ end

# ~\~ begin <<lit/main.md|dataset-utilities>>[0]
# enumerate the parameter values of a dataset for instance generation and
# algorithm benchmarking.
def dataset(id, **kwargs):
    params = dict({
        # defaults
        "id": [id],
        "alg": algs,
        "seed": [42],
        "n_runs": [3],
        "n_permutations": [1],
        "n_repetitions": [3],
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
# ~\~ end


# ~\~ begin <<lit/main.md|datasets>>[0]
n_samples = 3

# benchmark configurations
# we don't want a full cartesian product (too slow to fully explore), so we'll
# use a union of subsets, each tailored to the particular algorithm
configs = merge_datasets(dataset(
    "weight range",
    alg = ["bf", "bb", "dpc", "dpw"],
    max_weight = [500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000],
), dataset(
    "cost range",
    alg = ["bf", "bb", "dpc", "dpw"],
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
    seed = [97],
    n_items = [20],
    alg = ["bf", "bb", "dpw", "redux"],
    n_runs = [1],
    n_permutations = [20],
    n_repetitions = [10],
))
# ~\~ end

# ~\~ begin <<lit/main.md|measurement-loop>>[0]
iteration = 0
total = sum([r * p * rep for (r, p, rep) in zip(configs["n_runs"], configs["n_permutations"], configs["n_repetitions"])])
for config in [dict(zip(configs, v)) for v in zip(*configs.values())]:
    param_iter = iter(config.values())
    next(param_iter) # skip id
    if show_progress:
        print(end = "\033[2K") # clear the current line to get rid of the progress bar
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

# ~\~ begin <<lit/main.md|chart-labels>>[0]
plot_labels = dict(
    seed = "Seed",
    t = "Doba b??hu [s]",
    cost = "Cena ??e??en??",
    perm_id = "ID permutace",
    n_items = "Velikost instance",
    max_cost = "Maxim??ln?? cena",
    max_weight = "Maxim??ln?? v??ha",
    n_instances = "Po??et instanc?? v zad??n??",
    granularity = "Granularita",
    light_heavy_balance = "Rozlo??en?? v??hy p??edm??t??",
    capacity_weight_sum_ratio = "Pom??r kapacity a sou??tu vah",
)

alg_labels = dict(
    bf = "Brute force",
    bb = "Branch & bound",
    dpc = "Dynamic programming (cost)",
    dpw = "Dynamic programming (weight)",
    redux = "Greedy redux",
)
# ~\~ end

def plot(x_axis, y_axis, id, title, data = data, filename = None):
    if filename is None:
        filename = id.replace(" ", "_")
    print("\t{}".format(title))
    fig, ax = plt.subplots(figsize = figsize)
    ds = [d for d in data if d["id"] == id]
    # create a frame from the list
    df = pd.DataFrame(ds)

    # do a boxplot grouped by the algorithm name
    # ~\~ begin <<lit/main.md|plot-boxplot>>[0]
    sns.boxplot(
        x = x_axis,
        y = y_axis,
        data = df,
        hue = "alg",
        ax = ax,
        linewidth = 0.8,
    )
    # ~\~ end

    # render the datapoints as dots with horizontal jitter
    # ~\~ begin <<lit/main.md|plot-stripplot>>[0]
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
    # ~\~ end

    plt.title(title)
    plt.xlabel(plot_labels[x_axis])
    plt.ylabel(plot_labels[y_axis])

    # ~\~ begin <<lit/main.md|plot-caption>>[0]
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
    # ~\~ end

    handles, labels = ax.get_legend_handles_labels()
    labels = [alg_labels[l] for l in labels]

    plt.legend(handles[0 : int(len(handles) / 2)], labels[0 : int(len(labels) / 2)])
    plt.savefig("docs/assets/{}.svg".format(filename))

print("rendering plots")
# ~\~ begin <<lit/main.md|plots>>[0]
plot("n_items",     "t", "n_items range",           "Pr??m??rn?? doba b??hu vzhledem k velikosti instance")
plot("max_weight",  "t", "weight range",            "Pr??m??rn?? doba b??hu vzhledem k maxim??ln?? v??ze")
plot("max_cost",    "t", "cost range",              "Pr??m??rn?? doba b??hu vzhledem k maxim??ln?? cen??")


for balance in ["light", "heavy"]:
    plot(
        "granularity",
        "t",
        "granularity exploration",
        "Doba b??hu vzhledem ke granularit?? (preference {})".format(balance),
        data = [d for d in data if d["light_heavy_balance"] == balance],
        filename = "granularity_exploration_{}".format(balance),
    )

plot(
    "capacity_weight_sum_ratio",
    "t",
    "capacity weight sum ratio exploration",
    "Doba b??hu vzhledem k pom??ru kapacity a sou??tu vah",
)

plot(
    "perm_id",
    "t",
    "branch and bound robustness",
    "Doba b??hu p??es n??kolik permutac?? jedn?? instance",
)

plot(
    "perm_id",
    "cost",
    "branch and bound robustness",
    "Cena ??e??en?? p??es n??kolik permutac?? jedn?? instance",
    filename = "branch_and_bound_robustness_cost"
)
# ~\~ end
# ~\~ end
# ~\~ end
