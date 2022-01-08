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

# ~\~ begin <<lit/main.md|performance-chart>>[0]
# plot the measurements

figsize = (14, 8)

show_progress = os.environ.get("JUPYTER") == None

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

def invoke_solver(input, cfg):
    solver = run(
        [
            "target/release/main",
            "sa",
            str(cfg["max_iterations"]),
            str(cfg["scaling_factor"]),
            str(cfg["temperature_modifier"]),
            str(cfg["equilibrium_width"]),
        ],
        stdout = PIPE,
        input = input,
        encoding = "ascii",
        cwd = "solver/"
    )
    if solver.returncode != 0:
        print(solver)
        raise Exception("solver failed")

    lines = solver.stdout.split("\n")
    [_, time, _] = lines[-3].split()
    [cost, err]  = lines[-2].split()
    cost_temperature_progression = [list(map(float, entry.split())) for entry in lines[:-4]]
    return (float(time), float(cost), float(err), cost_temperature_progression)

def dataset(id, **kwargs):
    params = dict({
        # defaults
        "id": [id],
        "precise_plot": [True],
        "n_instances": [6], # FIXME: this is not a good default
        "max_iterations": [8000],
        "scaling_factor": [0.996],
        "temperature_modifier": [0.7],
        "equilibrium_width": [10],
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

configs = merge_datasets(dataset(
    "scaling_factor_exploration",
    scaling_factor = [0.85, 0.9, 0.95, 0.99, 0.992, 0.994, 0.996, 0.997, 0.998, 0.999],
), dataset(
    "temperature modifier exploration",
    n_instances = [30],
    temperature_modifier = [0.0001, 0.01, 1, 100, 10000],
), dataset(
    "equilibrium width exploration",
    n_instances = [40],
    precise_plot = [False],
    equilibrium_width = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
), dataset(
    "black box",
    precise_plot = [False],
    n_instances = [500],
))

# load the input
input = None
with open("solver/ds/NK35_inst.dat", "r") as f:
    input = f.read()

errors = []
cfgs = [dict(zip(configs, v)) for v in zip(*configs.values())]
iteration = 0
total = sum([cfg["n_instances"] for cfg in cfgs])

for config in cfgs:
    if show_progress:
        print(end = "\033[2K")
    print(config)
    progress_bar(iteration, total)

    params = "-".join([str(v) for _, v in config.items()])
    for instance in input.split("\n")[:config["n_instances"]]:
        id = instance.split()[0]
        (t, cost, err, cost_temperature_progression) = invoke_solver(instance, config)
        errors.append(dict(config, error = err))

        if config["precise_plot"]:
            # plot the cost / temperature progression:
            # we have two line graphs in a single plot
            # the x axis is just the index in the list

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize = figsize)
            for (i, label) in zip(range(42), ["cost", "best cost", "temperature"]):
                ax.plot(
                    range(len(cost_temperature_progression)),
                    [entry[i] for entry in cost_temperature_progression],
                    label = label,
                )
            ax.set_xlabel("iteration")
            ax.set_title(f"instance {id} with error {err}")
            ax.legend(loc = "lower right")

            plt.savefig(f"docs/assets/whitebox-{params}-{id}.svg")
            plt.close()

        iteration = iteration + 1
        progress_bar(iteration, total)

data = pd.DataFrame(errors)
def ridgeline(id, title, col, filename, x_label = "Chyba oproti optimálnímu řešení [%]"):
    df = data[data["id"] == id]
    series = df.groupby(col)["error"].mean()
    df["mean error"] = df[col].map(series)

    # plot the error distributions for each value of col
    plt.style.use("default")
    sns.set_theme(style = "white", rc = {"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.color_palette("crest", n_colors = len(df[col].unique()))

    # set up the layout
    g = sns.FacetGrid(
        df,
        row = col,
        hue = "mean error",
        palette = pal,
        height = 0.75,
        aspect = 15,
    )
    plt.xlim(-0.1, 1.0)
    # distributions
    g.map(sns.kdeplot, "error", clip = (-0.1, 1.0), bw_adjust = 1, clip_on = False, fill = True, alpha = 1, linewidth = 0.1)
    # contours
    g.map(sns.kdeplot, "error", clip = (-0.1, 1.0), bw_adjust = 1, clip_on = False, color = "w", lw = 1)
    # horizontal lines
    g.map(plt.axhline, y = 0, lw = 2, clip_on = False)
    # overlap
    g.fig.subplots_adjust(hspace = -0.3)

    for i, ax in enumerate(g.axes.flat):
        ax.text(-0.125, 5, df[col].unique()[i],
                fontsize = 15, color = ax.lines[-1].get_color(), va = "baseline")

    # remove titles, y ticks, spines
    g.set_titles("")
    g.set(yticks = [])
    g.despine(left = True, bottom = True)
    g.fig.suptitle(title, fontsize = 20, ha = "right")
    for ax in g.axes.flat:
        ax.xaxis.set_major_formatter(lambda x, pos: f"{x * 100:.0f}")
    g.set_xlabels(x_label)
    g.set_ylabels("")

    g.savefig(f"docs/assets/{filename}")
    plt.close()

ridgeline(
    "scaling_factor_exploration",
    "Vliv koeficientu chlazení na hustotu chyb",
    "scaling_factor",
    "whitebox-error-distributions.svg",
)

ridgeline(
    "temperature modifier exploration",
    "Vliv koeficientu počáteční teploty na hustotu chyb",
    "temperature_modifier",
    "whitebox-error-distributions-temperature.svg",
)

ridgeline(
    "equilibrium width exploration",
    "Vliv šířky ekvilibria na hustotu chyb",
    "equilibrium_width",
    "whitebox-error-distributions-equilibrium-width.svg",
)

# plot the error distribution
sns.kdeplot(
    data = data[data["id"] == "black box"],
    x = "error",
)
plt.savefig("docs/assets/blackbox-error-distribution.svg")

# ~\~ end
# ~\~ end
