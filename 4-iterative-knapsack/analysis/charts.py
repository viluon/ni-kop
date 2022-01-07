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

def invoke_solver(input, cfg):
    solver = run(
        ["target/release/main", "sa"] + list(cfg),
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

# load the input
input = None
with open("solver/ds/NK35_inst.dat", "r") as f:
    input = f.read()

errors = []
for cfg in product(
    # max iterations
    # ["15000"],
    ["3000"],
    # scaling factor
    ["0.85", "0.9", "0.95", "0.99", "0.991", "0.992", "0.993", "0.994", "0.995", "0.997"],
    # temperature modifier
    ["1"],
):
    print(cfg)
    params = '-'.join(cfg)
    for instance in input.split("\n")[:10]:
        id = instance.split()[0]
        (t, cost, err, cost_temperature_progression) = invoke_solver(instance, cfg)
        errors.append({"scaling factor": cfg[1], "error": err})
        print("took", t, "ms")

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

print(*errors, sep = "\n")

df = pd.DataFrame(errors)
series = df.groupby("scaling factor")["error"].mean()
df["mean error"] = df["scaling factor"].map(series)

# plot the error distributions for each scaling factor
plt.style.use("default")
sns.set_theme(style = "white", rc = {"axes.facecolor": (0, 0, 0, 0)})
pal = sns.color_palette("crest", n_colors = len(set([e["scaling factor"] for e in errors])))

# set up the layout
g = sns.FacetGrid(
    df,
    row = "scaling factor",
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
    ax.text(-0.125, 1, df["scaling factor"].unique()[i],
            fontsize = 15, color = ax.lines[-1].get_color())

# remove titles, y ticks, spines
g.set_titles("")
g.set(yticks = [])
g.despine(left = True, bottom = True)
g.fig.suptitle("Vliv koeficientu chlazení na hustotu chyb", fontsize = 20, ha = "right")
for ax in g.axes.flat:
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x * 100:.0f}")
g.set_xlabels("Chyba oproti optimálnímu řešení [%]")
g.set_ylabels("")

g.savefig("docs/assets/whitebox-error-distributions.svg")

# sns.kdeplot([100 * e for e in errors], shade = True)
# plt.savefig(f"docs/assets/whitebox-overview-{params}.svg")
# plt.close()

# ~\~ end
# ~\~ end
