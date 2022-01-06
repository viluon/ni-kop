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

for cfg in product(
    # max iterations
    ["10000"],
    # scaling factor
    ["0.998"],
):
    print(cfg)
    errors = []
    for instance in input.split("\n")[:4]:
        id = instance.split()[0]
        (t, cost, err, cost_temperature_progression) = invoke_solver(instance, cfg)
        errors.append(err)
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

        plt.savefig(f"docs/assets/whitebox-{'-'.join(cfg)}-{id}.svg")
        plt.close()

    print(*errors, sep = "\n")

# ~\~ end
# ~\~ end
