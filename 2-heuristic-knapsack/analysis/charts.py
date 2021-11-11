# ~\~ language=Python filename=analysis/charts.py
# ~\~ begin <<lit/main.md|analysis/charts.py>>[0]
# ~\~ begin <<lit/main.md|preprocessing>>[0]
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from pandas.core.tools.numeric import to_numeric

# load the mean runtime per algorithm. The data is stored in the
# algorithm/n/estimates.json file, where n is the size of the input.
# The mean is in the estimate.mean.point_estimate field.

# TODO: keep this in a better place (duplicities between here and bench.rs)
sets = [ "NK"
       , "ZKC"
       , "ZKW"
       ]

algs = [ "bb"
       , "dpc"
       , "dpw"
       , "fptas1"
       , "fptas2"
       , "greedy"
       , "redux"
       ]

n_values = [4, 10, 15, 20, 22, 25, 27, 30, 32]
data = {}

for s in sets:
    data[s] = {}
    for alg in algs:
        data[s][alg] = {}
        for n in n_values:
            est_file = os.path.join("solver", "target", "criterion", alg, str(n), "new", "estimates.json")
            if os.path.exists(est_file):
                with open(est_file, "r") as f:
                    estimates = json.load(f)
                    mean = estimates["mean"]["point_estimate"]
                    data[s][alg][n] = { "mean": mean / 1000 / 1000 / 1000
                                      }
                err_file = os.path.join("docs", "measurements", alg + "_" + str(n) + ".txt")
                with open(err_file, "r") as f:
                    measurements = pd.read_csv(f)
                    data[s][alg][n]["error"] = { "max": measurements["max"]
                                               , "avg": measurements["avg"]
                                               }

# ~\~ end

# ~\~ begin <<lit/main.md|performance-chart>>[0]

# plot the mean runtimes and max errors

for s in sets:
    figsize = (14, 8)
    fig, ax = plt.subplots(figsize = figsize)
    plt.title("Průměrná doba běhu")
    plt.xlabel("Velikost instance")
    plt.ylabel("Průměrná doba běhu (sec)")
    plt.xticks(n_values)
    for alg in algs:
        plt.plot([n for n in data[s][alg]], [data[s][alg][n]["mean"] for n in data[s][alg]], "--o", label=alg)
    plt.legend()
    plt.savefig("docs/assets/{}-mean_runtimes.svg".format(s))

    fig, ax = plt.subplots(figsize = figsize)
    plt.title("Závislost maximální chyby na velikosti instance")
    plt.xlabel("Velikost instance")
    plt.ylabel("Maximální chyba")
    plt.xticks(n_values)
    yticks = np.append(ax.get_yticks(), [0.1, 0.01])
    ax.set_yticks(yticks)
    ax.grid(linestyle = "dotted")
    for alg in algs:
        plt.plot([n for n in data[s][alg]], [data[s][alg][n]["error"]["max"] for n in data[s][alg]], label=alg)
    plt.legend()
    plt.savefig("docs/assets/{}-max_errors.svg".format(s))
# ~\~ end
# ~\~ end
