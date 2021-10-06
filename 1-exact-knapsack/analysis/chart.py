# ~\~ language=Python filename=analysis/chart.py
# ~\~ begin <<lit/main.md|analysis/chart.py>>[0]
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.tools.numeric import to_numeric

bench = pd.read_csv("docs/bench.csv", dtype = "string")
bench.rename({
        "algoritmus": "alg",
        "$n$": "n",
        "průměr": "avg",
        "$\pm \sigma$": "sigma",
        "medián": "median",
        "minimum": "min",
        "maximum": "max",
    },
    inplace = True,
    errors  = "raise",
    axis    = 1,
)

numeric_columns = ["n", "avg", "sigma", "min", "median", "max"]
bench[numeric_columns] = bench[numeric_columns].apply(lambda c: c.apply(lambda x: to_numeric(x.replace("**", "").replace(" ms", ""))))
# plt.figure()
# bench.groupby("alg").plot("n", "avg", kind = "line", ax = plt.gca())

df = bench

# Create a figure and a set of subplots.
fig, ax = plt.subplots()

# Group the dataframe by alg and create a line for each group.
for name, group in df.groupby("alg"):
    ax.errorbar(group["n"], group["avg"], yerr = group["sigma"], label = name)

# Set the axis labels.

ax.set_xlabel("n")
ax.set_ylabel("time (ms)")
ax.set_yscale("log")

# Add a legend.
ax.legend(loc="upper left")


plt.savefig("docs/graph.svg")
# ~\~ end
