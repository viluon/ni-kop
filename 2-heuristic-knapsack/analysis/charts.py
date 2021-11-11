# ~\~ language=Python filename=analysis/charts.py
# ~\~ begin <<lit/main.md|analysis/charts.py>>[0]
# ~\~ begin <<lit/main.md|preprocessing>>[0]
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from pandas.core.tools.numeric import to_numeric

# load the mean runtime per algorithm. The data is stored in the
# algorithm/n/estimates.json file, where n is the size of the input.
# The mean is in the estimate.mean.point_estimate field.

# TODO: keep this stored in a better place (duplicities between here and
# bench.rs)
algs = [ "bb"
       , "dpc"
       , "dpw"
       , "fptas1"
       , "fptas2"
       , "greedy"
       , "redux"
       ]

n_values = [4, 10, 15, 20]
data = {}

for alg in algs:
    data[alg] = {}
    for n in n_values:
        est_file = os.path.join("solver", "target", "criterion", alg, str(n), "new", "estimates.json")
        with open(est_file, "r") as f:
            estimates = json.load(f)
            mean = estimates["mean"]["point_estimate"]
            data[alg][n] = { "mean": mean / 1000 / 1000 / 1000
                           }
        err_file = os.path.join("docs", "measurements", alg + "_" + str(n) + ".txt")
        with open(err_file, "r") as f:
            measurements = pd.read_csv(f)
            data[alg][n]["error"] = { "max": measurements["max"]
                                    , "avg": measurements["avg"]
                                    }

# ~\~ end

# ~\~ begin <<lit/main.md|performance-chart>>[0]

# plot the mean runtimes

plt.figure()

for alg in algs:
    plt.plot(n_values, [data[alg][n]["mean"] for n in n_values], "-o", label=alg)

plt.xlabel("Input size")
plt.ylabel("Mean runtime (sec)")
plt.legend()
plt.title("Mean runtimes for various algorithms")
plt.savefig("docs/assets/mean_runtimes.svg")

plt.figure()
plt.title("Maximum error for each algorithm and n")
plt.xlabel("n")
plt.ylabel("error")
for alg in algs:
    plt.plot(n_values, [data[alg][n]["error"]["max"] for n in n_values], label=alg)
plt.legend()
plt.savefig("docs/assets/max_errors.svg")
# ~\~ end
# ~\~ end
