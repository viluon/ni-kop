# ~\~ language=Python filename=analysis/measure.py
# ~\~ begin <<lit/main.md|analysis/measure.py>>[0]
import os
from itertools import product, chain
from subprocess import run, PIPE
import json
import pandas as pd

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

def invoke_solver(cfg):
    solver = run(
        [
            "target/release/main",
            json.dumps(cfg),
        ],
        stdout = PIPE,
        encoding = "ascii",
        cwd = "solver/",
        env = {
            "RUST_BACKTRACE": "1",
            **os.environ,
        },
    )
    if solver.returncode != 0:
        print(solver)
        raise Exception("solver failed")

    results = []
    stats = []
    for line in solver.stdout.split("\n")[8:]:
        if line.startswith("done: "):
            [_, time, inst_id, satisfied, valid, weight, err] = line.split()
            results.append((float(time), int(inst_id), satisfied == "true", valid == "true", float(weight), float(err), stats))
            stats = []
        else:
            stats.append(list(map(float, line.split())))
    return results

def dataset(id, **kwargs):
    # defaults
    params = dict({
        "id": [id],
        "set": ["M"],
        "instance_params": [{"variables": 20, "clauses": 78}],
        "n_instances": [15],
        "generations": [200],
        "mutation_chance": [0.02],
        "population_size": [1000],
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

# ~\~ begin <<lit/main.md|datasets>>[0]
configs = merge_datasets(dataset(
#     "all",
#     generations = [6_000],
#     mutation_chance = [0.25],
#     disaster_interval = [900],
#     set = ["M", "N", "Q", "R", "A"],
#     instance_params = [{"variables": 20, "clauses": 78}, {"variables": 50, "clauses": 201}, {"variables": 75, "clauses": 310}, {"variables": 100, "clauses": 430}],
#     n_instances = [150],
#     population_size = [200],
# ), dataset(
#     "huge_A",
#     generations = [10_000],
#     mutation_chance = [0.25],
#     disaster_interval = [900],
#     set = ["A"],
#     instance_params = [{"variables": 100, "clauses": 430}],
#     n_instances = [150],
#     population_size = [200],
# ), dataset(
    "default",
    generations = [2_500],
    mutation_chance = [0.2],
    n_instances = [50],
    population_size = [200],
# ), dataset(
#     "exploration",
#     generations = [500, 1_000, 2_500, 5_000],
#     mutation_chance = [0.2, 0.1, 0.05, 0.01],
#     disaster_interval = [50, 100, 200, 400, 1000],
#     n_instances = [50],
#     population_size = [50, 100, 200, 400],
# ), dataset(
#     "dataset_N_large",
#     set = "N",
#     instance_params = [{"variables": 50, "clauses": 201}],
#     generations = [5_500],
#     mutation_chance = [0.2],
#     n_instances = [50],
#     population_size = [200],
# ), dataset(
#     "dataset_Q_large",
#     set = "Q",
#     instance_params = [{"variables": 50, "clauses": 201}],
#     generations = [1000],
#     mutation_chance = [0.03],
#     n_instances = [100],
# ), dataset(
#     "dataset_R_large",
#     set = "R",
#     instance_params = [{"variables": 50, "clauses": 201}],
#     generations = [1000],
#     mutation_chance = [0.03],
#     n_instances = [100],
# ), dataset(
#     "dataset_N_largest",
#     set = "N",
#     instance_params = [{"variables": 75, "clauses": 310}],
#     generations = [1000],
#     mutation_chance = [0.03],
#     n_instances = [200],
# ), dataset(
#     "dataset_Q_largest",
#     set = "Q",
#     instance_params = [{"variables": 75, "clauses": 310}],
#     generations = [1000],
#     mutation_chance = [0.03],
#     n_instances = [200],
# ), dataset(
#     "dataset_R_largest",
#     set = "R",
#     instance_params = [{"variables": 75, "clauses": 310}],
#     generations = [1000],
#     mutation_chance = [0.03],
#     n_instances = [200],
# ), dataset(
#     "dataset_A_huge",
#     set = "A",
#     instance_params = [{"variables": 100, "clauses": 430}],
#     generations = [1000],
#     mutation_chance = [0.03],
#     n_instances = [200],
), dataset(
    "mutation_exploration",
    n_instances = [50],
    mutation_chance = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
# ), dataset(
#     "dataset_N",
#     set = ["N"],
#     generations = [2_500],
#     n_instances = [40],
#     population_size = [200],
# ), dataset(
#     "dataset_Q",
#     set = ["Q"],
#     generations = [2_500],
#     n_instances = [40],
#     population_size = [200],
# ), dataset(
#     "dataset_R",
#     set = ["R"],
#     generations = [2_500],
#     n_instances = [40],
#     population_size = [200],
# ), dataset(
#     "dataset_A",
#     set = ["A"],
#     instance_params = [{"variables": 20, "clauses": 88}],
#     generations = [2_500],
#     n_instances = [40],
#     population_size = [200],
# ), dataset(
#     "dataset_Q_exploration",
#     set = ["Q"],
#     generations = [2000],
#     population_size = [2000],
#     mutation_chance = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
# ), dataset(
#     "dataset_R_exploration",
#     set = ["R"],
#     generations = [2000],
#     population_size = [2000],
#     mutation_chance = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
# ), dataset(
#     "dataset_A_exploration",
#     set = ["A"],
#     generations = [2000],
#     population_size = [2000],
#     mutation_chance = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
))
# ~\~ end

data = pd.DataFrame()
cfgs = [dict(zip(configs, v)) for v in zip(*configs.values())]
iteration = 0
total = sum([cfg["n_instances"] * cfg["generations"] for cfg in cfgs])

for config in cfgs:
    if show_progress:
        print(end = "\033[2K")
    print(json.dumps(config))
    progress_bar(iteration, total)

    for (t, inst_id, satisfied, valid, weight, err, stats) in invoke_solver(config):
        data = data.append(dict(config,
            error   = err,
            inst_id = inst_id,
            stats   = stats,
            time    = t,
            valid   = valid,
            weight  = weight,
        ), ignore_index = True)

        iteration = iteration + config["generations"]
        progress_bar(iteration, total)

data.to_pickle("docs/assets/measurements.pkl")

# ~\~ end
