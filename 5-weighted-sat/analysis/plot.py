# ~\~ language=Python filename=analysis/plot.py
# ~\~ begin <<lit/main.md|analysis/plot.py>>[0]

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
import textwrap as tr
import math
import os

data = pd.read_pickle("docs/assets/measurements.pkl")

show_progress = os.environ.get("JUPYTER") == None

plot_labels = dict(
    error           = "Chyba oproti optimálnímu řešení [%]",
    generations     = "Počet generací",
    mutation_chance = "Pravděpodobnost mutace",
    n_instances     = "Počet instancí",
    set             = "Datová sada",
    time            = "Doba běhu [s]",
    weight          = "Váha řešení",
)

scheduled_plots = []

def progress_bar(iteration, total, length = 60):
    if not show_progress:
        return
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '=' * filledLength + ' ' * (length - filledLength)
    print(f'\r[{bar}] {percent}%', end = "\r")
    if iteration == total:
        print()

def ridgeline(id, title, col, filename, x_label = "Chyba oproti optimálnímu řešení [%]", data = data, progress = lambda _: None):
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
        ax.annotate(
            df[col].unique()[i],
            (0, 0),
            (-16.5, 3),
            xycoords = "axes fraction",
            textcoords = "offset points",
            va = "baseline",
            fontsize = 15,
            color = ax.lines[-1].get_color(),
            path_effects = [
                PathEffects.withStroke(linewidth = 0.5, foreground = "w")
            ],
        )

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
    progress(1)

def boxplot(x_axis, y_axis, id, title, grouping_column, data = data, filename = None, progress = lambda _: None):
    if filename is None:
        filename = id.replace(" ", "_") + ".svg"
    print(f"\t{title}")
    fig, ax = plt.subplots(figsize = (14, 8))
    ds = [d for d in data if d["id"] == id]
    # create a frame from the list
    df = pd.DataFrame(ds)

    # do a grouped boxplot
    sns.boxplot(
        x = x_axis,
        y = y_axis,
        data = df,
        hue = grouping_column,
        ax = ax,
        linewidth = 0.8,
    )

    # render the datapoints as dots with horizontal jitter
    sns.stripplot(
        x = x_axis,
        y = y_axis,
        data = df,
        hue = grouping_column,
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
    # labels = [alg_labels[l] for l in labels]

    plt.legend(handles[0 : int(len(handles) / 2)], labels[0 : int(len(labels) / 2)])
    plt.savefig(f"docs/assets/{filename}.svg")
    progress(1)

def heatmap(id, title, filename, data = data, progress = lambda _: None):
    dataset = data[data["id"] == id]
    stats = list(dataset["stats"])
    n_instances = int(dataset["inst_id"].count())
    # print()
    # print(dataset.describe())
    # print(dataset.head())
    # print()
    n_generations = int(dataset["generations"].max())
    n_variables = len(stats[0][0]) - 2

    # math.sqrt(n_generations) * 0.5
    fig, axs = plt.subplots(1, 2 * n_instances,
        figsize = (n_instances * n_variables * 0.15, 8),
        gridspec_kw = {"left": 0.015, "right": 0.975, "width_ratios": [n_variables, 1] * n_instances},
    )
    fig.suptitle(title)

    for i, (_, inst_id), stats, (_, err), (_, sat) in zip(
        range(1, 10000),
        dataset["inst_id"].iteritems(),
        stats,
        dataset["error"].iteritems(),
        dataset["valid"].iteritems(),
    ):
        inst_id = int(inst_id)
        df = pd.DataFrame(stats)
        ax = axs[2 * (i - 1)]
        err_ax = axs[2 * i - 1]
        ax.set_title(f"inst. {inst_id}")

        sns.heatmap(
            df.iloc[:, 2:], # drop first column (with generation numbers) and second (with errors)
            ax = ax,
            vmin = 0,
            vmax = 1,
            square = True,
            cmap = "magma",
            xticklabels = False,
            yticklabels = df.iloc[:, 0].map(int) if i == 1 else False,
            cbar_kws = {"shrink": 0.5, "pad": 0.2},
            cbar = i == n_instances,
        )

        ax_pos = ax.get_position()
        err_pos = err_ax.get_position()
        err_ax.set_position([ax_pos.x0 + ax_pos.width * 1.08, err_pos.y0, err_pos.width, err_pos.height])
        mask = df.iloc[:, 1:2]
        # disable false-positive warning
        pd.options.mode.chained_assignment = None
        mask[1] = mask[1].map(lambda x: x > 1)
        sns.heatmap(
            df.iloc[:, 1:2], # second column (error)
            ax = err_ax,
            vmin = 0,
            vmax = 1,
            square = True,
            mask = mask,
            cmap = "viridis_r",
            xticklabels = False,
            yticklabels = False,
            cbar = False,
        )

        new_ticks = [i.get_text() for i in ax.get_yticklabels()]
        ax.set_yticks(range(0, len(new_ticks), 10), new_ticks[::10])
        ax.annotate(
            (f"{100 * err:.2f}%" if err < 2 else "Neznámé optimum, splněno") if sat
            else "Splňující řešení nenalezeno",
            (0, 0),
            (4, -10),
            xycoords = "axes fraction",
            textcoords = "offset points",
            va = "top",
        )
        progress(1)

    plt.savefig(f"docs/assets/{filename}.svg")
    plt.close()
    progress(1)

def scatter(id, title, filename, data = data, progress = lambda _: None):
    dataset = data[data["id"] == id]
    fig = plt.plot()
    sns.set_style("darkgrid")
    sns.regplot(
        x = 100 * dataset["error"],
        y = dataset["inst_id"],
        fit_reg = False,
        scatter_kws = {"color": "navy", "alpha": 0.7,"s": 10}
    )
    plt.title(title)
    plt.xlabel("Chyba [%]")
    plt.ylabel("ID instance")
    plt.savefig(f"docs/assets/{filename}.svg")
    plt.close()
    progress(1)

def schedule_ridgeline(*args, **kwargs):
    scheduled_plots.append({"type": "ridgeline", "total": 1, "args": args, "kwargs": kwargs})

def schedule_scatter(*args, **kwargs):
    scheduled_plots.append({"type": "scatter", "total": 1, "args": args, "kwargs": kwargs})

def schedule_boxplot(*args, **kwargs):
    scheduled_plots.append({"type": "boxplot",   "total": 1, "args": args, "kwargs": kwargs})

def schedule_heatmap(id, *args, data = data, **kwargs):
    dataset = data[data["id"] == id]
    n_instances = int(dataset["inst_id"].count())
    scheduled_plots.append({
        "type": "heatmap",
        "total": n_instances + 1,
        "args": [id] + list(args),
        "kwargs": dict(kwargs, data = data)
    })

def plottery():
    iteration = 0
    total = sum(p["total"] for p in scheduled_plots)
    progress_bar(iteration, total)
    for plot in scheduled_plots:
        def progress(i):
            nonlocal iteration
            iteration += i
            progress_bar(iteration, total)
        try:
            if plot["type"] == "ridgeline":
                ridgeline(*plot["args"], progress = progress, **plot["kwargs"])
            elif plot["type"] == "boxplot":
                boxplot(*plot["args"], progress = progress, **plot["kwargs"])
            elif plot["type"] == "heatmap":
                heatmap(*plot["args"], progress = progress, **plot["kwargs"])
            elif plot["type"] == "scatter":
                scatter(*plot["args"], progress = progress, **plot["kwargs"])
        except Exception as e:
            print(e)
            print("Failed to plot", plot)

# describe errors
for id in data["id"].unique():
    dataset = data[data["id"] == id]
    df = pd.DataFrame(dataset[dataset["error"] < 2][dataset["valid"] == True]["error"].describe())
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: 100 * x)
    df = df.T
    df["count"] = df["count"] / 100
    df["dataset"] = id
    df.to_csv(f"docs/assets/{id}_errors.csv")
    # describe satisfiability
    _sum = dataset["valid"].sum()
    _count = dataset["valid"].count()
    sat = 100 * (_sum / _count)
    print(f"{id}: {sat:.2f}% ({_sum} / {_count})")

schedule_ridgeline(
    "mutation_exploration",
    "Vliv šance mutace na hustotu chyb",
    "mutation_chance",
    "whitebox-mutation-chance-error.svg",
)

schedule_heatmap(
    "default",
    "Vývoj populace ve výchozím nastavení",
    "whitebox-heatmap-default-mix",
    data = data[data["inst_id"] <= 8]
)

# for _, mutation_chance in data[data["id"] == "mutation_exploration"]["mutation_chance"].iteritems():
#     schedule_heatmap(
#         "mutation_exploration",
#         f"Vývoj populace s šancí mutace {mutation_chance * 100}%",
#         f"whitebox-heatmap-mut-explr-{mutation_chance}",
#         data = data[data["mutation_chance"] == mutation_chance]
#     )

for dataset in ["M", "M", "N", "Q", "R", "A"]:
    _id = f"dataset_{dataset}"
    # schedule_heatmap(
    #     _id,
    #     f"Vývoj populace pro dataset {dataset}",
    #     f"whitebox-heatmap-dataset-{dataset}",
    #     data = data[data["id"] == _id][:8],
    # )

    # schedule_ridgeline(
    #     _id,
    #     f"Hustota chyb pro dataset {dataset}",
    #     f"mutation_chance",
    #     f"whitebox-error-density-evaluation-dataset-{dataset}.svg",
    # )

    # schedule_scatter(
    #     "all",
    #     f"Chyby v sadě {dataset}",
    #     f"whitebox-error-scatter-dataset-{dataset}",
    #     data = data[data["set"] == dataset][data["valid"] == True][data["error"] < 2],
    # )

# schedule_ridgeline(
#     "all",
#     "Hustota chyb podle datové sady",
#     "set",
#     f"whitebox-error-density-evaluation-all.svg",
#     data = data[data["valid"] == True],
# )

# schedule_heatmap(
#     "all",
#     f"Studie vývoje instancí v datasetu 100-430-A1",
#     f"whitebox-heatmap-dataset-100-430-A1-closeup",
#     data = data[data["set"] == "A"][data["instance_params"] == {"variables": 100, "clauses": 430}][:3],
# )

# schedule_heatmap(
#     "dataset_A",
#     "Vývoj populace pro dataset A",
#     "whitebox-heatmap-dataset-A",
#     data = data[data["inst_id"] <= 20]
#     # data = data[:8],
# )

# schedule_ridgeline(
#     "dataset_A",
#     "Hustota chyb pro dataset A",
#     "mutation_chance",
#     "whitebox-error-density-evaluation-dataset-A.svg",
# )

for ds in [
    # "dataset_N_large",
    # "dataset_Q_large",
    # "dataset_R_large",
    # "dataset_N_largest",
    # "dataset_Q_largest",
    # "dataset_R_largest",
    # "dataset_A_huge",
]:
    schedule_heatmap(
        ds,
        f"Studie vývoje instancí v datasetu {ds[8:]}",
        f"whitebox-heatmap-dataset-{ds}-closeup",
        data = data[data["id"] == ds][:4],
    )

# do the plottery
plottery()

# ~\~ end
