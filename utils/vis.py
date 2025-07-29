import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from matplotlib.transforms import ScaledTranslation
from pathlib import Path

# Dictionary to show nice labels
DS_MAP = {
    "llava-bench": "LLaVa-Bench",
    "mmbench": "MMBench",
    "seed": "SEED-Bench 2",
    "vqav2": "VQA v2",
}

MODEL_MAP = {
    "internvl2": "InternVL2",
    "llava-1.6": "LLaVa-1.6",
    "minigpt4": "MiniGPT4",
    "sharegpt4v": "ShareGPT4V",
}

# -- -- -- -- -- -- -- -- -- -- -- --
# Individual charts
# -- -- -- -- -- -- -- -- -- -- -- --


def save_graph_to_img(
    graph: nx.DiGraph, out_file: str, highlight_cycles: bool = False
) -> None:
    # Plot graph
    plt.figure(figsize=(15, 15))
    plt.axis("off")
    connection_style = "arc3,rad=0.1"

    pos = nx.kamada_kawai_layout(graph)
    nx.draw(
        graph,
        pos=pos,
        with_labels=True,
        node_size=800,
        node_color="skyblue",
        font_size=8,
        font_weight="bold",
        connectionstyle=connection_style,
    )

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=edge_labels)

    if highlight_cycles:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
            nx.draw_networkx_edges(
                graph,
                pos,
                arrows=True,
                edgelist=cycle,
                edge_color="r",
                width=1,
                connectionstyle=connection_style,
            )
        except Exception as e:
            print(e)

    plt.savefig(out_file)
    plt.close()


def plot_concepts_vs_relations(sample_stats_df: pd.DataFrame, output_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plt.scatter(
        sample_stats_df["num_relations"],
        sample_stats_df["num_concepts"],
        color="skyblue",
        edgecolor="black",
    )
    plt.title("Number of Concepts vs. Number of Relations")
    plt.xlabel("Number of Relations")
    plt.ylabel("Number of Concepts")
    plt.grid(True)

    # for i, row in sample_stats_df.iterrows():
    #     plt.text(row["num_relations"], row["num_concepts"], str(row["id_image"]),
    #              fontsize=10, ha='right', color='darkblue')

    # plt.show()
    plt.savefig(output_path, dpi=300)
    plt.close()


# -- -- -- -- -- -- -- -- -- -- -- --
# Plotting SHOULD-KNOWs
# -- -- -- -- -- -- -- -- -- -- -- --


def set_custom_patch_sk(b_plot: plt.boxplot, color_key: str = "concept") -> None:
    # Palette taken from https://venngage.com/tools/accessible-color-palette-generator
    color_map = {
        "concept": ["#b27795", "#c18750", "#7f9a46"],
        "relation": ["#2291a2", "#e0a2a3", "#d8d330"],
    }
    hatches = ["...", "///", "xxx"]

    for idx_col, patch in enumerate(b_plot["boxes"]):
        patch.set(facecolor=color_map[color_key][idx_col])
        patch.set(hatch=hatches[idx_col])


def plot_sk(count_concepts: dict, count_preds: dict, output_path: str | Path) -> None:
    _, axes = plt.subplots(1, len(count_concepts.keys()), sharey=True, figsize=(15, 4))

    ds_keys = list(count_concepts.keys())
    width = 0.85

    for idx, ax in enumerate(axes):
        ax.set_title(DS_MAP[ds_keys[idx]])
        ax.tick_params(bottom=False)
        ax.set_xticklabels(["", "Concepts", "", "", "Relations", ""])
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        cc_plot = count_concepts[ds_keys[idx]]
        rc_plot = count_preds[ds_keys[idx]]

        b_plot = ax.boxplot(
            cc_plot.values(),
            positions=[1, 2, 3],
            widths=width,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black"),
        )
        set_custom_patch_sk(b_plot, color_key="concept")

        b_plot = ax.boxplot(
            rc_plot.values(),
            positions=[4.5, 5.5, 6.5],
            widths=width,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black"),
        )
        set_custom_patch_sk(b_plot, color_key="relation")

    ax.legend(
        ["Original", "Validation", "Annotation"] * 2,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1, 0.5),
    )

    # ax.set_title("Should-Know Concepts and Relations")
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_sk_rels(all_relation_counts: dict, output_path: str | Path) -> None:
    _, axes = plt.subplots(
        1, len(all_relation_counts.keys()), sharey=True, figsize=(15, 4)
    )

    ds_keys = list(all_relation_counts.keys())
    print(ds_keys)
    width = 0.85

    for idx, ax in enumerate(axes):
        ax.set_title(DS_MAP[ds_keys[idx]])
        ax.tick_params(bottom=False)
        ax.set_xticklabels(["OG", "VAL", "ANN"])
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        rc_plot = all_relation_counts[ds_keys[idx]]

        b_plot = ax.boxplot(
            rc_plot.values(),
            positions=[1, 2, 3],
            widths=width,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black"),
        )
        set_custom_patch_sk(b_plot, color_key="relation")

    ax.legend(
        ["Original", "Validation", "Annotation"],
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1, 0.5),
    )

    # ax.set_title("Should-Know Relations")
    axes[0].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -- -- -- -- -- -- -- -- -- -- -- --
# Plotting REALLY-KNOWs
# -- -- -- -- -- -- -- -- -- -- -- --


def set_custom_patch_rk(b_plot: plt.boxplot) -> None:
    # Palette taken from https://venngage.com/tools/accessible-color-palette-generator
    color_map = ["#e6a6a3", "#f1c3c2", "#b0d3dc", "#8cbcc7"]
    hatches = ["...", "///", "xxx", "+++"]

    for idx_col, patch in enumerate(b_plot["boxes"]):
        patch.set(facecolor=color_map[idx_col])
        patch.set(hatch=hatches[idx_col])


def plot_rk(
    all_concept_counts: dict, all_relation_counts: dict, output_path: str | Path
) -> None:
    fig, axes = plt.subplots(
        1, len(all_concept_counts.keys()), sharey=True, figsize=(15, 4)
    )

    model_keys = list(all_concept_counts.keys())

    width = 0.8

    for idx, ax in enumerate(axes):
        ax.set_title(MODEL_MAP[model_keys[idx]])
        ax.tick_params(bottom=False)
        ax.set_xticklabels(["", "Concepts", "", "", "", "Relations", "", ""])
        # Create offset transform by 5 points in x direction
        dx = 15 / 72.0
        dy = 0 / 72.0
        offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        cc_plot = all_concept_counts[model_keys[idx]]
        rc_plot = all_relation_counts[model_keys[idx]]

        b_plot = ax.boxplot(
            cc_plot.values(),
            positions=[1, 2, 3, 4],
            widths=width,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black"),
        )
        set_custom_patch_rk(b_plot)

        b_plot = ax.boxplot(
            rc_plot.values(),
            positions=[5.5, 6.5, 7.5, 8.5],
            widths=width,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black"),
        )
        set_custom_patch_rk(b_plot)

    ax.legend(DS_MAP.values(), loc="center left", ncol=1, bbox_to_anchor=(1, 0.5))

    # ax.legend(DS_MAP.values(),
    #           loc="upper right",
    #           ncol=1)

    # ax.set_title("Boxplot of Concepts and Relations")
    axes[0].set_ylabel("Count")
    # plt.show()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_rk_rels(all_relation_counts: dict, output_path: str | Path) -> None:
    fig, axes = plt.subplots(
        1, len(all_relation_counts.keys()), sharey=True, figsize=(15, 4)
    )

    model_keys = list(all_relation_counts.keys())

    width = 0.8

    for idx, ax in enumerate(axes):
        ax.set_title(MODEL_MAP[model_keys[idx]])
        ax.tick_params(bottom=False)
        ax.set_xticklabels(["", "Relations", "", ""])
        # Create offset transform by 5 points in x direction
        dx = 15 / 72.0
        dy = 0 / 72.0
        offset = ScaledTranslation(dx, dy, fig.dpi_scale_trans)

        # apply offset transform to all x ticklabels.
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        rc_plot = all_relation_counts[model_keys[idx]]

        b_plot = ax.boxplot(
            rc_plot.values(),
            positions=[5.5, 6.5, 7.5, 8.5],
            widths=width,
            patch_artist=True,
            notch=True,
            medianprops=dict(color="black"),
        )
        set_custom_patch_rk(b_plot)

    ax.legend(DS_MAP.values(), loc="center left", ncol=1, bbox_to_anchor=(1, 0.5))

    # ax.legend(DS_MAP.values(),
    #           loc="upper right",
    #           ncol=1)

    # ax.set_title("Really-Know Relations")
    axes[0].set_ylabel("Count")
    # plt.show()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


# -- -- -- -- -- -- -- -- -- -- -- --
# Plotting Descriptive Statistics
# -- -- -- -- -- -- -- -- -- -- -- --


def set_custom_patch_sim(b_plot: plt.boxplot, pos: int) -> None:
    # Palette taken from https://venngage.com/tools/accessible-color-palette-generator
    color_map = ["#e6a6a3", "#f1c3c2", "#b0d3dc", "#8cbcc7"]
    hatch_density = 2
    hatches = [
        "." * hatch_density,
        "/" * hatch_density,
        "x" * hatch_density,
        "+" * hatch_density,
    ]

    for _, patch in enumerate(b_plot["boxes"]):
        patch.set(facecolor=color_map[pos - 1])
        patch.set(hatch=hatches[pos - 1])


def plot_similarity(
    data: dict, out_path: str | Path, measure_key: str = "cosine"
) -> None:
    _, axes = plt.subplots(1, len(data.keys()), sharey=True, figsize=(15, 4))

    model_keys = list(data.keys())
    width = 0.7

    for idx, ax in enumerate(axes):
        ax.set_title(MODEL_MAP[model_keys[idx]])
        ax.tick_params(bottom=False)
        # ax.set_xticklabels(list(DS_MAP.values()))
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        pos = 0
        curr_model_data = data[model_keys[idx]]
        for ds in curr_model_data:
            if measure_key in ["iou_nodes", "iou_edges"]:
                plot_dict = {
                    k: v["iou"][measure_key] for k, v in curr_model_data[ds].items()
                }
            else:
                # ged and cosine
                plot_dict = {k: v[measure_key] for k, v in curr_model_data[ds].items()}

            b_plot = ax.boxplot(
                plot_dict.values(),
                positions=[pos],
                widths=width,
                patch_artist=True,
                notch=True,
                medianprops=dict(color="black"),
            )
            ax.set_xticklabels("" * len(DS_MAP))
            pos += 1
            set_custom_patch_sim(b_plot, pos)

    ax.legend(DS_MAP.values(), loc="center left", ncol=1, bbox_to_anchor=(1, 0.5))

    axes[0].set_ylabel("cosine_sim(SK,RK)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_behaviour_counts(data, out_path) -> None:
    _, axes = plt.subplots(1, 3, sharey=True, layout="constrained", figsize=(15, 4))

    type_map = {0: "Type 1 Behaviours", 1: "Type 2 Behaviours", 2: "Type 3 Behaviours"}

    color_map = {
        "llava-bench": "#DC602E",
        "mmbench": "#D7B49E",
        "seed": "#B8D5B8",
        "vqav2": "#05A8AA",
    }

    width = 0.2

    for idx, ax in enumerate(axes):
        ax.set_title(type_map[idx])
        ax.tick_params(bottom=False)
        ax.set_ylim(top=150)
        x = np.arange(len(MODEL_MAP))
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)

        multiplier = 0
        curr_failure = data[idx]
        for ds, values in curr_failure.items():
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, color=color_map[ds], label=ds)
            ax.bar_label(bars, padding=3)

            # Draw line over group
            # ax.hlines(y, x1, x2)
            # for model_idx, start_x in enumerate(x + offset):
            #     ax.hlines(totals[models[model_idx]][ds], start_x-width*0.5, start_x+width*0.5, color='black')

            multiplier += 1

        ax.set_xticks(x + width * 1.5, list(MODEL_MAP.values()))

    ax.legend(list(DS_MAP.values()), loc="upper right", ncol=1)
    axes[0].set_ylabel("N. Samples per Behaviour Type")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
