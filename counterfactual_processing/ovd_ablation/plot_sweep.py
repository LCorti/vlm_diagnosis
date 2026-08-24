import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "ovd_th_sweep.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "ovd_th_sweep.pdf"
REQUIRED_COUNTS = {
    "initial",
    "with_exact_match",
    "with_ovd_match",
    "with_sim_match",
}
SIMILARITY_MEDIAN_KEY = "semantic_similarity_median"
DS_MAP = {
    "llava-bench": "LLaVa-Bench",
    "mmbench": "MMBench",
    "seed": "SEED-Bench 2",
    "vqav2": "VQA v2",
}
MODEL_MAP = {
    "internvl2": "InternVL2",
    "llava-1.6": "LLaVa-1.6",
    "qwen2_5_vl": "Qwen2.5-VL",
    "sharegpt4v": "ShareGPT4V",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot matching counts across OWLv2 thresholds for every "
            "model/dataset combination in a sweep JSONL file."
        )
    )
    parser.add_argument(
        "--threshold-divisor",
        type=float,
        default=10.0,
        help=(
            "Divide stored 'th' values by this number for the x-axis. "
            "The sweep stores 0..10 for actual thresholds 0.0..1.0 "
            "(default: 10)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Only plot these models (default: all models).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Only plot these datasets (default: all datasets).",
    )
    parser.add_argument(
        "--show-medians",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Display similarity median labels (default: disabled).",
    )
    parser.add_argument(
        "--individual-output-dir",
        type=Path,
        help=(
            "Also save each model/dataset plot and a standalone legend "
            "to this directory."
        ),
    )
    return parser.parse_args()


def load_sweep(path: Path) -> list[dict]:
    """Load and validate one sweep record per threshold."""
    records = []
    seen_thresholds = set()

    with path.open(encoding="utf-8") as sweep_file:
        for line_number, line in enumerate(sweep_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                ) from error

            if set(record) != {"th", "data"}:
                raise ValueError(
                    f"Line {line_number} must contain exactly the keys 'th' and 'data'."
                )
            if not isinstance(record["th"], (int, float)):
                raise TypeError(f"Line {line_number}: 'th' must be numeric.")
            if not isinstance(record["data"], dict):
                raise TypeError(f"Line {line_number}: 'data' must be an object.")

            threshold = record["th"]
            if threshold in seen_thresholds:
                raise ValueError(
                    f"Duplicate record for threshold {threshold} on line {line_number}."
                )
            seen_thresholds.add(threshold)
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}.")

    records.sort(key=lambda record: record["th"])
    validate_counts(records)
    return records


def validate_counts(records: list[dict]) -> None:
    combinations = None

    for record in records:
        current_combinations = set()
        for model, datasets in record["data"].items():
            if not isinstance(datasets, dict):
                raise TypeError(f"Datasets for model '{model}' must be an object.")
            for dataset, counts in datasets.items():
                current_combinations.add((model, dataset))
                if not isinstance(counts, dict):
                    raise TypeError(
                        f"Counts for '{model}/{dataset}' must be an object."
                    )
                missing = (REQUIRED_COUNTS | {SIMILARITY_MEDIAN_KEY}) - counts.keys()
                if missing:
                    raise ValueError(
                        f"Missing measurements for '{model}/{dataset}' at "
                        f"th={record['th']}: {', '.join(sorted(missing))}."
                    )
                if any(
                    not isinstance(counts[key], (int, float)) for key in REQUIRED_COUNTS
                ):
                    raise ValueError(
                        f"All counts for '{model}/{dataset}' at th={record['th']} "
                        "must be numeric."
                    )
                similarity_median = counts[SIMILARITY_MEDIAN_KEY]
                if similarity_median is not None and not isinstance(
                    similarity_median, (int, float)
                ):
                    raise ValueError(
                        f"'{SIMILARITY_MEDIAN_KEY}' for '{model}/{dataset}' at "
                        f"th={record['th']} must be numeric or null."
                    )

        if combinations is None:
            combinations = current_combinations
        elif current_combinations != combinations:
            missing = combinations - current_combinations
            extra = current_combinations - combinations
            raise ValueError(
                f"Model/dataset combinations differ at th={record['th']}; "
                f"missing={sorted(missing)}, extra={sorted(extra)}."
            )


def select_names(
    available: list[str], requested: list[str] | None, kind: str
) -> list[str]:
    if requested is None:
        return available

    unknown = set(requested) - set(available)
    if unknown:
        raise ValueError(
            f"Unknown {kind}(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(available)}."
        )
    return requested


def _plot_combination(
    axis,
    records: list[dict],
    thresholds: list[float],
    model: str,
    dataset: str,
    show_medians: bool,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    line_specs = [
        ("with_exact_match", "Exact match", "tab:blue", ".", ":", 1.8),
        ("with_ovd_match", "OVD match", "tab:orange", "o", "-", 1.8),
        (
            "with_sim_match",
            (
                "Similarity fallback (median labels)"
                if show_medians
                else "Similarity fallback"
            ),
            "tab:green",
            "^",
            "-",
            1.8,
        ),
    ]
    count_series = {
        key: [record["data"][model][dataset][key] for record in records]
        for key in REQUIRED_COUNTS
    }
    for key, label, color, marker, line_style, line_width in line_specs:
        axis.plot(
            thresholds,
            count_series[key],
            label=label,
            color=color,
            marker=marker,
            linestyle=line_style,
            linewidth=line_width,
            markersize=4,
        )

    if show_medians:
        similarity_medians = [
            record["data"][model][dataset][SIMILARITY_MEDIAN_KEY] for record in records
        ]
        for threshold, similarity_count, similarity_median in zip(
            thresholds,
            count_series["with_sim_match"],
            similarity_medians,
        ):
            median_label = (
                "N/A" if similarity_median is None else f"{similarity_median:.2f}"
            )
            axis.annotate(
                median_label,
                xy=(threshold, similarity_count),
                xytext=(2, -6),
                textcoords="offset points",
                color="black",
                fontsize=7,
                ha="center",
                va="top",
            )

    total_matched = [
        exact + ovd + similarity
        for exact, ovd, similarity in zip(
            count_series["with_exact_match"],
            count_series["with_ovd_match"],
            count_series["with_sim_match"],
        )
    ]
    axis.plot(
        thresholds,
        total_matched,
        label="Total matched",
        color="tab:red",
        marker="s",
        linewidth=2.2,
        markersize=4,
    )
    axis.plot(
        thresholds,
        count_series["initial"],
        label="Initial relations",
        color="#29D8D7",
        linestyle="--",
        linewidth=1.3,
    )

    axis.set_title(f"{MODEL_MAP[model]} / {DS_MAP[dataset]}", fontsize=10)
    axis.grid(alpha=0.25)
    axis.set_xticks(thresholds)
    axis.tick_params(axis="x", labelrotation=45)
    if show_ylabel:
        axis.set_ylabel("Count")
    if show_xlabel:
        axis.set_xlabel("OWLv2 threshold")


def plot_sweep(
    records: list[dict],
    output_path: Path,
    threshold_divisor: float,
    requested_models: list[str] | None,
    requested_datasets: list[str] | None,
    show_medians: bool = True,
    individual_output_dir: Path | None = None,
) -> plt.Figure:
    if threshold_divisor == 0:
        raise ValueError("--threshold-divisor cannot be zero.")

    first_data = records[0]["data"]
    available_models = sorted(first_data)
    available_datasets = sorted(
        {dataset for datasets in first_data.values() for dataset in datasets}
    )
    models = select_names(available_models, requested_models, "model")
    datasets = select_names(available_datasets, requested_datasets, "dataset")

    thresholds = [record["th"] / threshold_divisor for record in records]
    figure, axes = plt.subplots(
        len(models),
        len(datasets),
        figsize=(4.6 * len(datasets), 3.4 * len(models)),
        sharex=True,
        squeeze=False,
    )

    if individual_output_dir is not None:
        individual_output_dir.mkdir(parents=True, exist_ok=True)
    output_suffix = output_path.suffix or ".pdf"

    for row, model in enumerate(models):
        for column, dataset in enumerate(datasets):
            axis = axes[row][column]
            if dataset not in first_data[model]:
                axis.set_visible(False)
                continue

            _plot_combination(
                axis,
                records,
                thresholds,
                model,
                dataset,
                show_medians,
                show_xlabel=row == len(models) - 1,
                show_ylabel=column == 0,
            )

            if individual_output_dir is not None:
                individual_figure, individual_axis = plt.subplots(figsize=(4.6, 3.4))
                _plot_combination(
                    individual_axis,
                    records,
                    thresholds,
                    model,
                    dataset,
                    show_medians,
                    show_xlabel=True,
                    show_ylabel=True,
                )
                individual_figure.tight_layout()
                individual_figure.savefig(
                    individual_output_dir / f"{model}__{dataset}{output_suffix}",
                    dpi=200,
                    bbox_inches="tight",
                )
                plt.close(individual_figure)

    visible_axes = [axis for row in axes for axis in row if axis.get_visible()]
    if not visible_axes:
        raise ValueError("No model/dataset combinations selected.")
    handles, labels = visible_axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 5),
        bbox_to_anchor=(0.5, 0.975),
    )

    # Save the legend as a separate figure when individual charts are asked.
    if individual_output_dir is not None:
        legend_figure = plt.figure(figsize=(max(6.0, 1.8 * len(labels)), 1.0))
        legend_figure.legend(
            handles,
            labels,
            loc="center",
            ncol=len(labels),
            frameon=False,
        )
        legend_figure.savefig(
            individual_output_dir / f"legend{output_suffix}",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(legend_figure)

    # figure.suptitle("OWLv2 Output Threshold Sweep", y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return figure


if __name__ == "__main__":
    args = parse_args()
    try:
        records = load_sweep(DEFAULT_INPUT)
        figure = plot_sweep(
            records,
            DEFAULT_OUTPUT,
            args.threshold_divisor,
            args.models,
            args.datasets,
            show_medians=args.show_medians,
            individual_output_dir=args.individual_output_dir,
        )
        print(f"Saved plot to {DEFAULT_OUTPUT}")
        if args.individual_output_dir is not None:
            print(f"Saved individual plots to {args.individual_output_dir}")
        plt.close(figure)
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
