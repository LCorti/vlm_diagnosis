import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "ovd_th_sweep.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "ovd_th_sweep.png"
REQUIRED_COUNTS = {
    "initial",
    "with_exact_match",
    "with_ovd_match",
    "with_sim_match",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot matching counts across OWLv2 thresholds for every "
            "model/dataset combination in a sweep JSONL file."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Sweep JSONL file (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure (default: {DEFAULT_OUTPUT}).",
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
        "--show",
        action="store_true",
        help="Open the figure after saving it.",
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
                missing = REQUIRED_COUNTS - counts.keys()
                if missing:
                    raise ValueError(
                        f"Missing counts for '{model}/{dataset}' at th={record['th']}: "
                        f"{', '.join(sorted(missing))}."
                    )
                if any(
                    not isinstance(counts[key], (int, float)) for key in REQUIRED_COUNTS
                ):
                    raise ValueError(
                        f"All counts for '{model}/{dataset}' at th={record['th']} "
                        "must be numeric."
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


def plot_sweep(
    records: list[dict],
    output_path: Path,
    threshold_divisor: float,
    requested_models: list[str] | None,
    requested_datasets: list[str] | None,
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

    line_specs = [
        ("with_exact_match", "Exact match", "tab:blue", ".", ":", 1.8),
        ("with_ovd_match", "OVD match", "tab:orange", "o", "-", 1.8),
        ("with_sim_match", "Similarity fallback", "tab:green", "^", "-", 1.8),
    ]

    for row, model in enumerate(models):
        for column, dataset in enumerate(datasets):
            axis = axes[row][column]
            if dataset not in first_data[model]:
                axis.set_visible(False)
                continue

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
                color="0.4",
                linestyle="--",
                linewidth=1.3,
            )

            axis.set_title(f"{model} / {dataset}", fontsize=10)
            axis.grid(alpha=0.25)
            axis.set_xticks(thresholds)
            axis.tick_params(axis="x", labelrotation=45)
            if column == 0:
                axis.set_ylabel("Count")
            if row == len(models) - 1:
                axis.set_xlabel("OWLv2 threshold")

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
    figure.suptitle("Matching counts across OWLv2 thresholds", y=0.998)
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    return figure


def main() -> int:
    args = parse_args()
    try:
        records = load_sweep(args.input)
        figure = plot_sweep(
            records,
            args.output,
            args.threshold_divisor,
            args.models,
            args.datasets,
        )
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Saved plot to {args.output}")
    if args.show:
        plt.show()
    else:
        plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
