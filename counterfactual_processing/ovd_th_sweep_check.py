import argparse
import json
import sys
from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().with_name("ovd_th_sweep.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that initial >= exact + OVD + similarity fallback counts "
            "throughout an OVD threshold sweep."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Sweep JSONL file (default: {DEFAULT_INPUT}).",
    )
    return parser.parse_args()


def check_sweep(path: Path) -> tuple[list[str], list[tuple]]:
    violations = []
    summaries = []
    seen_triplets = {}

    with path.open(encoding="utf-8") as sweep_file:
        for line_number, line in enumerate(sweep_file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            threshold = record["th"]
            for model, datasets in record["data"].items():
                for dataset, counts in datasets.items():
                    count_values = (
                        counts["initial"],
                        counts["with_exact_match"],
                        counts["with_ovd_match"],
                        counts["with_sim_match"],
                    )
                    triplet = (model, dataset, threshold)
                    if triplet in seen_triplets:
                        if seen_triplets[triplet] != count_values:
                            raise ValueError(
                                f"Conflicting counts for {model}/{dataset} at "
                                f"th={threshold} on line {line_number}."
                            )
                        continue
                    seen_triplets[triplet] = count_values

                    initial, exact, ovd, similarity = count_values
                    matched = exact + ovd + similarity
                    summaries.append(
                        (
                            model,
                            dataset,
                            threshold,
                            initial,
                            exact,
                            ovd,
                            similarity,
                            matched,
                        )
                    )

                    if initial < matched:
                        violations.append(
                            f"line {line_number}, th={threshold}, "
                            f"{model}/{dataset}: initial={initial} < "
                            f"matched={matched} (exact={exact}, OVD={ovd}, "
                            f"similarity={similarity})"
                        )

    return violations, summaries


def format_count(count: int, initial: int) -> str:
    percentage = f"{100 * count / initial:.2f}%" if initial else "n/a"
    return f"{count}/{initial} ({percentage})"


def main() -> int:
    args = parse_args()
    try:
        violations, summaries = check_sweep(args.input)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"Error reading {args.input}: {error}", file=sys.stderr)
        return 2

    for (
        model,
        dataset,
        threshold,
        initial,
        exact,
        ovd,
        semantic,
        matched,
    ) in summaries:
        print(
            f"th={threshold} | {model}/{dataset} | "
            f"exact={format_count(exact, initial)} | "
            f"OVD={format_count(ovd, initial)} | "
            f"semantic={format_count(semantic, initial)} | "
            f"total={format_count(matched, initial)}"
        )

    if violations:
        print(f"Found {len(violations)} count violation(s):", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1

    print(f"All count checks passed for {args.input}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
