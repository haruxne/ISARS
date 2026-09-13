"""Command-line interface for the public ISARS reproduction package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .interval_selection import select_intervals
from .results import format_summary, load_reported_results, verify_reported_results

DEFAULT_RESULTS = Path(__file__).with_name("reported_results.json")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _verify_command(args: argparse.Namespace) -> int:
    report = load_reported_results(args.results)
    verified = verify_reported_results(report)
    print(format_summary(report))
    print("\nExact paired sign tests")
    for cohort, comparisons in verified.items():
        for baseline, result in comparisons.items():
            print(
                f"{cohort}/{baseline}: "
                f"W/T/L={result.wins}/{result.ties}/{result.losses}, "
                f"adjusted p={result.adjusted_p_value:.6g}"
            )
    print("\nVERIFIED: aggregate arithmetic only; original execution files were not rehashed")
    return 0


def _select_command(args: argparse.Namespace) -> int:
    payload = _load_json(args.input)
    selected = select_intervals(
        payload["scores"],
        frame_count=payload["frame_count"],
        target_position=payload["target_position"],
        window_size=payload.get("window_size", 16),
        stride=payload.get("stride", 1),
    )
    output = {name: interval.to_dict() for name, interval in selected.items()}
    rendered = json.dumps(output, indent=2) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="isars",
        description="Reproduce ISARS aggregate checks and interval selection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify = subparsers.add_parser("verify", help="verify reported aggregate arithmetic")
    verify.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    verify.set_defaults(handler=_verify_command)

    select = subparsers.add_parser("select", help="select matched-budget intervals")
    select.add_argument("--input", type=Path, required=True)
    select.add_argument("--output", type=Path)
    select.set_defaults(handler=_select_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
