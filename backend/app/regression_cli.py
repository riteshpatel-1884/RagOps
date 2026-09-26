"""
Phase 5: Regression Testing — main entry point.

Usage:
    # Record a version (runs full retrieval + generation eval, stores it)
    python regression_cli.py record v1 --chunk_size 400 --top_k 5
    python regression_cli.py record v2 --chunk_size 200 --top_k 3 --set_baseline

    # Check a version against the current baseline
    python regression_cli.py check v2

    # Check a version against a specific other version
    python regression_cli.py check v2 --against v1

    # List all recorded versions
    python regression_cli.py list

    # Change which version future checks compare against
    python regression_cli.py set-baseline v1
"""
import argparse

from pipeline import PipelineConfig
from regression import RegressionHistory, record_version, compare_versions, format_report


def parse_args():
    p = argparse.ArgumentParser(description="Phase 5: record pipeline versions and check for regressions")
    sub = p.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="Run a config and store it as a named version")
    rec.add_argument("name")
    rec.add_argument("--chunk_size", type=int, default=400)
    rec.add_argument("--chunk_overlap", type=int, default=80)
    rec.add_argument(
        "--embedder", type=str, default="bge-small-en-v1.5",
        choices=["bge-m3", "qwen3-embedding-0.6b", "bge-small-en-v1.5"],
    )
    rec.add_argument("--top_k", type=int, default=5)
    rec.add_argument("--generator", type=str, default="groq", choices=["groq"])
    rec.add_argument("--judge", type=str, default="offline", choices=["offline", "llm"])
    rec.add_argument("--notes", type=str, default="")
    rec.add_argument("--set_baseline", action="store_true")

    chk = sub.add_parser("check", help="Compare a version against the baseline (or another named version)")
    chk.add_argument("name")
    chk.add_argument("--against", type=str, default=None)

    sub.add_parser("list", help="List all recorded versions")

    base = sub.add_parser("set-baseline", help="Mark a version as the baseline for future checks")
    base.add_argument("name")

    rm = sub.add_parser("delete", help="Delete a recorded version")
    rm.add_argument("name")

    return p.parse_args()


def main():
    args = parse_args()
    history = RegressionHistory()

    if args.command == "record":
        config = PipelineConfig(
            chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
            embedder_name=args.embedder, top_k=args.top_k, generator_name=args.generator,
        )
        print(f"Running evaluation for version '{args.name}'...")
        try:
            record = record_version(
                args.name, config, judge=args.judge, notes=args.notes,
                set_as_baseline=args.set_baseline, history=history,
            )
        except RuntimeError as e:
            print(f"FAILED: {e}")
            return

        print(f"\nRecorded version '{record.name}':")
        for k, v in record.metrics.items():
            print(f"  {k:<15}: {v:.3f}" if isinstance(v, float) else f"  {k:<15}: {v}")
        if history.get_baseline_name() == record.name:
            print("\n(set as baseline)")

    elif args.command == "list":
        versions = history.list_versions()
        if not versions:
            print("No versions recorded yet. Try: python regression_cli.py record v1 --chunk_size 400 --top_k 5")
            return
        baseline = history.get_baseline_name()
        print(f"{'Version':<20}{'Timestamp':<22}{'':<14}Config")
        print("-" * 100)
        for v in versions:
            marker = "* baseline" if v["name"] == baseline else ""
            c = v["config"]
            config_str = f"chunk={c['chunk_size']}/{c['chunk_overlap']} top_k={c['top_k']} embedder={c['embedder_name']} gen={c['generator_name']}"
            print(f"{v['name']:<20}{v['timestamp'][:19]:<22}{marker:<14}{config_str}")

    elif args.command == "set-baseline":
        try:
            history.set_baseline(args.name)
            print(f"Baseline set to '{args.name}'")
        except ValueError as e:
            print(str(e))

    elif args.command == "delete":
        try:
            history.delete_version(args.name)
            print(f"Deleted version '{args.name}'")
        except ValueError as e:
            print(str(e))

    elif args.command == "check":
        new_version = history.get_version(args.name)
        if new_version is None:
            print(f"No such version: '{args.name}'. Run 'python regression_cli.py list' to see recorded versions.")
            return

        against_name = args.against or history.get_baseline_name()
        if against_name is None:
            print("No baseline set and no --against given. Record a first version, or pass --against <name>.")
            return
        if against_name == args.name:
            print("Can't compare a version against itself — pass a different --against.")
            return

        old_version = history.get_version(against_name)
        if old_version is None:
            print(f"No such version to compare against: '{against_name}'")
            return

        comparison = compare_versions(old_version["metrics"], new_version["metrics"])
        print(format_report(against_name, args.name, comparison))


if __name__ == "__main__":
    main()