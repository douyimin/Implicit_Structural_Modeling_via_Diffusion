"""Write a new reproducible 90/5/5 partition; never label it the historical split."""
import argparse
import json
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/split.json"))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    files = sorted(p.relative_to(args.data).as_posix() for p in args.data.rglob("*.npz"))
    if len(files) != 4000:
        parser.error(f"Expected the manuscript's 4000 NPZ samples, found {len(files)}")
    random.Random(args.seed).shuffle(files)
    split = dict(seed=args.seed, provenance="New release split; not the historical experiment manifest",
                 train=files[:3600], val=files[3600:3800], test=files[3800:])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(split, stream, indent=2)
        stream.write("\n")
    print(f"Saved 3600/200/200 split to {args.output}")


if __name__ == "__main__":
    main()
