from __future__ import annotations

import argparse
from pathlib import Path

from cognid_genrec.data.sequence_builder import (
    build_user_sequences,
    write_user_sequences,
)
from cognid_genrec.data.validate import validate_interactions, validate_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare content recommendation samples.")
    parser.add_argument("--items", required=True, help="Path to items CSV.")
    parser.add_argument("--interactions", required=True, help="Path to interactions CSV.")
    parser.add_argument("--out", required=True, help="Output directory for processed files.")
    args = parser.parse_args()

    items = validate_items(args.items)
    interactions = validate_interactions(args.interactions)
    sequences = build_user_sequences(interactions)

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)
    items.to_csv(output_dir / "items.csv", index=False)
    items.to_parquet(output_dir / "items.parquet", index=False)
    interactions.to_csv(output_dir / "interactions.csv", index=False)
    output_path = output_dir / "user_sequences.jsonl"
    write_user_sequences(sequences, output_path)
    print(f"Wrote {len(sequences)} user sequences to {output_path}")


if __name__ == "__main__":
    main()
