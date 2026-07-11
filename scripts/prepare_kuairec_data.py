from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cognid_genrec.evaluation.protocol import (
    build_sequential_candidate_manifest,
    write_candidate_manifest,
)
from cognid_genrec.kuairec.behaviors import attach_actions
from cognid_genrec.kuairec.features import build_item_features, build_user_features
from cognid_genrec.kuairec.loaders import load_raw_kuairec, resolve_raw_file
from cognid_genrec.kuairec.sequences import (
    build_kuairec_sequences,
    write_kuairec_sequences,
)
from cognid_genrec.kuairec.schemas import RAW_FILES


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KuaiRec raw CSV files.")
    parser.add_argument("--raw", default="data/raw/kuairec")
    parser.add_argument("--out", default="data/processed/kuairec")
    parser.add_argument("--matrix", choices=["small", "big"], default="small")
    parser.add_argument("--min-history", type=int, default=3)
    parser.add_argument("--max-users", type=int, default=0)
    parser.add_argument("--max-interactions", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw)
    matrix_path = resolve_raw_file(raw_dir, RAW_FILES[args.matrix], required=False)
    if matrix_path.exists():
        tables = load_raw_kuairec(raw_dir, matrix=args.matrix)
        data_note = f"raw={raw_dir}; matrix={matrix_path}"
    else:
        tables = sample_kuairec_tables()
        data_note = (
            f"raw files missing under {raw_dir}; wrote built-in KuaiRec-like sample "
            "for local smoke tests"
        )

    interactions = tables["interactions"].copy()
    if args.max_users > 0:
        keep_users = interactions["user_id"].drop_duplicates().head(args.max_users)
        interactions = interactions[interactions["user_id"].isin(keep_users)]
    if args.max_interactions > 0:
        interactions = interactions.head(args.max_interactions)

    interactions = attach_actions(interactions)
    sequences = build_kuairec_sequences(interactions, min_history=args.min_history)
    items = build_item_features(
        tables["item_categories"],
        tables["caption_category"],
        tables["item_daily_features"],
    )
    users = build_user_features(tables["user_features"], interactions)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    items.to_parquet(out_dir / "items.parquet", index=False)
    items.to_parquet(out_dir / "item_features.parquet", index=False)
    users.to_parquet(out_dir / "users.parquet", index=False)
    users.to_parquet(out_dir / "user_features.parquet", index=False)
    write_kuairec_sequences(sequences, out_dir / "user_sequences.jsonl")
    manifest = build_sequential_candidate_manifest(
        sequences=sequences,
        candidate_item_ids=interactions["video_id"].astype(str),
        dataset_matrix=args.matrix,
        random_seed=args.seed,
    )
    write_candidate_manifest(manifest, out_dir / "candidate_manifest.json")

    print(data_note)
    print(
        "Prepared KuaiRec data: "
        f"users={users['user_id'].nunique()}, "
        f"items={items['item_id'].nunique()}, "
        f"interactions={len(interactions)}, "
        f"sequences={len(sequences)}, "
        f"candidates={manifest.candidate_count}"
    )


def sample_kuairec_tables() -> dict[str, pd.DataFrame]:
    interactions = pd.DataFrame(
        [
            [1, 10, 100, 0.4],
            [1, 11, 110, 1.1],
            [1, 12, 120, 2.3],
            [1, 13, 130, 0.8],
            [1, 14, 140, 1.6],
            [2, 11, 100, 0.2],
            [2, 15, 105, 0.7],
            [2, 16, 111, 1.4],
            [2, 17, 120, 2.5],
            [2, 18, 132, 0.9],
            [3, 12, 90, 2.1],
            [3, 13, 99, 0.6],
            [3, 14, 120, 1.2],
            [3, 15, 130, 0.1],
            [3, 18, 145, 2.2],
            [4, 10, 80, 0.9],
            [4, 16, 96, 1.0],
            [4, 17, 115, 1.8],
            [4, 18, 135, 2.4],
            [4, 14, 155, 0.5],
        ],
        columns=["user_id", "video_id", "timestamp", "watch_ratio"],
    )
    categories = pd.DataFrame(
        {
            "video_id": [10, 11, 12, 13, 14, 15, 16, 17, 18],
            "feat": [
                "food|life",
                "food|home",
                "music|guitar",
                "sports|running",
                "travel|city",
                "education|math",
                "music|piano",
                "sports|basketball",
                "travel|nature",
            ],
        }
    )
    captions = pd.DataFrame(
        {
            "video_id": [10, 11, 12, 13, 14, 15, 16, 17, 18],
            "caption": [
                "quick home cooking",
                "street food review",
                "guitar practice clip",
                "morning running vlog",
                "city walk short video",
                "math trick explained",
                "piano cover practice",
                "basketball training tips",
                "nature travel highlights",
            ],
            "category": [
                "food",
                "food",
                "music",
                "sports",
                "travel",
                "education",
                "music",
                "sports",
                "travel",
            ],
        }
    )
    daily = pd.DataFrame(
        {
            "video_id": [10, 11, 12, 13, 14, 15, 16, 17, 18],
            "author_id": ["a1", "a2", "a3", "a4", "a5", "a6", "a3", "a4", "a5"],
            "video_duration": [12, 18, 22, 20, 25, 30, 24, 21, 28],
            "show_cnt": [100, 120, 90, 80, 70, 60, 85, 95, 75],
            "play_cnt": [80, 100, 70, 60, 50, 45, 65, 75, 58],
            "like_cnt": [12, 18, 16, 10, 8, 9, 13, 17, 11],
            "comment_cnt": [2, 3, 4, 1, 1, 2, 3, 2, 2],
            "share_cnt": [1, 2, 2, 1, 1, 1, 2, 2, 1],
        }
    )
    users = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 4],
            "user_active_degree": [
                "high_active",
                "middle_active",
                "middle_active",
                "low_active",
            ],
        }
    )
    return {
        "interactions": interactions,
        "item_categories": categories,
        "caption_category": captions,
        "item_daily_features": daily,
        "user_features": users,
    }


if __name__ == "__main__":
    main()
