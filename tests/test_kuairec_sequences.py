import json

import pandas as pd
from pandas.errors import ParserError


def test_build_kuairec_sequences_sorts_and_splits():
    from cognid_genrec.kuairec.behaviors import attach_actions
    from cognid_genrec.kuairec.sequences import build_kuairec_sequences

    frame = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1],
            "video_id": [10, 11, 12, 13],
            "timestamp": [100, 90, 110, 130],
            "watch_ratio": [0.8, 2.1, 1.2, 0.2],
        }
    )
    result = build_kuairec_sequences(attach_actions(frame), min_history=2)

    assert len(result) == 1
    sequence = result[0]
    assert sequence["item_ids"] == ["11", "10", "12", "13"]
    assert sequence["actions"] == [
        "high_interest",
        "valid_view",
        "complete_view",
        "short_view",
    ]
    assert sequence["train_history_item_ids"] == ["11", "10"]
    assert sequence["validation_history_item_ids"] == ["11", "10"]
    assert sequence["test_history_item_ids"] == ["11", "10", "12"]
    assert sequence["validation_item_id"] == "12"
    assert sequence["test_item_id"] == "13"
    assert sequence["test_action"] == "short_view"
    assert sequence["time_deltas"] == [0, 10, 10, 20]


def test_write_kuairec_sequences_jsonl(tmp_path):
    from cognid_genrec.kuairec.sequences import write_kuairec_sequences

    output_path = tmp_path / "user_sequences.jsonl"
    write_kuairec_sequences([{"user_id": "1", "item_ids": ["10"]}], output_path)

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert rows == [{"user_id": "1", "item_ids": ["10"]}]


def test_load_raw_kuairec_finds_nested_zenodo_layout(tmp_path):
    from cognid_genrec.kuairec.loaders import load_raw_kuairec

    data_dir = tmp_path / "KuaiRec 2.0" / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "small_matrix.csv").write_text(
        "user_id,video_id,play_duration,video_duration,time,date,timestamp,watch_ratio\n"
        "1,10,5,10,100,2020-01-01,100,0.5\n",
        encoding="utf-8",
    )
    (data_dir / "item_categories.csv").write_text(
        "video_id,feat\n10,food\n", encoding="utf-8"
    )
    (data_dir / "kuairec_caption_category.csv").write_text(
        "video_id,caption,category\n10,cooking,food\n",
        encoding="utf-8",
    )
    (data_dir / "item_daily_features.csv").write_text(
        "video_id,show_cnt,play_cnt,like_cnt\n10,10,8,1\n",
        encoding="utf-8",
    )
    (data_dir / "user_features.csv").write_text(
        "user_id,user_active_degree\n1,high_active\n",
        encoding="utf-8",
    )

    tables = load_raw_kuairec(tmp_path, matrix="small")

    assert tables["interactions"]["video_id"].tolist() == [10]
    assert tables["item_categories"]["feat"].tolist() == ["food"]
    assert tables["caption_category"]["caption"].tolist() == ["cooking"]
    assert tables["item_daily_features"]["show_cnt"].tolist() == [10]
    assert tables["user_features"]["user_active_degree"].tolist() == ["high_active"]


def test_read_optional_csv_falls_back_when_c_parser_overflows(tmp_path, monkeypatch):
    from cognid_genrec.kuairec.loaders import read_optional_csv

    csv_path = tmp_path / "kuairec_caption_category.csv"
    csv_path.write_text("video_id,caption\n10,cooking\n", encoding="utf-8")
    original_read_csv = pd.read_csv
    calls = []

    def fake_read_csv(path, *args, **kwargs):
        engine = kwargs.get("engine", "c")
        calls.append(engine)
        if engine == "c":
            raise ParserError("Buffer overflow caught - possible malformed input file")
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    frame = read_optional_csv(csv_path, {"video_id"})

    assert calls == ["c", "python"]
    assert frame["caption"].tolist() == ["cooking"]


def test_read_matrix_fills_missing_timestamp_from_time(tmp_path):
    from cognid_genrec.kuairec.loaders import read_matrix

    csv_path = tmp_path / "small_matrix.csv"
    csv_path.write_text(
        "user_id,video_id,play_duration,video_duration,time,timestamp,watch_ratio\n"
        "1,10,5,10,123,,0.5\n",
        encoding="utf-8",
    )

    frame = read_matrix(csv_path)

    assert frame.loc[0, "timestamp"] == 123
