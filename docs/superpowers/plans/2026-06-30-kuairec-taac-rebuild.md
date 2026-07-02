# KuaiRec TAAC Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `CognID-GenRec` from a small semantic-ID content recommendation prototype into a KuaiRec-based short-video generative retrieval project with multi-behavior sequence modeling, semantic IDs, InfoNCE training, ANN retrieval, offline evaluation, and resume-ready evidence.

**Architecture:** Keep the current package boundary, tests, reports, and FastAPI demo style, but replace the toy content sample path with a real KuaiRec data pipeline. The new core path is KuaiRec raw files -> multi-behavior interaction sequences -> item/user/time feature tokens -> semantic ID tokenizer -> Causal Transformer + InfoNCE user/item embeddings -> Faiss-or-NumPy ANN retrieval -> Top-K evaluation and API explanations.

**Tech Stack:** Python 3.10+, pandas, numpy, scikit-learn, PyTorch, FastAPI, pytest, pyarrow, optional `faiss-cpu`.

## 2026-06-30 Implementation Status

P0/P1/P2 have been implemented and rerun on the real KuaiRec open dataset `small_matrix.csv`. Fresh verification evidence:

- `python scripts\prepare_kuairec_data.py --raw data\raw\kuairec --out data\processed\kuairec --matrix small` -> wrote 7,176 users, 10,731 items, 4,676,570 interactions, and 1,411 leave-one-out sequences from `data\raw\kuairec\KuaiRec 2.0\data\small_matrix.csv`.
- `python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method kmeans --out data\processed\kuairec\semantic_ids.parquet` -> wrote Semantic IDs for 10,731 items.
- `python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 1 --batch-size 64 --out data\processed\kuairec\kuai_retriever.pt` -> wrote checkpoint plus `ann_items.npy` and `ann_item_ids.json`.
- `reports/metrics/kuairec_experiment_panel.md` contains real-data Popular, ItemCF, and ANN Transformer offline metrics plus segmented metrics.
- Current local verification is recorded in `docs/EXPERIMENT_LOG.md`; full `python -m pytest -q` must remain the final gate after any further edits.

The built-in KuaiRec-like sample remains only as a smoke-test fallback for environments without raw data.

---

## External Evidence

KuaiRec is useful because it is a real-world short-video recommendation dataset with an almost fully observed user-item matrix. The README lists `small_matrix.csv`, `big_matrix.csv`, `social_network.csv`, `user_features.csv`, `item_daily_features.csv`, `item_categories.csv`, and `kuairec_caption_category.csv`; it also states that `watch_ratio = play_duration / video_duration` can be treated as the interaction label, and users may define a binary like signal such as `watch_ratio > 2.0`.

TAAC 2025 paper notes provide the algorithm direction: `Causal Transformer + InfoNCE + ANN retrieval` is the baseline paradigm; top schemes commonly use `Action Conditioning`; `Semantic ID` can be built with residual quantization such as RQ-KMeans; inference should output a user embedding and use Faiss ANN retrieval; time features should include absolute time, relative interval, session structure, and Fourier encodings.

## New Project Positioning

中文名：基于 KuaiRec 的多行为短视频生成式推荐系统

英文名：CognID-GenRec

Resume narrative:

> 基于 KuaiRec 开源短视频推荐数据集，构建多行为 next-item 生成式推荐系统；将 `watch_ratio` 派生为浏览、有效播放、完播、高兴趣等行为语义，融合用户画像、视频类目、文本描述、热度统计和时间间隔特征，参考 TAAC 2025 全模态生成式推荐思路实现 Action Conditioning、Semantic ID、InfoNCE 对比学习和 ANN 检索，并与 Popular、ItemCF、SASRec-style baseline 进行 HR@10、NDCG@10、Coverage、Diversity 对比。

## Migration Policy

The current toy pipeline is not deleted immediately. It becomes a historical baseline and smoke-test path until the KuaiRec pipeline is stable.

Rules:

1. Keep existing tests passing while adding the new KuaiRec path.
2. Do not claim TAAC champion reproduction. The project only implements a lightweight, explainable subset: action conditioning, semantic ID, InfoNCE, and ANN retrieval.
3. Do not claim online Kuaishou production results. All claims must say `基于 KuaiRec 开源数据集离线实验`.
4. Use `numpy` brute-force nearest-neighbor retrieval as the required path; add `faiss-cpu` only as an optional acceleration layer so the project remains runnable on Windows.

## Target File Map

Create:

- `src/cognid_genrec/kuairec/__init__.py`: package marker for KuaiRec-specific data and modeling.
- `src/cognid_genrec/kuairec/schemas.py`: typed column contracts for KuaiRec raw and processed tables.
- `src/cognid_genrec/kuairec/loaders.py`: load raw KuaiRec CSV files, normalize column names, and validate required files.
- `src/cognid_genrec/kuairec/behaviors.py`: convert `watch_ratio` into action labels and action weights.
- `src/cognid_genrec/kuairec/sequences.py`: build user chronological sequences, leave-one-out split, and session/time features.
- `src/cognid_genrec/kuairec/features.py`: build item text/category/stat features and user profile features.
- `src/cognid_genrec/models/sequence_encoder.py`: Causal Transformer encoder for user sequence embeddings.
- `src/cognid_genrec/models/contrastive.py`: InfoNCE loss, in-batch negatives, and item embedding table.
- `src/cognid_genrec/retrieval/ann_index.py`: NumPy exact ANN index with optional Faiss adapter.
- `src/cognid_genrec/evaluation/segmented_metrics.py`: HR/NDCG/Recall/Coverage/Diversity plus action and user-activity slices.
- `scripts/prepare_kuairec_data.py`: raw KuaiRec -> processed parquet/jsonl artifacts.
- `scripts/train_kuairec_semantic_id.py`: item semantic ID generation from caption/category/stat features.
- `scripts/train_kuairec_retriever.py`: train Causal Transformer + InfoNCE retrieval model.
- `scripts/evaluate_kuairec_retriever.py`: run baselines and model evaluation.
- `scripts/serve_kuairec_recommender.py`: FastAPI service for the new model path.
- `tests/test_kuairec_behaviors.py`
- `tests/test_kuairec_sequences.py`
- `tests/test_kuairec_features.py`
- `tests/test_contrastive_model.py`
- `tests/test_ann_index.py`
- `tests/test_kuairec_metrics.py`
- `tests/test_kuairec_recommend_api.py`

Modify:

- `pyproject.toml`: add optional dependency group `kuairec = ["faiss-cpu>=1.8.0"]` only if the environment supports it; keep core tests independent from Faiss.
- `README.md`: change project headline and add KuaiRec setup commands.
- `docs/IMPLEMENTATION_PLAN.md`: mark old plan as historical baseline and link to this rebuild plan.
- `docs/EXPERIMENT_LOG.md`: add a new KuaiRec experiment section.
- `docs/RESUME_NOTES.md`: replace small-sample wording with staged KuaiRec resume boundaries after implementation.

## Data Artifacts

Expected raw input directory:

```text
data/raw/kuairec/
  big_matrix.csv
  small_matrix.csv
  social_network.csv
  user_features.csv
  user_features_raw.csv
  item_daily_features.csv
  item_categories.csv
  kuairec_caption_category.csv
```

Expected processed output:

```text
data/processed/kuairec/
  users.parquet
  items.parquet
  interactions.parquet
  user_sequences.jsonl
  item_features.parquet
  user_features.parquet
  semantic_ids.parquet
  ann_items.npy
  ann_item_ids.json
  kuai_retriever.pt
```

## Progress Board

| Phase | Status | Deliverable | Verification |
|---|---|---|---|
| P0-1 KuaiRec schema and loaders | [x] Implemented | raw CSV validation and normalized tables | `pytest tests/test_kuairec_behaviors.py tests/test_kuairec_sequences.py -q` |
| P0-2 multi-behavior sequence builder | [x] Implemented | action labels, weights, leave-one-out sequences | `python scripts/prepare_kuairec_data.py --raw data/raw/kuairec --out data/processed/kuairec --matrix small` |
| P0-3 KuaiRec baselines | [x] Implemented | Popular, ItemCF, ANN Transformer smoke baseline | `python scripts/evaluate_kuairec_retriever.py --method popular --data data/processed/kuairec` |
| P0-4 semantic ID on short videos | [x] Implemented | KMeans/RQ-style semantic IDs from caption/category/stat features | `pytest tests/test_kuairec_features.py -q` |
| P1-1 Causal Transformer + InfoNCE | [x] Implemented | user/item embeddings and training report | `pytest tests/test_contrastive_model.py -q` |
| P1-2 ANN retrieval | [x] Implemented | NumPy exact index and optional Faiss index | `pytest tests/test_ann_index.py -q` |
| P1-3 segmented evaluation | [x] Implemented | HR@10, NDCG@10, Recall@K, Coverage, Diversity by action/user slice | `pytest tests/test_kuairec_metrics.py -q` |
| P2-1 API and explanations | [x] Implemented | `/recommend` with action, semantic ID, ANN score, reason fields | `pytest tests/test_kuairec_recommend_api.py -q` |
| P2-2 resume/report material | [x] Implemented | experiment panel, resume notes, interview Q&A | `python -m pytest -q` |

## Task 1: KuaiRec Schema, Loaders, and Behavior Labels

**Files:**
- Create: `src/cognid_genrec/kuairec/schemas.py`
- Create: `src/cognid_genrec/kuairec/loaders.py`
- Create: `src/cognid_genrec/kuairec/behaviors.py`
- Test: `tests/test_kuairec_behaviors.py`

- [ ] **Step 1: Write failing behavior tests**

```python
import pandas as pd


def test_watch_ratio_to_action_label():
    from cognid_genrec.kuairec.behaviors import action_from_watch_ratio

    assert action_from_watch_ratio(0.1) == "short_view"
    assert action_from_watch_ratio(0.7) == "valid_view"
    assert action_from_watch_ratio(1.2) == "complete_view"
    assert action_from_watch_ratio(2.4) == "high_interest"


def test_attach_actions_adds_label_and_weight():
    from cognid_genrec.kuairec.behaviors import attach_actions

    frame = pd.DataFrame({"watch_ratio": [0.2, 0.8, 1.4, 2.2]})
    result = attach_actions(frame)

    assert result["action"].tolist() == [
        "short_view",
        "valid_view",
        "complete_view",
        "high_interest",
    ]
    assert result["action_weight"].tolist() == [0.2, 1.0, 1.5, 2.5]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
python -m pytest tests\test_kuairec_behaviors.py -q
```

Expected: fails with `ModuleNotFoundError: No module named 'cognid_genrec.kuairec'`.

- [ ] **Step 3: Implement behavior mapping**

```python
from __future__ import annotations

import pandas as pd

ACTION_WEIGHTS = {
    "short_view": 0.2,
    "valid_view": 1.0,
    "complete_view": 1.5,
    "high_interest": 2.5,
}


def action_from_watch_ratio(watch_ratio: float) -> str:
    value = float(watch_ratio)
    if value < 0.3:
        return "short_view"
    if value < 1.0:
        return "valid_view"
    if value < 2.0:
        return "complete_view"
    return "high_interest"


def attach_actions(interactions: pd.DataFrame) -> pd.DataFrame:
    if "watch_ratio" not in interactions.columns:
        raise ValueError("interactions missing column: watch_ratio")
    result = interactions.copy()
    result["action"] = result["watch_ratio"].map(action_from_watch_ratio)
    result["action_weight"] = result["action"].map(ACTION_WEIGHTS).astype(float)
    return result
```

- [ ] **Step 4: Add loader contracts**

`schemas.py` must define required column sets:

```python
INTERACTION_COLUMNS = {"user_id", "video_id", "timestamp", "watch_ratio"}
ITEM_CATEGORY_COLUMNS = {"video_id"}
CAPTION_CATEGORY_COLUMNS = {"video_id"}
USER_FEATURE_COLUMNS = {"user_id"}
ITEM_DAILY_COLUMNS = {"video_id"}
```

`loaders.py` must expose:

```python
from pathlib import Path
import pandas as pd

from cognid_genrec.kuairec.schemas import INTERACTION_COLUMNS


def read_matrix(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = INTERACTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"matrix missing columns: {sorted(missing)}")
    return frame
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests\test_kuairec_behaviors.py -q
```

Expected: `2 passed`.

## Task 2: Build Multi-Behavior User Sequences

**Files:**
- Create: `src/cognid_genrec/kuairec/sequences.py`
- Create: `scripts/prepare_kuairec_data.py`
- Test: `tests/test_kuairec_sequences.py`

- [ ] **Step 1: Write sequence tests**

```python
import pandas as pd


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
    assert sequence["actions"] == ["high_interest", "valid_view", "complete_view", "short_view"]
    assert sequence["train_history_item_ids"] == ["11", "10"]
    assert sequence["validation_item_id"] == "12"
    assert sequence["test_item_id"] == "13"
```

- [ ] **Step 2: Implement sequence builder**

`build_kuairec_sequences()` must:

1. Sort by `user_id`, `timestamp`, `video_id`.
2. Keep all actions but allow a `min_action_weight` filter in later experiments.
3. Store `item_ids`, `actions`, `action_weights`, `timestamps`, `time_deltas`.
4. Use leave-one-out split with last two events as validation and test targets.

Use this implementation shape:

```python
def build_kuairec_sequences(frame: pd.DataFrame, min_history: int = 3) -> list[dict]:
    required = {"user_id", "video_id", "timestamp", "action", "action_weight"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"interactions missing columns: {sorted(missing)}")

    sorted_frame = frame.sort_values(["user_id", "timestamp", "video_id"])
    sequences = []
    for user_id, user_events in sorted_frame.groupby("user_id", sort=True):
        if len(user_events) < min_history + 2:
            continue
        timestamps = [int(value) for value in user_events["timestamp"].tolist()]
        deltas = [0, *[max(0, right - left) for left, right in zip(timestamps, timestamps[1:])]]
        item_ids = [str(value) for value in user_events["video_id"].tolist()]
        actions = [str(value) for value in user_events["action"].tolist()]
        weights = [float(value) for value in user_events["action_weight"].tolist()]
        sequences.append(
            {
                "user_id": str(user_id),
                "item_ids": item_ids,
                "actions": actions,
                "action_weights": weights,
                "timestamps": timestamps,
                "time_deltas": deltas,
                "train_history_item_ids": item_ids[:-2],
                "validation_item_id": item_ids[-2],
                "test_item_id": item_ids[-1],
            }
        )
    return sequences
```

- [ ] **Step 3: Add preparation script**

`scripts/prepare_kuairec_data.py` arguments:

```text
--raw data/raw/kuairec
--out data/processed/kuairec
--matrix small
--min-history 3
--max-users 0
```

The script must write:

- `interactions.parquet`
- `user_sequences.jsonl`
- `items.parquet`
- `users.parquet`

- [ ] **Step 4: Verify**

Run:

```powershell
python -m pytest tests\test_kuairec_sequences.py -q
python scripts\prepare_kuairec_data.py --raw data\raw\kuairec --out data\processed\kuairec --matrix small --min-history 3
```

Expected: test passes; script prints counts for users, items, interactions, and sequences.

## Task 3: KuaiRec Feature Builder

**Files:**
- Create: `src/cognid_genrec/kuairec/features.py`
- Test: `tests/test_kuairec_features.py`

- [ ] **Step 1: Build item text and statistic features**

Input fields:

- `video_id`
- category columns from `item_categories.csv`
- caption/category fields from `kuairec_caption_category.csv`
- daily statistics from `item_daily_features.csv`

Output columns:

- `item_id`
- `title`
- `body`
- `topic`
- `author_id`
- `duration`
- `quality_score`
- `hot_score`
- `caption_text`
- `category_text`

The existing `TfidfItemTextEncoder` expects `title`, `body`, and `topic`. `features.py` should adapt KuaiRec fields into those names instead of rewriting the tokenizer first.

- [ ] **Step 2: Write feature tests**

```python
import pandas as pd


def test_build_item_features_outputs_content_schema():
    from cognid_genrec.kuairec.features import build_item_features

    categories = pd.DataFrame({"video_id": [1], "feat": ["food|life"]})
    captions = pd.DataFrame({"video_id": [1], "caption": ["home cooking"], "category": ["food"]})
    daily = pd.DataFrame({"video_id": [1], "show_cnt": [10], "play_cnt": [8], "like_cnt": [2]})

    result = build_item_features(categories, captions, daily)

    assert result.loc[0, "item_id"] == "1"
    assert "home cooking" in result.loc[0, "body"]
    assert result.loc[0, "topic"] == "food"
    assert 0.0 <= result.loc[0, "quality_score"] <= 1.0
```

- [ ] **Step 3: Verify**

Run:

```powershell
python -m pytest tests\test_kuairec_features.py -q
```

Expected: feature tests pass.

## Task 4: Baselines and Metrics for KuaiRec

**Files:**
- Modify: `src/cognid_genrec/models/baselines/popular.py`
- Modify: `src/cognid_genrec/models/baselines/itemcf.py`
- Create: `src/cognid_genrec/evaluation/segmented_metrics.py`
- Create: `scripts/evaluate_kuairec_retriever.py`
- Test: `tests/test_kuairec_metrics.py`

- [ ] **Step 1: Add HR@K and weighted HR@K**

```python
def hit_rate_at_k(recommended: list[str], target: str, k: int) -> float:
    return 1.0 if str(target) in [str(item) for item in recommended[:k]] else 0.0


def weighted_hit_rate_at_k(
    recommended: list[str],
    target: str,
    target_weight: float,
    k: int,
) -> float:
    return float(target_weight) * hit_rate_at_k(recommended, target, k)
```

- [ ] **Step 2: Add segmented report**

Report slices:

- all users
- action of test target
- user sequence length bucket: `short`, `medium`, `long`
- item popularity bucket: `head`, `middle`, `tail`

- [ ] **Step 3: Verify baselines**

Run:

```powershell
python scripts\evaluate_kuairec_retriever.py --method popular --data data\processed\kuairec --out reports\metrics\kuairec_popular.md
python scripts\evaluate_kuairec_retriever.py --method itemcf --data data\processed\kuairec --out reports\metrics\kuairec_itemcf.md
python -m pytest tests\test_kuairec_metrics.py -q
```

Expected: baseline reports include `HR@10`, `NDCG@10`, `Recall@20`, `Coverage`, `Diversity`, and segmented rows.

## Task 5: Semantic ID for Short Videos

**Files:**
- Modify: `src/cognid_genrec/tokenizer/item_text_encoder.py`
- Modify: `src/cognid_genrec/tokenizer/semantic_id_tokenizer.py`
- Create: `scripts/train_kuairec_semantic_id.py`
- Test: extend `tests/test_semantic_id_tokenizer.py`

- [ ] **Step 1: Reuse existing tokenizer on KuaiRec feature schema**

Keep `SemanticIDTokenizer` and `RQVAESemanticIDTokenizer`, but require the KuaiRec feature builder to output `item_id`, `title`, `body`, and `topic`.

- [ ] **Step 2: Add RQ-style semantic ID experiment**

Run:

```powershell
python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method kmeans --out data\processed\kuairec\semantic_ids.parquet
python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method rqvae --out data\processed\kuairec\semantic_ids_rqvae.parquet
```

Expected reports:

- `reports/metrics/kuairec_semantic_id_kmeans.md`
- `reports/metrics/kuairec_semantic_id_rqvae.md`

Required metrics:

- item_count
- semantic_id_unique
- collision_rate
- topic_purity
- head/middle/tail collision rate

## Task 6: Causal Transformer + InfoNCE Retrieval Model

**Files:**
- Create: `src/cognid_genrec/models/sequence_encoder.py`
- Create: `src/cognid_genrec/models/contrastive.py`
- Create: `scripts/train_kuairec_retriever.py`
- Test: `tests/test_contrastive_model.py`

- [ ] **Step 1: Write model shape test**

```python
import torch


def test_sequence_encoder_outputs_user_embedding():
    from cognid_genrec.models.sequence_encoder import SequenceEncoder

    model = SequenceEncoder(num_items=100, num_actions=4, hidden_dim=16, max_len=8)
    item_ids = torch.tensor([[1, 2, 3, 0]])
    action_ids = torch.tensor([[2, 1, 3, 0]])
    time_deltas = torch.tensor([[0.0, 1.0, 2.0, 0.0]])
    output = model(item_ids=item_ids, action_ids=action_ids, time_deltas=time_deltas)

    assert output.shape == (1, 16)
```

- [ ] **Step 2: Implement minimal encoder**

The model must include:

- item embedding
- action embedding for Action Conditioning
- time delta projection
- causal transformer encoder
- last valid token pooling

- [ ] **Step 3: Implement InfoNCE**

`contrastive.py` must expose:

```python
def info_nce_loss(user_embeddings, positive_item_embeddings, temperature=0.07):
    logits = user_embeddings @ positive_item_embeddings.T
    logits = logits / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return torch.nn.functional.cross_entropy(logits, labels)
```

- [ ] **Step 4: Add training script**

Training command:

```powershell
python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 3 --batch-size 128 --hidden-dim 128 --out data\processed\kuairec\kuai_retriever.pt
```

Expected report:

```text
reports/metrics/kuairec_retriever_train.md
```

The report must include training loss, batch count, item count, user sequence count, and model config.

## Task 7: ANN Retrieval Index

**Files:**
- Create: `src/cognid_genrec/retrieval/ann_index.py`
- Test: `tests/test_ann_index.py`

- [ ] **Step 1: Implement NumPy exact index**

```python
import numpy as np


class NumpyANNIndex:
    def __init__(self, item_ids: list[str], embeddings: np.ndarray) -> None:
        self.item_ids = [str(item_id) for item_id in item_ids]
        vectors = embeddings.astype(np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
        self.embeddings = vectors / norms

    def search(self, query: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        vector = query.astype(np.float32)
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        scores = self.embeddings @ vector
        order = np.argsort(-scores)[:top_k]
        return [(self.item_ids[index], float(scores[index])) for index in order]
```

- [ ] **Step 2: Add optional Faiss adapter**

If `faiss` imports successfully, expose `FaissANNIndex`; otherwise keep the NumPy path as the default. Tests must not require Faiss.

- [ ] **Step 3: Verify**

Run:

```powershell
python -m pytest tests\test_ann_index.py -q
```

Expected: NumPy search returns the nearest item by cosine similarity.

## Task 8: KuaiRec Recommendation API

**Files:**
- Create: `scripts/serve_kuairec_recommender.py`
- Modify: `src/cognid_genrec/service/api.py` or create `src/cognid_genrec/service/kuairec_api.py`
- Test: `tests/test_kuairec_recommend_api.py`

- [ ] **Step 1: Define request and response**

Request:

```json
{
  "user_id": "123",
  "history_item_ids": ["10", "20", "30"],
  "history_actions": ["valid_view", "complete_view", "high_interest"],
  "top_k": 10
}
```

Response item fields:

- `item_id`
- `score`
- `semantic_id`
- `source`
- `action_context`
- `ann_rank`
- `reason`

- [ ] **Step 2: Add API test**

```python
from fastapi.testclient import TestClient


def test_kuairec_api_returns_ann_recommendations():
    from cognid_genrec.service.kuairec_api import create_kuairec_app

    client = TestClient(create_kuairec_app(data_dir="data/processed/kuairec_sample"))
    response = client.post(
        "/recommend",
        json={
            "user_id": "1",
            "history_item_ids": ["10", "20"],
            "history_actions": ["valid_view", "high_interest"],
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["user_id"] == "1"
    assert payload["items"]
    assert {"item_id", "score", "semantic_id", "source", "action_context", "ann_rank", "reason"} <= set(payload["items"][0])
```

- [ ] **Step 3: Verify**

Run:

```powershell
python -m pytest tests\test_kuairec_recommend_api.py -q
python scripts\serve_kuairec_recommender.py --data data\processed\kuairec
```

Expected: API test passes; local server starts and serves `/recommend`.

## Task 9: Documentation and Resume Evidence

**Files:**
- Modify: `README.md`
- Modify: `docs/IMPLEMENTATION_PLAN.md`
- Modify: `docs/EXPERIMENT_LOG.md`
- Modify: `docs/RESUME_NOTES.md`
- Create: `reports/metrics/kuairec_experiment_panel.md`

- [ ] **Step 1: README update**

README must include:

```powershell
python -m pip install -e ".[dev]"
python scripts\prepare_kuairec_data.py --raw data\raw\kuairec --out data\processed\kuairec --matrix small
python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method kmeans --out data\processed\kuairec\semantic_ids.parquet
python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 3 --batch-size 128 --out data\processed\kuairec\kuai_retriever.pt
python scripts\evaluate_kuairec_retriever.py --method ann_transformer --data data\processed\kuairec --out reports\metrics\kuairec_ann_transformer.md
```

- [ ] **Step 2: Resume notes update after verification**

Only after the corresponding tests and reports exist, use this resume wording:

```text
基于 KuaiRec 开源短视频推荐数据集构建多行为生成式推荐系统，按 user_id + timestamp 生成用户行为序列，并利用 watch_ratio 派生 short_view、valid_view、complete_view、high_interest 等行为标签；参考 TAAC 2025 全模态生成式推荐思路，实现 Action Conditioning、Semantic ID、InfoNCE 对比学习和 ANN 检索，统一评估 HR@10、NDCG@10、Recall@K、Coverage、Diversity 及分层指标。
```

- [ ] **Step 3: Keep risk boundaries**

`docs/RESUME_NOTES.md` must explicitly say:

```text
不得把项目表述为竞赛冠军级方案、企业线上系统、线上业务指标提升或全量多模态大模型训练；只能写“基于 KuaiRec 开源数据集的离线推荐实验”和“参考 TAAC 2025 思路实现轻量核心模块”。
```

## Verification Plan

Fast checks:

```powershell
python -m pytest tests\test_kuairec_behaviors.py tests\test_kuairec_sequences.py -q
python -m pytest tests\test_kuairec_features.py tests\test_ann_index.py -q
```

Model-path checks:

```powershell
python scripts\prepare_kuairec_data.py --raw data\raw\kuairec --out data\processed\kuairec --matrix small --min-history 3
python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method kmeans --out data\processed\kuairec\semantic_ids.parquet
python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 1 --batch-size 64 --out data\processed\kuairec\kuai_retriever.pt
python scripts\evaluate_kuairec_retriever.py --method ann_transformer --data data\processed\kuairec --out reports\metrics\kuairec_ann_transformer.md
```

Full regression:

```powershell
python -m pytest -q
```

## Resume Milestones

Stage A, after Tasks 1-4:

> 可写“基于 KuaiRec 构建短视频推荐离线评测链路，完成多行为标签构造、用户序列生成、Popular/ItemCF baseline 和 HR@10/NDCG@10/Coverage/Diversity 评估”。

Stage B, after Tasks 5-7:

> 可写“实现 Semantic ID、Action Conditioning、InfoNCE 训练和 ANN 检索，将逐样本打分推荐改造为用户向量到候选视频向量的检索式召回”。

Stage C, after Tasks 8-9:

> 可写“封装 FastAPI 推荐接口，输出 Top-K 短视频推荐、行为上下文、semantic ID、ANN 分数和解释字段，形成可演示的端到端生成式推荐原型”。

## Stop Conditions

Pause and report instead of pushing ahead when:

1. KuaiRec raw files are missing from `data/raw/kuairec`.
2. `small_matrix.csv` column names differ from expected `user_id`, `video_id`, `timestamp`, `watch_ratio`.
3. Training exceeds local memory on Windows. In that case, add `--max-users` and `--max-interactions` sampling arguments before continuing.
4. Faiss install fails. Continue with `NumpyANNIndex`; do not block the project on Faiss.
