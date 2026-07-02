# CognID-GenRec Implementation Plan

> 当前主计划：`docs/superpowers/plans/2026-06-30-kuairec-taac-rebuild.md`。

## 当前定位

`CognID-GenRec` 是基于 KuaiRec 开源短视频推荐数据集的多行为生成式推荐离线实验项目。旧版小样本 semantic-ID 内容推荐原型保留为 historical baseline；当前主线是 KuaiRec raw CSV -> 多行为标签 -> 用户序列 -> item/user features -> Semantic ID -> Causal Transformer + InfoNCE -> NumPy ANN -> Top-K 评估 -> FastAPI `/recommend`。

## 已完成阶段

| 阶段 | 状态 | 证据 |
|---|---|---|
| P0 KuaiRec schema/loaders/behaviors/sequences/features | 已完成 | `src/cognid_genrec/kuairec/*.py`、`scripts/prepare_kuairec_data.py` |
| P0 Popular/ItemCF baseline 与分层指标 | 已完成 | `scripts/evaluate_kuairec_retriever.py`、`src/cognid_genrec/evaluation/segmented_metrics.py` |
| P1 Semantic ID + Causal Transformer + InfoNCE + ANN | 已完成 | `scripts/train_kuairec_semantic_id.py`、`scripts/train_kuairec_retriever.py`、`src/cognid_genrec/models/sequence_encoder.py` |
| P2 FastAPI、解释字段、实验面板、简历材料 | 已完成 | `src/cognid_genrec/service/kuairec_api.py`、`reports/metrics/kuairec_experiment_panel.md`、`docs/RESUME_NOTES.md` |

## 最新真实数据验证

数据范围：KuaiRec `small_matrix.csv`。

```powershell
python scripts\prepare_kuairec_data.py --raw data\raw\kuairec --out data\processed\kuairec --matrix small
python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method kmeans --out data\processed\kuairec\semantic_ids.parquet
python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 1 --batch-size 64 --out data\processed\kuairec\kuai_retriever.pt
python scripts\evaluate_kuairec_retriever.py --method popular --data data\processed\kuairec --out reports\metrics\kuairec_popular.md
python scripts\evaluate_kuairec_retriever.py --method itemcf --data data\processed\kuairec --out reports\metrics\kuairec_itemcf.md
python scripts\evaluate_kuairec_retriever.py --method ann_transformer --data data\processed\kuairec --out reports\metrics\kuairec_ann_transformer.md
python -m pytest -q
```

最新规模：

- 交互：4,676,570
- 用户：7,176
- 视频：10,731
- 可评估 leave-one-out 序列：1,411
- Semantic ID item 数：10,731

最新指标见 `reports/metrics/kuairec_experiment_panel.md`。

## 命名说明

项目对外名称和目录保留为 `CognID-GenRec`，因为它表达的是 Semantic/Cognitive ID + Generative Recommendation 的方法型定位；`KuaiRec` 作为当前真实数据集和实验场景写在副标题与文档中。内部 Python import 包继续使用 `cognid_genrec`。

## 边界

所有结果只代表基于 KuaiRec 开源数据集的离线实验。当前 ANN Transformer 只训练 1 epoch，不能写成效果优于传统 baseline。不能将项目表述为企业线上推荐系统或线上业务指标提升。
