# CognID-GenRec

基于 KuaiRec 开源短视频数据集的多行为生成式推荐离线实验项目。旧版小样本 semantic ID 内容推荐原型保留为 historical baseline，新链路在 `data/processed/kuairec` 下运行。

## 当前状态

- 数据：已使用真实 KuaiRec `small_matrix.csv` 跑通，原始文件位于 `data/raw/kuairec/KuaiRec 2.0/data`，外部补充文件位于 `data/raw/kuairec`。
- 规模：4,676,570 条交互、7,176 个用户、10,731 个视频，构造 1,411 条 leave-one-out 用户序列。
- 模块：schema/loaders、`watch_ratio` 多行为标签、序列构建、item/user feature builder、Popular/ItemCF baseline、Semantic ID、Causal Transformer + InfoNCE、NumPy ANN、分层评估、FastAPI `/recommend`。
- 边界：所有结果都是基于 KuaiRec 开源数据集的离线实验，不代表任何线上业务指标或线上系统效果。

## 安装

```powershell
python -m pip install -e ".[dev]"
```

可选 Faiss 加速：

```powershell
python -m pip install -e ".[kuairec]"
```

Faiss 不是主线依赖；不可用时默认使用 `NumpyANNIndex`。

## 数据文件

当前已检查到的真实文件：

```text
data/raw/kuairec/
  kuairec_caption_category.csv
  user_features_raw.csv
  video_raw_categories_multi.csv
  KuaiRec 2.0/data/
    small_matrix.csv
    big_matrix.csv
    social_network.csv
    user_features.csv
    item_daily_features.csv
    item_categories.csv
    kuairec_caption_category.csv
```

`small_matrix.csv` 约 406 MB，`big_matrix.csv` 约 1.08 GB。默认先用 small matrix 做离线实验。

## 运行命令

```powershell
python scripts\prepare_kuairec_data.py --raw data\raw\kuairec --out data\processed\kuairec --matrix big
python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method kmeans --out data\processed\kuairec\semantic_ids.parquet
python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 1 --batch-size 64 --out data\processed\kuairec\kuai_retriever.pt
python scripts\evaluate_kuairec_retriever.py --method popular --data data\processed\kuairec --out reports\metrics\kuairec_sequential_next_item_popular.md
python scripts\evaluate_kuairec_retriever.py --method itemcf --data data\processed\kuairec --out reports\metrics\kuairec_sequential_next_item_itemcf.md
python scripts\evaluate_kuairec_retriever.py --method ann_transformer --data data\processed\kuairec --out reports\metrics\kuairec_sequential_next_item_ann_transformer.md
```

`prepare_kuairec_data.py` 会生成 `candidate_manifest.json`。评估命令默认加载该清单，按
`sequential_next_item` 协议让所有方法使用同一 full-sort candidate universe，并确保
Popular/ItemCF 只使用 `train_history_item_ids` 拟合。旧 processed data 需要先重新执行 prepare。

## 最新离线指标

数据范围：KuaiRec `small_matrix.csv`，1,411 条可评估用户序列。ItemCF baseline 使用每用户最近 100 个去重 item 和每 item Top-200 邻居的本机性能保护。

| Method | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| Popular | 0.7123 | 0.3453 | 0.9348 | 0.1373 | 0.5617 |
| ItemCF | 0.9865 | 0.6197 | 0.9979 | 0.0673 | 0.7838 |
| ANN Transformer | 0.0007 | 0.0002 | 0.0043 | 0.0038 | 0.5395 |

ANN Transformer 当前只训练 1 epoch，用于证明训练、向量化和 ANN 检索闭环可运行；低指标说明它还不是可用于效果宣称的充分训练模型。

## API

```powershell
python scripts\serve_kuairec_recommender.py --data data\processed\kuairec
```

请求：

```json
{
  "user_id": "1",
  "history_item_ids": ["10", "20"],
  "history_actions": ["valid_view", "high_interest"],
  "top_k": 10
}
```

响应 item 包含 `item_id`、`score`、`semantic_id`、`source`、`action_context`、`ann_rank`、`reason`。

## 验证

```powershell
python -m pytest -q
```

## 文档

- [KuaiRec TAAC Rebuild Plan](docs/superpowers/plans/2026-06-30-kuairec-taac-rebuild.md)
- [实验日志](docs/EXPERIMENT_LOG.md)
- [实验面板](reports/metrics/kuairec_experiment_panel.md)
