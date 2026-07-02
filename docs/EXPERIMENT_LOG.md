# CognID-GenRec 实验日志

更新时间：2026-06-30

## 记录规则

每次实验记录必须写清楚数据范围、命令、核心参数、指标、结论和边界。所有指标只代表离线实验，不写线上业务效果。

## 最新记录：真实 KuaiRec small matrix 闭环

日期：2026-06-30

阶段：KuaiRec P0/P1/P2 真实数据重跑

实验目标：基于已下载到 `data/raw/kuairec` 的 KuaiRec 真实原始数据，重新跑通多行为生成式推荐实验闭环，并用真实离线结果覆盖 smoke-sample 报告。

代码路径：

- `src/cognid_genrec/kuairec/loaders.py`
- `src/cognid_genrec/kuairec/behaviors.py`
- `src/cognid_genrec/kuairec/sequences.py`
- `src/cognid_genrec/kuairec/features.py`
- `src/cognid_genrec/models/sequence_encoder.py`
- `src/cognid_genrec/models/contrastive.py`
- `src/cognid_genrec/models/baselines/itemcf.py`
- `src/cognid_genrec/retrieval/ann_index.py`
- `src/cognid_genrec/evaluation/segmented_metrics.py`
- `scripts/prepare_kuairec_data.py`
- `scripts/train_kuairec_semantic_id.py`
- `scripts/train_kuairec_retriever.py`
- `scripts/evaluate_kuairec_retriever.py`

数据版本：

- `small_matrix.csv`：406,155,844 bytes，真实 KuaiRec small matrix。
- `big_matrix.csv`：1,083,521,226 bytes，已下载但本轮未训练。
- 额外特征：`item_categories.csv`、`item_daily_features.csv`、`user_features.csv`、`kuairec_caption_category.csv`、`user_features_raw.csv`、`video_raw_categories_multi.csv`。
- 输出规模：4,676,570 条交互、7,176 个用户、10,731 个视频、1,411 条 leave-one-out 用户序列。

运行命令：

```powershell
python scripts\prepare_kuairec_data.py --raw data\raw\kuairec --out data\processed\kuairec --matrix small
python scripts\train_kuairec_semantic_id.py --items data\processed\kuairec\items.parquet --method kmeans --out data\processed\kuairec\semantic_ids.parquet
python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 1 --batch-size 64 --out data\processed\kuairec\kuai_retriever.pt
python scripts\evaluate_kuairec_retriever.py --method popular --data data\processed\kuairec --out reports\metrics\kuairec_popular.md
python scripts\evaluate_kuairec_retriever.py --method itemcf --data data\processed\kuairec --out reports\metrics\kuairec_itemcf.md
python scripts\evaluate_kuairec_retriever.py --method ann_transformer --data data\processed\kuairec --out reports\metrics\kuairec_ann_transformer.md
```

核心参数：

- 行为标签：`watch_ratio < 0.3 -> short_view`，`< 1.0 -> valid_view`，`< 2.0 -> complete_view`，`>= 2.0 -> high_interest`。
- 序列切分：按 `user_id + timestamp + video_id` 排序，最后两条分别作为 validation/test。
- Semantic ID：TF-IDF item text/category/stat features + KMeans，多级语义 ID。
- Retriever：Causal Transformer + Action Conditioning + InfoNCE，`epochs=1`，`batch_size=64`，`hidden_dim=128`，`max_len=64`。
- ANN：NumPy exact cosine index；Faiss 未作为必要依赖。
- ItemCF：每用户最近 100 个去重 item、每 item Top-200 邻居，避免真实长序列平方级计算拖垮本机评估。

核心结果：

| Method | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| Popular | 0.7123 | 0.3453 | 0.9348 | 0.1373 | 0.5617 |
| ItemCF | 0.9865 | 0.6197 | 0.9979 | 0.0673 | 0.7838 |
| ANN Transformer | 0.0007 | 0.0002 | 0.0043 | 0.0038 | 0.5395 |

Semantic ID 结果：

- `item_count=10731`
- `semantic_id_unique=198`
- `collision_rate=0.9815`
- `topic_purity=0.4283`
- `collision_rate_head=0.9614`
- `collision_rate_middle=0.9622`
- `collision_rate_tail=0.9703`

训练结果：

- `batch_count=23`
- `loss=59.1890`
- `avg_loss=276.1961`
- `item_count=10731`
- `user_sequence_count=1411`

结论：

真实 KuaiRec small matrix 已完成 prepare、Semantic ID、Causal Transformer + InfoNCE 训练、NumPy ANN artifact、Popular/ItemCF/ANN Transformer 评估和分层报告生成。ANN Transformer 仅训练 1 epoch，当前指标主要证明链路可运行，不能作为效果优于传统 baseline 的结论。ItemCF 在本 split 下指标很高，但它是离线共现 baseline，不代表线上收益。

本轮修正：

- loader 支持官方 zip 解压后的 `KuaiRec 2.0/data` 嵌套目录。
- loader 在 pandas C parser 遇到 caption CSV buffer overflow 时回退到 Python parser。
- loader 用 `time/event_time` 兜底缺失的 `timestamp`。
- ItemCF 加入长历史和邻居截断，避免真实数据评估长时间卡住。

下一步：

- 如需扩大实验，可在 `big_matrix.csv` 上先加入 `--max-users` 或 `--max-interactions` 采样，再逐步扩大。
- ANN Transformer 需要更充分训练、负采样/候选召回改进和更合理的验证方案后，才能讨论模型效果。

## Historical Baseline

旧版小样本内容推荐原型仍保留，包含内容 schema、Popular/ItemCF baseline、KMeans/RQ-style Semantic ID、transition decoder、重排和旧 `/recommend` API。它只作为 historical baseline 和回归测试路径，不删除现有可运行代码。
