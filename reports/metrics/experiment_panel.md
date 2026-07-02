# CognID-GenRec Historical Baseline Experiment Panel

更新时间：2026-06-30

## 当前数据版本

- 样本数据：`data/samples/*.csv`
- 处理后数据：`data/processed/user_sequences.jsonl`
- Semantic ID：`data/processed/semantic_ids.parquet`
- RQ-VAE-style Semantic ID：`data/processed/semantic_ids_rqvae.parquet`
- 生成式召回 artifact：`data/processed/generative_retriever.json`

## 离线指标

| 方法 | Recall@10 | NDCG@10 | Coverage | Diversity | 说明 |
|---|---:|---:|---:|---:|---|
| Popular baseline | 1.0000 | 1.0000 | 0.3333 | 1.0000 | 加权热度召回 |
| ItemCF baseline | 1.0000 | 0.6309 | 0.3333 | 1.0000 | item 共现召回 |
| Generative prototype | 1.0000 | 0.6309 | 0.3333 | 1.0000 | semantic ID transition decoder |
| Generative rerank | 1.0000 | 1.0000 | 0.3333 | 1.0000 | 生成式召回后规则重排 |

## Semantic ID 质量

| 方法 | item_count | semantic_id_unique | collision_rate | topic_purity | 说明 |
|---|---:|---:|---:|---:|---|
| KMeans semantic ID | 6.0000 | 6.0000 | 0.0000 | 1.0000 | TF-IDF + 多层 KMeans |
| RQ-VAE-style semantic ID | 6.0000 | 6.0000 | 0.0000 | 1.0000 | 小型 PyTorch encoder/decoder + residual codebook |

## 服务验证

- 接口：`POST /recommend`
- 示例输出：`reports/metrics/recommend_example.json`
- 返回字段：`item_id`、`score`、`semantic_id`、`reason`、`source`、`recall_path`、`rerank_reason`

## 边界说明

当前指标来自 6 条 item、1 条可评测用户序列的小样本，只证明数据处理、semantic ID、召回、重排、评测和服务链路可运行，不能作为真实线上效果提升结论。RQ-VAE-style 版本是轻量残差量化原型，尚未完整复现 `phonism/genrec` 训练器；TIGER 级神经生成模型、SASRec 对照和大规模实验尚未实现。
