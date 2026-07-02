# KuaiRec Experiment Panel

Scope: offline experiments on the KuaiRec open dataset `small_matrix.csv`.

Data scale:

- interactions: 4,676,570
- users: 7,176
- items: 10,731
- evaluable leave-one-out sequences: 1,411
- Semantic ID items: 10,731

Training/evaluation notes:

- Retriever command: `python scripts\train_kuairec_retriever.py --data data\processed\kuairec --epochs 1 --batch-size 64 --out data\processed\kuairec\kuai_retriever.pt`
- ANN path: NumPy exact cosine index; Faiss is optional and not required.
- ItemCF path: capped to each user's latest 100 deduplicated items and each item's Top-200 neighbors for local Windows evaluation.
- Boundary: offline dataset metrics only; no online business metric or production-system claim.

| Method | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| ann_transformer | 0.0007 | 0.0002 | 0.0043 | 0.0038 | 0.5395 |
| itemcf | 0.9865 | 0.6197 | 0.9979 | 0.0673 | 0.7838 |
| popular | 0.7123 | 0.3453 | 0.9348 | 0.1373 | 0.5617 |

## Popular Segments

| Segment | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| action=complete_view | 0.8455 | 0.4824 | 0.9719 | 0.0840 | 0.5494 |
| action=high_interest | 0.8923 | 0.5047 | 1.0000 | 0.0404 | 0.5814 |
| action=short_view | 0.4649 | 0.1736 | 0.8266 | 0.0699 | 0.5749 |
| action=valid_view | 0.7232 | 0.3278 | 0.9513 | 0.1056 | 0.5610 |
| activity=long | 0.7123 | 0.3453 | 0.9348 | 0.1373 | 0.5617 |
| popularity=head | 0.7118 | 0.3439 | 0.9335 | 0.1345 | 0.5626 |
| popularity=middle | 0.7241 | 0.3789 | 0.9655 | 0.0347 | 0.5396 |

## ItemCF Segments

| Segment | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| action=complete_view | 0.9775 | 0.6354 | 0.9944 | 0.0404 | 0.7589 |
| action=high_interest | 0.9846 | 0.6182 | 1.0000 | 0.0171 | 0.7391 |
| action=short_view | 0.9852 | 0.5914 | 1.0000 | 0.0343 | 0.8078 |
| action=valid_view | 0.9917 | 0.6227 | 0.9986 | 0.0527 | 0.7912 |
| activity=long | 0.9865 | 0.6197 | 0.9979 | 0.0673 | 0.7838 |
| popularity=head | 0.9867 | 0.6231 | 0.9978 | 0.0655 | 0.7833 |
| popularity=middle | 0.9828 | 0.5404 | 1.0000 | 0.0141 | 0.7966 |

## ANN Transformer Segments

| Segment | HR@10 | NDCG@10 | Recall@20 | Coverage | Diversity |
|---|---:|---:|---:|---:|---:|
| action=complete_view | 0.0000 | 0.0000 | 0.0028 | 0.0035 | 0.5364 |
| action=high_interest | 0.0000 | 0.0000 | 0.0000 | 0.0029 | 0.5385 |
| action=short_view | 0.0000 | 0.0000 | 0.0037 | 0.0034 | 0.5419 |
| action=valid_view | 0.0014 | 0.0004 | 0.0056 | 0.0036 | 0.5402 |
| activity=long | 0.0007 | 0.0002 | 0.0043 | 0.0038 | 0.5395 |
| popularity=head | 0.0007 | 0.0002 | 0.0044 | 0.0038 | 0.5391 |
| popularity=middle | 0.0000 | 0.0000 | 0.0000 | 0.0032 | 0.5474 |
