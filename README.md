# MSA-scVI 单细胞无监督聚类 Pipeline

本项目实现了一个面向单细胞 RNA-seq 数据的 **MSA-scVI** 聚类分析流程。流程结合 CellMarker / PanglaoDB marker 数据库、无监督 marker program 打分、scVI 潜在表示学习、多尺度稳定性聚类、共识聚类以及混合簇二次细分，用于在尽量不依赖真实标签的情况下获得可解释的细胞聚类结果。

> 当前代码支持 `.h5ad` 与 `.csv` 输入；若提供真实标签列，仅用于最终监督指标评估，不参与无监督聚类训练与调参。

---

## 目录结构

```text
.
├── main.py                 # 主入口：解析命令行参数并运行完整 pipeline
├── config.py               # PipelineConfig 配置项
├── io_utils.py             # 数据读取、目录/JSON/marker 保存工具
├── preprocessing.py        # AnnData 预处理、QC、归一化、HVG 选择
├── marker_selection.py     # marker 数据库读取、合并、筛选、program 打分
├── latent_dim.py           # 基于 PCA 的潜在维度估计
├── scvi_model.py           # scVI 训练与 marker prototype latent refinement
├── clustering.py           # 多尺度 Leiden 聚类、共识聚类、二次细分
├── merge_guard.py          # 可选的小簇合并保护逻辑
├── evaluation.py           # ARI/NMI/FMI/ACC、UMAP、热图等评估与可视化
└── run_h5_pipeline.py      # 批量运行 h5ad 数据集的示例脚本
```

---

## 功能概览

### 核心流程

1. **读取输入数据**
   - 支持 `.h5ad` 与 `.csv`。
   - 自动识别常见 label、batch、barcode 列。
   - 对 `.h5ad`，若存在 `layers["counts"]`，优先作为 scVI counts；否则使用 `adata.X` 作为 pseudo-counts。

2. **预处理**
   - 细胞 / 基因过滤。
   - 线粒体比例过滤。
   - counts 数据执行 `normalize_total`、`log1p`、HVG 选择。
   - 已处理但无 counts 层的 `.h5ad` 会避免重复 normalize / log。

3. **marker 数据库处理**
   - 读取 CellMarker CSV。
   - 可选读取 PanglaoDB TSV。
   - 多来源 marker 支持度归一化后合并。
   - marker 映射到当前数据集基因空间。
   - 无监督筛选更具体、更适合当前数据的 marker。

4. **marker program 打分与高置信伪标签**
   - 为每个细胞计算不同 marker program 的平均表达分数。
   - 按 program 内部分位数和 margin 选择高置信伪标签。
   - 伪标签用于 latent prototype refinement，不使用真实标签。

5. **scVI 表示学习**
   - 自动或指定潜在维度。
   - 支持 batch covariate。
   - 使用 marker prototype 对 latent 表示做轻量修正。

6. **多尺度稳定聚类**
   - 在多个 `n_neighbors` 与 `resolution` 组合上重复运行 Leiden。
   - 根据稳定性、marker specificity、batch mixing、过分裂惩罚进行无监督打分。
   - 选取得分最高的若干结果构建共识矩阵。
   - 基于共识矩阵生成最终聚类。

7. **混合簇二次细分**
   - 对 marker program 熵高、top gap 小的疑似混合簇进行局部再聚类。
   - 只有当局部结果稳定、可解释、子簇足够大且 marker specificity 有提升时才接受细分。

8. **可选 merge guard**
   - 对过小且 marker 解释不充分的小簇，尝试合并到相似的大簇。

9. **输出报告与结果文件**
   - 保存 JSON 报告、聚类标签、marker 分数、embedding、UMAP 图、聚类注释等。
   - 若提供真实标签列，可额外输出 ARI、NMI、FMI、ACC 与 true-vs-cluster 热图。

---

## 环境依赖

建议使用 Python 3.10 或以上版本，并在独立虚拟环境中运行。

### 安装示例

```bash
conda create -n msa-scvi python=3.10 -y
conda activate msa-scvi

pip install numpy pandas scipy scikit-learn matplotlib anndata scanpy scvi-tools torch
```

`scanpy.tl.leiden` 通常还需要 Leiden 相关依赖：

```bash
pip install leidenalg igraph
```

如果你的环境中已经安装了 GPU 版本 PyTorch，`scvi-tools` 会自动使用可用设备；否则会在 CPU 上运行。

---

## 输入数据要求

### 1. `.h5ad` 输入

推荐输入为 AnnData `.h5ad` 文件。

常见要求：

- `adata.X` 为表达矩阵。
- 若有原始 counts，建议放在 `adata.layers["counts"]`。
- 细胞元信息放在 `adata.obs`。
- 基因名放在 `adata.var_names`。

可选元信息列：

| 类型 | 说明 |
|---|---|
| `label_col` | 真实细胞类型标签，仅用于最终评估，不参与训练 |
| `batch_col` | batch / donor / sample 信息，用于 scVI batch correction |
| `barcode` | 细胞 ID；若不存在会使用 `obs_names` |

### 2. `.csv` 输入

CSV 支持两种模式：

#### 模式 A：元信息列 + 表达矩阵

如果前若干列是元信息，后面是表达矩阵，可通过 `--csv_expr_start_col` 指定表达矩阵起始列。

```bash
python main.py \
  --input data.csv \
  --cellmarker_db Cell_marker_Human.csv \
  --output_dir results \
  --csv_expr_start_col 3
```

#### 模式 B：自动识别元信息

如果不指定 `--csv_expr_start_col`，程序会尝试自动识别常见的 label、batch、barcode 列，其余列作为表达基因。

---

## Marker 数据库要求

### CellMarker CSV

`--cellmarker_db` 是必需参数。CSV 至少需要包含以下列：

```text
species
tissue_class
cell_name
Symbol
```

示例：

```text
species,tissue_class,cell_name,Symbol
Human,Pancreas,Beta cell,INS
Human,Pancreas,Alpha cell,GCG
```

### PanglaoDB TSV（可选）

可通过 `--panglaodb_tsv` 加入 PanglaoDB marker 数据。TSV 至少需要包含：

```text
species
official gene symbol
cell type
organ
```

若存在以下列，会用于支持度加权：

```text
canonical marker
ubiquitousness index
specificity_human
specificity_mouse
sensitivity_human
sensitivity_mouse
```

---

## 快速开始

### 基础运行

```bash
python main.py \
  --input path/to/input.h5ad \
  --cellmarker_db path/to/Cell_marker_Human.csv \
  --output_dir path/to/output \
  --species Human \
  --tissue Pancreas
```

### 指定真实标签与 batch 列

真实标签只用于评估，不参与无监督训练：

```bash
python main.py \
  --input path/to/input.h5ad \
  --cellmarker_db path/to/Cell_marker_Human.csv \
  --output_dir path/to/output \
  --label_col cell_type \
  --batch_col sample_id \
  --species Human \
  --tissue Pancreas
```

### 同时使用 CellMarker 与 PanglaoDB

```bash
python main.py \
  --input path/to/input.h5ad \
  --cellmarker_db path/to/Cell_marker_Human.csv \
  --panglaodb_tsv path/to/PanglaoDB_markers.tsv \
  --output_dir path/to/output \
  --species Human \
  --tissue Pancreas
```

### 启用 merge guard

```bash
python main.py \
  --input path/to/input.h5ad \
  --cellmarker_db path/to/Cell_marker_Human.csv \
  --output_dir path/to/output \
  --enable_merge_guard
```

---

## 命令行参数

`main.py` 支持以下主要参数：

| 参数 | 是否必需 | 默认值 | 说明 |
|---|---:|---|---|
| `--input` | 是 | 无 | 输入 `.h5ad` 或 `.csv` 文件 |
| `--cellmarker_db` | 是 | 无 | CellMarker CSV 数据库路径 |
| `--panglaodb_tsv` | 否 | `None` | PanglaoDB TSV marker 文件路径 |
| `--output_dir` | 是 | 无 | 输出目录 |
| `--label_col` | 否 | `None` | 真实标签列名，仅用于评估 |
| `--batch_col` | 否 | `None` | batch 列名 |
| `--species` | 否 | `Human` | 物种，例如 `Human` 或 `Mouse` |
| `--tissue` | 否 | `Pancreas` | 组织关键词，多个关键词用逗号分隔 |
| `--csv_expr_start_col` | 否 | `0` | CSV 表达矩阵起始列 |
| `--n_top_hvg` | 否 | `2000` | 高变基因数量 |
| `--enable_merge_guard` | 否 | 关闭 | 启用小簇合并保护 |

---

## 重要配置项

多数高级参数定义在 `config.py` 的 `PipelineConfig` 中。常用参数如下：

### 预处理

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `min_genes_per_cell` | `200` | 每个细胞最少检测基因数 |
| `min_cells_per_gene` | `3` | 每个基因最少出现细胞数 |
| `max_mt_pct` | `20.0` | 最大线粒体比例；可设为 `None` |
| `target_sum` | `1e4` | normalize_total 目标总量 |
| `n_top_hvg` | `2000` | HVG 数量 |

### marker 选择

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `max_markers_per_type` | `50` | 每类从 CellMarker 保留的 marker 上限 |
| `panglaodb_max_markers_per_type` | `50` | 每类从 PanglaoDB 保留的 marker 上限 |
| `n_specific_markers_per_program` | `15` | 每个 program 选择的特异 marker 数量 |
| `min_specific_markers_floor` | `3` | 每个 program 至少保留 marker 数 |
| `marker_overlap_penalty` | `0.5` | 共享 marker 的软惩罚 |
| `marker_auto_filter_enabled` | `True` | 是否启用 marker 自动质控 |

### scVI

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `latent_dim_min` | `10` | 自动估计 latent dim 下限 |
| `latent_dim_max` | `30` | 自动估计 latent dim 上限 |
| `scvi_max_epochs` | `200` | scVI 最大训练 epoch |
| `scvi_gene_likelihood` | `nb` | scVI gene likelihood |
| `scvi_n_layers` | `2` | scVI 网络层数 |
| `prototype_refine_strength` | `0.10` | marker prototype 修正强度 |

### 聚类搜索

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `neighbors_grid` | `[10, 15, 20]` | Leiden 邻居数搜索网格 |
| `resolution_grid` | `[0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]` | Leiden resolution 搜索网格 |
| `clustering_repeats` | `5` | 每个参数组合重复次数 |
| `score_alpha` | `0.55` | 稳定性权重 |
| `score_beta` | `0.35` | marker specificity 权重 |
| `score_gamma` | `0.10` | 过分裂惩罚权重 |
| `consensus_top_k` | `5` | 用于共识聚类的 top 参数组合数 |

### 二次细分

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `secondary_refine_enabled` | `True` | 是否启用混合簇二次细分 |
| `secondary_min_cluster_size` | `60` | 允许细分的父簇最小细胞数 |
| `secondary_entropy_threshold` | `0.90` | program entropy 触发阈值 |
| `secondary_top_gap_threshold` | `0.05` | top marker gap 触发阈值 |
| `secondary_max_subclusters` | `3` | 局部细分最大子簇数 |
| `secondary_min_improvement` | `0.05` | 接受细分所需 marker specificity 提升 |

---

## 输出文件说明

运行完成后，输出目录中可能包含以下文件。

### 总报告

| 文件 | 说明 |
|---|---|
| `report.json` | 主报告，包含数据规模、marker 信息、latent dim、聚类搜索 top 结果、最终簇数和评估指标 |

### 聚类与细胞级结果

| 文件 | 说明 |
|---|---|
| `cluster_labels.npy` | 最终聚类标签 |
| `cell_metadata_with_results.csv` | 每个细胞的元信息、预测簇、top marker program、confidence label |
| `cluster_annotations.csv` | 每个 cluster 的主导 marker program 注释 |
| `clustering_search_results.csv` | 多尺度聚类参数搜索结果 |
| `secondary_refinement_report.csv` | 二次细分决策与统计信息 |

### marker 相关结果

| 文件 | 说明 |
|---|---|
| `selected_markers.json` | 最终选择的 marker |
| `specific_marker_scores.csv` | marker specificity 评分表 |
| `marker_program_scores.csv` | 每个细胞的 marker program 分数 |
| `unsuitable_marker_report.csv` | 若启用 marker 自动过滤，保存不适合 marker 的检测报告 |

### embedding 与共识矩阵

| 文件 | 说明 |
|---|---|
| `embeddings_scvi.npy` | scVI latent embedding |
| `embeddings_refined.npy` | marker prototype refinement 后的 embedding |
| `consensus_matrix.npy` | top 聚类结果构建的共识矩阵 |

### 可视化

| 文件 | 说明 |
|---|---|
| `embedding_plot.png` | 按最终 cluster 着色的 UMAP 图 |
| `true_label_plot.png` | 若提供真实标签，按真实标签着色的 UMAP 图 |
| `true_vs_cluster_heatmap.png` | 若提供真实标签，真实标签与预测簇的交叉热图 |

---

## 批量运行示例

`run_h5_pipeline.py` 提供了批量处理多个 `.h5ad` 数据集的示例。使用前需要修改脚本顶部路径：

```python
H5AD_DIR = r"path/to/h5ad_dir"
OUTPUT_ROOT = r"path/to/output_root"
CELLMARKER_DB = r"path/to/Cell_marker_Mouse.csv"
PANGLAODB_TSV = None
```

然后配置数据集列表：

```python
DATASETS = [
    {"name": "Heart", "input_path": os.path.join(H5AD_DIR, "Heart.h5ad"), "tissue_keywords": ["Heart"]},
    {"name": "Lung", "input_path": os.path.join(H5AD_DIR, "Lung.h5ad"), "tissue_keywords": ["Lung"]},
]
```

运行：

```bash
python run_h5_pipeline.py
```

批量脚本会在每个数据集输出目录下保存结果，并在 `OUTPUT_ROOT` 下生成：

```text
msa_scvi_all_summary.csv
```

---

## 方法设计说明

### 为什么不直接使用真实标签？

本 pipeline 的核心目标是无监督聚类。真实标签如果存在，只在最后用于计算 ARI、NMI、FMI、ACC 等评估指标，不参与：

- marker 选择；
- 高置信伪标签构建；
- scVI 训练；
- 聚类参数搜索；
- 二次细分决策。

### 聚类评分如何计算？

多尺度聚类搜索中的总分由以下部分组成：

```text
total_score = alpha * stability
            + beta * marker_specificity
            + 0.10 * batch_mixing
            - gamma * oversplit_penalty
```

其中：

- `stability`：重复聚类之间的平均 ARI；
- `marker_specificity`：每个簇是否有明确占优的 marker program；
- `batch_mixing`：簇内 batch 混合程度；
- `oversplit_penalty`：过小簇比例惩罚。

### 二次细分何时会接受？

只有同时满足以下条件，局部细分才会被接受：

- 局部聚类稳定性达到阈值；
- 子簇 marker dominance 足够高；
- 子簇 top marker gap 足够明显；
- marker specificity 相比父簇有足够提升；
- 子簇数量和大小不过度碎片化。

---

## 常见问题

### 1. 为什么 CellMarker 过滤后没有 marker？

可能原因：

- `--species` 与数据库中的 `species` 不匹配；
- `--tissue` 与数据库中的 `tissue_class` 不匹配；
- 数据基因名与 marker 基因名不一致；
- marker 数据库缺少必需列。

建议检查：

```bash
python -c "import pandas as pd; print(pd.read_csv('Cell_marker_Human.csv').head()); print(pd.read_csv('Cell_marker_Human.csv').columns)"
```

### 2. 为什么 `.h5ad` 没有 counts 层也能跑？

如果 `.h5ad` 不含 `layers["counts"]`，程序会把 `adata.X` 复制为 pseudo-counts，并在 `adata.uns` 中记录该情况。对于已经 processed 的数据，预处理阶段会避免重复 normalize / log。

### 3. 为什么没有监督指标？

只有在命令行中提供 `--label_col`，并且该列存在于 `adata.obs` 或 CSV 中时，才会输出监督指标。

### 4. scVI 训练很慢怎么办？

可以尝试：

- 降低 `scvi_max_epochs`；
- 减小 `n_top_hvg`；
- 先对数据下采样；
- 使用 GPU 版本 PyTorch；
- 减少 `neighbors_grid`、`resolution_grid` 或 `clustering_repeats`。

### 5. 聚类太碎怎么办？

可以尝试：

- 降低 `resolution_grid` 上限；
- 增大 `score_gamma`；
- 启用 `--enable_merge_guard`；
- 增大 `secondary_min_cluster_size`；
- 提高 `secondary_min_improvement`。

### 6. 聚类太粗怎么办？

可以尝试：

- 提高 `resolution_grid` 上限；
- 增大 `neighbors_grid` 的搜索范围；
- 降低 `secondary_entropy_threshold`；
- 降低 `secondary_top_gap_threshold`；
- 增大 `secondary_max_subclusters`。

---

## 可复现性

默认随机种子为：

```text
random_seed = 42
```

scVI、NumPy、PyTorch 以及 Leiden 聚类相关步骤均尽量使用该随机种子。由于 GPU、底层库和并行计算差异，不同机器上仍可能存在轻微数值差异。

---

## 许可证

请根据你的项目实际情况补充许可证信息，例如：

```text
MIT License
```

或：

```text
For academic research use only.
```

---

## 引用与致谢

本项目依赖以下开源生态：

- Scanpy / AnnData
- scvi-tools
- PyTorch
- scikit-learn
- SciPy
- pandas / NumPy
- matplotlib

如用于论文或报告，请同时引用相应工具和 marker 数据库来源。
