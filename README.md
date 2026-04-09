# 县域农业数据 PM2.5 预测项目

本项目围绕 `data.csv` 中的县域农业与环境表格数据，构建一个面向本科毕业设计的 `PM2.5` 浓度预测研究仓库。仓库包含完整的数据审计、特征工程、基线实验、GeoPFNMix 融合模型、图表生成、实验日志、项目日记和论文草稿。

## 研究目标

- 预测对象：县域尺度 `PM2.5`
- 任务类型：表格回归
- 核心难点：中小样本、强地理层级、随机划分导致地理泄漏、跨区域泛化困难
- 主评估协议：`GroupKFold(CITY)`

项目不是单纯在随机划分上追求高分，而是强调“在未见过的城市上，模型还能否保持稳定预测能力”。

## 当前仓库中的主要结论

以下结果来自仓库当前可复现脚本在本地环境中的最新运行产物：

- 随机划分下最优单模型：`TabPFN`
  - `RMSE = 9.3250`
  - `MAE = 6.5747`
  - `R² = 0.5261`
- 主协议 `GroupKFold(CITY)` 下最优单模型：`TabPFN`
  - `RMSE = 12.5009`
  - `MAE = 9.6614`
  - `R² = 0.1415`
- 当前最优融合模型：`GeoPFNMix-Lite-TabPFN`
  - `RMSE = 12.4848`
  - `MAE = 9.4729`
  - `R² = 0.1442`
- 相对单模型 `TabPFN`，`GeoPFNMix-Lite-TabPFN` 在主协议下：
  - `RMSE` 降低 `0.13%`
  - `MAE` 降低 `1.95%`
  - Wilcoxon 配对检验 `p = 9.89e-06`
- 在不使用 `TabPFN` 先验时，当前最强可执行融合配置为 `GeoPFNMix-CatBoost`
  - `RMSE = 12.5200`
  - `MAE = 9.7104`
  - `R² = 0.1390`

与随机划分相比，`TabPFN` 在 `GroupKFold(CITY)` 下的 RMSE 上升约 `34.06%`，说明地理泄漏确实显著存在；也说明本项目的主结论不是“随机划分高分”，而是“严格地理协议下谁更稳健”。

## 仓库结构

- `data.csv`
  - 原始数据文件，包含省、市、县三级地理字段与数值特征
- `src/pm25_geopfnmix/`
  - 核心 Python 包
- `scripts/run_eda.py`
  - 数据审计、描述统计和基础可视化
- `scripts/run_baselines.py`
  - 跑通统一特征工程下的多组基线模型
- `scripts/run_geopfnmix.py`
  - 运行 GeoPFNMix 家族模型、消融实验和显著性检验
- `scripts/run_analysis_assets.py`
  - 基于实验结果生成论文用图表与分析表
- `artifacts/figures/`
  - 自动生成的图像输出
- `artifacts/tables/`
  - 自动生成的结果表、OOF 预测和统计检验结果
- `reports/实验日志.md`
  - 实验记录
- `reports/项目日记.md`
  - 项目决策与推进记录
- `paper/毕业论文初稿.md`
  - 论文草稿

## 数据字段说明

原始数据共 13 列：

- 标识列：`OBJECTID_1`
- 类别列：`PROVINCE`、`CITY`、`COUNTY`
- 数值列：`AET`、`ppt`、`tem`、`wind`、`NOX`、`SO2`、`fertilzier`、`manure`
- 目标列：`PM2.5`

建模时：

- `OBJECTID_1` 不进入模型
- `COUNTY` 不作为主类别特征参与编码
- 主要类别特征只保留 `PROVINCE` 和 `CITY`

这样做是为了降低“县级近唯一标识”带来的伪记忆问题。

## 安装方式

推荐在单独环境中运行：

```bash
python -m pip install -e .
```

这会以可编辑模式安装本项目，并自动拉起 `catboost`、`lightgbm`、`xgboost`、`shap`、`tabpfn` 等依赖。

## 一键复现实验流程

按下面顺序执行即可：

```bash
python scripts/run_eda.py
python scripts/run_baselines.py
python scripts/run_geopfnmix.py
python scripts/run_analysis_assets.py
```

各脚本作用如下：

1. `run_eda.py`
   - 输出数据审计摘要、描述统计、目标分布图、数值相关图、城市组大小分布图
2. `run_baselines.py`
   - 跑 `Ridge`、`RandomForest`、`HistGBDT`、`XGBoost`、`LightGBM`、`CatBoost`、`TabPFN`
   - 在 `RandomKFold`、`GroupKFold(CITY)`、`GroupKFold(PROVINCE)` 三套协议下输出结果
3. `run_geopfnmix.py`
   - 运行 GeoPFNMix 家族模型与显著性检验
4. `run_analysis_assets.py`
   - 读取前面脚本生成的表格，自动绘制论文图表和补充分析表

## 主要输出文件

### 基线与协议比较

- `artifacts/tables/baseline_summary.csv`
- `artifacts/tables/protocol_best_models.csv`
- `artifacts/tables/protocol_generalization_gap.csv`
- `artifacts/figures/baseline_protocol_heatmap.png`

### GeoPFNMix 消融与显著性

- `artifacts/tables/geopfnmix_ablation_summary.csv`
- `artifacts/tables/geopfnmix_significance.json`
- `artifacts/tables/geopfnmix_significance_pairs.csv`
- `artifacts/figures/group_city_model_comparison.png`
- `artifacts/figures/geopfnmix_ablation_metrics.png`

### 误差分析与可解释性

- `artifacts/figures/best_geopfnmix_scatter.png`
- `artifacts/figures/geopfnmix_error_by_target_bin.png`
- `artifacts/tables/geopfnmix_error_by_target_bin.csv`
- `artifacts/figures/province_level_mae.png`
- `artifacts/tables/province_level_mae.csv`
- `artifacts/tables/province_level_mae_top_bottom.csv`
- `artifacts/figures/best_global_expert_shap_bar.png`
- `artifacts/tables/best_global_expert_shap_top10.csv`

## Python 接口

### 1. 读取数据

```python
from pm25_geopfnmix.data import load_dataset

dataset = load_dataset()
print(dataset.features.shape)
print(dataset.target.name)
```

返回对象为 `DatasetBundle`，包含：

- `frame`：原始完整数据
- `features`：用于建模的特征表
- `target`：目标列

### 2. 构建模型

统一使用 `make_model(name)`：

```python
from pm25_geopfnmix.models import make_model

model = make_model("geopfnmix_no_residual_tabpfn")
model.fit(dataset.features, dataset.target)
pred = model.predict(dataset.features.head(10))
```

当前支持的主要模型名：

- `ridge`
- `rf`
- `hgbt`
- `xgboost`
- `lightgbm`
- `catboost`
- `tabpfn`
- `geopfnmix_no_prior`
- `geopfnmix_no_residual`
- `geopfnmix_no_residual_tabpfn`
- `geopfnmix`
- `geopfnmix_tabpfn`
- `geopfnmix_catboost`
- `geopfnmix_catboost_tabpfn`

### 3. 交叉验证评估

```python
from pm25_geopfnmix.evaluation import evaluate_cv
from pm25_geopfnmix.models import make_model

result = evaluate_cv(
    dataset=dataset,
    model_name="geopfnmix_no_residual_tabpfn",
    model_factory=lambda: make_model("geopfnmix_no_residual_tabpfn"),
    split_name="group_city",
)

print(result.summary)
print(result.fold_metrics.head())
print(result.oof_frame.head())
```

可选 `split_name`：

- `random_kfold`
- `group_city`
- `group_province`

### 4. GeoPFNMix 直接实例化

如果你想显式控制结构，可以直接实例化 `GeoPFNMixRegressor`：

```python
from pm25_geopfnmix.models import GeoPFNMixRegressor

model = GeoPFNMixRegressor(
    global_model_name="catboost",
    use_residual_branch=True,
    use_prior_branch=True,
    inner_splits=3,
)
```

核心参数说明：

- `global_model_name`
  - 全局专家，可选 `rf`、`catboost` 等
- `use_residual_branch`
  - 是否启用省市两级残差校正
- `use_prior_branch`
  - 是否启用先验专家分支
- `inner_splits`
  - stacking 训练时的内层 `GroupKFold` 折数

## GeoPFNMix 工作流程

GeoPFNMix 家族模型遵循统一结构：

1. 统一特征工程
   - 分位裁剪
   - `log1p`
   - 污染协同项
   - 农业投入与气象交互项
   - 省内 z-score
   - 样本支撑度特征
2. 全局专家
   - `RF` 或 `CatBoost`
3. 地理层级残差分支
   - 省级 + 城市级经验贝叶斯式收缩校正
4. 异构先验专家
   - 当前默认可执行先验为 `LightGBM`
   - 若提供 `HF_TOKEN` 与 `TABPFN_TOKEN`，可启用 `TabPFN`
   - 本仓库会优先从根目录 `token.py` 中导入本地令牌
5. 轻量 stacking
   - 使用低容量线性模型融合多路专家输出

## 当前推荐模型配置

如果你已经完成 `TabPFN` 无人值守授权，建议优先使用：

```python
model = make_model("geopfnmix_no_residual_tabpfn")
```

它对应：

- 全局专家：`RF`
- 残差分支：关闭
- 先验专家：`TabPFN`
- 融合器：线性 stacking

如果你暂时不使用 `TabPFN`，建议使用：

```python
model = make_model("geopfnmix_catboost")
```

它对应：

- 全局专家：`CatBoost`
- 残差分支：开启
- 先验专家：开启
- 融合器：线性 stacking

如果你想要更简洁的对照配置，可以使用：

```python
model = make_model("geopfnmix_no_residual")
```

也就是论文中常提到的 `GeoPFNMix-Lite`。

## 无人值守运行注意事项

本仓库当前已经支持 `TabPFN` 的无人值守执行，策略如下：

- 根目录允许放置一个本地私有 `token.py`
  - 可提供 `HF_TOKEN`
  - 也可同时提供 `TABPFN_TOKEN`
- `token.py` 已加入 `.gitignore`
  - 不会被默认提交到 Git
- 如果只有 `HF_TOKEN`
  - 可以通过 Hugging Face 完成基础认证
  - 但 `TabPFN` 头less 推理仍可能不可用
- 如果同时存在 `HF_TOKEN` 与 `TABPFN_TOKEN`
  - `tabpfn` 基线与 `GeoPFNMix` 的 PFN 先验分支都可直接运行
- 如果 `TabPFN` 在当前环境中失败
  - GeoPFNMix 的先验专家会自动回退为 `LightGBM`
  - 主流程不会被中断

这保证了脚本在本地、服务器和答辩前的无人值守重跑环境中都能稳定工作。

额外说明：

- 由于仓库根目录存在 `token.py`，在仓库根目录直接执行某些 `python -` 或 `python -c` 临时探针时，可能会与 Python 标准库同名模块 `token` 发生遮蔽
- 日常复现实验请优先使用 `python scripts/run_*.py`
- 如果必须写临时探针，建议显式清理 `sys.path` 中的仓库根目录后再导入标准库相关模块

## 复现与环境说明

- 结果会受到 Python 版本、库版本和运行平台的影响
- `reports/实验日志.md` 中保留了项目早期环境记录和当前本地重跑记录
- 当前仓库中的图表、表格和论文草稿，建议一律以 `artifacts/` 中最新文件为准

## 论文与文档对应关系

- 论文正文：`paper/毕业论文初稿.md`
- 实验日志：`reports/实验日志.md`
- 决策与推进记录：`reports/项目日记.md`
- 项目阶段性评估：`reports/项目瓶颈、困难与改进方向分析.md`

如果后续你要继续扩展毕业设计，最直接的方向是：

- 增加更严格的空间泛化协议
- 引入经纬度、土地利用、交通或遥感等外部空间特征
- 继续研究 `TabPFN` 先验与更复杂残差结构之间的适配关系
- 为最终模型增加部署脚本或简单演示界面
