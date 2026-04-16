# 多源农业数据 PM2.5 预测项目

本项目围绕 `data/` 目录中的多源农业与环境表格数据，构建一个面向本科毕业设计的 `PM2.5` 浓度预测研究仓库。仓库包含完整的数据审计、字段标准化、特征工程、基线实验、GeoPFNMix 融合模型、图表生成、实验日志、项目日记和论文终稿。

## 研究目标

- 预测对象：区域尺度 `PM2.5`
- 任务类型：表格回归
- 核心难点：多源口径不一致、强地理层级、随机划分导致地理泄漏、跨区域泛化困难
- 主评估协议：`GroupKFold(CITY)`

这里的 `CITY` 是统一后的二级地理组字段。项目不是单纯在随机划分上追求高分，而是强调“在未见过的地理组上，模型还能否保持稳定预测能力”。

## 当前仓库中的主要结论

以下结果来自仓库当前 `artifacts/` 中的最新正式运行产物，即 Colab A100 GPU 重跑后同步回本地仓库的结果：

- 随机划分下最优单模型：`TabPFN`
  - `RMSE = 4.8426`
  - `MAE = 2.1188`
  - `R² = 0.8425`
- 主协议 `GroupKFold(CITY)` 下最优单模型：`CatBoost`
  - `RMSE = 6.0109`
  - `MAE = 3.2069`
  - `R² = 0.7542`
- 更严格的 `GroupKFold(PROVINCE)` 下最优单模型：`CatBoost`
  - `RMSE = 6.8052`
  - `MAE = 4.5704`
  - `R² = 0.6820`
- 当前最优融合模型：`GeoPFNMix-CatBoost-TabPFN`
  - `RMSE = 5.9225`
  - `MAE = 3.1078`
  - `R² = 0.7622`
- 相对单模型 `CatBoost`，`GeoPFNMix-CatBoost-TabPFN` 在主协议下：
  - `RMSE` 降低 `1.47%`
  - `MAE` 降低 `3.09%`
  - Wilcoxon 配对检验 `p = 2.14e-21`
- 在不使用 `TabPFN` 先验时，当前最强可执行融合配置为 `GeoPFNMix-CatBoost`
  - `RMSE = 5.9551`
  - `MAE = 3.1092`
  - `R² = 0.7593`
- 新增补充变体 `GeoPFNMix-Lite-CatBoost`
  - `RMSE = 5.9526`
  - `MAE = 3.1176`
  - `R² = 0.7595`
- 新增补充变体 `GeoPFNMix-Lite-CatBoost-TabPFN`
  - `RMSE = 5.9281`
  - `MAE = 3.1220`
  - `R² = 0.7618`
  - 当前为第二优融合配置
- `GeoPFNMix-Lite` 与 `GeoPFNMix-Lite-TabPFN` 仍保留为重要的 RF 骨干对照配置，用于验证复杂度控制和先验吸收能力。

最新补充实验还说明：

- `GeoPFNMix-Lite-CatBoost-TabPFN` 相对 `GeoPFNMix-Lite-CatBoost` 的提升达到统计显著（`p = 1.17e-02`）
- 当前最优两个 GeoPFNMix 家族模型已经变为：
  - `GeoPFNMix-CatBoost-TabPFN`
  - `GeoPFNMix-Lite-CatBoost-TabPFN`
- 按数据源拆分后，GeoPFNMix 家族在 `Australia`、`Brazil` 和 `China` 上整体占优，而单模型 `CatBoost` 在 `EU` 与 `USA` 上更稳健

需要补充说明的是：在本轮正式重跑中，`GeoPFNMix-CatBoost-TabPFN` 仍然取得最佳均值结果，但它相对 `GeoPFNMix-CatBoost` 的额外优势未达到统计显著（`p = 1.16e-01`）。因此，更稳妥的结论是：当前最强的是“CatBoost 骨干的 GeoPFNMix 家族”，而 `TabPFN` 先验的收益会随着结构位置不同而变化，在 Lite-CatBoost 路线上更容易表现出可检出的稳定增益。

与随机划分相比，最佳单模型在 `GroupKFold(CITY)` 下的 RMSE 上升约 `24.13%`，在 `GroupKFold(PROVINCE)` 下上升约 `40.53%`，说明地理泄漏确实显著存在；也说明本项目的主结论不是“随机划分高分”，而是“严格地理协议下谁更稳健”。

## 仓库结构

- `data/`
  - 多源原始数据目录，当前包含 `Australia.csv`、`Brazil.xlsx`、`China.xlsx`、`EU.xlsx`、`USA.xlsx`
- `src/pm25_geopfnmix/`
  - 核心 Python 包
- `scripts/run_eda.py`
  - 数据审计、描述统计和基础可视化
- `scripts/run_baselines.py`
  - 跑通统一特征工程下的多组基线模型
- `scripts/run_geopfnmix.py`
  - 运行 GeoPFNMix 家族模型、补充变体、消融实验和显著性检验
- `scripts/run_analysis_assets.py`
  - 基于实验结果生成论文用图表、分析表、最佳模型按数据源拆分表现，以及“最优两个 GeoPFNMix 模型 + CatBoost + TabPFN”的来源对照
- `artifacts/figures/`
  - 自动生成的图像输出
- `artifacts/tables/`
  - 自动生成的结果表、OOF 预测和统计检验结果
- `reports/实验日志.md`
  - 实验记录
- `reports/项目日记.md`
  - 项目决策与推进记录
- `paper/毕业论文定稿.md`
  - 当前论文提交版

## 数据字段说明

标准化后的建模数据共 14 列：

- 标识列：`OBJECTID_1`
- 类别列：`COUNTRY`、`PROVINCE`、`CITY`、`COUNTY`
- 数值列：`AET`、`ppt`、`tem`、`wind`、`NOX`、`SO2`、`fertilzier`、`manure`
- 目标列：`PM2.5`

建模时：

- `OBJECTID_1` 不进入模型
- `COUNTY` 不作为主类别特征参与编码
- 主要类别特征保留 `COUNTRY`、`PROVINCE` 和 `CITY`

这样做是为了在保留跨域来源信息的同时，降低“最细粒度地理 ID”带来的伪记忆问题。

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
python scripts/run_baselines.py --device auto
python scripts/run_geopfnmix.py --device auto
python scripts/run_analysis_assets.py
```

各脚本作用如下：

1. `run_eda.py`
   - 输出数据审计摘要、描述统计、目标分布图、数值相关图、二级地理组大小分布图
2. `run_baselines.py`
  - 跑 `Ridge`、`RandomForest`、`HistGBDT`、`XGBoost`、`LightGBM`、`CatBoost`、`TabPFN`
  - 在 `RandomKFold`、`GroupKFold(CITY)`、`GroupKFold(PROVINCE)` 三套协议下输出结果
  - 支持 `--device auto/cpu/gpu`
3. `run_geopfnmix.py`
  - 运行 GeoPFNMix 家族模型与显著性检验
  - 当前也包含补充变体 `GeoPFNMix-Lite-CatBoost` 与 `GeoPFNMix-Lite-CatBoost-TabPFN`
  - 支持 `--device auto/cpu/gpu`
4. `run_analysis_assets.py`
   - 读取前面脚本生成的表格，自动绘制论文图表、补充分析表、最佳模型按数据源拆分表现，以及“最优两个 GeoPFNMix 模型 + CatBoost + TabPFN”的按来源对照

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
- `artifacts/figures/best_model_country_metrics.png`
- `artifacts/tables/best_model_country_metrics.csv`
- `artifacts/figures/selected_models_country_metrics.png`
- `artifacts/tables/selected_models_country_metrics.csv`
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

model = make_model("geopfnmix_catboost_tabpfn", device_preference="auto")
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
- `geopfnmix_lite_catboost`
- `geopfnmix_lite_catboost_tabpfn`
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
   - 本仓库会优先从根目录 `HF_token.py` 中导入本地令牌
5. 轻量 stacking
   - 使用低容量线性模型融合多路专家输出

## 当前推荐模型配置

如果你已经完成 `TabPFN` 无人值守授权，建议优先使用：

```python
model = make_model("geopfnmix_catboost_tabpfn", device_preference="gpu")
```

它对应：

- 全局专家：`CatBoost`
- 残差分支：开启
- 先验专家：`TabPFN`
- 融合器：线性 stacking

如果你暂时不使用 `TabPFN`，建议使用：

```python
model = make_model("geopfnmix_catboost")
```

它对应：

- 全局专家：`CatBoost`
- 残差分支：开启
- 先验专家：`LightGBM`
- 融合器：线性 stacking

如果你想要更简洁的 RF 骨干对照配置，可以使用：

```python
model = make_model("geopfnmix_no_residual")
```

也就是论文中常提到的 `GeoPFNMix-Lite`。

如果你想补测“保留 Lite 结构，但把全局专家从 `RF` 提升到 `CatBoost`”这一问题，可以使用：

```python
model = make_model("geopfnmix_lite_catboost")
```

它对应：

- 全局专家：`CatBoost`
- 残差分支：关闭
- 先验专家：`LightGBM`
- 融合器：线性 stacking

这个补充变体主要用于回答：`Lite` 的收益究竟更多来自“低复杂度结构本身”，还是也会显著受益于更强全局专家。

## 无人值守运行注意事项

本仓库当前已经支持 `TabPFN` 的无人值守执行，策略如下：

- 根目录允许放置一个本地私有 `HF_token.py`
  - 可提供 `HF_TOKEN`
  - 也可同时提供 `TABPFN_TOKEN`
- `HF_token.py` 已加入 `.gitignore`
  - 不会被默认提交到 Git
- 代码仍兼容早期 `token.py`
  - 但推荐统一迁移到 `HF_token.py`
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

- 正是为了避免与 Python 标准库 `token` 同名冲突，推荐使用 `HF_token.py`
- 日常复现实验请优先使用 `python scripts/run_*.py`
- 如果必须写临时探针，建议显式清理 `sys.path` 中的仓库根目录后再导入标准库相关模块

## 复现与环境说明

- 结果会受到 Python 版本、库版本和运行平台的影响
- `reports/实验日志.md` 中保留了项目早期环境记录和当前本地重跑记录
- 当前仓库中的图表、表格和论文文稿，建议一律以 `artifacts/` 中最新文件为准

## 论文与文档对应关系

- 论文正文：`paper/毕业论文定稿.md`
- 实验日志：`reports/实验日志.md`
- 决策与推进记录：`reports/项目日记.md`
- 项目阶段性评估：`reports/项目瓶颈、困难与改进方向分析.md`

如果后续你要继续扩展毕业设计，最直接的方向是：

- 增加更严格的空间泛化协议
- 引入经纬度、土地利用、交通或遥感等外部空间特征
- 继续研究 `TabPFN` 先验与更复杂残差结构之间的适配关系
- 为最终模型增加部署脚本或简单演示界面
