# 县域农业数据 PM2.5 预测项目

本项目面向毕业设计任务：基于 `data.csv` 中的县域农业与环境特征，构建空气 `PM2.5` 含量预测模型，并输出可复现的实验、日志、图表与论文草稿。

## 当前方案

- 主任务：表格回归
- 主评估协议：`GroupKFold(CITY)`，用于抑制地理泄漏
- 强基线：Ridge / RandomForest / HistGBDT / XGBoost / LightGBM / CatBoost
- 创新模型：`GeoPFNMix`
  - 全局非线性专家：CatBoost
  - 地理层级残差校正：省-市两级经验贝叶斯残差校准
  - 先验专家：优先使用 TabPFN；若当前环境无 `TABPFN_TOKEN`，自动回退为 LightGBM 专家
  - 动态融合：基于 OOF 元特征的轻量 stacking

## 目录

- `src/pm25_geopfnmix/`: 核心代码
- `scripts/run_eda.py`: 数据审计与图表
- `scripts/run_baselines.py`: 基线实验
- `scripts/run_geopfnmix.py`: 创新模型、消融与显著性检验
- `reports/实验日志.md`: 关键实验记录
- `reports/项目日记.md`: 决策与推进日志
- `paper/毕业论文初稿.md`: 论文草稿
- `artifacts/`: 结果表、图、模型与中间输出

## 运行方式

```bash
python -m pip install -e .
python scripts/run_eda.py
python scripts/run_baselines.py
python scripts/run_geopfnmix.py
```

## TabPFN 说明

`tabpfn>=7` 在无头环境中通常需要通过 `TABPFN_TOKEN` 完成授权后才能下载模型。如果环境变量不存在，项目会自动跳过 TabPFN 可执行实验，并回退到开放可运行的专家模型，以保证实验流程不中断。
